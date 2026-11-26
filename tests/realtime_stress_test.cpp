// SPDX-License-Identifier: Apache-2.0
// Deterministic CPU-only stress for the production realtime voice controller.
// Fake devices and sockets provide scheduled boundaries; VoiceClient and
// VoicePlayback retain their production threading, ACK and cancellation code.
#include "client/voice.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "client/http.hpp"

namespace {

using namespace umminfer::client;
using namespace umminfer::client::detail;
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
using PcmCallback = VoiceMicrophone::PcmCallback;
using ErrorCallback = VoiceMicrophone::ErrorCallback;
using Completion = VoiceAudioOutput::Completion;

struct Options {
  uint64_t seed = 0x5eed38;
  size_t iterations = 3;
  size_t sessions = 6;
  bool trace = false;
};

struct Context {
  mutable std::mutex mu;
  uint64_t seed = 0;
  size_t iteration = 0;
  size_t round = 0;
  size_t session = 0;
  size_t stage = 0;
  std::deque<std::string> recent;

  void action(uint64_t seed_value, size_t iteration_value, size_t round_value,
              size_t session_value, size_t stage_value, std::string label) {
    std::lock_guard lock(mu);
    seed = seed_value;
    iteration = iteration_value;
    round = round_value;
    session = session_value;
    stage = stage_value;
    std::ostringstream line;
    line << "seed=" << seed_value << " iteration=" << iteration_value
         << " round=" << round_value << " session=" << session_value
         << " stage=" << stage_value << " action=" << label;
    recent.push_back(line.str());
    if (recent.size() > 32) recent.pop_front();
  }

  std::string diagnostic(const std::string& message) const {
    std::lock_guard lock(mu);
    std::ostringstream out;
    out << message << " [seed=" << seed << " iteration=" << iteration
        << " round=" << round << " session=" << session << " stage=" << stage
        << "]\nrecent deterministic actions (oldest first):\n";
    for (const std::string& line : recent) out << line << '\n';
    return out.str();
  }
};

Context g_context;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(g_context.diagnostic(message));
}

void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

void wait_until(const std::function<bool()>& predicate, const std::string& message) {
  const auto deadline = Clock::now() + std::chrono::seconds(10);
  while (!predicate()) {
    if (Clock::now() >= deadline) fail("timeout: " + message);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void quiet_millisecond_boundary() {
  // A negative assertion needs a scheduling opportunity. Positive ordering is
  // always established by a predicate, never by this bounded quiet window.
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}
  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  size_t index(size_t bound) { return static_cast<size_t>(next() % bound); }
 private:
  uint64_t state_;
};

uint64_t trace_mix(uint64_t hash, std::string_view text) {
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

std::string even_bytes(std::string value) {
  if (value.size() % 2 != 0) value.push_back('.');
  return value;
}

json pcm_delta(const std::string& response, const std::string& bytes,
               uint64_t cursor) {
  return {{"type", "audio.delta"},
          {"response_id", response},
          {"format", "pcm16"},
          {"sample_rate", 16000},
          {"channels", 1},
          {"audio", base64_encode(bytes)},
          {"text_cursor", cursor}};
}

class ScriptedSocket;

class SocketHandle final : public VoiceSocket {
 public:
  explicit SocketHandle(std::shared_ptr<ScriptedSocket> socket)
      : socket_(std::move(socket)) {}
  void send_text(std::string_view text) override;
  void run(const std::function<void(std::string_view)>& on_message) override;
  void close() noexcept override;
 private:
  std::shared_ptr<ScriptedSocket> socket_;
};

class ScriptedSocket final : public std::enable_shared_from_this<ScriptedSocket> {
 public:
  std::unique_ptr<VoiceSocket> handle() {
    return std::make_unique<SocketHandle>(shared_from_this());
  }

  void send_text(std::string_view text) {
    std::lock_guard lock(mu_);
    sent_.push_back(std::string(text));
    cv_.notify_all();
  }

  void run(const std::function<void(std::string_view)>& on_message) {
    std::unique_lock lock(mu_);
    running_ = true;
    cv_.notify_all();
    while (!closed_) {
      if (!incoming_.empty()) {
        std::string message = std::move(incoming_.front());
        incoming_.pop_front();
        lock.unlock();
        on_message(message);
        lock.lock();
        ++processed_;
        cv_.notify_all();
        continue;
      }
      if (disconnected_) {
        throw Error(ErrorCode::kNetwork, "scheduled disconnect", true);
      }
      cv_.wait_for(lock, std::chrono::milliseconds(2));
    }
  }

  void close() noexcept {
    std::lock_guard lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  size_t emit(const json& event) {
    std::lock_guard lock(mu_);
    check(!closed_, "attempted to schedule an event on a closed socket");
    incoming_.push_back(event.dump());
    ++scheduled_;
    cv_.notify_all();
    return scheduled_;
  }

  void emit_wait(const json& event) {
    const size_t ticket = emit(event);
    std::unique_lock lock(mu_);
    check(cv_.wait_for(lock, std::chrono::seconds(10),
                       [&] { return processed_ >= ticket; }),
          "socket event was not processed");
  }

  void disconnect() {
    std::lock_guard lock(mu_);
    disconnected_ = true;
    cv_.notify_all();
  }

  std::vector<json> messages() const {
    std::lock_guard lock(mu_);
    std::vector<json> out;
    for (const std::string& text : sent_) out.push_back(json::parse(text));
    return out;
  }

  size_t sent_count() const {
    std::lock_guard lock(mu_);
    return sent_.size();
  }

  bool closed() const {
    std::lock_guard lock(mu_);
    return closed_;
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::string> sent_;
  std::deque<std::string> incoming_;
  size_t scheduled_ = 0;
  size_t processed_ = 0;
  bool running_ = false;
  bool closed_ = false;
  bool disconnected_ = false;
};

void SocketHandle::send_text(std::string_view text) { socket_->send_text(text); }
void SocketHandle::run(const std::function<void(std::string_view)>& on_message) {
  socket_->run(on_message);
}
void SocketHandle::close() noexcept { socket_->close(); }

class ScriptedMicrophone;

class MicrophoneHandle final : public VoiceMicrophone {
 public:
  explicit MicrophoneHandle(std::shared_ptr<ScriptedMicrophone> microphone)
      : microphone_(std::move(microphone)) {}
  void start(PcmCallback on_pcm, ErrorCallback on_error) override;
  void stop() noexcept override;
 private:
  std::shared_ptr<ScriptedMicrophone> microphone_;
};

class ScriptedMicrophone final
    : public std::enable_shared_from_this<ScriptedMicrophone> {
 public:
  std::unique_ptr<VoiceMicrophone> handle() {
    return std::make_unique<MicrophoneHandle>(shared_from_this());
  }

  void start(PcmCallback on_pcm, ErrorCallback on_error) {
    std::lock_guard lock(mu_);
    ++starts_;
    callback_ = std::move(on_pcm);
    error_callback_ = std::move(on_error);
    stale_callback_ = callback_;
  }

  void stop() noexcept {
    std::lock_guard lock(mu_);
    ++stops_;
    callback_ = nullptr;
    error_callback_ = nullptr;
  }

  void emit(std::string bytes, bool stale = false) {
    PcmCallback callback;
    {
      std::lock_guard lock(mu_);
      callback = stale ? stale_callback_ : callback_;
    }
    if (callback) callback(std::move(bytes));
  }

  int starts() const {
    std::lock_guard lock(mu_);
    return starts_;
  }
  int stops() const {
    std::lock_guard lock(mu_);
    return stops_;
  }

 private:
  mutable std::mutex mu_;
  PcmCallback callback_;
  PcmCallback stale_callback_;
  ErrorCallback error_callback_;
  int starts_ = 0;
  int stops_ = 0;
};

void MicrophoneHandle::start(PcmCallback on_pcm, ErrorCallback on_error) {
  microphone_->start(std::move(on_pcm), std::move(on_error));
}
void MicrophoneHandle::stop() noexcept { microphone_->stop(); }

class ScriptedOutput;

class OutputHandle final : public VoiceAudioOutput {
 public:
  explicit OutputHandle(std::shared_ptr<ScriptedOutput> output)
      : output_(std::move(output)) {}
  void enqueue(VoicePcmChunk chunk, Completion completion) override;
  void stop() noexcept override;
  void close() noexcept override;
 private:
  std::shared_ptr<ScriptedOutput> output_;
};

class ScriptedOutput final : public std::enable_shared_from_this<ScriptedOutput> {
 public:
  struct Item {
    VoicePcmChunk chunk;
    Completion completion;
    size_t callbacks = 0;
  };

  std::unique_ptr<VoiceAudioOutput> handle() {
    return std::make_unique<OutputHandle>(shared_from_this());
  }

  void enqueue(VoicePcmChunk chunk, Completion completion) {
    std::lock_guard lock(mu_);
    check(!closed_, "audio was enqueued after output close");
    items_.push_back(Item{std::move(chunk), std::move(completion), 0});
    cv_.notify_all();
  }

  void stop() noexcept {
    std::lock_guard lock(mu_);
    ++stops_;
  }

  void close() noexcept {
    std::lock_guard lock(mu_);
    ++closes_;
    closed_ = true;
  }

  void complete(size_t index, bool consumed = true) {
    Completion completion;
    {
      std::lock_guard lock(mu_);
      check(index < items_.size(), "device completion index is out of range");
      ++items_[index].callbacks;
      completion = items_[index].completion;
    }
    if (completion) completion(consumed);
  }

  size_t size() const {
    std::lock_guard lock(mu_);
    return items_.size();
  }
  int stops() const {
    std::lock_guard lock(mu_);
    return stops_;
  }
  int closes() const {
    std::lock_guard lock(mu_);
    return closes_;
  }
  std::vector<std::string> pcm() const {
    std::lock_guard lock(mu_);
    std::vector<std::string> out;
    for (const Item& item : items_) out.push_back(item.chunk.pcm);
    return out;
  }
  size_t callback_count(size_t index) const {
    std::lock_guard lock(mu_);
    return items_.at(index).callbacks;
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Item> items_;
  int stops_ = 0;
  int closes_ = 0;
  bool closed_ = false;
};

void OutputHandle::enqueue(VoicePcmChunk chunk, Completion completion) {
  output_->enqueue(std::move(chunk), std::move(completion));
}
void OutputHandle::stop() noexcept { output_->stop(); }
void OutputHandle::close() noexcept { output_->close(); }

struct Session final {
  size_t index;
  size_t iteration;
  bool echo;
  std::string tag;
  VoiceConfig config;
  std::unique_ptr<VoiceClient> client;
  std::thread thread;
  std::exception_ptr failure;

  mutable std::mutex factory_mu;
  std::vector<std::shared_ptr<ScriptedSocket>> sockets;
  std::vector<std::shared_ptr<ScriptedMicrophone>> microphones;
  std::vector<std::shared_ptr<ScriptedOutput>> outputs;

  mutable std::mutex event_mu;
  std::vector<std::string> transcripts;
  std::vector<std::string> completions;
  std::vector<std::string> response_text;
  std::vector<VoiceBargeInCutPoint> cut_points;
  size_t stage = 0;
  size_t cancelled_ack_count = 0;
  size_t cancelled_output_count = 0;
  size_t cancelled_text_count = 0;

  Session(size_t index_value, size_t iteration_value, bool echo_value)
      : index(index_value), iteration(iteration_value), echo(echo_value) {
    std::ostringstream name;
    name << 's' << std::setfill('0') << std::setw(2) << index << 'i'
         << std::setw(4) << iteration;
    tag = name.str();
    config.base_url = "http://127.0.0.1:8123";
    config.model = "stress-voice";
    config.echo_suppression = echo;
    config.reconnect_attempts = 2;
    config.reconnect_delay = Milliseconds{0};
    VoiceHandlers handlers;
    handlers.on_transcript = [this](const std::string& text) {
      std::lock_guard lock(event_mu);
      transcripts.push_back(text);
    };
    handlers.on_response_completed = [this](const std::string& id) {
      std::lock_guard lock(event_mu);
      completions.push_back(id);
    };
    handlers.on_response_text = [this](const std::string& text) {
      std::lock_guard lock(event_mu);
      response_text.push_back(text);
    };
    handlers.on_response_cancelled = [this](const VoiceBargeInCutPoint& point) {
      std::lock_guard lock(event_mu);
      cut_points.push_back(point);
    };
    client = std::make_unique<VoiceClient>(
        config, std::move(handlers),
        [this](const std::string&, Milliseconds, CancelFlag, CancelFlag external,
               const std::string&) {
          auto value = std::make_shared<ScriptedSocket>();
          {
            std::lock_guard lock(factory_mu);
            sockets.push_back(value);
          }
          return value->handle();
        },
        [this](const std::string&) {
          auto value = std::make_shared<ScriptedMicrophone>();
          {
            std::lock_guard lock(factory_mu);
            microphones.push_back(value);
          }
          return value->handle();
        },
        [this](const std::string&) {
          auto value = std::make_shared<ScriptedOutput>();
          {
            std::lock_guard lock(factory_mu);
            outputs.push_back(value);
          }
          return value->handle();
        });
  }

  ~Session() {
    if (thread.joinable()) client->request_stop();
    if (thread.joinable()) thread.join();
  }

  std::shared_ptr<ScriptedSocket> socket(size_t connection) const {
    std::lock_guard lock(factory_mu);
    return sockets.at(connection);
  }
  std::shared_ptr<ScriptedMicrophone> microphone(size_t connection) const {
    std::lock_guard lock(factory_mu);
    return microphones.at(connection);
  }
  std::shared_ptr<ScriptedOutput> output(size_t connection) const {
    std::lock_guard lock(factory_mu);
    return outputs.at(connection);
  }
  size_t socket_count() const {
    std::lock_guard lock(factory_mu);
    return sockets.size();
  }
  size_t output_count() const {
    std::lock_guard lock(factory_mu);
    return outputs.size();
  }

  std::string response(size_t generation, size_t connection = 0) const {
    return tag + "-c" + std::to_string(connection) + "-r" +
           std::to_string(generation);
  }
  std::string answer(size_t generation) const {
    return tag + " answer " + std::to_string(generation) + " acknowledged prefix";
  }
  std::string far(size_t generation) const {
    return even_bytes(tag + "-far-" + std::to_string(generation));
  }

  void start() {
    thread = std::thread([this] {
      try {
        client->run();
      } catch (...) {
        failure = std::current_exception();
      }
    });
    wait_until([&] { return socket_count() == 1 && socket(0)->sent_count() == 1; },
               "initial session.update");
    const json update = socket(0)->messages().at(0);
    const auto& turn = update["session"]["turn_detection"];
    check(turn.value("echo_suppression", true) == echo,
          "initial echo mode differs from the scheduled mode");
    check(echo || turn.contains("echo_suppression"),
          "echo-disabled session omitted its explicit false opt-in");
    check(!echo || !turn.contains("echo_suppression"),
          "default echo-enabled wire shape gained a key");
    ready(0);
  }

  void ready(size_t connection) {
    wait_until([&] { return socket_count() > connection; }, "scheduled reconnect");
    json turn = {{"type", "server_vad"}, {"silence_ms", 400}};
    if (!echo) turn["echo_suppression"] = false;
    socket(connection)->emit_wait(
        {{"type", "transcription_session.updated"},
         {"session", {{"turn_detection", turn},
                      {"audio_output", {{"playback_ack", true}}}}}});
    wait_until([&] { return microphone(connection)->starts() == 1; },
               "microphone start after negotiation");
  }

  void toggle_echo() {
    echo = !echo;
    client->set_echo_suppression(echo);
    wait_until([&] { return socket(0)->sent_count() >= 2; },
               "active echo toggle update");
    json turn = {{"type", "server_vad"}, {"silence_ms", 400}};
    if (!echo) turn["echo_suppression"] = false;
    socket(0)->emit_wait(
        {{"type", "transcription_session.updated"},
         {"session", {{"turn_detection", turn},
                      {"audio_output", {{"playback_ack", true}}}}}});
    check(client->confirmed_echo_suppression() == std::optional<bool>(echo),
          "active echo toggle was not confirmed");
  }

  void asr_final(const std::string& user) {
    socket(0)->emit_wait({{"type", "transcription.segment"},
                          {"text", user}, {"is_final", true}});
    socket(0)->emit_wait({{"type", "transcription.completed"}, {"text", user}});
    std::lock_guard lock(event_mu);
    check(std::count(transcripts.begin(), transcripts.end(), user) == 1,
          "ASR final was duplicated or lost");
  }

  void append_microphone(size_t connection, const std::string& bytes,
                         bool stale = false) {
    microphone(connection)->emit(even_bytes(bytes), stale);
  }

  std::vector<json> of_type(size_t connection, const std::string& type) const {
    std::vector<json> out;
    for (const json& event : socket(connection)->messages()) {
      if (event.value("type", "") == type) out.push_back(event);
    }
    return out;
  }

  size_t ack_count(size_t connection, const std::string& id) const {
    size_t count = 0;
    for (const json& event : of_type(connection, "output_audio_buffer.playback_ack")) {
      if (event.value("response_id", "") == id) ++count;
    }
    return count;
  }

  void wait_ack(size_t connection, const std::string& id, uint64_t cursor,
                size_t count) {
    wait_until([&] {
      size_t matches = 0;
      for (const json& event : of_type(connection, "output_audio_buffer.playback_ack")) {
        if (event.value("response_id", "") == id &&
            event.value("text_cursor", uint64_t{0}) == cursor) {
          ++matches;
        }
      }
      return matches == count;
    }, "expected consumed-playback ACK");
  }

  size_t completion_count(const std::string& id) const {
    std::lock_guard lock(event_mu);
    return static_cast<size_t>(std::count(completions.begin(), completions.end(), id));
  }
  size_t text_count() const {
    std::lock_guard lock(event_mu);
    return response_text.size();
  }
  size_t cut_point_count() const {
    std::lock_guard lock(event_mu);
    return cut_points.size();
  }

  void step() {
    const std::string r0 = response(0);
    const std::string r1 = response(1);
    const std::string r2 = response(2);
    const std::string r3 = response(3, 1);
    const std::string r4 = response(4, 1);
    switch (stage) {
      case 0: {
        append_microphone(0, tag + "-mic-zero");
        wait_until([&] { return of_type(0, "input_audio_buffer.append").size() == 1; },
                   "first microphone append");
        const json appended = of_type(0, "input_audio_buffer.append").at(0);
        check(base64_decode(appended["audio"].get<std::string>(), 4096) ==
                  even_bytes(tag + "-mic-zero"),
              "microphone bytes changed or crossed sessions");
        asr_final(tag + " user zero");
        break;
      }
      case 1: {
        socket(0)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r0}, {"text", answer(0)}});
        socket(0)->emit_wait(pcm_delta(r0, far(0), 0));
        socket(0)->emit_wait(pcm_delta(r0, far(10), 4));
        wait_until([&] { return output(0)->size() == 2; }, "two far-end chunks");
        break;
      }
      case 2: {
        socket(0)->emit_wait({{"type", "audio.done"}, {"response_id", r0},
                              {"text_cursor", answer(0).size()}});
        const json complete = {{"type", "response.completed"}, {"response_id", r0}};
        socket(0)->emit_wait(complete);
        socket(0)->emit_wait(complete);  // delayed duplicate transport frame
        check(completion_count(r0) == 1,
              "duplicate response.completed produced a double completion");
        break;
      }
      case 3: {
        output(0)->complete(1);  // delayed out-of-order device callback
        quiet_millisecond_boundary();
        check(ack_count(0, r0) == 0,
              "out-of-order far-end completion acknowledged the queue head");
        output(0)->complete(0);
        wait_ack(0, r0, 0, 1);
        wait_ack(0, r0, 4, 1);
        wait_ack(0, r0, answer(0).size(), 1);
        output(0)->complete(0);  // duplicate device callback
        quiet_millisecond_boundary();
        check(ack_count(0, r0) == 3, "duplicate device callback duplicated an ACK");
        toggle_echo();
        break;
      }
      case 4: {
        for (int repeat = 0; repeat < 2; ++repeat) {
          socket(0)->emit_wait({{"type", "response.playback_ack"},
                                {"response_id", r0}, {"text_cursor", 4},
                                {"final", false}});
        }
        socket(0)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r1}, {"text", answer(1)}});
        socket(0)->emit_wait(pcm_delta(r1, far(1), 3));
        wait_until([&] { return output(0)->size() == 3; }, "barge-in far-end chunk");
        break;
      }
      case 5: {
        output(0)->complete(2);
        wait_ack(0, r1, 3, 1);
        for (uint64_t cursor : {uint64_t{3}, uint64_t{3}, uint64_t{0}}) {
          socket(0)->emit_wait({{"type", "response.playback_ack"},
                                {"response_id", r1}, {"text_cursor", cursor},
                                {"final", false}});
        }
        break;
      }
      case 6: {
        socket(0)->emit_wait({{"type", "input_audio_buffer.speech_started"},
                              {"interrupted_response_id", r1}});
        check(output(0)->stops() >= 1, "speech start did not stop old playback");
        const json cancelled = {{"type", "response.cancelled"},
                                {"response_id", r1}, {"reason", "barge_in"},
                                {"text_cursor", 3},
                                {"acknowledged_text", answer(1).substr(0, 3)}};
        socket(0)->emit_wait(cancelled);
        socket(0)->emit_wait(cancelled);  // repeated server cancellation
        check(cut_point_count() == 1 && client->barge_in_count() == 1,
              "repeated response.cancelled was not idempotent");
        {
          std::lock_guard lock(event_mu);
          check(cut_points.at(0).response_id == r1 &&
                    cut_points.at(0).spoken_bytes == 3,
                "barge-in cut point differs from the acknowledged prefix");
        }
        cancelled_ack_count = ack_count(0, r1);
        cancelled_output_count = output(0)->size();
        cancelled_text_count = text_count();
        break;
      }
      case 7: {
        socket(0)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r1}, {"text", "STALE"}});
        socket(0)->emit_wait(pcm_delta(r1, far(99), 99));
        socket(0)->emit_wait({{"type", "audio.done"}, {"response_id", r1},
                              {"text_cursor", 99}});
        socket(0)->emit_wait({{"type", "response.completed"}, {"response_id", r1}});
        output(0)->complete(2);
        quiet_millisecond_boundary();
        check(output(0)->size() == cancelled_output_count &&
                  text_count() == cancelled_text_count &&
                  ack_count(0, r1) == cancelled_ack_count &&
                  completion_count(r1) == 0,
              "stale source emitted text/audio/ACK/completion after cancellation");
        break;
      }
      case 8: {
        socket(0)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r2}, {"text", answer(2)}});
        socket(0)->emit_wait(pcm_delta(r2, far(2), 2));
        wait_until([&] { return output(0)->size() == 4; }, "pre-reconnect audio");
        break;
      }
      case 9: {
        socket(0)->disconnect();
        wait_until([&] { return socket_count() == 2 && output_count() == 2; },
                   "reconnect and replacement devices");
        check(microphone(0)->stops() == 1 && output(0)->closes() == 1 &&
                  socket(0)->closed(),
              "reconnect did not destroy the old connection resources");
        ready(1);
        break;
      }
      case 10: {
        const size_t old_sent = socket(0)->sent_count();
        append_microphone(0, tag + "-stale-mic", true);
        output(0)->complete(3);
        quiet_millisecond_boundary();
        check(socket(0)->sent_count() == old_sent,
              "stale microphone/device callback emitted after reconnect");
        check(of_type(1, "input_audio_buffer.append").empty() &&
                  of_type(1, "output_audio_buffer.playback_ack").empty(),
              "reconnect replayed old microphone audio or ACKs");
        append_microphone(1, tag + "-mic-new");
        wait_until([&] { return of_type(1, "input_audio_buffer.append").size() == 1; },
                   "new microphone data after reconnect");
        break;
      }
      case 11: {
        const std::string user = tag + " user after reconnect";
        socket(1)->emit_wait({{"type", "transcription.completed"}, {"text", user}});
        socket(1)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r3}, {"text", answer(3)}});
        socket(1)->emit_wait(pcm_delta(r3, far(3), 7));
        wait_until([&] { return output(1)->size() == 1; }, "post-reconnect far-end");
        break;
      }
      case 12: {
        socket(1)->emit_wait({{"type", "audio.done"}, {"response_id", r3},
                              {"text_cursor", answer(3).size()}});
        socket(1)->emit_wait({{"type", "response.completed"}, {"response_id", r3}});
        output(1)->complete(0);
        wait_ack(1, r3, 7, 1);
        wait_ack(1, r3, answer(3).size(), 1);
        socket(1)->emit_wait({{"type", "response.playback_ack"},
                              {"response_id", r3},
                              {"text_cursor", answer(3).size()}, {"final", true}});
        socket(1)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r3}, {"text", "LATE AFTER FINAL"}});
        check(completion_count(r3) == 1,
              "post-reconnect response completed other than exactly once");
        break;
      }
      case 13: {
        socket(1)->emit_wait({{"type", "response.text.delta"},
                              {"response_id", r4}, {"text", answer(4)}});
        socket(1)->emit_wait(pcm_delta(r4, far(4), 11));
        wait_until([&] { return output(1)->size() == 2; },
                   "teardown audio left in flight");
        break;
      }
      default:
        break;
    }
    ++stage;
  }

  bool done() const { return stage >= 14; }

  void stop_with_inflight_callback() {
    const std::string r4 = response(4, 1);
    client->request_stop();
    if (thread.joinable()) thread.join();
    check(failure == nullptr, "scheduled Stop Voice returned an exception");
    const size_t acknowledgements = ack_count(1, r4);
    output(1)->complete(1);
    append_microphone(1, tag + "-post-stop", true);
    quiet_millisecond_boundary();
    check(ack_count(1, r4) == acknowledgements,
          "late callback emitted an ACK after teardown");
    check(microphone(1)->stops() == 1 && output(1)->closes() == 1 &&
              socket(1)->closed(),
          "teardown did not close microphone, playback and socket");
  }

  void validate_isolation() const {
    std::map<std::string, uint64_t> last_cursor;
    for (size_t connection = 0; connection < socket_count(); ++connection) {
      for (const json& event : socket(connection)->messages()) {
        const std::string type = event.value("type", "");
        if (type == "output_audio_buffer.playback_ack") {
          const std::string id = event.at("response_id").get<std::string>();
          check(id.rfind(tag + "-", 0) == 0, "ACK response id crossed sessions");
          const uint64_t value = event.at("text_cursor").get<uint64_t>();
          auto previous = last_cursor.find(id);
          check(previous == last_cursor.end() || value >= previous->second,
                "ACK cursor moved backwards");
          last_cursor[id] = value;
        } else if (type == "input_audio_buffer.append") {
          const std::string bytes =
              base64_decode(event.at("audio").get<std::string>(), 4096);
          check(bytes.rfind(tag, 0) == 0, "microphone bytes crossed sessions");
        }
      }
      for (const std::string& bytes : output(connection)->pcm()) {
        check(bytes.rfind(tag, 0) == 0, "far-end PCM crossed sessions");
      }
    }
    std::lock_guard lock(event_mu);
    for (const std::string& id : completions) {
      check(id.rfind(tag + "-", 0) == 0, "completion crossed sessions");
    }
    for (const std::string& text : transcripts) {
      check(text.rfind(tag, 0) == 0, "ASR final crossed sessions");
    }
  }
};

void test_bounded_playback_queue_and_inflight_close() {
  auto owned = std::make_shared<ScriptedOutput>();
  std::mutex ack_mu;
  std::vector<std::pair<std::string, uint64_t>> acknowledgements;
  {
    VoicePlayback playback(owned->handle(),
                           [&](const std::string& id, uint64_t cursor) {
      std::lock_guard lock(ack_mu);
      acknowledgements.emplace_back(id, cursor);
    }, nullptr);
    for (size_t i = 0; i < 4096; ++i) {
      playback.push("bounded", VoicePcmChunk{std::string(2, '\x01'), 16000, 1}, i);
    }
    bool rejected = false;
    try {
      playback.push("bounded", VoicePcmChunk{std::string(2, '\x02'), 16000, 1}, 4096);
    } catch (const Error& error) {
      rejected = error.code() == ErrorCode::kBadResponse;
    }
    check(rejected && playback.pending_count() == 4096,
          "saturated playback queue did not fail closed at its fixed bound");
    playback.cancel("bounded");
    check(playback.pending_count() == 0, "cancellation did not drain a saturated queue");
    owned->complete(0);
    quiet_millisecond_boundary();
    std::lock_guard lock(ack_mu);
    check(acknowledgements.empty(), "saturated cancelled queue emitted an ACK");
  }

  auto blocking = std::make_shared<ScriptedOutput>();
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool entered = false;
  bool release = false;
  std::atomic<bool> close_returned{false};
  VoicePlayback playback(blocking->handle(),
                         [&](const std::string&, uint64_t) {
    std::unique_lock lock(gate_mu);
    entered = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [&] { return release; });
  }, nullptr);
  playback.push("close-inflight", VoicePcmChunk{std::string(2, '\x03'), 16000, 1}, 5);
  blocking->complete(0);
  {
    std::unique_lock lock(gate_mu);
    check(gate_cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }),
          "ACK sink did not enter the in-flight close gate");
  }
  std::thread closer([&] {
    playback.close();
    close_returned = true;
  });
  quiet_millisecond_boundary();
  check(!close_returned, "close returned while its ACK callback was still in flight");
  {
    std::lock_guard lock(gate_mu);
    release = true;
  }
  gate_cv.notify_all();
  closer.join();
  check(close_returned && blocking->closes() == 1,
        "in-flight close did not retire exactly one output");
  blocking->complete(0);
  playback.close();
  check(blocking->closes() == 1, "repeated close closed the output twice");
}

struct RunSummary {
  uint64_t trace = 0xcbf29ce484222325ULL;
  size_t actions = 0;
};

RunSummary run_iteration(const Options& options, size_t iteration) {
  RunSummary summary;
  const uint64_t seed = options.seed + iteration * 0x9e3779b97f4a7c15ULL;
  SplitMix64 random(seed);
  std::vector<std::unique_ptr<Session>> sessions;
  for (size_t index = 0; index < options.sessions; ++index) {
    sessions.push_back(std::make_unique<Session>(
        index, iteration, ((index + iteration) % 2) == 0));
  }
  for (size_t index = 0; index < sessions.size(); ++index) {
    g_context.action(seed, iteration, 0, index, 0, "start+negotiate");
    sessions[index]->start();
  }

  size_t round = 0;
  while (true) {
    std::vector<size_t> order;
    for (size_t index = 0; index < sessions.size(); ++index) {
      if (!sessions[index]->done()) order.push_back(index);
    }
    if (order.empty()) break;
    for (size_t i = order.size(); i > 1; --i) {
      std::swap(order[i - 1], order[random.index(i)]);
    }
    for (const size_t index : order) {
      Session& session = *sessions[index];
      const size_t stage = session.stage;
      std::ostringstream label;
      label << "session=" << index << " stage=" << stage;
      g_context.action(seed, iteration, round, index, stage, label.str());
      summary.trace = trace_mix(summary.trace, label.str());
      session.step();
      ++summary.actions;
      if (options.trace) {
        std::cout << "TRACE seed=" << seed << " iteration=" << iteration
                  << " round=" << round << " session=" << index
                  << " stage=" << stage << '\n';
      }
    }
    ++round;
  }

  std::vector<size_t> teardown(sessions.size());
  for (size_t i = 0; i < teardown.size(); ++i) teardown[i] = i;
  for (size_t i = teardown.size(); i > 1; --i) {
    std::swap(teardown[i - 1], teardown[random.index(i)]);
  }
  for (const size_t index : teardown) {
    g_context.action(seed, iteration, round, index, sessions[index]->stage,
                     "stop with callback in flight");
    sessions[index]->stop_with_inflight_callback();
    sessions[index]->validate_isolation();
  }
  return summary;
}

uint64_t parse_u64(const std::string& text, const char* name) {
  size_t consumed = 0;
  try {
    const unsigned long long value = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) throw std::invalid_argument("suffix");
    return static_cast<uint64_t>(value);
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&](const std::string& prefix) -> std::optional<std::string> {
      if (argument.rfind(prefix, 0) != 0) return std::nullopt;
      return argument.substr(prefix.size());
    };
    if (const auto text = value("--seed=")) {
      options.seed = parse_u64(*text, "seed");
    } else if (const auto text = value("--iterations=")) {
      options.iterations = static_cast<size_t>(parse_u64(*text, "iterations"));
      if (options.iterations == 0 || options.iterations > 1000000) {
        throw std::runtime_error("iterations must be in 1..1000000");
      }
    } else if (const auto text = value("--sessions=")) {
      options.sessions = static_cast<size_t>(parse_u64(*text, "sessions"));
      if (options.sessions < 2 || options.sessions > 64) {
        throw std::runtime_error("sessions must be in 2..64");
      }
    } else if (argument == "--trace") {
      options.trace = true;
    } else if (argument == "--help") {
      std::cout << "realtime_stress_test [--seed=0x5eed38] [--iterations=3] "
                   "[--sessions=6] [--trace]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    g_context.action(options.seed, 0, 0, 0, 0, "bounded queue and close");
    test_bounded_playback_queue_and_inflight_close();
    uint64_t trace = 0xcbf29ce484222325ULL;
    size_t actions = 0;
    for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
      RunSummary summary = run_iteration(options, iteration);
      trace = trace_mix(trace, std::to_string(summary.trace));
      actions += summary.actions;
    }
    std::cout << "REALTIME_STRESS PASS seed=" << options.seed
              << " iterations=" << options.iterations
              << " sessions=" << options.sessions
              << " scheduler_actions=" << actions
              << " echo_modes=on,off queue_bound=4096 trace=0x"
              << std::hex << trace << std::dec << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "REALTIME_STRESS FAIL " << error.what() << '\n';
    return 1;
  }
}
