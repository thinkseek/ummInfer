// SPDX-License-Identifier: Apache-2.0
#include "client/voice.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void wait_until(const std::function<bool()>& predicate, const std::string& message) {
  const auto deadline = Clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (Clock::now() >= deadline) throw std::runtime_error("timeout: " + message);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

VoicePcmChunk pcm(size_t bytes = 4, int rate = 24000, int channels = 1) {
  return VoicePcmChunk{std::string(bytes, '\x01'), rate, channels};
}

class FakeOutput final : public VoiceAudioOutput {
 public:
  struct Item {
    VoicePcmChunk chunk;
    Completion completion;
  };

  void enqueue(VoicePcmChunk chunk, Completion completion) override {
    std::lock_guard lock(mu_);
    if (throw_on_enqueue) throw Error(ErrorCode::kConfig, "injected output enqueue failure");
    items.push_back(Item{std::move(chunk), std::move(completion)});
  }

  void stop() noexcept override {
    std::lock_guard lock(mu_);
    ++stops;
  }

  void close() noexcept override {
    std::lock_guard lock(mu_);
    ++closes;
  }

  void complete(size_t index, bool consumed = true) {
    Completion completion;
    {
      std::lock_guard lock(mu_);
      check(index < items.size(), "fake output item index is out of range");
      completion = items[index].completion;
    }
    if (completion) completion(consumed);
  }

  size_t size() const {
    std::lock_guard lock(mu_);
    return items.size();
  }
  int stop_count() const {
    std::lock_guard lock(mu_);
    return stops;
  }
  int close_count() const {
    std::lock_guard lock(mu_);
    return closes;
  }

  mutable std::mutex mu_;
  std::vector<Item> items;
  int stops = 0;
  int closes = 0;
  bool throw_on_enqueue = false;
};

struct Ack {
  std::string response_id;
  uint64_t cursor = 0;
};

struct PlaybackFixture {
  std::unique_ptr<FakeOutput> owned = std::make_unique<FakeOutput>();
  FakeOutput* output = owned.get();
  std::mutex mu;
  std::vector<Ack> acks;
  std::exception_ptr error;
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  std::atomic<bool> block_ack{false};
  bool ack_entered = false;
  bool release_ack = false;
  VoicePlayback playback;

  PlaybackFixture()
      : playback(std::move(owned),
                 [this](const std::string& id, uint64_t cursor) {
                   if (block_ack) {
                     std::unique_lock lock(gate_mu);
                     ack_entered = true;
                     gate_cv.notify_all();
                     gate_cv.wait(lock, [this] { return release_ack; });
                   }
                   std::lock_guard lock(mu);
                   acks.push_back(Ack{id, cursor});
                 },
                 [this](std::exception_ptr value) {
                   std::lock_guard lock(mu);
                   error = value;
                 }) {}
  ~PlaybackFixture() { playback.close(); }

  std::vector<Ack> acknowledged() {
    std::lock_guard lock(mu);
    return acks;
  }
};

void test_cursor_monotonicity_and_duplicate_consumption() {
  PlaybackFixture f;
  f.playback.push("r1", pcm(), 0);
  f.playback.push("r1", pcm(), 8);
  f.playback.push("r1", pcm(), 8);
  f.playback.push("r1", pcm(), 4);  // A late lower cursor cannot move backwards.
  check(f.acknowledged().empty(), "enqueue alone emitted an ACK");
  f.output->complete(1);
  check(f.acknowledged().empty(), "out-of-order completion skipped the queue head");
  f.output->complete(0);
  wait_until([&] { return f.acknowledged().size() == 2; }, "ordered cursor ACKs");
  f.output->complete(2);
  f.output->complete(3);
  f.output->complete(0);  // repeated device callback
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  auto acks = f.acknowledged();
  check(acks.size() == 2 && acks[0].cursor == 0 && acks[1].cursor == 8,
        "ACK cursors were not monotonic and duplicate-free");
  f.playback.done("r1", 8);
  wait_until([&] { return f.acknowledged().size() == 3; }, "equal final cursor ACK");
  f.playback.done("r1", 8);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  acks = f.acknowledged();
  check(acks.size() == 3 && acks[2].cursor == 8,
        "final equal cursor must be sent exactly once");
}

void test_delayed_and_final_acks() {
  PlaybackFixture f;
  f.playback.push("r1", pcm(), 5);
  f.playback.done("r1", 12);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.acknowledged().empty(), "audio.done acknowledged unplayed PCM");
  f.output->complete(0);
  wait_until([&] { return f.acknowledged().size() == 2; }, "drained final ACK");
  const auto acks = f.acknowledged();
  check(acks[0].cursor == 5 && acks[1].cursor == 12,
        "delta ACK did not precede the final cursor");

  PlaybackFixture empty;
  empty.playback.done("empty", 0);
  wait_until([&] { return empty.acknowledged().size() == 1; }, "empty response final ACK");
  check(empty.acknowledged()[0].cursor == 0, "empty response did not ACK its final zero cursor");
}

void test_partial_consumption_never_acknowledges() {
  PlaybackFixture f;
  f.playback.push("r1", pcm(), 7);
  f.playback.done("r1", 9);
  f.output->complete(0, false);
  wait_until([&] {
    std::lock_guard lock(f.mu);
    return f.error != nullptr;
  }, "partial-consumption error");
  check(f.acknowledged().empty(), "partially consumed audio emitted an ACK");
  check(f.playback.pending_count() == 0, "partially consumed item wedged the queue");
  check(f.output->stop_count() == 1, "partial output did not stop the device");
}

void test_cancellation_races_and_stale_callbacks() {
  PlaybackFixture f;
  f.playback.push("old", pcm(), 4);
  f.playback.cancel("old");
  f.playback.cancel("old");
  check(f.output->stop_count() == 1, "repeated cancellation stopped the device repeatedly");
  f.output->complete(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.acknowledged().empty(), "cancelled callback emitted an ACK");

  f.playback.push("current", pcm(), 2);
  f.output->complete(1);
  wait_until([&] { return f.acknowledged().size() == 1; }, "new response after cancellation");
  f.playback.cancel("old");
  check(f.acknowledged().back().response_id == "current",
        "stale cancellation affected the current response");
  f.playback.push("current", pcm(), 6);
  f.playback.begin("next");
  f.output->complete(2);  // stale current completion after next begins
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.acknowledged().size() == 1, "old response ACK arrived after a response change");
  f.playback.push("next", pcm(), 1);
  f.output->complete(3);
  wait_until([&] { return f.acknowledged().size() == 2; }, "next response ACK");
  check(f.acknowledged().back().response_id == "next", "next response id was not preserved");
}

void test_cancel_stops_immediately_during_blocked_ack() {
  PlaybackFixture f;
  f.block_ack = true;
  f.playback.push("r1", pcm(), 3);
  f.output->complete(0);
  {
    std::unique_lock lock(f.gate_mu);
    check(f.gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return f.ack_entered; }),
          "ACK did not enter the blocked socket sink");
  }
  std::atomic<bool> cancel_returned{false};
  std::thread canceller([&] {
    f.playback.cancel("r1");
    cancel_returned = true;
  });
  wait_until([&] { return f.output->stop_count() == 1; },
             "immediate output stop while ACK was blocked");
  check(!cancel_returned, "cancel returned before the in-flight ACK serialized out");
  {
    std::lock_guard lock(f.gate_mu);
    f.release_ack = true;
  }
  f.gate_cv.notify_all();
  canceller.join();
  check(cancel_returned, "cancel did not finish after ACK release");
  check(f.acknowledged().size() == 1, "the already-started ACK was lost or duplicated");
}

class FakeSocket final : public VoiceSocket {
 public:
  explicit FakeSocket(CancelFlag external = nullptr) : external_cancel(external) {}

  void send_text(std::string_view text) override {
    std::lock_guard lock(mu_);
    if (throw_on_send) throw Error(ErrorCode::kNetwork, "injected send failure", true);
    sent.push_back(std::string(text));
    cv_.notify_all();
  }

  void run(const std::function<void(std::string_view)>& on_message) override {
    std::unique_lock lock(mu_);
    running = true;
    cv_.notify_all();
    while (!closed) {
      if (!incoming.empty()) {
        std::string message = std::move(incoming.front());
        incoming.pop_front();
        lock.unlock();
        on_message(message);
        lock.lock();
        continue;
      }
      if (disconnected) throw Error(ErrorCode::kNetwork, "injected disconnect", true);
      if (external_cancel && external_cancel->load(std::memory_order_relaxed)) return;
      cv_.wait_for(lock, std::chrono::milliseconds(5));
    }
  }

  void close() noexcept override {
    std::lock_guard lock(mu_);
    closed = true;
    cv_.notify_all();
  }

  void emit(const json& event) {
    std::lock_guard lock(mu_);
    incoming.push_back(event.dump());
    cv_.notify_all();
  }

  void disconnect() {
    std::lock_guard lock(mu_);
    disconnected = true;
    cv_.notify_all();
  }

  std::vector<json> messages() const {
    std::lock_guard lock(mu_);
    std::vector<json> out;
    for (const auto& text : sent) out.push_back(json::parse(text));
    return out;
  }

  size_t sent_count() const {
    std::lock_guard lock(mu_);
    return sent.size();
  }
  bool is_closed() const {
    std::lock_guard lock(mu_);
    return closed;
  }

  CancelFlag external_cancel;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::string> sent;
  std::deque<std::string> incoming;
  bool running = false;
  bool closed = false;
  bool disconnected = false;
  bool throw_on_send = false;
};

class FakeMicrophone final : public VoiceMicrophone {
 public:
  void start(PcmCallback on_pcm, ErrorCallback on_error) override {
    PcmCallback immediate;
    std::string initial;
    {
      std::lock_guard lock(mu_);
      ++starts;
      callback = std::move(on_pcm);
      error_callback = std::move(on_error);
      stale_callback = callback;
      immediate = callback;
      initial = on_start_pcm;
    }
    if (immediate && !initial.empty()) immediate(std::move(initial));
  }

  void stop() noexcept override {
    std::lock_guard lock(mu_);
    ++stops;
    callback = nullptr;
    error_callback = nullptr;
  }

  void emit(std::string bytes) {
    PcmCallback current;
    {
      std::lock_guard lock(mu_);
      current = callback;
    }
    if (current) current(std::move(bytes));
  }

  void emit_stale(std::string bytes) {
    PcmCallback old;
    {
      std::lock_guard lock(mu_);
      old = stale_callback;
    }
    if (old) old(std::move(bytes));
  }

  void fail() {
    ErrorCallback current;
    {
      std::lock_guard lock(mu_);
      current = error_callback;
    }
    if (current) {
      current(std::make_exception_ptr(Error(ErrorCode::kConfig, "injected microphone failure")));
    }
  }

  int start_count() const {
    std::lock_guard lock(mu_);
    return starts;
  }
  int stop_count() const {
    std::lock_guard lock(mu_);
    return stops;
  }

  mutable std::mutex mu_;
  PcmCallback callback;
  PcmCallback stale_callback;
  ErrorCallback error_callback;
  std::string on_start_pcm;
  int starts = 0;
  int stops = 0;
};

class BorrowedSocket final : public VoiceSocket {
 public:
  explicit BorrowedSocket(std::shared_ptr<FakeSocket> socket) : socket_(std::move(socket)) {}
  void send_text(std::string_view text) override { socket_->send_text(text); }
  void run(const std::function<void(std::string_view)>& on_message) override {
    socket_->run(on_message);
  }
  void close() noexcept override { socket_->close(); }
 private:
  std::shared_ptr<FakeSocket> socket_;
};

class BorrowedMicrophone final : public VoiceMicrophone {
 public:
  explicit BorrowedMicrophone(std::shared_ptr<FakeMicrophone> microphone)
      : microphone_(std::move(microphone)) {}
  void start(PcmCallback on_pcm, ErrorCallback on_error) override {
    microphone_->start(std::move(on_pcm), std::move(on_error));
  }
  void stop() noexcept override { microphone_->stop(); }
 private:
  std::shared_ptr<FakeMicrophone> microphone_;
};

class BorrowedOutput final : public VoiceAudioOutput {
 public:
  explicit BorrowedOutput(std::shared_ptr<FakeOutput> output) : output_(std::move(output)) {}
  void enqueue(VoicePcmChunk chunk, Completion completion) override {
    output_->enqueue(std::move(chunk), std::move(completion));
  }
  void stop() noexcept override { output_->stop(); }
  void close() noexcept override { output_->close(); }
 private:
  std::shared_ptr<FakeOutput> output_;
};

struct ClientFixture {
  VoiceConfig config;
  mutable std::mutex factory_mu;
  std::vector<std::shared_ptr<FakeSocket>> sockets;
  std::vector<std::shared_ptr<FakeMicrophone>> microphones;
  std::vector<std::shared_ptr<FakeOutput>> outputs;
  std::vector<std::string> urls;
  std::vector<std::string> bearers;
  std::mutex output_mu;
  std::vector<std::string> response_text;
  std::vector<std::string> statuses;
  std::vector<bool> echo_updates;
  std::vector<VoiceBargeInCutPoint> cancelled_cut_points;
  std::vector<double> mic_levels;
  std::unique_ptr<VoiceClient> client;
  std::thread thread;
  std::exception_ptr failure;

  explicit ClientFixture(int reconnects = 0, bool echo_suppression = true) {
    config.base_url = "http://127.0.0.1:8123";
    config.model = "voice model";
    config.api_key = "test-secret-never-in-url";
    config.echo_suppression = echo_suppression;
    config.reconnect_attempts = reconnects;
    config.reconnect_delay = Milliseconds{0};
    VoiceHandlers handlers;
    handlers.on_response_text = [this](const std::string& text) {
      std::lock_guard lock(output_mu);
      response_text.push_back(text);
    };
    handlers.on_status = [this](const std::string& text) {
      std::lock_guard lock(output_mu);
      statuses.push_back(text);
    };
    handlers.on_echo_suppression = [this](bool enabled) {
      std::lock_guard lock(output_mu);
      echo_updates.push_back(enabled);
    };
    handlers.on_response_cancelled = [this](const VoiceBargeInCutPoint& cut_point) {
      std::lock_guard lock(output_mu);
      cancelled_cut_points.push_back(cut_point);
    };
    handlers.on_mic_level = [this](double level) {
      std::lock_guard lock(output_mu);
      mic_levels.push_back(level);
    };
    client = std::make_unique<VoiceClient>(
        config, std::move(handlers),
        [this](const std::string& url, Milliseconds, CancelFlag, CancelFlag external,
               const std::string& bearer) {
          auto socket = std::make_shared<FakeSocket>(external);
          {
            std::lock_guard lock(factory_mu);
            urls.push_back(url);
            bearers.push_back(bearer);
            sockets.push_back(socket);
          }
          return std::make_unique<BorrowedSocket>(std::move(socket));
        },
        [this](const std::string&) {
          auto microphone = std::make_shared<FakeMicrophone>();
          {
            std::lock_guard lock(factory_mu);
            microphones.push_back(microphone);
          }
          return std::make_unique<BorrowedMicrophone>(std::move(microphone));
        },
        [this](const std::string&) {
          auto output = std::make_shared<FakeOutput>();
          {
            std::lock_guard lock(factory_mu);
            outputs.push_back(output);
          }
          return std::make_unique<BorrowedOutput>(std::move(output));
        });
  }

  ~ClientFixture() {
    if (thread.joinable()) client->request_stop();
    if (thread.joinable()) thread.join();
  }

  size_t socket_count() const {
    std::lock_guard lock(factory_mu);
    return sockets.size();
  }
  size_t microphone_count() const {
    std::lock_guard lock(factory_mu);
    return microphones.size();
  }
  size_t output_count() const {
    std::lock_guard lock(factory_mu);
    return outputs.size();
  }
  std::shared_ptr<FakeSocket> socket(size_t index) const {
    std::lock_guard lock(factory_mu);
    return sockets.at(index);
  }
  std::shared_ptr<FakeMicrophone> microphone(size_t index) const {
    std::lock_guard lock(factory_mu);
    return microphones.at(index);
  }
  std::shared_ptr<FakeOutput> output(size_t index) const {
    std::lock_guard lock(factory_mu);
    return outputs.at(index);
  }

  void start() {
    thread = std::thread([this] {
      try {
        client->run();
      } catch (...) {
        failure = std::current_exception();
      }
    });
    wait_until([this] { return socket_count() > 0 && socket(0)->sent_count() >= 1; },
               "voice session update");
  }

  void ready(size_t index = 0) {
    wait_until([this, index] { return socket_count() > index; }, "fake socket connection");
    socket(index)->emit({{"type", "transcription_session.updated"},
                         {"session", {{"audio_output", {{"playback_ack", true}}}}}});
    wait_until([this, index] {
      return microphone_count() > index && microphone(index)->start_count() == 1;
    }, "microphone start after negotiation");
  }

  void stop() {
    client->request_stop();
    if (thread.joinable()) thread.join();
  }

  void join() {
    if (thread.joinable()) thread.join();
  }

  std::vector<json> messages(size_t index = 0) const { return socket(index)->messages(); }

  std::vector<json> of_type(size_t index, const std::string& type) const {
    std::vector<json> out;
    for (const auto& event : messages(index)) {
      if (event.value("type", "") == type) out.push_back(event);
    }
    return out;
  }
};

void test_negotiation_microphone_playback_and_stop_cleanup() {
  ClientFixture f;
  f.start();
  check(f.urls[0] ==
            "ws://127.0.0.1:8123/v1/realtime?intent=transcription&model=voice%20model",
        "voice WebSocket URL was malformed");
  check(f.urls[0].find(f.config.api_key) == std::string::npos,
        "credential leaked into the WebSocket URL");
  check(f.bearers[0] == f.config.api_key, "credential did not use the Authorization channel");
  const auto update = f.messages()[0];
  check(update["session"]["input_audio_format"] == "pcm16" &&
            update["session"]["audio_output"]["playback_ack"] == true &&
            update["session"]["turn_detection"]["type"] == "server_vad",
        "voice session did not negotiate Stage B playback ACK");
  check(f.microphone(0)->start_count() == 0, "microphone started before negotiation");
  f.socket(0)->emit({{"type", "input_audio_buffer.speech_stopped"}});
  const std::string captured("\x01\x00\xff\x7f", 4);
  f.microphone(0)->on_start_pcm = captured;
  f.ready();

  wait_until([&] { return !f.of_type(0, "input_audio_buffer.append").empty(); },
             "microphone append");
  const auto append = f.of_type(0, "input_audio_buffer.append")[0];
  check(base64_decode(append["audio"].get<std::string>(), 1024) == captured,
        "microphone bytes were not streamed as exact PCM16");

  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "r1"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x02'))}, {"text_cursor", 5}});
  wait_until([&] { return f.output_count() > 0 && f.output(0)->size() == 1; }, "audio enqueue");
  check(f.of_type(0, "output_audio_buffer.playback_ack").empty(),
        "audio receipt emitted a playback ACK");
  f.output(0)->complete(0);
  wait_until([&] { return f.of_type(0, "output_audio_buffer.playback_ack").size() == 1; },
             "consumed playback ACK");
  const auto ack = f.of_type(0, "output_audio_buffer.playback_ack")[0];
  check(ack["response_id"] == "r1" && ack["text_cursor"] == 5,
        "playback ACK has the wrong response or cursor");

  f.stop();
  check(f.failure == nullptr, "explicit Stop Voice returned an error");
  check(f.microphone(0)->stop_count() == 1 && f.output(0)->close_count() == 1 &&
            f.socket(0)->is_closed(),
        "Stop Voice did not clean microphone, playback and socket");
}

void test_reconnect_cleans_old_session_and_never_replays_audio() {
  ClientFixture f(1);
  f.start();
  f.ready();
  f.microphone(0)->emit(std::string(4, '\x03'));
  wait_until([&] { return f.of_type(0, "input_audio_buffer.append").size() == 1; },
             "first connection microphone append");
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "old"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x04'))}, {"text_cursor", 2}});
  wait_until([&] { return f.output(0)->size() == 1; }, "old connection playback");
  f.socket(0)->disconnect();
  wait_until([&] { return f.socket_count() == 2 && f.microphone_count() == 2; },
             "voice reconnect");
  check(f.microphone(0)->stop_count() == 1 && f.output(0)->close_count() == 1,
        "reconnect did not clean old audio resources");
  check(f.of_type(1, "input_audio_buffer.append").empty(),
        "reconnect replayed captured audio before a new utterance");
  f.microphone(0)->emit_stale(std::string(4, '\x05'));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.of_type(1, "input_audio_buffer.append").empty(),
        "stale microphone callback reached the new connection");
  f.ready(1);
  f.microphone(1)->emit(std::string(4, '\x06'));
  wait_until([&] { return f.of_type(1, "input_audio_buffer.append").size() == 1; },
             "new capture after reconnect");
  f.stop();
  check(f.failure == nullptr, "reconnect followed by Stop Voice failed");
}

void test_late_events_and_repeated_cancellation_are_inert() {
  ClientFixture f;
  f.start();
  f.ready();
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "old"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 4}});
  wait_until([&] { return f.output(0)->size() == 1; }, "first response audio");
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "old"}});
  wait_until([&] { return f.output(0)->stop_count() == 1; }, "response.cancelled stop");
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "old"}});
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "old"}, {"text", "late"}});
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "old"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x02'))}, {"text_cursor", 8}});
  f.socket(0)->emit({{"type", "audio.done"}, {"response_id", "old"}, {"text_cursor", 8}});
  f.output(0)->complete(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  check(f.output(0)->size() == 1 && f.output(0)->stop_count() == 1,
        "late or repeated cancellation resurrected old playback");
  check(f.of_type(0, "output_audio_buffer.playback_ack").empty(),
        "stale callback acknowledged a cancelled response");
  {
    std::lock_guard lock(f.output_mu);
    check(f.response_text.empty(), "late response text reached the UI");
  }
  f.stop();
}

void test_speech_started_is_an_immediate_local_barge_in() {
  ClientFixture f;
  f.start();
  f.ready();
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "old"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 4}});
  wait_until([&] { return f.output(0)->size() == 1; }, "barge-in response audio");
  // No response.cancelled has arrived yet: the speech-start edge alone must
  // stop and invalidate old queued playback.
  f.socket(0)->emit({{"type", "input_audio_buffer.speech_started"}});
  wait_until([&] { return f.output(0)->stop_count() == 1; }, "speech-start local stop");
  f.output(0)->complete(0);
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "old"}});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.output(0)->stop_count() == 1 &&
            f.of_type(0, "output_audio_buffer.playback_ack").empty(),
        "server cancellation repeated the local stop or emitted a stale ACK");
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "current"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x03'))}, {"text_cursor", 2}});
  wait_until([&] { return f.output(0)->size() == 2; }, "new response after barge-in");
  f.socket(0)->emit({{"type", "input_audio_buffer.speech_started"},
                      {"interrupted_response_id", "old"}});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(f.output(0)->stop_count() == 1,
        "a stale interrupted_response_id cancelled the newer response");
  f.stop();
}

void test_final_ack_retires_response_and_drops_late_text() {
  ClientFixture f;
  f.start();
  f.ready();
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "done"},
                      {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                      {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 4}});
  f.socket(0)->emit({{"type", "audio.done"}, {"response_id", "done"}, {"text_cursor", 4}});
  wait_until([&] { return f.output(0)->size() == 1; }, "final response audio");
  f.output(0)->complete(0);
  wait_until([&] { return f.of_type(0, "output_audio_buffer.playback_ack").size() == 2; },
             "delta and equal final ACK");
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "done"},
                      {"text_cursor", 4}, {"final", true}});
  wait_until([&] { return f.output(0)->stop_count() == 1; }, "final response retirement");
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "done"},
                      {"text", "late"}});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  {
    std::lock_guard lock(f.output_mu);
    check(f.response_text.empty(), "text arrived after the final playback ACK retired response");
  }
  f.stop();
}

void test_error_paths_and_cleanup() {
  {
    ClientFixture f;
    f.start();
    f.socket(0)->emit({{"type", "transcription_session.updated"},
                        {"session", {{"audio_output", {{"playback_ack", false}}}}}});
    f.join();
    check(f.failure != nullptr && f.microphone(0)->start_count() == 0 &&
              f.microphone(0)->stop_count() == 1 && f.output(0)->close_count() == 1,
          "missing playback_ack negotiation was not a clean failure");
  }
  {
    ClientFixture f;
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "bad"},
                        {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                        {"audio", "not base64!"}, {"text_cursor", 1}});
    f.join();
    check(f.failure != nullptr && f.microphone(0)->stop_count() == 1 &&
              f.output(0)->close_count() == 1,
          "invalid base64 did not close microphone and playback");
  }
  {
    ClientFixture f;
    f.start();
    f.ready();
    f.microphone(0)->fail();
    f.join();
    check(f.failure != nullptr && f.microphone(0)->stop_count() == 1 &&
              f.socket(0)->is_closed(),
          "microphone failure did not tear down the connection");
  }
  {
    ClientFixture f;
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "partial"},
                        {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                        {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 1}});
    wait_until([&] { return f.output(0)->size() == 1; }, "partial output enqueue");
    f.output(0)->complete(0, false);
    f.join();
    check(f.failure != nullptr && f.microphone(0)->stop_count() == 1 &&
              f.of_type(0, "output_audio_buffer.playback_ack").empty(),
          "partial output did not fail without ACK");
  }
}

void test_external_cancellation_cleans_up() {
  std::atomic<bool> cancelled{false};
  ClientFixture f;
  f.config.cancel = &cancelled;
  // Rebuild with the external cancellation pointer, matching the process signal
  // flag passed to the production WebSocket.
  f.client = std::make_unique<VoiceClient>(
      f.config, VoiceHandlers{},
      [&f](const std::string&, Milliseconds, CancelFlag, CancelFlag external,
           const std::string&) {
        auto socket = std::make_shared<FakeSocket>(external);
        {
          std::lock_guard lock(f.factory_mu);
          f.sockets.push_back(socket);
        }
        return std::make_unique<BorrowedSocket>(std::move(socket));
      },
      [&f](const std::string&) {
        auto microphone = std::make_shared<FakeMicrophone>();
        {
          std::lock_guard lock(f.factory_mu);
          f.microphones.push_back(microphone);
        }
        return std::make_unique<BorrowedMicrophone>(std::move(microphone));
      },
      [&f](const std::string&) {
        auto output = std::make_shared<FakeOutput>();
        {
          std::lock_guard lock(f.factory_mu);
          f.outputs.push_back(output);
        }
        return std::make_unique<BorrowedOutput>(std::move(output));
      });
  f.thread = std::thread([&f] {
    try {
      f.client->run();
    } catch (...) {
      f.failure = std::current_exception();
    }
  });
  wait_until([&f] { return f.socket_count() > 0 && f.socket(0)->sent_count() == 1; },
             "external-cancel connection");
  f.ready();
  cancelled = true;
  f.join();
  check(f.failure != nullptr && f.microphone(0)->stop_count() == 1 &&
            f.output(0)->close_count() == 1,
        "external signal cancellation did not clean resources");
}

void test_echo_suppression_default_omits_wire_key() {
  ClientFixture f;
  f.start();
  const auto update = f.messages()[0];
  check(update["type"] == "session.update", "initial message was not session.update");
  const auto& turn = update["session"]["turn_detection"];
  check(turn.is_object() && turn["type"] == "server_vad" && turn["silence_ms"] == 400 &&
            !turn.contains("echo_suppression"),
        "default echo_suppression=true must omit echo_suppression on the wire");
  check(!f.client->confirmed_echo_suppression().has_value(),
        "echo_suppression was confirmed before transcription_session.updated");
  f.ready();
  check(f.client->confirmed_echo_suppression() == std::optional<bool>(true),
        "omitted turn_detection did not confirm echo_suppression=true");
  check(f.client->barge_in_count() == 0, "idle session reported non-zero barge_in_count");
  f.stop();
  check(f.failure == nullptr, "default echo_suppression session failed on stop");
}

void test_echo_suppression_false_negotiation_and_rejection() {
  // Explicit false emits echo_suppression=false and succeeds when server confirms false.
  {
    ClientFixture f(0, false);
    f.start();
    const auto update = f.messages()[0];
    const auto& turn = update["session"]["turn_detection"];
    check(turn.contains("echo_suppression") && turn["echo_suppression"].is_boolean() &&
              !turn["echo_suppression"].get<bool>(),
          "echo_suppression=false did not emit explicit false on initial session.update");
    f.socket(0)->emit({{"type", "transcription_session.updated"},
                       {"session",
                        {{"audio_output", {{"playback_ack", true}}},
                         {"turn_detection",
                          {{"type", "server_vad"},
                           {"silence_ms", 550},
                           {"echo_suppression", false}}}}}});
    wait_until([&] { return f.microphone(0)->start_count() == 1; },
               "microphone start when server confirms echo_suppression=false");
    check(f.client->confirmed_echo_suppression() == std::optional<bool>(false),
          "confirmed_echo_suppression did not record false");
    check(f.client->negotiated_silence_ms() == 550,
          "negotiated_silence_ms did not track server-confirmed value");
    {
      std::lock_guard lock(f.output_mu);
      check(!f.statuses.empty() &&
                f.statuses.back().find("echo suppression: off") != std::string::npos,
            "session ready status did not report echo suppression off");
    }
    f.stop();
    check(f.failure == nullptr, "echo_suppression=false session failed cleanly on stop");
  }

  // Server omitting echo_suppression when client requested false must fail cleanly.
  {
    ClientFixture f(0, false);
    f.start();
    f.socket(0)->emit({{"type", "transcription_session.updated"},
                       {"session",
                        {{"audio_output", {{"playback_ack", true}}},
                         {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 400}}}}}});
    f.join();
    check(f.failure != nullptr && f.microphone(0)->start_count() == 0 &&
              f.microphone(0)->stop_count() == 1 && f.output(0)->close_count() == 1,
          "server omitting echo_suppression for echo_suppression=false did not fail cleanly");
    std::string message;
    try {
      std::rethrow_exception(f.failure);
    } catch (const std::exception& error) {
      message = error.what();
    }
    check(message.find("server did not apply turn_detection.echo_suppression=false") !=
              std::string::npos,
          "unexpected error message for unconfirmed echo_suppression=false: " + message);
  }

  // Server returning echo_suppression=true when client requested false must fail cleanly.
  {
    ClientFixture f(0, false);
    f.start();
    f.socket(0)->emit({{"type", "transcription_session.updated"},
                       {"session",
                        {{"audio_output", {{"playback_ack", true}}},
                         {"turn_detection",
                          {{"type", "server_vad"},
                           {"silence_ms", 400},
                           {"echo_suppression", true}}}}}});
    f.join();
    check(f.failure != nullptr && f.microphone(0)->start_count() == 0,
          "server confirming true for echo_suppression=false did not fail cleanly");
  }

  // Non-boolean echo_suppression must fail cleanly as a protocol error.
  {
    ClientFixture f;
    f.start();
    f.socket(0)->emit({{"type", "transcription_session.updated"},
                       {"session",
                        {{"audio_output", {{"playback_ack", true}}},
                         {"turn_detection",
                          {{"type", "server_vad"},
                           {"silence_ms", 400},
                           {"echo_suppression", "off"}}}}}});
    f.join();
    check(f.failure != nullptr && f.microphone(0)->start_count() == 0 &&
              f.output(0)->close_count() == 1,
          "non-boolean turn_detection.echo_suppression did not fail cleanly");
  }
}

void test_mid_response_echo_suppression_toggle_preserves_active_response() {
  ClientFixture f;
  f.start();
  f.socket(0)->emit({{"type", "transcription_session.updated"},
                     {"session",
                      {{"audio_output", {{"playback_ack", true}}},
                       {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 650}}}}}});
  wait_until([&] { return f.microphone(0)->start_count() == 1; }, "initial negotiation");
  check(f.client->confirmed_echo_suppression() == std::optional<bool>(true) &&
            f.client->negotiated_silence_ms() == 650,
        "initial negotiation did not record default echo_suppression=true and silence_ms=650");

  // Begin an active spoken response before toggling echo suppression mid-response.
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "r1"}, {"text", "hello "}});
  f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "r1"},
                     {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                     {"audio", base64_encode(std::string(4, '\x02'))}, {"text_cursor", 6}});
  wait_until([&] { return f.output(0)->size() == 1; }, "active response first chunk");

  // Toggle OFF mid-response: must send ONLY narrow turn_detection with negotiated silence_ms.
  f.client->set_echo_suppression(false);
  wait_until([&] { return f.of_type(0, "session.update").size() == 2; },
             "mid-response echo_suppression=false update");
  const auto toggle_off = f.of_type(0, "session.update")[1];
  check(toggle_off["session"].size() == 1 && toggle_off["session"].contains("turn_detection") &&
            toggle_off["session"]["turn_detection"] ==
                json{{"type", "server_vad"}, {"silence_ms", 650}, {"echo_suppression", false}},
        "mid-response toggle off did not send narrow turn_detection-only session.update");

  f.socket(0)->emit({{"type", "transcription_session.updated"},
                     {"session",
                      {{"audio_output", {{"playback_ack", true}}},
                       {"turn_detection",
                        {{"type", "server_vad"},
                         {"silence_ms", 650},
                         {"echo_suppression", false}}}}}});
  wait_until([&] { return f.client->confirmed_echo_suppression() == std::optional<bool>(false); },
             "mid-response echo_suppression=false confirmation");

  // Toggle back ON mid-response: server omits echo_suppression in reply per effective_session.
  f.client->set_echo_suppression(true);
  wait_until([&] { return f.of_type(0, "session.update").size() == 3; },
             "mid-response echo_suppression=true update");
  const auto toggle_on = f.of_type(0, "session.update")[2];
  check(toggle_on["session"].size() == 1 && toggle_on["session"].contains("turn_detection") &&
            toggle_on["session"]["turn_detection"] ==
                json{{"type", "server_vad"}, {"silence_ms", 650}, {"echo_suppression", true}},
        "mid-response toggle on did not send narrow turn_detection-only session.update");

  f.socket(0)->emit({{"type", "transcription_session.updated"},
                     {"session",
                      {{"audio_output", {{"playback_ack", true}}},
                       {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 650}}}}}});
  wait_until([&] { return f.client->confirmed_echo_suppression() == std::optional<bool>(true); },
             "mid-response echo_suppression=true confirmation");

  // Verify microphone was not restarted and active response playback/ACK was not clobbered.
  check(f.microphone(0)->start_count() == 1 && f.output(0)->stop_count() == 0,
        "mid-response transcription_session.updated restarted mic or stopped active playback");
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "r1"}, {"text", "world"}});
  f.socket(0)->emit({{"type", "audio.done"}, {"response_id", "r1"}, {"text_cursor", 11}});
  f.output(0)->complete(0);
  wait_until([&] { return f.of_type(0, "output_audio_buffer.playback_ack").size() == 2; },
             "active response ACKs after mid-response echo_suppression toggle");
  const auto acks = f.of_type(0, "output_audio_buffer.playback_ack");
  check(acks[0]["response_id"] == "r1" && acks[0]["text_cursor"] == 6 &&
            acks[1]["response_id"] == "r1" && acks[1]["text_cursor"] == 11,
        "active response ACKs were corrupted by mid-response toggle");
  {
    std::lock_guard lock(f.output_mu);
    check(f.response_text.size() == 2 && f.response_text[0] == "hello " &&
              f.response_text[1] == "world",
          "active response text stream was clobbered by mid-response toggle");
  }
  f.stop();
  check(f.failure == nullptr, "mid-response toggle test failed on stop");
}

void test_observed_barge_in_count_telemetry() {
  // No cancellations => barge_in_count stays 0.
  {
    ClientFixture f;
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "r1"},
                       {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                       {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 4}});
    f.socket(0)->emit({{"type", "audio.done"}, {"response_id", "r1"}, {"text_cursor", 4}});
    wait_until([&] { return f.output(0)->size() == 1; }, "clean response audio");
    f.output(0)->complete(0);
    wait_until([&] { return f.of_type(0, "output_audio_buffer.playback_ack").size() == 2; },
               "clean response ACKs");
    f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r1"},
                       {"text_cursor", 4}, {"final", true}});
    wait_until([&] { return f.output(0)->stop_count() == 1; }, "clean response retirement");
    check(f.client->barge_in_count() == 0,
          "barge_in_count incremented without response.cancelled");
    f.stop();
    check(f.client->barge_in_count() == 0, "barge_in_count changed after stop");
  }

  // Real response.cancelled events increment barge_in_count, and reconnect resets it.
  {
    ClientFixture f(1);
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "r1"},
                       {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                       {"audio", base64_encode(std::string(4, '\x01'))}, {"text_cursor", 3}});
    wait_until([&] { return f.output(0)->size() == 1; }, "first response audio");
    f.socket(0)->emit({{"type", "input_audio_buffer.speech_started"},
                       {"interrupted_response_id", "r1"}});
    wait_until([&] { return f.output(0)->stop_count() == 1; }, "speech_started local stop");
    // speech_started alone does not fabricate a barge_in_count; server response.cancelled does.
    check(f.client->barge_in_count() == 0,
          "speech_started incremented barge_in_count before response.cancelled");
    f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r1"}});
    wait_until([&] { return f.client->barge_in_count() == 1; }, "first barge-in count");

    f.socket(0)->emit({{"type", "audio.delta"}, {"response_id", "r2"},
                       {"format", "pcm16"}, {"sample_rate", 24000}, {"channels", 1},
                       {"audio", base64_encode(std::string(4, '\x02'))}, {"text_cursor", 5}});
    wait_until([&] { return f.output(0)->size() == 2; }, "second response audio");
    f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r2"}});
    wait_until([&] { return f.client->barge_in_count() == 2; }, "second barge-in count");

    // Reconnect resets session barge_in_count to 0.
    f.socket(0)->disconnect();
    wait_until([&] { return f.socket_count() == 2 && f.microphone_count() == 2; },
               "reconnect after barge-ins");
    f.ready(1);
    check(f.client->barge_in_count() == 0, "reconnect did not reset barge_in_count");
    f.socket(1)->emit({{"type", "response.cancelled"}, {"response_id", "r3"}});
    wait_until([&] { return f.client->barge_in_count() == 1; }, "post-reconnect barge-in count");
    f.stop();
    check(f.client->barge_in_count() == 1, "post-stop barge_in_count did not preserve summary");
  }
}

std::string make_pcm16_le(const std::vector<int16_t>& samples) {
  std::string out;
  out.reserve(samples.size() * 2);
  for (int16_t sample : samples) {
    const uint16_t u = static_cast<uint16_t>(sample);
    out.push_back(static_cast<char>(u & 0xffu));
    out.push_back(static_cast<char>((u >> 8) & 0xffu));
  }
  return out;
}

void test_compute_pcm16_rms_level_and_cut_point_format() {
  check(compute_pcm16_rms_level("") == 0.0, "empty PCM16 must return 0.0");
  check(compute_pcm16_rms_level(std::string(1, '\x7f')) == 0.0,
        "1-byte PCM16 must return 0.0");
  check(compute_pcm16_rms_level(make_pcm16_le({0, 0, 0, 0})) == 0.0,
        "all-zero PCM16 must return 0.0");

  const double half = compute_pcm16_rms_level(make_pcm16_le({16384, 16384, 16384, 16384}));
  check(std::abs(half - 0.5) < 1e-9, "half-scale PCM16 RMS must be 0.5");

  const double full_neg = compute_pcm16_rms_level(make_pcm16_le({-32768, -32768}));
  check(std::abs(full_neg - 1.0) < 1e-9, "full-scale negative PCM16 RMS must be 1.0");

  const double full_pos = compute_pcm16_rms_level(make_pcm16_le({32767, 32767}));
  check(std::abs(full_pos - (32767.0 / 32768.0)) < 1e-9 && full_pos <= 1.0,
        "full-scale positive PCM16 RMS must match 32767/32768");

  const double mixed = compute_pcm16_rms_level(make_pcm16_le({16384, -16384, 0, 0}));
  check(std::abs(mixed - std::sqrt(0.125)) < 1e-9, "mixed PCM16 RMS calculation was inaccurate");

  std::string with_trailing_odd = make_pcm16_le({16384});
  with_trailing_odd.push_back('\xff');
  check(std::abs(compute_pcm16_rms_level(with_trailing_odd) - 0.5) < 1e-9,
        "trailing odd byte must be safely ignored");

  check(format_barge_in_cut_point(0) == "Interrupted before audio",
        "0 spoken bytes must format as 'Interrupted before audio'");
  check(format_barge_in_cut_point(1) == "Interrupted · 1 B spoken",
        "1 spoken byte formatted incorrectly");
  check(format_barge_in_cut_point(42) == "Interrupted · 42 B spoken",
        "42 spoken bytes formatted incorrectly");
  check(format_barge_in_cut_point(2048) == "Interrupted · 2048 B spoken",
        "2048 spoken bytes formatted incorrectly");
}

void test_barge_in_cut_point_from_playback_ack_and_cancelled_cursor() {
  ClientFixture f;
  f.start();
  f.ready();
  check(!f.client->last_barge_in_cut_point().has_value(),
        "idle session must not have a barge-in cut-point");

  // 1. Cut-point derived monotonically from response.playback_ack across speech_started.
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "r1"}, {"text", "Hello world"}});
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r1"},
                     {"text_cursor", 12}, {"final", false}});
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r1"},
                     {"text_cursor", 24}, {"final", false}});
  // Out-of-order lower cursor must not regress tracked spoken bytes.
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r1"},
                     {"text_cursor", 10}, {"final", false}});
  f.socket(0)->emit({{"type", "input_audio_buffer.speech_started"},
                     {"interrupted_response_id", "r1"}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return !f.statuses.empty() && f.statuses.back() == "Interrupted; listening";
  }, "speech_started status");
  check(!f.client->last_barge_in_cut_point().has_value(),
        "speech_started alone must not populate last_barge_in_cut_point");
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r1"}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 1;
  }, "first response.cancelled cut-point");
  {
    const auto cp = f.client->last_barge_in_cut_point();
    check(cp.has_value() && cp->response_id == "r1" && cp->spoken_bytes == 24 &&
              cp->label == "Interrupted · 24 B spoken",
          "playback_ack cut-point did not preserve monotonic 24 B spoken");
    std::lock_guard lock(f.output_mu);
    check(f.cancelled_cut_points[0].response_id == "r1" &&
              f.cancelled_cut_points[0].spoken_bytes == 24 &&
              f.cancelled_cut_points[0].label == "Interrupted · 24 B spoken",
          "on_response_cancelled callback did not receive exact cut-point");
  }

  // 2. response.cancelled.text_cursor overrides higher cursor, and max keeps higher ACK.
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r2"},
                     {"text_cursor", 16}, {"final", false}});
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r2"},
                     {"text_cursor", 30}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 2;
  }, "second response.cancelled cut-point override");
  {
    const auto cp = f.client->last_barge_in_cut_point();
    check(cp.has_value() && cp->response_id == "r2" && cp->spoken_bytes == 30 &&
              cp->label == "Interrupted · 30 B spoken",
          "response.cancelled.text_cursor did not override lower playback_ack");
  }

  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r3"},
                     {"text_cursor", 40}, {"final", false}});
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r3"},
                     {"text_cursor", 12}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 3;
  }, "third response.cancelled max cursor");
  {
    const auto cp = f.client->last_barge_in_cut_point();
    check(cp.has_value() && cp->response_id == "r3" && cp->spoken_bytes == 40 &&
              cp->label == "Interrupted · 40 B spoken",
          "response.cancelled did not keep max of playback_ack and event text_cursor");
  }

  // 3. Zero-audio cancellation emits "Interrupted before audio".
  f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "r4"}, {"text", "Early"}});
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r4"}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 4;
  }, "zero-audio response.cancelled cut-point");
  {
    const auto cp = f.client->last_barge_in_cut_point();
    check(cp.has_value() && cp->response_id == "r4" && cp->spoken_bytes == 0 &&
              cp->label == "Interrupted before audio",
          "zero-audio cancellation did not format as 'Interrupted before audio'");
  }

  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r5"},
                     {"text_cursor", 0}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 5;
  }, "explicit-zero response.cancelled cut-point");
  {
    const auto cp = f.client->last_barge_in_cut_point();
    check(cp.has_value() && cp->response_id == "r5" && cp->spoken_bytes == 0 &&
              cp->label == "Interrupted before audio",
          "explicit text_cursor=0 did not format as 'Interrupted before audio'");
  }

  f.stop();
  check(f.failure == nullptr, "cut-point session failed on stop");
  check(!f.client->last_barge_in_cut_point().has_value(),
        "cleanup on stop did not reset last_barge_in_cut_point");
}

void test_barge_in_cut_point_non_emission_and_invalid_cursor_rejection() {
  // response.completed, final response.playback_ack, and response.failed must never emit cut-points.
  {
    ClientFixture f;
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "ok"}, {"text", "Done"}});
    f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "ok"},
                       {"text_cursor", 4}, {"final", false}});
    f.socket(0)->emit({{"type", "response.completed"}, {"response_id", "ok"}});
    f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "ok"},
                       {"text_cursor", 4}, {"final", true}});
    f.socket(0)->emit({{"type", "response.text.delta"}, {"response_id", "bad"}, {"text", "Oops"}});
    f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "bad"},
                       {"text_cursor", 4}, {"final", false}});
    f.socket(0)->emit({{"type", "response.failed"}, {"response_id", "bad"},
                       {"error", {{"message", "synthetic failure"}}}});
    wait_until([&] {
      std::lock_guard lock(f.output_mu);
      return !f.statuses.empty() && f.statuses.back() == "Listening" &&
             f.response_text.size() == 2;
    }, "completed and failed turns settled");
    check(!f.client->last_barge_in_cut_point().has_value(),
          "response.completed or response.failed populated last_barge_in_cut_point");
    check(f.client->barge_in_count() == 0,
          "response.completed or response.failed incremented barge_in_count");
    {
      std::lock_guard lock(f.output_mu);
      check(f.cancelled_cut_points.empty(),
            "response.completed or response.failed invoked on_response_cancelled");
    }
    // Verify response.failed erased tracked cursor so a hypothetical later cancelled starts at 0.
    f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "bad"}});
    wait_until([&] {
      std::lock_guard lock(f.output_mu);
      return f.cancelled_cut_points.size() == 1;
    }, "cancelled after failed");
    check(f.client->last_barge_in_cut_point()->spoken_bytes == 0,
          "response.failed did not clear tracked playback_ack cursor");
    f.stop();
    check(f.failure == nullptr, "non-emission test failed on stop");
  }

  // Invalid response.cancelled.text_cursor values must fail cleanly with ErrorCode::kBadResponse.
  const std::vector<json> invalid_cursors = {-1, "12", true, 1.5, nullptr};
  for (const auto& bad_cursor : invalid_cursors) {
    ClientFixture f;
    f.start();
    f.ready();
    f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r_bad"},
                       {"text_cursor", bad_cursor}});
    f.join();
    check(f.failure != nullptr, "invalid response.cancelled.text_cursor did not fail");
    bool saw_bad_response = false;
    try {
      std::rethrow_exception(f.failure);
    } catch (const Error& error) {
      saw_bad_response = (error.code() == ErrorCode::kBadResponse);
    }
    check(saw_bad_response,
          "invalid response.cancelled.text_cursor did not fail with ErrorCode::kBadResponse: " +
              bad_cursor.dump());
    check(f.client->barge_in_count() == 0 && !f.client->last_barge_in_cut_point().has_value(),
          "invalid response.cancelled.text_cursor mutated barge-in telemetry");
    {
      std::lock_guard lock(f.output_mu);
      check(f.cancelled_cut_points.empty(),
            "invalid response.cancelled.text_cursor invoked on_response_cancelled");
    }
    check(f.microphone(0)->stop_count() == 1 && f.output(0)->close_count() == 1,
          "invalid response.cancelled.text_cursor did not clean audio resources");
  }
}

void test_mic_rms_level_callback_and_reconnect_reset() {
  ClientFixture f(1);
  f.start();
  f.ready();
  check(f.client->last_mic_rms_level() == 0.0,
        "initial last_mic_rms_level must be 0.0 before capture");

  f.microphone(0)->emit(make_pcm16_le({0, 0, 0, 0}));
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.mic_levels.size() == 1;
  }, "silence mic RMS callback");
  check(f.client->last_mic_rms_level() == 0.0, "silence frame must report 0.0 RMS");

  f.microphone(0)->emit(make_pcm16_le({16384, 16384, 16384, 16384}));
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.mic_levels.size() == 2;
  }, "half-scale mic RMS callback");
  check(std::abs(f.client->last_mic_rms_level() - 0.5) < 1e-9,
        "half-scale captured frame did not update last_mic_rms_level to 0.5");

  f.microphone(0)->emit(make_pcm16_le({-32768, -32768}));
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.mic_levels.size() == 3;
  }, "full-scale mic RMS callback");
  check(std::abs(f.client->last_mic_rms_level() - 1.0) < 1e-9,
        "full-scale captured frame did not update last_mic_rms_level to 1.0");
  {
    std::lock_guard lock(f.output_mu);
    check(f.mic_levels[0] == 0.0 && std::abs(f.mic_levels[1] - 0.5) < 1e-9 &&
              std::abs(f.mic_levels[2] - 1.0) < 1e-9,
          "on_mic_level callback sequence did not match real captured PCM16 RMS");
  }

  // Populate both last_barge_in_cut_point and an in-flight playback_ack before disconnect.
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r1"},
                     {"text_cursor", 15}, {"final", false}});
  f.socket(0)->emit({{"type", "response.cancelled"}, {"response_id", "r1"}});
  f.socket(0)->emit({{"type", "response.playback_ack"}, {"response_id", "r_stale"},
                     {"text_cursor", 99}, {"final", false}});
  wait_until([&] {
    const auto cp = f.client->last_barge_in_cut_point();
    return cp.has_value() && cp->spoken_bytes == 15;
  }, "pre-reconnect cut-point");

  // Reconnect must reset last_mic_rms_level, last_barge_in_cut_point, and acked_text_cursors.
  f.socket(0)->disconnect();
  wait_until([&] { return f.socket_count() == 2 && f.microphone_count() == 2; },
             "reconnect after mic RMS and cut-point");
  f.ready(1);
  check(f.client->last_mic_rms_level() == 0.0,
        "reconnect did not reset last_mic_rms_level to 0.0");
  check(!f.client->last_barge_in_cut_point().has_value(),
        "reconnect did not reset last_barge_in_cut_point");

  // r_stale from connection 0 must not leak its 99-byte cursor into connection 1.
  f.socket(1)->emit({{"type", "response.cancelled"}, {"response_id", "r_stale"}});
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.cancelled_cut_points.size() == 2;
  }, "post-reconnect cancelled for stale id");
  check(f.client->last_barge_in_cut_point().has_value() &&
            f.client->last_barge_in_cut_point()->spoken_bytes == 0 &&
            f.client->last_barge_in_cut_point()->label == "Interrupted before audio",
        "reconnect leaked prior connection's acked_text_cursors");

  f.microphone(1)->emit(make_pcm16_le({16384, 16384}));
  wait_until([&] {
    std::lock_guard lock(f.output_mu);
    return f.mic_levels.size() == 4;
  }, "post-reconnect mic RMS callback");
  check(std::abs(f.client->last_mic_rms_level() - 0.5) < 1e-9,
        "post-reconnect captured frame did not update last_mic_rms_level");

  f.stop();
  check(f.failure == nullptr, "mic RMS reconnect test failed on stop");
  check(f.client->last_mic_rms_level() == 0.0,
        "cleanup on stop did not reset last_mic_rms_level to 0.0");
  check(!f.client->last_barge_in_cut_point().has_value(),
        "cleanup on stop did not reset last_barge_in_cut_point");
}

using Test = std::pair<const char*, std::function<void()>>;

}  // namespace

int main() {
  const std::vector<Test> tests = {
      {"cursor_monotonicity_and_duplicate_consumption", test_cursor_monotonicity_and_duplicate_consumption},
      {"delayed_and_final_acks", test_delayed_and_final_acks},
      {"partial_consumption_never_acknowledges", test_partial_consumption_never_acknowledges},
      {"cancellation_races_and_stale_callbacks", test_cancellation_races_and_stale_callbacks},
      {"cancel_stops_immediately_during_blocked_ack", test_cancel_stops_immediately_during_blocked_ack},
      {"negotiation_microphone_playback_and_stop_cleanup", test_negotiation_microphone_playback_and_stop_cleanup},
      {"reconnect_cleans_old_session_and_never_replays_audio", test_reconnect_cleans_old_session_and_never_replays_audio},
      {"late_events_and_repeated_cancellation_are_inert", test_late_events_and_repeated_cancellation_are_inert},
      {"speech_started_is_an_immediate_local_barge_in", test_speech_started_is_an_immediate_local_barge_in},
      {"final_ack_retires_response_and_drops_late_text", test_final_ack_retires_response_and_drops_late_text},
      {"error_paths_and_cleanup", test_error_paths_and_cleanup},
      {"external_cancellation_cleans_up", test_external_cancellation_cleans_up},
      {"echo_suppression_default_omits_wire_key", test_echo_suppression_default_omits_wire_key},
      {"echo_suppression_false_negotiation_and_rejection", test_echo_suppression_false_negotiation_and_rejection},
      {"mid_response_echo_suppression_toggle_preserves_active_response", test_mid_response_echo_suppression_toggle_preserves_active_response},
      {"observed_barge_in_count_telemetry", test_observed_barge_in_count_telemetry},
      {"compute_pcm16_rms_level_and_cut_point_format", test_compute_pcm16_rms_level_and_cut_point_format},
      {"barge_in_cut_point_from_playback_ack_and_cancelled_cursor", test_barge_in_cut_point_from_playback_ack_and_cancelled_cursor},
      {"barge_in_cut_point_non_emission_and_invalid_cursor_rejection", test_barge_in_cut_point_non_emission_and_invalid_cursor_rejection},
      {"mic_rms_level_callback_and_reconnect_reset", test_mic_rms_level_callback_and_reconnect_reset},
  };
  size_t passed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      ++passed;
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
      return 1;
    }
  }
  std::cout << "voice_client_test: " << passed << "/" << tests.size() << " passed\n";
  return 0;
}
