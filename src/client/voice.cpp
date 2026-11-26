// SPDX-License-Identifier: Apache-2.0
#include "voice.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "http.hpp"
#include "websocket.hpp"

namespace umminfer::client::detail {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMaxAudioChunkBytes = 16u << 20;
constexpr size_t kMaxBufferedAudioBytes = 64u << 20;
constexpr size_t kMaxBufferedChunks = 4096;
constexpr size_t kMaxCancelledResponses = 64;

uint64_t cursor(const nlohmann::json& event, const char* field) {
  if (!event.contains(field) || !event[field].is_number_integer() ||
      (event[field].is_number_unsigned() &&
       event[field].get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ||
      (!event[field].is_number_unsigned() && event[field].get<int64_t>() < 0)) {
    throw Error(ErrorCode::kBadResponse,
                std::string("realtime voice event has an invalid ") + field);
  }
  return event[field].get<uint64_t>();
}

std::string response_id(const nlohmann::json& event, bool required = true) {
  if (!event.contains("response_id") || !event["response_id"].is_string() ||
      (required && event["response_id"].get_ref<const std::string&>().empty())) {
    if (!required && !event.contains("response_id")) return {};
    throw Error(ErrorCode::kBadResponse, "realtime voice event has an invalid response_id");
  }
  return event["response_id"].get<std::string>();
}

std::string event_error(const nlohmann::json& event) {
  if (event.contains("error") && event["error"].is_object() &&
      event["error"].contains("message") && event["error"]["message"].is_string()) {
    return event["error"]["message"].get<std::string>();
  }
  if (event.contains("error") && event["error"].is_string()) {
    return event["error"].get<std::string>();
  }
  return "unknown realtime server error";
}

class CurlVoiceSocket final : public VoiceSocket {
 public:
  CurlVoiceSocket(const std::string& url, Milliseconds timeout, CancelFlag connection_cancel,
                  CancelFlag external_cancel, const std::string& bearer)
      : socket_(url, timeout, connection_cancel, external_cancel, bearer) {}
  void send_text(std::string_view text) override { socket_.send_text(text); }
  void run(const std::function<void(std::string_view)>& on_message) override {
    socket_.run(on_message);
  }
  void close() noexcept override { socket_.close(); }

 private:
  WebSocket socket_;
};

}  // namespace

double compute_pcm16_rms_level(std::string_view pcm16_le) noexcept {
  const size_t sample_count = pcm16_le.size() / 2;
  if (sample_count == 0) return 0.0;
  double sum_squares = 0.0;
  for (size_t i = 0; i < sample_count; ++i) {
    const uint16_t lo = static_cast<unsigned char>(pcm16_le[i * 2]);
    const uint16_t hi = static_cast<unsigned char>(pcm16_le[i * 2 + 1]);
    const int16_t sample = static_cast<int16_t>(lo | (hi << 8));
    const double normalized = static_cast<double>(sample) / 32768.0;
    sum_squares += normalized * normalized;
  }
  const double rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
  if (!std::isfinite(rms) || rms <= 0.0) return 0.0;
  return std::min(1.0, rms);
}

std::string format_barge_in_cut_point(uint64_t spoken_bytes) {
  if (spoken_bytes > 0) {
    return "Interrupted · " + std::to_string(spoken_bytes) + " B spoken";
  }
  return "Interrupted before audio";
}

struct VoicePlayback::Impl : std::enable_shared_from_this<VoicePlayback::Impl> {
  struct Item {
    uint64_t sequence = 0;
    uint64_t epoch = 0;
    std::string response_id;
    uint64_t text_cursor = 0;
    size_t bytes = 0;
    bool completed = false;
    bool fully_consumed = false;
  };

  Impl(std::unique_ptr<VoiceAudioOutput> device, AckSink sink, ErrorSink errors)
      : output(std::move(device)), ack_sink(std::move(sink)), error_sink(std::move(errors)) {
    if (!output) throw Error(ErrorCode::kConfig, "voice audio output is unavailable");
    if (!ack_sink) {
      throw Error(ErrorCode::kConfig, "voice playback acknowledgement sink is unavailable");
    }
    worker = std::thread([this] { run(); });
  }

  ~Impl() { close(); }

  bool cancelled_locked(const std::string& id) const {
    return cancelled_set.find(id) != cancelled_set.end();
  }

  void remember_cancelled_locked(const std::string& id) {
    if (id.empty() || cancelled_set.count(id)) return;
    cancelled_set.insert(id);
    cancelled_order.push_back(id);
    if (cancelled_order.size() > kMaxCancelledResponses) {
      cancelled_set.erase(cancelled_order.front());
      cancelled_order.pop_front();
    }
  }

  // send_mu orders the state transition with any already-executing socket ACK.
  // Device stop happens after the old response is invalidated, so its stale
  // completion cannot create a new ACK even if the callback was already queued.
  bool begin_locked(const std::string& id) {
    if (closed || id.empty() || cancelled_locked(id)) return false;
    if (active == id) return true;
    remember_cancelled_locked(active);
    ++epoch;
    active = id;
    pending.clear();
    pending_bytes = 0;
    final_cursor.reset();
    final_sent = false;
    incomplete = false;
    last_ack.reset();
    max_cursor = 0;
    return true;
  }

  void begin(const std::string& id) {
    if (id.empty()) throw Error(ErrorCode::kBadResponse, "realtime response id is empty");
    std::lock_guard send_lock(send_mu);
    bool reset = false;
    {
      std::lock_guard lock(mu);
      if (!active.empty() && active != id && !cancelled_locked(id)) reset = true;
      if (!begin_locked(id)) return;
    }
    if (reset) output->stop();
  }

  void push(const std::string& id, VoicePcmChunk chunk, uint64_t text_cursor) {
    if (id.empty()) throw Error(ErrorCode::kBadResponse, "audio.delta response id is empty");
    if (chunk.sample_rate <= 0 || chunk.sample_rate > 192000 || chunk.channels <= 0 ||
        chunk.channels > 8 || chunk.pcm.empty() ||
        chunk.pcm.size() % (static_cast<size_t>(chunk.channels) * 2) != 0) {
      throw Error(ErrorCode::kBadResponse, "audio.delta has invalid PCM16 metadata");
    }
    if (chunk.pcm.size() > kMaxAudioChunkBytes) {
      throw Error(ErrorCode::kBadResponse, "audio.delta exceeds the decoded chunk limit");
    }

    std::lock_guard send_lock(send_mu);
    Item item;
    bool reset = false;
    {
      std::lock_guard lock(mu);
      if (!active.empty() && active != id && !cancelled_locked(id)) reset = true;
      if (!begin_locked(id) || final_cursor) return;
      if (pending.size() >= kMaxBufferedChunks ||
          chunk.pcm.size() > kMaxBufferedAudioBytes -
                                std::min(pending_bytes, kMaxBufferedAudioBytes)) {
        throw Error(ErrorCode::kBadResponse, "realtime voice playback queue is full");
      }
      item.sequence = next_sequence++;
      item.epoch = epoch;
      item.response_id = id;
      item.text_cursor = text_cursor;
      item.bytes = chunk.pcm.size();
      pending_bytes += chunk.pcm.size();
      pending.push_back(item);
      max_cursor = std::max(max_cursor, text_cursor);
    }
    if (reset) output->stop();
    const size_t item_bytes = item.bytes;
    try {
      const uint64_t sequence = item.sequence;
      const uint64_t item_epoch = item.epoch;
      std::weak_ptr<Impl> weak = shared_from_this();
      output->enqueue(std::move(chunk), [weak, sequence, item_epoch](bool consumed) {
        if (auto self = weak.lock()) self->complete(sequence, item_epoch, consumed);
      });
    } catch (...) {
      std::lock_guard lock(mu);
      if (!pending.empty() && pending.back().sequence == item.sequence) {
        pending_bytes -= std::min(pending_bytes, item_bytes);
        pending.pop_back();
      }
      throw;
    }
  }

  void done(const std::string& id, uint64_t text_cursor) {
    if (id.empty()) throw Error(ErrorCode::kBadResponse, "audio.done response id is empty");
    std::lock_guard send_lock(send_mu);
    bool reset = false;
    {
      std::lock_guard lock(mu);
      if (!active.empty() && active != id && !cancelled_locked(id)) reset = true;
      if (!begin_locked(id) || final_cursor) return;
      if (text_cursor < max_cursor) {
        throw Error(ErrorCode::kBadResponse, "audio.done final cursor precedes an audio delta");
      }
      final_cursor = text_cursor;
      max_cursor = std::max(max_cursor, text_cursor);
    }
    if (reset) output->stop();
    cv.notify_all();
  }

  void complete(uint64_t sequence, uint64_t item_epoch, bool consumed) noexcept {
    try {
      std::lock_guard lock(mu);
      if (closed || item_epoch != epoch) return;
      // Normally the front item completes first. Searching also makes delayed
      // and deliberately reordered fake-device callbacks deterministic.
      for (Item& item : pending) {
        if (item.sequence != sequence) continue;
        if (item.completed) return;
        item.completed = true;
        item.fully_consumed = consumed;
        break;
      }
    } catch (...) {
      return;
    }
    cv.notify_all();
  }

  void cancel(const std::string& id) noexcept {
    if (id.empty()) return;
    try {
      bool stop = false;
      {
        std::lock_guard lock(mu);
        if (closed) return;
        if (cancelled_locked(id) && active != id) return;
        remember_cancelled_locked(id);
        if (active == id) {
          active.clear();
          ++epoch;
          pending.clear();
          pending_bytes = 0;
          final_cursor.reset();
          final_sent = true;
          last_ack.reset();
          max_cursor = 0;
          stop = true;
        } else {
          for (const Item& item : pending) {
            if (item.response_id == id) pending_bytes -= std::min(pending_bytes, item.bytes);
          }
          pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const Item& item) {
                          return item.response_id == id;
                        }), pending.end());
        }
      }
      // Stop first, even when an earlier ACK write is blocked. Taking send_mu
      // afterwards guarantees no stale ACK can be written after cancel returns.
      if (stop) output->stop();
      std::lock_guard send_lock(send_mu);
      cv.notify_all();
    } catch (...) {
      // Cancellation is best effort and must remain safe from a signal/EOF path.
    }
  }

  [[noreturn]] void output_incomplete() {
    throw Error(ErrorCode::kConfig,
                "voice audio output stopped before consuming a complete PCM chunk");
  }

  void fail(std::exception_ptr error) noexcept {
    {
      std::lock_guard lock(mu);
      if (closed) return;
      closed = true;
      ++epoch;
      pending.clear();
      pending_bytes = 0;
    }
    output->stop();
    cv.notify_all();
    if (error_sink) {
      try {
        error_sink(error);
      } catch (...) {
      }
    }
  }

  void run() noexcept {
    while (true) {
      Item item;
      bool final = false;
      std::string id;
      uint64_t text_cursor = 0;
      uint64_t item_epoch = 0;
      {
        std::unique_lock lock(mu);
        cv.wait(lock, [this] {
          return closed || (!pending.empty() && pending.front().completed) ||
                 (pending.empty() && final_cursor && !final_sent && !active.empty());
        });
        if (closed) return;
        if (!pending.empty() && pending.front().completed) {
          item = pending.front();
          pending.pop_front();
          pending_bytes -= std::min(pending_bytes, item.bytes);
          id = item.response_id;
          text_cursor = item.text_cursor;
          item_epoch = item.epoch;
          if (!item.fully_consumed) {
            incomplete = true;
            lock.unlock();
            try {
              output_incomplete();
            } catch (...) {
              fail(std::current_exception());
            }
            return;
          }
        } else {
          final = true;
          id = active;
          text_cursor = *final_cursor;
          item_epoch = epoch;
        }
      }

      bool send = false;
      try {
        std::lock_guard send_lock(send_mu);
        {
          std::lock_guard lock(mu);
          if (closed || active != id || epoch != item_epoch || cancelled_locked(id)) continue;
          if (final) {
            if (final_sent || !final_cursor || *final_cursor != text_cursor ||
                !pending.empty() || incomplete || text_cursor < last_ack.value_or(0)) {
              continue;
            }
            final_sent = true;
          } else {
            if (last_ack && text_cursor <= *last_ack) continue;
          }
          last_ack = last_ack ? std::max(*last_ack, text_cursor) : text_cursor;
          send = true;
        }
        if (send) ack_sink(id, text_cursor);
      } catch (...) {
        fail(std::current_exception());
        return;
      }
    }
  }

  void close() noexcept {
    try {
      std::lock_guard send_lock(send_mu);
      {
        std::lock_guard lock(mu);
        closed = true;
        ++epoch;
        active.clear();
        pending.clear();
        pending_bytes = 0;
        final_cursor.reset();
      }
      output->stop();
      cv.notify_all();
    } catch (...) {
    }
    if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) worker.join();
    {
      std::lock_guard lock(mu);
      if (output_closed) return;
      output_closed = true;
    }
    output->close();
  }

  std::unique_ptr<VoiceAudioOutput> output;
  AckSink ack_sink;
  ErrorSink error_sink;
  mutable std::mutex mu;
  std::mutex send_mu;
  std::condition_variable cv;
  std::thread worker;
  std::deque<Item> pending;
  size_t pending_bytes = 0;
  uint64_t next_sequence = 0;
  uint64_t epoch = 0;
  std::string active;
  std::set<std::string> cancelled_set;
  std::deque<std::string> cancelled_order;
  std::optional<uint64_t> final_cursor;
  std::optional<uint64_t> last_ack;
  uint64_t max_cursor = 0;
  bool final_sent = false;
  bool incomplete = false;
  bool closed = false;
  bool output_closed = false;
};

VoicePlayback::VoicePlayback(std::unique_ptr<VoiceAudioOutput> output, AckSink ack_sink,
                             ErrorSink error_sink)
    : impl_(std::make_shared<Impl>(std::move(output), std::move(ack_sink),
                                   std::move(error_sink))) {}
VoicePlayback::~VoicePlayback() { close(); }
void VoicePlayback::begin(const std::string& id) { impl_->begin(id); }
void VoicePlayback::push(const std::string& id, VoicePcmChunk chunk, uint64_t text_cursor) {
  impl_->push(id, std::move(chunk), text_cursor);
}
void VoicePlayback::done(const std::string& id, uint64_t final_cursor) {
  impl_->done(id, final_cursor);
}
void VoicePlayback::cancel(const std::string& id) { impl_->cancel(id); }
void VoicePlayback::close() noexcept { if (impl_) impl_->close(); }
std::string VoicePlayback::active_response() const {
  std::lock_guard lock(impl_->mu);
  return impl_->active;
}
bool VoicePlayback::is_cancelled(const std::string& id) const {
  std::lock_guard lock(impl_->mu);
  return impl_->cancelled_locked(id);
}
size_t VoicePlayback::pending_count() const {
  std::lock_guard lock(impl_->mu);
  return impl_->pending.size();
}

struct VoiceClient::Impl {
  struct Connection : std::enable_shared_from_this<Connection> {
    explicit Connection(Impl* owner_) : owner(owner_) {}
    Impl* owner;
    std::unique_ptr<VoiceSocket> socket;
    std::unique_ptr<VoiceMicrophone> microphone;
    std::unique_ptr<VoicePlayback> playback;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> active{true};
    std::atomic<bool> ready{false};
    std::mutex mu;
    std::exception_ptr error;
    bool microphone_started = false;
    bool initial_negotiated = false;
    bool initial_echo_suppression = true;
    int negotiated_silence_ms = 400;
    std::optional<bool> confirmed_echo_suppression;
    std::string active_response;
    std::map<std::string, uint64_t> acked_text_cursors;
  };

  Impl(VoiceConfig input, VoiceHandlers hooks, VoiceSocketFactory sockets,
       VoiceMicrophoneFactory microphones, VoiceAudioOutputFactory outputs)
      : config(std::move(input)), handlers(std::move(hooks)), socket_factory(std::move(sockets)),
        microphone_factory(std::move(microphones)), output_factory(std::move(outputs)) {
    if (config.model.empty()) throw Error(ErrorCode::kConfig, "voice model is required");
    if (config.voice.empty()) throw Error(ErrorCode::kConfig, "voice name must not be empty");
    if (config.max_new_tokens <= 0 || config.max_new_tokens > 1000000) {
      throw Error(ErrorCode::kConfig, "voice max_new_tokens must be in 1..1000000");
    }
    if (config.silence_ms < 100 || config.silence_ms > 5000) {
      throw Error(ErrorCode::kConfig, "voice silence_ms must be in 100..5000");
    }
    if (config.connect_timeout.count() <= 0 || config.reconnect_delay.count() < 0) {
      throw Error(ErrorCode::kConfig, "voice timeouts are invalid");
    }
    if (config.reconnect_attempts < 0 || config.reconnect_attempts > 10) {
      throw Error(ErrorCode::kConfig, "voice reconnect_attempts must be in 0..10");
    }
    desired_echo_suppression.store(config.echo_suppression, std::memory_order_relaxed);
    negotiated_silence.store(config.silence_ms, std::memory_order_relaxed);
  }

  bool stopping() const {
    return stop.load(std::memory_order_relaxed) ||
           (config.cancel && config.cancel->load(std::memory_order_relaxed));
  }

  void status(const std::string& text) {
    if (handlers.on_status) handlers.on_status(text);
  }

  void note_error(const std::string& text) {
    if (handlers.on_error) handlers.on_error(detail::redact(text, config.api_key));
  }

  void fail(const std::shared_ptr<Connection>& connection, std::exception_ptr error) noexcept {
    if (!connection || !connection->active.exchange(false)) return;
    connection->ready = false;
    connection->cancelled = true;
    {
      std::lock_guard lock(connection->mu);
      if (!connection->error) connection->error = error;
    }
    connection->socket->close();
  }

  void send_pcm(const std::shared_ptr<Connection>& connection, std::string&& pcm) {
    if (pcm.empty() || stopping() || !connection->active || !connection->ready) return;
    if (pcm.size() % 2 != 0) {
      fail(connection, std::make_exception_ptr(
                           Error(ErrorCode::kBadResponse, "microphone returned an odd PCM16 byte count")));
      return;
    }
    try {
      const double rms = compute_pcm16_rms_level(pcm);
      last_mic_rms.store(rms, std::memory_order_relaxed);
      if (handlers.on_mic_level) handlers.on_mic_level(rms);
      static const size_t max_append_pcm = [] {
        constexpr size_t max_message = 1u << 20;
        const size_t envelope = nlohmann::json{
            {"type", "input_audio_buffer.append"}, {"audio", ""}}.dump().size();
        return (((max_message - envelope) / 4) * 3) & ~size_t{1};
      }();
      for (size_t offset = 0; offset < pcm.size();) {
        const size_t count = std::min(max_append_pcm, pcm.size() - offset);
        const nlohmann::json message = {
            {"type", "input_audio_buffer.append"},
            {"audio", detail::base64_encode(std::string_view(pcm).substr(offset, count))}};
        connection->socket->send_text(message.dump());
        offset += count;
      }
    } catch (...) {
      if (connection->active && !stopping()) fail(connection, std::current_exception());
    }
  }

  void cancel_response(const std::shared_ptr<Connection>& connection, const std::string& id) {
    if (id.empty()) return;
    {
      std::lock_guard lock(connection->mu);
      if (connection->active_response == id) connection->active_response.clear();
    }
    connection->playback->cancel(id);
  }

  void begin_response(const std::shared_ptr<Connection>& connection, const std::string& id) {
    connection->playback->begin(id);
    std::lock_guard lock(connection->mu);
    if (!connection->playback->is_cancelled(id)) connection->active_response = id;
  }

  void start_microphone(const std::shared_ptr<Connection>& connection, bool echo_suppression) {
    {
      std::lock_guard lock(connection->mu);
      if (connection->microphone_started) return;
      connection->microphone_started = true;
    }
    std::weak_ptr<Connection> weak = connection;
    // AudioQueue/ALSA may deliver immediately from start(): accept those first
    // samples instead of clipping the utterance while ready is still false.
    connection->ready = true;
    connection->microphone->start([weak](std::string&& pcm) {
      if (auto current = weak.lock(); current && current->active) {
        current->owner->send_pcm(current, std::move(pcm));
      }
    }, [weak](std::exception_ptr error) {
      if (auto current = weak.lock(); current && current->active) {
        current->owner->fail(current, error);
      }
    });
    status(echo_suppression ? "Listening (echo suppression: on)"
                            : "Listening (echo suppression: off)");
  }

  void handle_event(const std::shared_ptr<Connection>& connection, std::string_view message) {
    if (!connection->active || stopping()) return;
    nlohmann::json event = nlohmann::json::parse(message);
    if (!event.is_object() || !event.contains("type") || !event["type"].is_string()) return;
    const std::string type = event["type"].get<std::string>();

    if (type == "transcription_session.updated") {
      if (!event.contains("session") || !event["session"].is_object() ||
          !event["session"].contains("audio_output") ||
          !event["session"]["audio_output"].is_object() ||
          !event["session"]["audio_output"].contains("playback_ack") ||
          !event["session"]["audio_output"]["playback_ack"].is_boolean() ||
          !event["session"]["audio_output"]["playback_ack"].get<bool>()) {
        throw Error(ErrorCode::kConfig,
                    "server did not negotiate audio_output.playback_ack=true");
      }
      const auto& session = event["session"];
      int silence_ms = config.silence_ms;
      bool echo_suppression = true;
      bool initial = false;
      bool requested_initial_echo = true;
      {
        std::lock_guard lock(connection->mu);
        silence_ms = connection->negotiated_silence_ms;
        initial = !connection->initial_negotiated;
        requested_initial_echo = connection->initial_echo_suppression;
      }
      if (session.contains("turn_detection")) {
        const auto& turn = session["turn_detection"];
        if (!turn.is_object()) {
          throw Error(ErrorCode::kBadResponse,
                      "transcription_session.updated has invalid turn_detection");
        }
        if (turn.contains("silence_ms")) {
          const auto& raw = turn["silence_ms"];
          const bool valid_silence =
              raw.is_number_integer() &&
              (raw.is_number_unsigned()
                   ? raw.get<uint64_t>() >= 100 && raw.get<uint64_t>() <= 5000
                   : raw.get<int64_t>() >= 100 && raw.get<int64_t>() <= 5000);
          if (!valid_silence) {
            throw Error(ErrorCode::kBadResponse,
                        "transcription_session.updated has invalid turn_detection.silence_ms");
          }
          silence_ms = raw.get<int>();
        }
        if (turn.contains("echo_suppression")) {
          if (!turn["echo_suppression"].is_boolean()) {
            throw Error(
                ErrorCode::kBadResponse,
                "transcription_session.updated has non-boolean turn_detection.echo_suppression");
          }
          echo_suppression = turn["echo_suppression"].get<bool>();
        }
      }
      if (initial && !requested_initial_echo && echo_suppression) {
        throw Error(ErrorCode::kConfig,
                    "server did not apply turn_detection.echo_suppression=false");
      }
      {
        std::lock_guard lock(connection->mu);
        connection->initial_negotiated = true;
        connection->negotiated_silence_ms = silence_ms;
        connection->confirmed_echo_suppression = echo_suppression;
      }
      negotiated_silence.store(silence_ms, std::memory_order_relaxed);
      confirmed_echo_state.store(echo_suppression ? 1 : 0, std::memory_order_relaxed);
      if (handlers.on_echo_suppression) handlers.on_echo_suppression(echo_suppression);
      if (initial) {
        start_microphone(connection, echo_suppression);
      } else {
        status(echo_suppression ? "Echo suppression: on" : "Echo suppression: off");
      }
      return;
    }
    if (type == "input_audio_buffer.speech_started") {
      std::string id;
      if (event.contains("interrupted_response_id")) {
        if (!event["interrupted_response_id"].is_string()) {
          throw Error(ErrorCode::kBadResponse,
                      "speech_started has an invalid interrupted_response_id");
        }
        id = event["interrupted_response_id"].get<std::string>();
      } else {
        // Compatibility with the initial L1 event, which had no explicit id.
        std::lock_guard lock(connection->mu);
        id = connection->active_response;
      }
      cancel_response(connection, id);
      status("Interrupted; listening");
      return;
    }
    if (type == "transcription.segment") {
      if (event.contains("text") && event["text"].is_string() && handlers.on_partial) {
        handlers.on_partial(event["text"].get<std::string>());
      }
      return;
    }
    if (type == "transcription.completed") {
      if (event.contains("text") && event["text"].is_string() && handlers.on_transcript) {
        handlers.on_transcript(event["text"].get<std::string>());
      }
      status("Thinking");
      return;
    }
    if (type == "input_audio_buffer.speech_stopped") {
      status("Hearing");
      return;
    }
    if (type == "error") {
      throw Error(ErrorCode::kServer, "realtime voice error: " + event_error(event), false);
    }
    static const std::set<std::string> response_events = {
        "response.text.delta", "audio.delta", "audio.done", "response.completed",
        "response.cancelled", "response.failed", "response.playback_ack"};
    if (!response_events.count(type)) return;

    const bool cancellation = type == "response.cancelled";
    const std::string id = response_id(event, !cancellation && type != "response.failed");
    if (!id.empty() && !cancellation && connection->playback->is_cancelled(id)) return;

    if (type == "response.text.delta") {
      if (!event.contains("text") || !event["text"].is_string()) {
        throw Error(ErrorCode::kBadResponse, "response.text.delta is missing text");
      }
      begin_response(connection, id);
      if (handlers.on_response_text) handlers.on_response_text(event["text"].get<std::string>());
    } else if (type == "audio.delta") {
      if (!event.contains("audio") || !event["audio"].is_string() ||
          !event.contains("format") || event["format"] != "pcm16" ||
          !event.contains("sample_rate") || !event["sample_rate"].is_number_integer() ||
          !event.contains("channels") || !event["channels"].is_number_integer()) {
        throw Error(ErrorCode::kBadResponse, "audio.delta requires PCM16 audio metadata");
      }
      VoicePcmChunk chunk;
      chunk.pcm = detail::base64_decode(event["audio"].get<std::string>(), kMaxAudioChunkBytes);
      chunk.sample_rate = event["sample_rate"].get<int>();
      chunk.channels = event["channels"].get<int>();
      const uint64_t text_cursor = cursor(event, "text_cursor");
      begin_response(connection, id);
      connection->playback->push(id, std::move(chunk), text_cursor);
      status("Speaking");
    } else if (type == "audio.done") {
      connection->playback->done(id, cursor(event, "text_cursor"));
    } else if (type == "response.completed") {
      if (handlers.on_response_completed) handlers.on_response_completed(id);
    } else if (type == "response.cancelled") {
      uint64_t event_cursor = 0;
      if (event.contains("text_cursor")) {
        event_cursor = cursor(event, "text_cursor");
      }
      barge_ins.fetch_add(1, std::memory_order_relaxed);
      std::string target = id;
      uint64_t acked_cursor = 0;
      {
        std::lock_guard lock(connection->mu);
        if (target.empty()) target = connection->active_response;
        if (!target.empty()) {
          const auto it = connection->acked_text_cursors.find(target);
          if (it != connection->acked_text_cursors.end()) {
            acked_cursor = it->second;
            connection->acked_text_cursors.erase(it);
          }
        }
      }
      cancel_response(connection, target);
      const uint64_t spoken_bytes = std::max(acked_cursor, event_cursor);
      VoiceBargeInCutPoint cut_point{target, spoken_bytes,
                                     format_barge_in_cut_point(spoken_bytes)};
      {
        std::lock_guard lock(telemetry_mu);
        last_cut_point = cut_point;
      }
      if (handlers.on_response_cancelled) handlers.on_response_cancelled(cut_point);
      status("Interrupted; listening");
    } else if (type == "response.failed") {
      std::string target = id;
      {
        std::lock_guard lock(connection->mu);
        if (target.empty()) target = connection->active_response;
        if (!target.empty()) connection->acked_text_cursors.erase(target);
      }
      cancel_response(connection, target);
      note_error("Realtime response failed: " + event_error(event));
      status("Listening");
    } else if (type == "response.playback_ack") {
      const uint64_t ack_cursor = cursor(event, "text_cursor");
      const bool final = event.value("final", false);
      {
        std::lock_guard lock(connection->mu);
        if (final) {
          if (!id.empty()) connection->acked_text_cursors.erase(id);
        } else if (!id.empty()) {
          auto& tracked = connection->acked_text_cursors[id];
          tracked = std::max(tracked, ack_cursor);
        }
      }
      if (final) {
        // Generation completion is not retirement: only the server's final ACK
        // echo makes all later frames for this response stale.
        cancel_response(connection, id);
        status("Listening");
      }
    }
  }

  void cleanup(const std::shared_ptr<Connection>& connection) noexcept {
    if (!connection) return;
    connection->active = false;
    connection->ready = false;
    connection->cancelled = true;
    if (connection->microphone) connection->microphone->stop();
    if (connection->playback) connection->playback->close();
    if (connection->socket) connection->socket->close();
    last_mic_rms.store(0.0, std::memory_order_relaxed);
    try {
      std::lock_guard lock(telemetry_mu);
      last_cut_point.reset();
    } catch (...) {
    }
    try {
      std::lock_guard lock(connection->mu);
      connection->acked_text_cursors.clear();
    } catch (...) {
    }
  }

  void run() {
    const std::string url = websocket_url(config.base_url, config.model);
    int reconnects = 0;
    while (!stopping()) {
      barge_ins.store(0, std::memory_order_relaxed);
      confirmed_echo_state.store(-1, std::memory_order_relaxed);
      negotiated_silence.store(config.silence_ms, std::memory_order_relaxed);
      last_mic_rms.store(0.0, std::memory_order_relaxed);
      {
        std::lock_guard lock(telemetry_mu);
        last_cut_point.reset();
      }
      auto connection = std::make_shared<Connection>(this);
      {
        std::lock_guard lock(current_mu);
        current = connection;
      }
      std::exception_ptr failure;
      try {
        connection->socket = socket_factory(url, config.connect_timeout, &connection->cancelled,
                                            config.cancel, config.api_key);
        if (!connection->socket) {
          throw Error(ErrorCode::kConfig, "voice WebSocket factory returned no socket");
        }
        connection->microphone = microphone_factory(config.input_device);
        if (!connection->microphone) {
          throw Error(ErrorCode::kConfig, "voice microphone is unavailable");
        }
        std::weak_ptr<Connection> weak = connection;
        connection->playback = std::make_unique<VoicePlayback>(
            output_factory(config.output_device),
            [weak](const std::string& id, uint64_t text_cursor) {
              auto current = weak.lock();
              if (!current || !current->active || current->cancelled) return;
              const nlohmann::json ack = {{"type", "output_audio_buffer.playback_ack"},
                                          {"response_id", id}, {"text_cursor", text_cursor}};
              current->socket->send_text(ack.dump());
            },
            [weak](std::exception_ptr error) {
              if (auto current = weak.lock()) current->owner->fail(current, error);
            });
        const bool want_echo_suppression =
            desired_echo_suppression.load(std::memory_order_relaxed);
        {
          std::lock_guard lock(connection->mu);
          connection->initial_echo_suppression = want_echo_suppression;
          connection->negotiated_silence_ms = config.silence_ms;
        }
        nlohmann::json turn_detection = {{"type", "server_vad"},
                                         {"silence_ms", config.silence_ms}};
        if (!want_echo_suppression) {
          turn_detection["echo_suppression"] = false;
        }
        const nlohmann::json update = {
            {"type", "session.update"},
            {"session", {{"input_audio_format", "pcm16"},
                         {"turn_detection", std::move(turn_detection)},
                         {"max_new_tokens", config.max_new_tokens},
                         {"audio_output", {{"voice", config.voice},
                                           {"playback_ack", true}}}}}};
        status(reconnects == 0 ? "Negotiating realtime voice" : "Reconnecting realtime voice");
        connection->socket->send_text(update.dump());
        connection->socket->run([this, connection](std::string_view message) {
          try {
            handle_event(connection, message);
          } catch (...) {
            fail(connection, std::current_exception());
          }
        });
        if (!failure) {
          std::lock_guard lock(connection->mu);
          failure = connection->error;
        }
        if (!failure && !stopping()) {
          failure = std::make_exception_ptr(
              Error(ErrorCode::kNetwork, "realtime voice WebSocket disconnected", true));
        }
      } catch (...) {
        failure = std::current_exception();
      }
      cleanup(connection);
      {
        std::lock_guard lock(current_mu);
        current.reset();
      }
      if (stopping()) break;
      if (!failure) continue;

      bool retryable = true;
      std::string message = "realtime voice connection failed";
      try {
        std::rethrow_exception(failure);
      } catch (const Error& error) {
        retryable = error.retryable() || error.code() == ErrorCode::kNetwork ||
                    error.code() == ErrorCode::kBadResponse ||
                    error.code() == ErrorCode::kServer || error.code() == ErrorCode::kTimeout;
        message = detail::redact(error.what(), config.api_key);
      } catch (const std::exception& error) {
        retryable = false;
        message = detail::redact(error.what(), config.api_key);
      }
      if (!retryable || reconnects >= config.reconnect_attempts) {
        std::rethrow_exception(failure);
      }
      ++reconnects;
      note_error(message + "; reconnecting");
      const auto deadline = Clock::now() + config.reconnect_delay;
      while (!stopping() && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (config.cancel && config.cancel->load(std::memory_order_relaxed)) {
      throw Error(ErrorCode::kCancelled, "request cancelled");
    }
  }

  void request_stop() noexcept {
    stop = true;
    std::shared_ptr<Connection> connection;
    {
      std::lock_guard lock(current_mu);
      connection = current.lock();
    }
    if (connection) {
      connection->active = false;
      connection->ready = false;
      connection->cancelled = true;
      if (connection->playback) connection->playback->cancel(connection->playback->active_response());
      if (connection->socket) connection->socket->close();
    }
  }

  void set_echo_suppression(bool enabled) {
    desired_echo_suppression.store(enabled, std::memory_order_relaxed);
    std::shared_ptr<Connection> connection;
    {
      std::lock_guard lock(current_mu);
      connection = current.lock();
    }
    if (!connection || !connection->active || !connection->ready || !connection->socket ||
        stopping()) {
      return;
    }
    int silence_ms = config.silence_ms;
    {
      std::lock_guard lock(connection->mu);
      silence_ms = connection->negotiated_silence_ms;
    }
    const nlohmann::json update = {
        {"type", "session.update"},
        {"session",
         {{"turn_detection",
           {{"type", "server_vad"},
            {"silence_ms", silence_ms},
            {"echo_suppression", enabled}}}}}};
    try {
      connection->socket->send_text(update.dump());
    } catch (...) {
      if (connection->active && !stopping()) fail(connection, std::current_exception());
    }
  }

  std::optional<bool> confirmed_echo_suppression() const noexcept {
    const int state = confirmed_echo_state.load(std::memory_order_relaxed);
    if (state < 0) return std::nullopt;
    return state != 0;
  }

  int negotiated_silence_ms() const noexcept {
    return negotiated_silence.load(std::memory_order_relaxed);
  }

  uint64_t barge_in_count() const noexcept {
    return barge_ins.load(std::memory_order_relaxed);
  }

  std::optional<VoiceBargeInCutPoint> last_barge_in_cut_point() const noexcept {
    try {
      std::lock_guard lock(telemetry_mu);
      return last_cut_point;
    } catch (...) {
      return std::nullopt;
    }
  }

  double last_mic_rms_level() const noexcept {
    return last_mic_rms.load(std::memory_order_relaxed);
  }

  VoiceConfig config;
  VoiceHandlers handlers;
  VoiceSocketFactory socket_factory;
  VoiceMicrophoneFactory microphone_factory;
  VoiceAudioOutputFactory output_factory;
  std::atomic<bool> stop{false};
  std::atomic<bool> desired_echo_suppression{true};
  std::atomic<int> confirmed_echo_state{-1};
  std::atomic<int> negotiated_silence{400};
  std::atomic<uint64_t> barge_ins{0};
  std::atomic<double> last_mic_rms{0.0};
  mutable std::mutex telemetry_mu;
  std::optional<VoiceBargeInCutPoint> last_cut_point;
  std::mutex current_mu;
  std::weak_ptr<Connection> current;
};

VoiceClient::VoiceClient(VoiceConfig config, VoiceHandlers handlers)
    : VoiceClient(std::move(config), std::move(handlers),
                  [](const std::string& url, Milliseconds timeout, CancelFlag connection_cancel,
                     CancelFlag external_cancel, const std::string& bearer) {
                    return std::make_unique<CurlVoiceSocket>(url, timeout, connection_cancel,
                                                             external_cancel, bearer);
                  },
                  [](const std::string& device) { return make_voice_microphone(device); },
                  [](const std::string& device) { return make_voice_audio_output(device); }) {}

VoiceClient::VoiceClient(VoiceConfig config, VoiceHandlers handlers,
                         VoiceSocketFactory socket_factory,
                         VoiceMicrophoneFactory microphone_factory,
                         VoiceAudioOutputFactory output_factory)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(handlers),
                                   std::move(socket_factory), std::move(microphone_factory),
                                   std::move(output_factory))) {}
VoiceClient::~VoiceClient() { request_stop(); }
void VoiceClient::run() { impl_->run(); }
void VoiceClient::request_stop() noexcept { if (impl_) impl_->request_stop(); }
void VoiceClient::set_echo_suppression(bool enabled) {
  if (impl_) impl_->set_echo_suppression(enabled);
}
std::optional<bool> VoiceClient::confirmed_echo_suppression() const noexcept {
  return impl_ ? impl_->confirmed_echo_suppression() : std::nullopt;
}
int VoiceClient::negotiated_silence_ms() const noexcept {
  return impl_ ? impl_->negotiated_silence_ms() : 400;
}
uint64_t VoiceClient::barge_in_count() const noexcept {
  return impl_ ? impl_->barge_in_count() : 0;
}
std::optional<VoiceBargeInCutPoint> VoiceClient::last_barge_in_cut_point() const noexcept {
  return impl_ ? impl_->last_barge_in_cut_point() : std::nullopt;
}
double VoiceClient::last_mic_rms_level() const noexcept {
  return impl_ ? impl_->last_mic_rms_level() : 0.0;
}

std::string VoiceClient::websocket_url(const std::string& base_url, const std::string& model) {
  if (model.empty()) throw Error(ErrorCode::kConfig, "voice model is required");
  std::string url = detail::validate_base_url(base_url, true);
  if (url.rfind("http://", 0) != 0) {
    throw Error(ErrorCode::kConfig, "realtime voice requires loopback http://");
  }
  url.replace(0, 4, "ws");
  url += "/v1/realtime?intent=transcription&model=" + detail::url_encode(model);
  return url;
}

}  // namespace umminfer::client::detail
