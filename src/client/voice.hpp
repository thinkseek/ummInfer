// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "umminfer/client/client.hpp"

namespace umminfer::client::detail {

// The voice transport deliberately exposes only the operations used by the
// full-duplex controller. Tests can therefore reproduce event and failure
// ordering without libcurl or a real server.
class VoiceSocket {
 public:
  virtual ~VoiceSocket() = default;
  virtual void send_text(std::string_view text) = 0;
  virtual void run(const std::function<void(std::string_view)>& on_message) = 0;
  virtual void close() noexcept = 0;
};

class VoiceMicrophone {
 public:
  using PcmCallback = std::function<void(std::string&& pcm16_16khz_mono)>;
  using ErrorCallback = std::function<void(std::exception_ptr)>;
  virtual ~VoiceMicrophone() = default;
  virtual void start(PcmCallback on_pcm, ErrorCallback on_error) = 0;
  virtual void stop() noexcept = 0;
};

struct VoicePcmChunk {
  std::string pcm;
  int sample_rate = 0;
  int channels = 0;
};

class VoiceAudioOutput {
 public:
  // fully_consumed is true only after the last sample has passed the device's
  // playback-consumption boundary. enqueue() itself is non-blocking. stop()
  // must promptly interrupt pending output; invalidated callbacks are harmless.
  using Completion = std::function<void(bool fully_consumed)>;
  virtual ~VoiceAudioOutput() = default;
  virtual void enqueue(VoicePcmChunk chunk, Completion on_consumed) = 0;
  virtual void stop() noexcept = 0;
  virtual void close() noexcept = 0;
};

// Owns one response's isolated playback/ACK state. Device callbacks only mark
// completion; a separate worker serializes ACK writes and final-queue draining.
class VoicePlayback {
 public:
  using AckSink = std::function<void(const std::string& response_id, uint64_t cursor)>;
  using ErrorSink = std::function<void(std::exception_ptr)>;

  VoicePlayback(std::unique_ptr<VoiceAudioOutput> output, AckSink ack_sink,
                ErrorSink error_sink);
  ~VoicePlayback();
  VoicePlayback(const VoicePlayback&) = delete;
  VoicePlayback& operator=(const VoicePlayback&) = delete;

  void begin(const std::string& response_id);
  void push(const std::string& response_id, VoicePcmChunk chunk, uint64_t text_cursor);
  void done(const std::string& response_id, uint64_t final_cursor);
  void cancel(const std::string& response_id);
  void close() noexcept;

  std::string active_response() const;
  bool is_cancelled(const std::string& response_id) const;
  size_t pending_count() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

struct VoiceConfig {
  std::string base_url;
  std::string model;
  std::string api_key;
  std::string voice = "vivian";
  std::string input_device;
  std::string output_device;
  int max_new_tokens = 256;
  int silence_ms = 400;
  Milliseconds connect_timeout{10000};
  int reconnect_attempts = 2;
  Milliseconds reconnect_delay{250};
  CancelFlag cancel = nullptr;
};

struct VoiceHandlers {
  std::function<void(const std::string&)> on_partial;
  std::function<void(const std::string&)> on_transcript;
  std::function<void(const std::string&)> on_response_text;
  std::function<void(const std::string&)> on_response_completed;
  std::function<void(const std::string&)> on_status;
  std::function<void(const std::string&)> on_error;
};

using VoiceSocketFactory = std::function<std::unique_ptr<VoiceSocket>(
    const std::string& url, Milliseconds connect_timeout, CancelFlag connection_cancel,
    CancelFlag external_cancel, const std::string& bearer)>;
using VoiceMicrophoneFactory =
    std::function<std::unique_ptr<VoiceMicrophone>(const std::string& device)>;
using VoiceAudioOutputFactory =
    std::function<std::unique_ptr<VoiceAudioOutput>(const std::string& device)>;

class VoiceClient {
 public:
  VoiceClient(VoiceConfig config, VoiceHandlers handlers = {});
  // Injection is intentional public surface for the deterministic fake-device
  // suite. Production construction uses the libcurl/ALSA/AudioQueue defaults.
  VoiceClient(VoiceConfig config, VoiceHandlers handlers, VoiceSocketFactory socket_factory,
              VoiceMicrophoneFactory microphone_factory,
              VoiceAudioOutputFactory output_factory);
  ~VoiceClient();
  VoiceClient(const VoiceClient&) = delete;
  VoiceClient& operator=(const VoiceClient&) = delete;

  void run();
  void request_stop() noexcept;
  static std::string websocket_url(const std::string& base_url, const std::string& model);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<VoiceMicrophone> make_voice_microphone(const std::string& device);
std::unique_ptr<VoiceAudioOutput> make_voice_audio_output(const std::string& device);

}  // namespace umminfer::client::detail
