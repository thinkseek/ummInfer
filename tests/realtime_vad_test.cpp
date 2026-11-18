// SPDX-License-Identifier: Apache-2.0
// Blackbox API tests for the /v1/realtime WebSocket endpoint: server_vad turn
// detection (silence, speech onset, split 10 ms frames, silence_ms hold, noise
// floor adaptation), spoken-response multi-turn session_history, WAV-to-PCM16
// artifact stripping, TTS leg-failure propagation, and public WAV parsing.
#include <curl/curl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "umminfer/audio/audio.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/realtime_api.hpp"
#include "umminfer/util/codec.hpp"

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

int g_failures = 0;

void check(bool condition, const std::string& message) {
  if (condition) return;
  std::fprintf(stderr, "FAIL %s\n", message.c_str());
  ++g_failures;
}

std::string tone(int milliseconds, double dbfs, double hz = 440.0) {
  const int samples = milliseconds * 16;
  const double amplitude = 32767.0 * std::pow(10.0, dbfs / 20.0) * std::sqrt(2.0);
  std::string out;
  out.resize(static_cast<size_t>(samples) * 2);
  for (int i = 0; i < samples; ++i) {
    const double value =
        amplitude * std::sin(2.0 * 3.14159265358979323846 * hz * i / 16000.0);
    const auto sample = static_cast<int16_t>(
        value > 32767.0 ? 32767.0 : value < -32768.0 ? -32768.0 : value);
    out[static_cast<size_t>(i) * 2] = static_cast<char>(sample & 0xff);
    out[static_cast<size_t>(i) * 2 + 1] =
        static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xff);
  }
  return out;
}

std::string silence(int milliseconds) {
  return std::string(static_cast<size_t>(milliseconds) * 32, '\0');
}

class WsClient {
 public:
  explicit WsClient(const std::string& url) {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    curl_ = curl_easy_init();
    if (!curl_) throw std::runtime_error("curl_easy_init failed");
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_CONNECT_ONLY, 2L);
    curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl_, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl_, CURLOPT_PROTOCOLS_STR, "ws");
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, 5000L);
    const CURLcode rc = curl_easy_perform(curl_);
    if (rc != CURLE_OK) {
      curl_easy_cleanup(curl_);
      curl_ = nullptr;
      throw std::runtime_error(std::string("ws connect failed: ") +
                               curl_easy_strerror(rc));
    }
    long status = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &status);
    if (status != 101) {
      curl_easy_cleanup(curl_);
      curl_ = nullptr;
      throw std::runtime_error("ws upgrade status " + std::to_string(status));
    }
    curl_easy_getinfo(curl_, CURLINFO_ACTIVESOCKET, &sock_);
  }

  ~WsClient() {
    if (curl_) {
      size_t sent = 0;
      const unsigned char normal[] = {0x03, 0xe8};
      (void)curl_ws_send(curl_, normal, sizeof(normal), &sent, 0, CURLWS_CLOSE);
      curl_easy_cleanup(curl_);
    }
  }

  void send_json(const json& payload) {
    const std::string text = payload.dump();
    size_t offset = 0;
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (offset < text.size()) {
      size_t sent = 0;
      const CURLcode rc =
          curl_ws_send(curl_, text.data() + offset, text.size() - offset, &sent,
                       0, CURLWS_TEXT);
      if (rc == CURLE_OK) {
        offset += sent;
        continue;
      }
      if (rc == CURLE_AGAIN && Clock::now() < deadline) {
        poll_sock(POLLOUT, 20);
        continue;
      }
      throw std::runtime_error(std::string("ws send failed: ") +
                               curl_easy_strerror(rc));
    }
  }

  std::optional<json> recv_json(int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    std::string message;
    while (true) {
      char buf[16384];
      size_t got = 0;
      const curl_ws_frame* meta = nullptr;
      const CURLcode rc = curl_ws_recv(curl_, buf, sizeof(buf), &got, &meta);
      if (rc == CURLE_OK && meta != nullptr) {
        if (meta->flags & CURLWS_CLOSE) return std::nullopt;
        if (meta->flags & CURLWS_PING) {
          size_t sent = 0;
          (void)curl_ws_send(curl_, buf, got, &sent, 0, CURLWS_PONG);
          continue;
        }
        if (meta->flags & CURLWS_TEXT) {
          message.append(buf, got);
          if (meta->bytesleft == 0 && !(meta->flags & CURLWS_CONT)) {
            return json::parse(message);
          }
        }
        continue;
      }
      if (rc == CURLE_AGAIN) {
        const auto now = Clock::now();
        if (now >= deadline) return std::nullopt;
        const int wait_ms = static_cast<int>(
            std::min<int64_t>(25, std::chrono::duration_cast<std::chrono::milliseconds>(
                                      deadline - now)
                                      .count()));
        poll_sock(POLLIN, std::max(1, wait_ms));
        continue;
      }
      return std::nullopt;
    }
  }

  std::vector<json> drain(int window_ms) {
    std::vector<json> events;
    const auto deadline = Clock::now() + std::chrono::milliseconds(window_ms);
    while (Clock::now() < deadline) {
      const int left = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                Clock::now())
              .count());
      auto next = recv_json(std::max(1, left));
      if (!next) break;
      events.push_back(std::move(*next));
    }
    return events;
  }

 private:
  void poll_sock(short events, int timeout_ms) {
    if (sock_ == CURL_SOCKET_BAD) {
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return;
    }
    pollfd pfd{};
    pfd.fd = static_cast<int>(sock_);
    pfd.events = events;
    (void)::poll(&pfd, 1, timeout_ms);
  }

  CURL* curl_ = nullptr;
  curl_socket_t sock_ = CURL_SOCKET_BAD;
};

struct RealtimeHarness {
  std::string ipc_dir;
  std::unique_ptr<umminfer::Coordinator> coordinator;
  std::unique_ptr<umminfer::Stage> asr_stage;
  std::unique_ptr<umminfer::Stage> vlm_stage;
  std::unique_ptr<umminfer::Stage> tts_stage;
  std::unique_ptr<umminfer::UnifiedEngine> unified;
  std::unique_ptr<umminfer::RealtimeApi> api;
  std::thread asr_thread;
  std::thread vlm_thread;
  std::thread tts_thread;
  std::thread coord_thread;
  std::thread api_thread;

  std::atomic<int> asr_finals{0};
  std::atomic<bool> fail_tts{false};
  std::mutex mu;
  std::vector<size_t> vlm_message_counts;
  std::string tts_pcm_payload = std::string("\x10\x00\x20\x00\x30\x00\x40\x00", 8);

  RealtimeHarness() {
    static std::atomic<int> seq{0};
    ipc_dir = "/tmp/umm-rt-bb-" + std::to_string(::getpid()) + "-" +
              std::to_string(++seq);
    std::filesystem::create_directories(ipc_dir);
    const std::string done_ep = "ipc://" + ipc_dir + "/done";
    const std::string abort_ep = "ipc://" + ipc_dir + "/abort";
    const std::string asr_ep = "ipc://" + ipc_dir + "/asr";
    const std::string vlm_ep = "ipc://" + ipc_dir + "/vlm";
    const std::string tts_ep = "ipc://" + ipc_dir + "/tts";

    coordinator = std::make_unique<umminfer::Coordinator>(done_ep, abort_ep);
    coordinator->register_stage("asr", asr_ep, /*require_ready=*/true);
    coordinator->register_stage("vlm", vlm_ep, /*require_ready=*/true);
    coordinator->register_stage("tts", tts_ep, /*require_ready=*/true);

    umminfer::UnifiedConfig ucfg;
    ucfg.asr_stage = "asr";
    ucfg.vlm_stage = "vlm";
    ucfg.tts_stage = "tts";
    ucfg.session_history.enabled = true;
    ucfg.session_history.max_turns = 8;
    unified = std::make_unique<umminfer::UnifiedEngine>(*coordinator, ucfg);

    umminfer::ApiConfig acfg;
    acfg.host = "127.0.0.1";
    acfg.port = 0;
    acfg.model = "served-model";
    acfg.entry_stage = "asr";
    acfg.request_timeout_s = 5.0;
    api = std::make_unique<umminfer::RealtimeApi>(*coordinator, acfg);
    api->attach_unified(*unified);
    api->install_stream_sink();
    unified->on_event = [this](const umminfer::UnifiedEvent& ev) {
      api->forward_event(ev);
    };
    unified->on_terminal = [this](const umminfer::UnifiedEvent& ev) {
      api->forward_event(ev);
    };

    coordinator->start();

    auto make_stage_cfg = [&](const std::string& name, const std::string& ep) {
      umminfer::Stage::Config scfg;
      scfg.name = name;
      scfg.recv_endpoint = ep;
      scfg.coordinator_endpoint = done_ep;
      scfg.abort_endpoint = abort_ep;
      scfg.announce_ready = true;
      return scfg;
    };

    asr_stage = std::make_unique<umminfer::Stage>(
        make_stage_cfg("asr", asr_ep),
        [this](const umminfer::Value&, const umminfer::AbortCheck&) {
          const std::string req_id = umminfer::bound_request_id();
          const bool is_preview = req_id.rfind("realtime-preview-", 0) == 0;
          const int idx = is_preview ? asr_finals.load() + 1 : ++asr_finals;
          umminfer::Value::Map out;
          out["text"] = umminfer::Value("turn-" + std::to_string(idx));
          out["generated_token_count"] = umminfer::Value(int64_t{2});
          out["finish_reason"] = umminfer::Value("stop");
          return umminfer::Value(std::move(out));
        });

    vlm_stage = std::make_unique<umminfer::Stage>(
        make_stage_cfg("vlm", vlm_ep),
        [this](const umminfer::Value& work, const umminfer::AbortCheck&) {
          size_t msg_count = 0;
          if (const auto* msgs = work.find("messages");
              msgs != nullptr && msgs->is_array()) {
            msg_count = msgs->as_array().size();
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            vlm_message_counts.push_back(msg_count);
          }
          umminfer::Value::Map out;
          out["content"] = umminfer::Value("spoken reply.");
          out["finish_reason"] = umminfer::Value("stop");
          return umminfer::Value(std::move(out));
        });

    tts_stage = std::make_unique<umminfer::Stage>(
        make_stage_cfg("tts", tts_ep),
        [this](const umminfer::Value&, const umminfer::AbortCheck&) {
          if (fail_tts.load()) {
            throw std::runtime_error("synthetic tts failure");
          }
          // Return a 24 kHz WAV container WITHOUT "streamed": true so
          // RealtimeApi exercises headerless PCM16 extraction on kArtifact.
          std::string wav = umminfer::audio::wav_from_pcm16(tts_pcm_payload, 24000);
          umminfer::Value::Map out;
          out["media_type"] = umminfer::Value("audio");
          out["mime_type"] = umminfer::Value("audio/wav");
          out["format"] = umminfer::Value("wav");
          out["sample_rate"] = umminfer::Value(int64_t{24000});
          out["channels"] = umminfer::Value(int64_t{1});
          out["data"] = umminfer::Value::bin(std::move(wav));
          return umminfer::Value(std::move(out));
        });

    asr_stage->start();
    vlm_stage->start();
    tts_stage->start();
    asr_thread = std::thread([this] { asr_stage->run(); });
    vlm_thread = std::thread([this] { vlm_stage->run(); });
    tts_thread = std::thread([this] { tts_stage->run(); });

    std::thread coord_runner([this] { coordinator->run(); });
    coord_thread = std::move(coord_runner);
    const auto ready_deadline = Clock::now() + std::chrono::seconds(5);
    while (Clock::now() < ready_deadline) {
      if (coordinator->stage_ready("asr") &&
          coordinator->stage_ready("vlm") &&
          coordinator->stage_ready("tts")) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!api->bind()) throw std::runtime_error("RealtimeApi bind failed");
    api_thread = std::thread([this] { api->listen(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  ~RealtimeHarness() {
    if (api) api->stop();
    if (api_thread.joinable()) api_thread.join();
    if (asr_stage) asr_stage->stop();
    if (vlm_stage) vlm_stage->stop();
    if (tts_stage) tts_stage->stop();
    if (asr_thread.joinable()) asr_thread.join();
    if (vlm_thread.joinable()) vlm_thread.join();
    if (tts_thread.joinable()) tts_thread.join();
    if (coordinator) coordinator->stop();
    if (coord_thread.joinable()) coord_thread.join();
    std::error_code ec;
    std::filesystem::remove_all(ipc_dir, ec);
  }

  std::string url() const {
    return "ws://127.0.0.1:" + std::to_string(api->bound_port()) +
           "/v1/realtime?intent=transcription&model=served-model";
  }
};

void append_pcm(WsClient& ws, std::string_view pcm) {
  ws.send_json({{"type", "input_audio_buffer.append"},
                {"audio", umminfer::util::base64_encode(pcm)}});
}

bool has_type(const std::vector<json>& events, const std::string& type) {
  for (const auto& ev : events) {
    if (ev.value("type", "") == type) return true;
  }
  return false;
}

void test_vad_over_websocket(RealtimeHarness& h) {
  // 1. Digital silence never triggers speech_started; split 10 ms frame does.
  {
    WsClient ws(h.url());
    ws.send_json(
        {{"type", "session.update"},
         {"session",
          {{"input_audio_format", "pcm16"},
           {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 500}}}}}});
    const auto updated = ws.recv_json(2000);
    check(updated.has_value() &&
              updated->value("type", "") == "transcription_session.updated",
          "session.update acknowledged server_vad");

    const std::string silent = silence(2000);
    for (size_t off = 0; off < silent.size(); off += 3200) {
      append_pcm(ws, std::string_view(silent).substr(off, 3200));
    }
    const auto during_silence = ws.drain(80);
    check(!has_type(during_silence, "input_audio_buffer.speech_started"),
          "digital silence produced no speech_started event over WebSocket");

    // Split the first 10 ms voiced frame (320 bytes) across two 160-byte appends.
    const std::string voiced = tone(300, -18.0);
    append_pcm(ws, std::string_view(voiced).substr(0, 160));
    check(!has_type(ws.drain(50), "input_audio_buffer.speech_started"),
          "half of a 10 ms frame did not prematurely trigger speech_started");

    append_pcm(ws, std::string_view(voiced).substr(160, 160));
    const auto onset = ws.recv_json(1500);
    check(onset.has_value() &&
              onset->value("type", "") == "input_audio_buffer.speech_started",
          "second half of split 10 ms voiced frame triggered speech_started");

    // Finish the 300 ms utterance, then pause 400 ms (< silence_ms=500): turn stays open.
    append_pcm(ws, std::string_view(voiced).substr(320));
    append_pcm(ws, silence(400));
    const auto short_pause = ws.drain(80);
    check(!has_type(short_pause, "input_audio_buffer.speech_stopped"),
          "pause shorter than silence_ms held the turn open");

    // Add 200 ms more silence (>= 500 ms total): turn closes and completes ASR.
    append_pcm(ws, silence(200));
    bool saw_stopped = false;
    bool saw_completed = false;
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (Clock::now() < deadline && (!saw_stopped || !saw_completed)) {
      const auto ev = ws.recv_json(500);
      if (!ev) continue;
      if (ev->value("type", "") == "input_audio_buffer.speech_stopped") {
        saw_stopped = true;
      } else if (ev->value("type", "") == "transcription.completed") {
        saw_completed = !ev->value("text", "").empty();
      }
    }
    check(saw_stopped, "silence_ms pause emitted input_audio_buffer.speech_stopped");
    check(saw_completed, "VAD endpoint committed final ASR transcription.completed");
  }

  // 2. Noise floor adapts down on -70 dBFS hum and still detects -20 dBFS speech.
  {
    WsClient ws(h.url());
    ws.send_json(
        {{"type", "session.update"},
         {"session",
          {{"input_audio_format", "pcm16"},
           {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 200}}}}}});
    (void)ws.recv_json(2000);

    const std::string hum = tone(1000, -70.0, 60.0);
    for (size_t off = 0; off < hum.size(); off += 3200) {
      append_pcm(ws, std::string_view(hum).substr(off, 3200));
    }
    check(!has_type(ws.drain(80), "input_audio_buffer.speech_started"),
          "low-level -70 dBFS hum did not trigger speech_started");

    append_pcm(ws, tone(200, -20.0));
    const auto onset = ws.recv_json(1500);
    check(onset.has_value() &&
              onset->value("type", "") == "input_audio_buffer.speech_started",
          "speech above adapted noise floor triggered speech_started");
  }
}

void test_spoken_realtime_history_wav_stripping_and_leg_failure(RealtimeHarness& h) {
  WsClient ws(h.url());
  ws.send_json(
      {{"type", "session.update"},
       {"session",
        {{"input_audio_format", "pcm16"},
         {"turn_detection", {{"type", "server_vad"}, {"silence_ms", 150}}},
         {"audio_output", {{"voice", "vivian"}}}}}});
  const auto updated = ws.recv_json(2000);
  check(updated.has_value() &&
            updated->value("type", "") == "transcription_session.updated",
        "spoken session.update accepted audio_output + server_vad");

  auto run_turn = [&](bool expect_failure) {
    append_pcm(ws, tone(200, -18.0));
    append_pcm(ws, silence(250));
    std::string decoded_pcm;
    bool saw_done = false;
    bool saw_completed = false;
    bool saw_failed = false;
    const auto deadline = Clock::now() + std::chrono::seconds(4);
    while (Clock::now() < deadline && !saw_completed && !saw_failed) {
      const auto ev = ws.recv_json(500);
      if (!ev) continue;
      const std::string type = ev->value("type", "");
      if (type == "audio.delta") {
        check(ev->value("format", "") == "pcm16", "audio.delta format is pcm16");
        check(ev->value("sample_rate", 0) == 24000, "audio.delta sample_rate is 24000");
        decoded_pcm += umminfer::util::base64_decode(ev->value("audio", ""));
      } else if (type == "audio.done") {
        saw_done = true;
      } else if (type == "response.completed") {
        saw_completed = true;
      } else if (type == "response.failed") {
        saw_failed = true;
      }
    }
    if (expect_failure) {
      check(saw_failed && !saw_completed,
            "TTS leg failure emitted response.failed instead of response.completed");
    } else {
      check(saw_done && saw_completed,
            "spoken turn emitted audio.done and response.completed");
      check(decoded_pcm == h.tts_pcm_payload,
            "WAV container header was stripped to exact PCM16 samples");
      check(decoded_pcm.rfind("RIFF", 0) != 0,
            "audio.delta never leaked RIFF header bytes");
    }
  };

  run_turn(/*expect_failure=*/false);
  run_turn(/*expect_failure=*/false);
  {
    std::lock_guard<std::mutex> lock(h.mu);
    check(h.vlm_message_counts.size() >= 2 &&
              h.vlm_message_counts[0] == 1 &&
              h.vlm_message_counts[1] == 3,
          "spoken realtime session enabled session_history across turns");
  }

  h.fail_tts.store(true);
  run_turn(/*expect_failure=*/true);
  h.fail_tts.store(false);
}

void test_public_wav_pcm16_api() {
  const std::string pcm("\x01\x02\x7f\x00\x00\x80\x34\x12", 8);
  const std::string wav24k = umminfer::audio::wav_from_pcm16(pcm, 24000);
  const auto parsed = umminfer::audio::parse_wav_pcm16(wav24k);
  check(parsed.sample_rate == 24000, "24 kHz WAV sample_rate parsed");
  check(parsed.channels == 1, "24 kHz WAV channels parsed");
  check(parsed.pcm == pcm, "24 kHz WAV payload excludes RIFF header");

  bool rejected_16k = false;
  try {
    (void)umminfer::audio::pcm16_from_wav(wav24k);
  } catch (const std::invalid_argument&) {
    rejected_16k = true;
  }
  check(rejected_16k, "pcm16_from_wav enforces 16 kHz mono ASR contract");

  bool rejected_corrupt = false;
  try {
    (void)umminfer::audio::parse_wav_pcm16("RIFF\x04\x00\x00\x00WAVE");
  } catch (const std::invalid_argument&) {
    rejected_corrupt = true;
  }
  check(rejected_corrupt, "truncated WAV without fmt/data rejected");
}

}  // namespace

int main() {
  test_public_wav_pcm16_api();
  RealtimeHarness harness;
  test_vad_over_websocket(harness);
  test_spoken_realtime_history_wav_stripping_and_leg_failure(harness);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d realtime blackbox check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("realtime VAD & spoken session blackbox checks passed\n");
  return 0;
}
