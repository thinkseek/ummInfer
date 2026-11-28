// SPDX-License-Identifier: Apache-2.0
// Unit / contract tests for the unified multimodal engine v1.
//
// These tests drive the engine through the real Coordinator over ipc sockets,
// but the ASR and VLM "stages" are bound PULL sockets and a PUSH socket: no
// model worker runs. They verify the ASR -> VLM orchestration contract: event
// order, deterministic aggregation, cancellation, stage failures,
// backpressure, and duplicate/out-of-order handling. They are not real-model
// E2E tests.
#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#if defined(__APPLE__)
#include <libproc.h>
#include <sys/sysctl.h>
#endif
#include <variant>
#include <vector>

#include "umminfer/comm/comm.hpp"
#include "umminfer/audio/audio.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/serve/realtime_api.hpp"
#include "umminfer/util/codec.hpp"
#include "umminfer/obs/behavior_evidence.hpp"
#include "umminfer/obs/metrics.hpp"
#include "umminfer/obs/request_telemetry.hpp"
#include "pipeline/narrative_parser.hpp"
#include "pipeline/unified_schema.hpp"
#include "serve/echo_robust_vad.hpp"
#include "serve/energy_vad.hpp"
#include "serve/online_attestation.hpp"
#include <curl/curl.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <sys/socket.h>

using namespace umminfer;
using json = nlohmann::json;

namespace test_http {

enum class HttpMethod { kGet, kPost, kDelete };

struct HttpRequest {
  std::string url;
  HttpMethod method = HttpMethod::kPost;
  std::string body;
  std::string bearer;
  std::map<std::string, std::string> headers;
  client::Milliseconds timeout{10000};
  client::Milliseconds connect_timeout{2000};
  bool no_proxy = true;
  bool allow_http_errors = false;
  std::function<bool(std::string_view)> on_body;
};

struct HttpResponse {
  long status = 0;
  std::string body;
  std::map<std::string, std::string> headers;
};

inline client::Milliseconds retry_after(
    const std::map<std::string, std::string>& headers) {
  for (const auto& [k, v] : headers) {
    std::string lower = k;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "retry-after") {
      try {
        const long secs = std::stol(v);
        if (secs > 0) return client::Milliseconds{secs * 1000};
      } catch (...) {
      }
    }
  }
  return client::Milliseconds{0};
}

inline HttpResponse perform(const HttpRequest& req) {
  std::string host = "127.0.0.1";
  int port = 80;
  std::string path = "/";
  const size_t scheme_end = req.url.find("://");
  const size_t host_begin = scheme_end == std::string::npos ? 0 : scheme_end + 3;
  const size_t slash = req.url.find('/', host_begin);
  const std::string authority =
      slash == std::string::npos ? req.url.substr(host_begin)
                                 : req.url.substr(host_begin, slash - host_begin);
  if (slash != std::string::npos) path = req.url.substr(slash);
  const size_t colon = authority.rfind(':');
  if (colon != std::string::npos) {
    host = authority.substr(0, colon);
    port = std::stoi(authority.substr(colon + 1));
  } else {
    host = authority;
  }

  httplib::Client cli(host, port);
  const auto to_sec_usec = [](client::Milliseconds ms) {
    const auto count = std::max<int64_t>(1, ms.count());
    return std::pair<time_t, time_t>{static_cast<time_t>(count / 1000),
                                     static_cast<time_t>((count % 1000) * 1000)};
  };
  const auto [c_sec, c_usec] = to_sec_usec(req.connect_timeout);
  const auto [t_sec, t_usec] = to_sec_usec(req.timeout);
  cli.set_connection_timeout(c_sec, c_usec);
  cli.set_read_timeout(t_sec, t_usec);
  cli.set_write_timeout(t_sec, t_usec);

  httplib::Headers hdrs;
  std::string content_type = "application/json";
  for (const auto& [k, v] : req.headers) {
    if (k == "Content-Type" || k == "content-type") {
      content_type = v;
    } else {
      hdrs.emplace(k, v);
    }
  }
  if (!req.bearer.empty()) {
    hdrs.emplace("Authorization", "Bearer " + req.bearer);
  }

  HttpResponse out;
  httplib::Result res;
  if (req.method == HttpMethod::kGet) {
    if (req.on_body) {
      res = cli.Get(
          path, hdrs,
          [&](const httplib::Response& r) {
            out.status = r.status;
            for (const auto& [hk, hv] : r.headers) out.headers[hk] = hv;
            return true;
          },
          [&](const char* data, size_t len) {
            out.body.append(data, len);
            return req.on_body(std::string_view(data, len));
          });
    } else {
      res = cli.Get(path, hdrs);
    }
  } else if (req.method == HttpMethod::kDelete) {
    res = req.body.empty() ? cli.Delete(path, hdrs)
                           : cli.Delete(path, hdrs, req.body, content_type);
  } else if (req.on_body) {
    res = cli.Post(path, hdrs, req.body, content_type,
                   [&](const char* data, size_t len) {
                     out.body.append(data, len);
                     return req.on_body(std::string_view(data, len));
                   });
  } else {
    res = cli.Post(path, hdrs, req.body, content_type);
  }

  if (!res && out.status == 0) {
    throw client::Error(client::ErrorCode::kNetwork,
                        httplib::to_string(res.error()), true, 0);
  }
  if (res) {
    out.status = res->status;
    if (out.body.empty()) out.body = res->body;
    for (const auto& [hk, hv] : res->headers) out.headers[hk] = hv;
  }
  if (!req.allow_http_errors && (out.status < 200 || out.status >= 300)) {
    const auto code =
        (out.status == 429 || out.status == 503) ? client::ErrorCode::kRateLimited
        : (out.status == 401 || out.status == 403)
            ? client::ErrorCode::kAuthentication
        : out.status >= 500 ? client::ErrorCode::kServer
                            : client::ErrorCode::kBadRequest;
    throw client::Error(
        code, "HTTP " + std::to_string(out.status) + ": " + out.body,
        out.status == 429 || out.status >= 500, static_cast<int>(out.status),
        retry_after(out.headers));
  }
  return out;
}

class WebSocket {
 public:
  WebSocket(const std::string& url, client::Milliseconds timeout,
            const std::atomic<bool>* cancel)
      : cancel_(cancel) {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    curl_ = curl_easy_init();
    if (!curl_) throw std::runtime_error("curl_easy_init failed");
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_CONNECT_ONLY, 2L);
    curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl_, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl_, CURLOPT_PROTOCOLS_STR, "ws");
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(std::max<int64_t>(1, timeout.count())));
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS,
                     static_cast<long>(std::max<int64_t>(1, timeout.count())));
    const CURLcode rc = curl_easy_perform(curl_);
    if (rc != CURLE_OK) {
      curl_easy_cleanup(curl_);
      curl_ = nullptr;
      throw std::runtime_error(curl_easy_strerror(rc));
    }
    curl_easy_getinfo(curl_, CURLINFO_ACTIVESOCKET, &sock_);
  }

  ~WebSocket() {
    close();
    if (curl_) curl_easy_cleanup(curl_);
  }

  void send_text(const std::string& text) {
    std::lock_guard<std::mutex> lock(io_mu_);
    size_t off = 0;
    while (off < text.size() && !stopped()) {
      size_t sent = 0;
      const CURLcode rc = curl_ws_send(
          curl_, text.data() + off, text.size() - off, &sent, 0, CURLWS_TEXT);
      if (rc == CURLE_OK) {
        off += sent;
      } else if (rc == CURLE_AGAIN) {
        poll_sock(POLLOUT, 20);
      } else {
        throw std::runtime_error(curl_easy_strerror(rc));
      }
    }
  }

  void run(const std::function<void(std::string_view)>& on_msg) {
    std::string accum;
    while (!stopped()) {
      char buf[16384];
      size_t got = 0;
      const curl_ws_frame* meta = nullptr;
      CURLcode rc = CURLE_OK;
      {
        std::lock_guard<std::mutex> lock(io_mu_);
        rc = curl_ws_recv(curl_, buf, sizeof(buf), &got, &meta);
      }
      if (rc == CURLE_OK && meta != nullptr) {
        if (meta->flags & CURLWS_CLOSE) return;
        if (meta->flags & CURLWS_TEXT) {
          accum.append(buf, got);
          if (meta->bytesleft == 0 && !(meta->flags & CURLWS_CONT)) {
            on_msg(accum);
            accum.clear();
          }
        }
        continue;
      }
      if (rc == CURLE_AGAIN) {
        poll_sock(POLLIN, 20);
        continue;
      }
      return;
    }
  }

  void close() {
    // Do not call ::shutdown() on libcurl's CURLINFO_ACTIVESOCKET fd: libcurl
    // owns that descriptor and may close it before ~WebSocket(), allowing the
    // OS to hand the same fd number to another subsystem (e.g. a libzmq
    // signaler socketpair). Setting closed_ wakes run() within its 20 ms poll.
    closed_.store(true, std::memory_order_relaxed);
  }

 private:
  bool stopped() const {
    return closed_.load(std::memory_order_relaxed) ||
           (cancel_ != nullptr && cancel_->load(std::memory_order_relaxed));
  }
  void poll_sock(short events, int ms) {
    if (sock_ == CURL_SOCKET_BAD) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
      return;
    }
    pollfd pfd{};
    pfd.fd = static_cast<int>(sock_);
    pfd.events = events;
    (void)::poll(&pfd, 1, ms);
  }
  CURL* curl_ = nullptr;
  curl_socket_t sock_ = CURL_SOCKET_BAD;
  const std::atomic<bool>* cancel_ = nullptr;
  std::atomic<bool> closed_{false};
  std::mutex io_mu_;
};

}  // namespace test_http

namespace {

int g_failures = 0;
// Names the running test so a failure that repeats the same message from many
// call sites still says which test produced it.
const char* g_current_test = "<none>";

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << " [" << g_current_test << "]\n";
    ++g_failures;
  }
}

// Bound for every assertion that waits for the engine to change state. The
// coordinator delivers stage submits asynchronously, so these waits poll for
// the state change instead of measuring elapsed time. Measured submit
// arrivals: 101-411 ms under 13 CPU-bound processes; the previous single
// 2 s receive fired under `zig build test`, which runs the test binaries in
// parallel.
constexpr std::chrono::seconds kStateWait{30};

// Runs one test with its name set for check() and the wait diagnostics.
#define RUN(call)           \
  do {                      \
    g_current_test = #call; \
    call;                   \
  } while (0)

void wait_until(const std::function<bool()>& condition, const char* message) {
  const auto deadline = std::chrono::steady_clock::now() + kStateWait;
  while (std::chrono::steady_clock::now() < deadline) {
    if (condition()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // Reaching this line means the condition stayed false for the whole budget,
  // not that the check ran too early.
  std::cerr << "DIAG: condition still false after " << kStateWait.count()
            << " s: " << message << "\n";
  check(false, message);
}

// Callbacks run on the coordinator pump thread; the lock covers the handoff.
struct EventLog {
  std::mutex mu;
  std::vector<UnifiedEvent> events;
  std::map<std::string, UnifiedEvent> terminals;
  // Every on_terminal invocation, counted separately from the first-only
  // terminal map. Without this, emplace makes terminal_count() always <= 1 and
  // a duplicate-terminal bug could not be observed.
  std::map<std::string, int> terminal_callbacks;

  void add(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
  void finish(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    // Keep the first terminal for state checks, but count every callback.
    terminals.emplace(event.request_id, event);
    ++terminal_callbacks[event.request_id];
  }
  bool has_terminal(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.count(id) != 0;
  }
  std::optional<UnifiedEvent> terminal_for(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminals.find(id);
    return it == terminals.end() ? std::nullopt : std::optional(it->second);
  }
  size_t terminal_count(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminal_callbacks.find(id);
    return it == terminal_callbacks.end() ? 0 : static_cast<size_t>(it->second);
  }
  // Backwards-compatible single-request view used by the other tests.
  bool has_terminal() {
    std::lock_guard<std::mutex> lock(mu);
    return !terminals.empty();
  }
  std::optional<UnifiedEvent> terminal_event() {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.empty() ? std::nullopt
                             : std::optional(terminals.begin()->second);
  }
  size_t size() {
    std::lock_guard<std::mutex> lock(mu);
    return events.size();
  }
  std::vector<UnifiedEvent> snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
};

// One mkdtemp(3) directory per Harness: a unique 0700 directory this run
// owns. Socket files keep their basenames inside it, and teardown removes only
// those paths plus the owned directory. No predictable global path is ever
// pre-deleted, so a foreign process cannot lose its sockets.
struct IpcDir {
  std::string path;
  std::vector<std::string> files;

  static IpcDir create(const char* tag) {
    std::string templ =
        std::string("/tmp/omni-unified-") + tag + "-XXXXXX";
    std::vector<char> buffer(templ.begin(), templ.end());
    buffer.push_back('\0');
    char* made = mkdtemp(buffer.data());
    if (made == nullptr) {
      throw std::runtime_error(std::string("mkdtemp failed: ") +
                               std::strerror(errno));
    }
    IpcDir dir;
    dir.path = made;  // mkdtemp itself applies 0700
    for (const char* name : {"asr", "vlm", "image", "tts", "done", "abort"}) {
      dir.files.push_back(dir.path + "/" + name);
    }
    return dir;
  }

  ~IpcDir() {
    for (const auto& file : files) std::remove(file.c_str());
    // rmdir only succeeds when the directory holds nothing this run did not
    // create; it never recurses into a foreign path.
    ::rmdir(path.c_str());
  }

  std::string endpoint(const char* name) const {
    return "ipc://" + path + "/" + name;
  }
};

Value::Map part(const std::string& type) {
  Value::Map out;
  out["type"] = Value(type);
  return out;
}

StagePayload audio_request(const std::string& modality = "audio") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value(modality));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

Value image_part(const std::string& url = "data:image/png;base64,AA") {
  Value::Map image_url;
  image_url["url"] = Value(url);
  Value::Map out = part("image_url");
  out["image_url"] = Value(std::move(image_url));
  return Value(std::move(out));
}

StagePayload image_part_request(const std::string& url = "data:image/png;base64,AA") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part(url));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// One request carrying real PCM for the ASR leg and one image part for the VLM
// leg. The image rides on the original messages; the ASR leg never sees it.
StagePayload joint_request(const std::string& session_id = "",
                           const std::string& image_url = "") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part(image_url.empty() ? "data:image/png;base64,AA"
                                                 : image_url));
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value("audio"));
  modalities.push_back(Value("image"));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (!session_id.empty()) inputs["session_id"] = Value(session_id);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// A joint request that explicitly opts into multi-turn history. The engine
// must have session history enabled or submit() rejects it.
StagePayload history_request(const std::string& session_id) {
  StagePayload payload = joint_request(session_id);
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["session_history"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}


// Real PCM selects the ASR leg. The optional flag is the documented opt-in for
// incremental ASR events through the unified chat route.
StagePayload audio_leg_request(bool asr_partials) {
  StagePayload payload = audio_request();
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (asr_partials) inputs["asr_partials"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

Value::Map text_chunk(const std::string& text, bool is_final) {
  Value::Map chunk;
  chunk["text"] = Value(text);
  if (is_final) chunk["is_final"] = Value(true);
  return chunk;
}

Value::Map text_result(const std::string& text) {
  Value::Map result;
  result["text"] = Value(text);
  return result;
}

// A VLM-only request that opts into image, audio and/or video output legs.
StagePayload mixed_request(bool want_image, bool want_audio,
                           const std::string& image_prompt = "",
                           const std::string& voice = "Tina",
                           bool want_video = false,
                           const std::string& video_prompt = "",
                           const std::string& session_id = "") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value("describe the scene");
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array selected;
  selected.push_back(Value("text"));
  if (want_image) selected.push_back(Value("image"));
  if (want_audio) selected.push_back(Value("audio"));
  if (want_video) selected.push_back(Value("video"));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["output_modalities"] = Value(std::move(selected));
  if (!session_id.empty()) inputs["session_id"] = Value(session_id);
  if (want_image) {
    Value::Map options;
    if (!image_prompt.empty()) options["prompt"] = Value(image_prompt);
    options["size"] = Value("1024x1024");
    options["output_format"] = Value("png");
    inputs["image_options"] = Value(std::move(options));
  }
  if (want_audio) {
    Value::Map options;
    options["voice"] = Value(voice);
    options["format"] = Value("wav");
    options["sample_rate"] = Value(int64_t{24000});
    inputs["audio_options"] = Value(std::move(options));
  }
  if (want_video) {
    Value::Map options;
    if (!video_prompt.empty()) options["prompt"] = Value(video_prompt);
    options["size"] = Value("1280x720");
    options["seconds"] = Value(int64_t{4});
    options["fps"] = Value(int64_t{24});
    options["output_format"] = Value("mp4");
    inputs["video_options"] = Value(std::move(options));
  }
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// A worker result carrying an inline Bin artifact, as the frozen contract
// specifies. The bytes are fake stage output, never a real model artifact.
Value artifact_result(const std::string& media_type,
                      const std::string& mime_type, std::string bytes,
                      Value::Map extra = {}) {
  Value::Map result = std::move(extra);
  result["media_type"] = Value(media_type);
  result["mime_type"] = Value(mime_type);
  result["data"] = Value::bin(std::move(bytes));
  return Value(std::move(result));
}

// Real Coordinator with two fake stage inboxes and one event pusher. No
// compute runs anywhere.
struct Harness {
  IpcDir dir;
  PullSocket asr_recv;
  PullSocket vlm_recv;
  PullSocket image_recv;
  PullSocket tts_recv;
  PullSocket video_recv;
  PushSocket event_push;
  Coordinator coordinator;
  std::thread pump;
  // broadcast_abort pushes AbortMessage on the same stage PULL sockets, so the
  // fake stages observe cancels here. The PUSH path is the reliable ordered
  // one; the test deliberately does not depend on PUB slow-joiner timing.
  std::mutex abort_mu;
  std::vector<std::string> aborted_legs;

  explicit Harness(const char* tag, bool start_pump_thread = false)
      : dir(IpcDir::create(tag)),
        asr_recv(dir.endpoint("asr"), true),
        vlm_recv(dir.endpoint("vlm"), true),
        image_recv(dir.endpoint("image"), true),
        tts_recv(dir.endpoint("tts"), true),
        video_recv(dir.endpoint("video"), true),
        event_push(dir.endpoint("done")),
        coordinator(dir.endpoint("done"), dir.endpoint("abort")) {
    asr_recv.start();
    vlm_recv.start();
    image_recv.start();
    tts_recv.start();
    video_recv.start();
    coordinator.register_stage("asr", dir.endpoint("asr"));
    coordinator.register_stage("vlm", dir.endpoint("vlm"));
    // Optional legs stay out of the readiness gate, matching production wiring.
    coordinator.register_stage("image", dir.endpoint("image"));
    coordinator.register_stage("tts", dir.endpoint("tts"));
    coordinator.register_stage("video", dir.endpoint("video"));
    coordinator.start();
    if (start_pump_thread) start_pump();
    event_push.connect();
  }

  // Config with all optional output stages enabled.
  UnifiedConfig mixed_config() const {
    UnifiedConfig config;
    config.image_stage = "image";
    config.tts_stage = "tts";
    config.video_stage = "video";
    return config;
  }

  void start_pump() {
    pump = std::thread([this] { coordinator.run(); });
  }

  ~Harness() {
    coordinator.stop();
    if (pump.joinable()) pump.join();
    event_push.close();
    asr_recv.close();
    vlm_recv.close();
    image_recv.close();
    tts_recv.close();
    video_recv.close();
  }

  std::optional<SubmitMessage> next_submit(PullSocket& socket) {
    // Abort frames share the stage PULL socket: broadcast_abort pushes to every
    // stage socket, and the real Stage drains them in handle_message. The fake
    // stage records them and skips to the next submit frame.
    const char* leg = (&socket == &asr_recv)     ? "asr"
                      : (&socket == &vlm_recv)   ? "vlm"
                      : (&socket == &image_recv) ? "image"
                      : (&socket == &video_recv) ? "video"
                                                 : "tts";
    const auto deadline = std::chrono::steady_clock::now() + kStateWait;
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        // The loop only exits here after the full poll budget with no submit
        // frame on the socket, so "not yet" cannot be reported as "never".
        std::cerr << "DIAG: no " << leg << " submit within "
                  << kStateWait.count() << " s; coordinator still had "
                  << coordinator.in_flight() << " request(s) in flight\n";
        return std::nullopt;
      }
      const auto left =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
              .count();
      auto msg = socket.recv_timeout(static_cast<int>(left < 100 ? left : 100));
      if (!msg) continue;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) return *submit;
    }
  }

  // Drains one stage inbox until the abort for leg_id is observed. Unlike
  // wait_abort it polls only the given socket, so a tts/image abort is not
  // missed and no other stage's frames are consumed.
  bool wait_abort_on(PullSocket& socket, const std::string& leg_id) {
    const auto deadline = std::chrono::steady_clock::now() + kStateWait;
    while (!saw_abort(leg_id) && std::chrono::steady_clock::now() < deadline) {
      auto msg = socket.recv_timeout(50);
      if (!msg) continue;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
      }
    }
    return saw_abort(leg_id);
  }

  // Drains both stage inboxes until the abort for leg_id is observed. The loop
  // ends on the state change or the budget, not after a fixed slice count.
  bool wait_abort(const std::string& leg_id) {
    const auto deadline = std::chrono::steady_clock::now() + kStateWait;
    while (!saw_abort(leg_id) && std::chrono::steady_clock::now() < deadline) {
      for (PullSocket* socket : {&asr_recv, &vlm_recv}) {
        auto msg = socket->recv_timeout(50);
        if (!msg) continue;
        if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
          std::lock_guard<std::mutex> lock(abort_mu);
          aborted_legs.push_back(abort->request_id);
        }
      }
    }
    return saw_abort(leg_id);
  }

  bool saw_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    for (const auto& id : aborted_legs) {
      if (id == leg_id) return true;
    }
    return false;
  }

  void record_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    aborted_legs.push_back(leg_id);
  }

  // Negative assertion: only the PUSH abort path can arrive, never a submit.
  bool no_submit(PullSocket& socket) {
    while (true) {
      auto msg = socket.recv_timeout(200);
      if (!msg) return true;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        (void)submit;
        return false;
      }
    }
  }


  void stream(const std::string& id, const std::string& stage, Value chunk) {
    StreamMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.chunk = std::move(chunk);
    event_push.send(Message{std::move(msg)});
  }

  void complete(const std::string& id, const std::string& stage, bool success,
                Value result,
                std::optional<std::string> error = std::nullopt) {
    CompleteMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.success = success;
    msg.result = std::move(result);
    msg.error = std::move(error);
    event_push.send(Message{std::move(msg)});
  }
};

// Drives one request through the fake ASR and VLM stages to a successful
// terminal, returning the VLM leg submit payload for prompt inspection.
StagePayload complete_turn(Harness& h, UnifiedEngine& engine,
                           const StagePayload& request,
                           const std::string& transcript,
                           const std::string& answer) {
  const std::string id = engine.submit(request);
  check(h.next_submit(h.asr_recv).has_value(), "history turn: no ASR submit");
  h.complete(id, "asr", true, text_result(transcript));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "history turn: no VLM submit");
  StagePayload vlm_payload;
  if (vlm) vlm_payload = StagePayload::from_value(vlm->data);
  h.complete(id + ":vlm", "vlm", true, text_result(answer));
  check(engine.wait(id, 5.0), "history turn did not complete");
  engine.release(id);
  return vlm_payload;
}

// The role/content pairs of a VLM prompt, in wire order.
std::vector<std::pair<std::string, std::string>> prompt_roles(
    const StagePayload& vlm_payload) {
  std::vector<std::pair<std::string, std::string>> out;
  const Value* messages = vlm_payload.request.inputs.find("messages");
  if (messages == nullptr) return out;
  for (const Value& message : messages->as_array()) {
    const Value& content = *message.find("content");
    // History turns carry plain strings; the current turn may carry the
    // content-part array, whose last text part is the transcript.
    std::string text;
    if (content.is_str()) {
      text = content.as_str();
    } else {
      for (const Value& part : content.as_array()) {
        const Value* part_text = part.find("text");
        if (part_text != nullptr) text = part_text->as_str();
      }
    }
    out.emplace_back(message.find("role")->as_str(), text);
  }
  return out;
}

// Streaming ASR partials/final -> VLM: ordering, gap-free seq, aggregation.
void test_streaming_asr_to_vlm() {
  Harness h("stream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.stream(id, "asr", text_chunk("hello world", true));
  wait_until([&] { return log.size() == 3; },
             "ASR partial/final events were not delivered");

  // The VLM leg starts when the ASR completion arrives, not on the final chunk.
  h.complete(id, "asr", true, text_result("hello world"));
  // The final transcript replaces the concatenated partial deltas, and the VLM
  // leg starts with exactly that text as its prompt.
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    check(vlm_submit->request_id == id + ":vlm", "VLM leg id is not a child id");
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    const Value& first = messages.as_array().at(0);
    check(first.find("content")->as_str() == "hello world",
          "VLM prompt is not the final ASR transcript");
  }

  const auto asr_leg_info = h.coordinator.info(id);
  check(asr_leg_info && asr_leg_info->state == "completed",
        "the ASR parent leg did not complete before the VLM barrier");
  const auto vlm_phase_info = engine.info(id);
  check(vlm_phase_info && vlm_phase_info->request_id == id &&
            vlm_phase_info->state == "running" &&
            vlm_phase_info->current_stage == kVlmStageId,
        "info() exposed the completed ASR parent leg instead of unified VLM state");

  h.stream(id + ":vlm", "vlm", text_chunk("hi", false));
  h.stream(id + ":vlm", "vlm", text_chunk(" there", false));
  h.complete(id + ":vlm", "vlm", true, text_result("hi there"));
  check(engine.wait(id, 5.0), "unified request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 6) {
    check(false, "wrong event count for the full ASR -> VLM flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected_kinds[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal,   UnifiedEvent::Kind::kVlmDelta,
      UnifiedEvent::Kind::kVlmDelta,   UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "unified event sequence is not gap-free from zero");
    check(events[i].kind == expected_kinds[i], "unified event kind out of order");
    check(events[i].request_id == id, "an event leaked the internal child id");
  }
  check(events[2].is_final && events[2].text == "hello world",
        "ASR final event has the wrong text");
  check(events[5].text == "hi there", "complete event lost the VLM aggregation");
  const auto final_info = engine.info(id);
  check(final_info && final_info->state == "completed" &&
            final_info->current_stage == kVlmStageId &&
            final_info->result.find("text") != nullptr &&
            final_info->result.find("text")->as_str() == "hi there",
        "info() did not retain the unified VLM terminal state");
  engine.release(id);
}

// Opt-in ASR partials: the ASR leg request streams, partial deltas precede one
// final, and the final replaces the concatenated partials for the VLM prompt.
void test_asr_partials_opt_in() {
  Harness h("partials");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "opt-in ASR partial request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(sent.request.params.at("stream").as_bool(),
          "opt-in request did not ask the ASR stage to stream");
    check(sent.request.inputs.find("asr_partials") == nullptr,
          "the ASR stage received the opt-in hint instead of audio only");
  }

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.complete(id, "asr", true, text_result("hello"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "opt-in ASR leg did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload sent = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *sent.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "hello",
          "partial deltas were not replaced by the final transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "opt-in ASR partial request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 4) {
    check(false, "wrong event count for the ASR partial flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal, UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "ASR partial sequence is not gap-free");
    check(events[i].kind == expected[i], "ASR partial event out of order");
    check(events[i].request_id == id, "ASR partial leaked the child id");
  }
  check(events[0].text == "hel" && events[1].text == "lo" &&
            !events[0].is_final && !events[1].is_final,
        "ASR partial events do not carry the delta text");
  check(events[2].is_final && events[2].text == "hello",
        "ASR final does not replace the partial text");
  engine.release(id);
}

// Default request: the ASR leg is not asked to stream and no partial is emitted.
void test_asr_partials_default_off() {
  Harness h("partialsoff");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(false));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "default ASR request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(!sent.request.params.at("stream").as_bool(),
          "a default request asked the ASR stage to stream");
  }
  h.complete(id, "asr", true, text_result("hello"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "default ASR leg did not reach the VLM leg");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "default request did not complete");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kAsrPartial,
          "a default request emitted an ASR partial");
  }
  engine.release(id);
}

// Cancel after a partial: exactly one cancelled terminal, no VLM submit.
void test_asr_partials_cancel() {
  Harness h("partialscancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) {
    log.finish(event);
  };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  wait_until([&] { return log.size() == 1; },
             "the ASR partial event was not delivered before the cancel");
  check(engine.cancel(id), "cancel was rejected before the ASR leg finished");
  check(h.wait_abort(id), "the ASR stage did not observe the abort");
  h.complete(id, "asr", false, Value{}, std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled request did not settle");
  h.no_submit(h.vlm_recv);

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
        "cancel after a partial did not produce a cancelled terminal");
  check(log.terminal_count(id) == 1,
        "cancel after a partial produced more than one terminal");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kVlmDelta,
          "a cancelled partial request emitted a VLM delta");
  }
  check(engine.in_flight() == 0, "cancelled request stayed in flight");
  engine.release(id);
}

// An ASR stream failure after a partial: exactly one error terminal.
void test_asr_partials_error() {
  Harness h("partialserror");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  h.complete(id, "asr", false, Value{}, std::string("backend reset"));
  check(engine.wait(id, 5.0), "failed partial request did not settle");

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->error == "backend reset",
        "ASR stream failure did not produce the error terminal");
  check(log.terminal_count(id) == 1,
        "ASR stream failure produced more than one terminal");
  check(h.no_submit(h.vlm_recv), "a failed ASR leg started the VLM leg");
  engine.release(id);
}

// A non-streaming ASR stage emits only a completion; the engine synthesizes
// the ASR final from its result text.
void test_nonstream_asr_synthesizes_final() {
  Harness h("nonstream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("spoken text"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "non-streaming request did not complete");

  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "synthesized final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a non-streaming result");
  engine.release(id);
}

// The ASR contract accepts a plain string result, matching
// serve_detail::asr_result_text. The engine must extract it and use it as the
// VLM prompt.
void test_asr_plain_string_result() {
  Harness h("plainstring");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, Value("spoken text"));  // plain string, no map

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "plain-string result did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "spoken text",
          "the VLM prompt is not the plain-string ASR transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "plain-string request did not complete");
  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "plain-string final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a plain-string result");
  engine.release(id);
}

// A result with no usable text is a contract failure: the engine must emit
// kError on the ASR stage and must never start the VLM leg, rather than
// forwarding an empty prompt.
void test_asr_malformed_result_fails() {
  std::vector<Value> bad_results;
  bad_results.emplace_back(Value::Map{});
  bad_results.emplace_back(Value(42));
  for (const Value& bad : bad_results) {
    Harness h("malformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, bad);
    check(engine.wait(id, 5.0), "a malformed ASR result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal.has_value() &&
              terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId,
          "a malformed ASR result did not produce an ASR-stage kError");
    check(terminal && !terminal->error.empty(),
          "the malformed-result error carried no message");
    check(h.no_submit(h.vlm_recv),
          "a malformed ASR result still started the VLM leg");
    engine.release(id);
  }
}

// A malformed streaming final must fail before the completion can start VLM.
void test_streaming_asr_malformed_final_fails() {
  std::vector<Value> bad_finals;
  Value::Map missing_text;
  missing_text["is_final"] = Value(true);
  bad_finals.emplace_back(std::move(missing_text));
  Value::Map numeric_text;
  numeric_text["is_final"] = Value(true);
  numeric_text["text"] = Value(42);
  bad_finals.emplace_back(std::move(numeric_text));

  for (const Value& bad : bad_finals) {
    Harness h("streammalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.stream(id, "asr", bad);
    // The same PUSH -> PULL queue makes this completion a FIFO barrier: the old
    // code treats the malformed final as valid and starts VLM here.
    h.complete(id, "asr", true, text_result("must not run"));
    check(engine.wait(id, 5.0), "a malformed streaming final did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId && !terminal->error.empty(),
          "a malformed streaming final did not produce one ASR kError");
    check(log.terminal_count(id) == 1,
          "a malformed streaming final produced duplicate terminals");
    check(log.size() == 1 && log.snapshot()[0].kind == UnifiedEvent::Kind::kError,
          "a malformed streaming final emitted an ASR final or success event");
    check(h.no_submit(h.vlm_recv),
          "a malformed streaming final still started the VLM leg");
    engine.release(id);
  }
}

void test_unknown_id_wait_fails() {
  Harness h("unknownwait");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  check(!engine.wait("unknown", 0.0), "wait() reported success for an unknown id");
  check(!engine.info("unknown"), "info() returned state for an unknown id");
}

// cancel() broadcasts an abort to both stages and settles the request.
void test_cancel_propagates_to_both_stages() {
  Harness h("cancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(engine.cancel(id), "cancel rejected an active request");
  check(h.wait_abort(id), "abort did not reach the ASR stage");
  wait_until([&] { return log.has_terminal(); },
             "cancellation did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kCancelled,
        "cancel produced the wrong terminal event");
  check(engine.info(id)->state == "aborted", "coordinator state is not aborted");
  check(!engine.cancel(id), "a second cancel was not idempotent");
  engine.release(id);

  // Second request: cancel during the VLM leg aborts the VLM child id.
  const std::string id2 = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id2, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  check(engine.cancel(id2), "cancel rejected the VLM-leg request");
  check(h.wait_abort(id2 + ":vlm"),
        "VLM-leg cancel did not abort the active stage leg");
  engine.release(id2);
}

// An ASR failure stops the flow before any VLM submit.
void test_asr_failure() {
  Harness h("asrfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", false, Value{}, std::string("asr broke"));
  check(engine.wait(id, 5.0), "ASR failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kAsrStageId &&
            log.terminal_event()->error == "asr broke",
        "ASR failure produced the wrong terminal event");
  check(h.no_submit(h.vlm_recv),
        "a failed ASR leg still submitted to the VLM stage");
  engine.release(id);
}

// A VLM failure is observable with the VLM stage id.
void test_vlm_failure() {
  Harness h("vlmfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", false, Value{}, std::string("vlm broke"));
  check(engine.wait(id, 5.0), "VLM failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kVlmStageId &&
            log.terminal_event()->error == "vlm broke",
        "VLM failure produced the wrong terminal event");
  engine.release(id);
}

// VLM results accept exactly the public bare-string/text/content shapes.
void test_vlm_result_text_contract() {
  std::vector<std::pair<Value, std::string>> valid_results;
  valid_results.emplace_back(Value("bare answer"), "bare answer");
  valid_results.emplace_back(Value(text_result("text answer")), "text answer");
  Value::Map content_result;
  content_result["content"] = Value("content answer");
  valid_results.emplace_back(Value(std::move(content_result)), "content answer");

  for (const auto& [result, expected] : valid_results) {
    Harness h("vlmresult");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.complete(id + ":vlm", "vlm", true, result);
    check(engine.wait(id, 5.0), "a valid VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->text == expected,
          "a valid VLM result shape lost its text");
    check(log.terminal_count(id) == 1,
          "a valid VLM result produced duplicate terminals");
    engine.release(id);
  }

  std::vector<Value> bad_results;
  bad_results.emplace_back(Value(42));
  bad_results.emplace_back(Value::Map{});
  Value::Map numeric_text;
  numeric_text["text"] = Value(42);
  bad_results.emplace_back(std::move(numeric_text));
  for (const Value& bad : bad_results) {
    Harness h("vlmmalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.stream(id + ":vlm", "vlm", text_chunk("delta", false));
    h.complete(id + ":vlm", "vlm", true, bad);
    check(engine.wait(id, 5.0), "a malformed VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kVlmStageId && !terminal->error.empty(),
          "a malformed VLM result did not produce exactly one VLM kError");
    check(log.terminal_count(id) == 1,
          "a malformed VLM result produced duplicate terminals");
    engine.release(id);
  }
}

// The admission cap is a hard bound: overflow throws, a finished slot reopens.
void test_queue_cap_backpressure() {
  Harness h("queue");
  UnifiedConfig config;
  config.max_in_flight = 2;
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  const std::string a = engine.submit(audio_request());
  const std::string b = engine.submit(audio_request());
  check(engine.in_flight() == 2, "admission count is wrong");
  bool full = false;
  try {
    engine.submit(audio_request());
  } catch (const QueueFullError&) {
    full = true;
  }
  check(full, "the queue cap did not reject an overflow request");

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(a, "asr", true, text_result("t"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(a + ":vlm", "vlm", true, text_result("r"));
  check(engine.wait(a, 5.0), "request did not complete");
  engine.release(a);
  wait_until([&] { return engine.in_flight() == 1; },
             "a finished request did not free an admission slot");
  const std::string c = engine.submit(audio_request());
  check(engine.in_flight() == 2, "a freed slot did not admit a new request");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  engine.release(b);
  engine.release(c);
}

// Concurrent submissions share one engine mutex. The admission cap must stay a
// hard bound under a burst, and every admitted request must still complete with
// the same result the serial path produces.
void test_queue_cap_concurrent_submit() {
  Harness h("queueburst");
  UnifiedConfig config;
  config.max_in_flight = 2;
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  constexpr int kBurst = 8;
  std::atomic<int> admitted{0};
  std::atomic<int> rejected{0};
  std::atomic<bool> go{false};
  std::mutex ids_mu;
  std::vector<std::string> ids;
  std::vector<std::thread> threads;
  threads.reserve(kBurst);
  for (int i = 0; i < kBurst; ++i) {
    threads.emplace_back([&] {
      while (!go.load()) std::this_thread::yield();
      try {
        const std::string id = engine.submit(mixed_request(false, false));
        {
          std::lock_guard<std::mutex> lock(ids_mu);
          ids.push_back(id);
        }
        ++admitted;
      } catch (const QueueFullError&) {
        ++rejected;
      }
    });
  }
  go = true;
  for (auto& thread : threads) thread.join();

  check(admitted.load() == 2, "the admission cap admitted more than its limit");
  check(rejected.load() == kBurst - 2,
        "a burst over the cap did not reject the excess requests");
  check(engine.in_flight() == 2, "the admission count does not match the cap");

  for (int i = 0; i < 2; ++i) {
    auto submit = h.next_submit(h.vlm_recv);
    check(submit.has_value(), "an admitted request never reached the VLM stage");
    if (!submit) continue;
    h.stream(submit->request_id, "vlm", text_chunk("ok", false));
    h.complete(submit->request_id, "vlm", true, text_result("ok"));
  }
  for (const auto& id : ids) {
    check(engine.wait(id, 5.0), "an admitted request did not complete");
    const auto info = engine.info(id);
    check(info && info->state == "completed",
          "an admitted request did not reach the completed state");
    check(info && info->result.find("text") != nullptr &&
              info->result.find("text")->as_str() == "ok",
          "a concurrent request returned a different result than the serial path");
  }
  for (const auto& id : ids) engine.release(id);
  wait_until([&] { return engine.in_flight() == 0; },
             "released requests did not free their admission slots");
}

// Overload contract: when every admission slot is in use, the HTTP boundary
// answers 503 with Retry-After instead of queueing the request without bound.
void test_public_http_overload_503() {
  Harness h("publicoverload");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedConfig engine_config;
  engine_config.max_in_flight = 1;
  UnifiedEngine engine(h.coordinator, engine_config);
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "overload server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string body =
      R"({"model":"qwen3-unified","messages":[{"role":"user","content":"hello"}],"stream":false})";
  auto send = [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body;
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  };

  // The first request holds the only slot until its VLM leg completes.
  auto first = std::async(std::launch::async, send);
  auto submit = h.next_submit(h.vlm_recv);
  check(submit.has_value(), "the first request did not reach the VLM stage");

  // The HTTP client turns a non-2xx into client::Error, so the 503 contract is
  // read from the error object: status, overloaded marker and Retry-After.
  bool overloaded_503 = false;
  bool named_overload = false;
  bool retry_after = false;
  try {
    const auto response = send();
    overloaded_503 = response.status == 503L;
    named_overload =
        response.body.find("\"type\":\"overloaded\"") != std::string::npos;
    retry_after = test_http::retry_after(response.headers).count() > 0;
  } catch (const client::Error& error) {
    overloaded_503 = error.status() == 503;
    named_overload = std::string(error.what()).find("overloaded") != std::string::npos;
    retry_after = error.retry_after().count() > 0;
  }
  check(overloaded_503, "an overloaded engine did not answer 503");
  check(named_overload, "the overload response did not name the overload type");
  check(retry_after, "the overload response did not carry Retry-After");

  if (submit) {
    h.complete(submit->request_id, "vlm", true, text_result("answer"));
  }
  const auto completed = first.get();
  api.stop();
  server_thread.join();
  check(completed.status == 200L,
        "the admitted request did not complete after the overload was rejected");
  wait_until([&] { return engine.in_flight() == 0; },
             "the admitted request did not free its slot");
}

// A repeated ASR final is idempotent; an out-of-order chunk is dropped.
void test_duplicate_final_and_out_of_order() {
  Harness h("dup");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("first", true));
  h.stream(id, "asr", text_chunk("second", true));  // duplicate final
  wait_until([&] { return log.size() == 1; },
             "the duplicate ASR final was not idempotent");
  check(log.snapshot()[0].text == "first",
        "the duplicate final overwrote the transcript");

  // A VLM-stage chunk before the VLM leg exists is out of order and dropped.
  h.stream(id, "vlm", text_chunk("early", false));
  h.complete(id, "asr", true, text_result("first"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  // An ASR chunk after the VLM leg started is late and dropped.
  h.stream(id, "asr", text_chunk("late", false));
  h.stream(id + ":vlm", "vlm", text_chunk("answer", false));
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "request did not complete");

  size_t finals = 0, deltas = 0;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) ++finals;
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) ++deltas;
  }
  check(finals == 1, "more than one ASR final event was emitted");
  check(deltas == 1, "an out-of-order chunk leaked into the event stream");
  engine.release(id);
}

// Image + audio: the ASR leg receives PCM only, the VLM leg receives the
// original image part plus the final transcript as a text part. Video stays
// rejected.
void test_joint_image_audio() {
  Harness h("joint");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(joint_request());
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR stage received no submit");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.find("audio") != nullptr &&
              payload.request.inputs.find("messages") == nullptr,
          "the ASR leg did not receive PCM only");
  }
  h.complete(id, "asr", true, text_result("what is on screen"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2,
          "the VLM prompt is not image part + transcript text part");
    if (content.is_array() && content.as_array().size() == 2) {
      const Value& image = content.as_array()[0];
      const Value& text = content.as_array()[1];
      check(image.find("type")->as_str() == "image_url" &&
                image.find("image_url")->find("url")->as_str() ==
                    "data:image/png;base64,AA",
            "the VLM lost the original image part");
      check(text.find("type")->as_str() == "text" &&
                text.find("text")->as_str() == "what is on screen",
            "the VLM lost the ASR transcript");
    }
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 2 &&
              modalities.as_array()[0].as_str() == "text" &&
              modalities.as_array()[1].as_str() == "image",
          "the VLM did not declare both text and image modalities");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("a settings screen"));
  check(engine.wait(id, 5.0), "joint request did not complete");
  const auto events = log.snapshot();
  check(!events.empty() &&
            events.front().kind == UnifiedEvent::Kind::kAsrFinal &&
            events.back().kind == UnifiedEvent::Kind::kComplete,
        "joint event order is not asr_final .. complete");
  engine.release(id);
}

// A remote HTTP image URL is a protocol value the engine must forward
// verbatim. The engine never fetches it; only the VLM worker does.
void test_joint_image_remote_url() {
  Harness h("jointurl");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string url = "http://198.51.100.10:18080/man_ironing.png";
  const std::string id = engine.submit(joint_request("", url));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("what is on screen"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2,
          "the remote-URL VLM prompt is not image part + transcript text part");
    if (content.is_array() && content.as_array().size() == 2) {
      const Value& image = content.as_array()[0];
      check(image.find("type")->as_str() == "image_url" &&
                image.find("image_url")->find("url")->as_str() == url,
            "the VLM lost or rewrote the remote image URL");
    }
  }
  h.complete(id + ":vlm", "vlm", true, text_result("a man ironing"));
  check(engine.wait(id, 5.0), "remote-URL joint request did not complete");
  engine.release(id);
}

// Malformed media contracts are rejected before any stage sees the work.
void test_media_contract_errors() {
  Harness h("mediaerr");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});

  auto rejects = [&](StagePayload payload, const char* message) {
    bool rejected = false;
    try {
      engine.submit(std::move(payload));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, message);
  };

  rejects(audio_request("video"), "a video modality was accepted");
  rejects(image_part_request(""), "an empty image URL was accepted");

  StagePayload no_url = image_part_request("data:image/png;base64,AA");
  no_url.request.inputs =
      Value(Value::Map{{"messages", Value::Array{Value::Map{
                          {"role", Value("user")},
                          {"content", Value::Array{part("image_url")}}}}}});
  rejects(std::move(no_url), "an image part without image_url.url was accepted");

  StagePayload two_images;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value::Array{image_part(), image_part(), Value(part("input_audio"))};
  two_images.request.inputs = Value::Map{{"messages", Value::Array{Value(std::move(message))}}};
  rejects(std::move(two_images), "a request with two image parts was accepted");

  StagePayload empty;
  rejects(std::move(empty), "a request without messages was accepted");
}

// A VLM-only request built from one public content shape. When as_string is
// true the content is a plain OpenAI string; otherwise it is a text part plus
// an optional image part. No binary audio input is attached.
StagePayload vlm_only_request(bool as_string, bool with_image) {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  if (as_string) {
    message["content"] = Value("what is two plus two");
  } else {
    Value::Array content;
    Value::Map text;
    text["type"] = Value("text");
    text["text"] = Value("describe this image");
    content.push_back(Value(std::move(text)));
    if (with_image) content.push_back(image_part());
    message["content"] = Value(std::move(content));
  }
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// Plain string content without audio: no ASR leg exists, the VLM leg receives
// the text directly and the event stream carries no ASR events.
void test_vlm_only_string_content() {
  Harness h("vlmstring");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(vlm_only_request(true, false));
  check(h.no_submit(h.asr_recv), "a VLM-only request reached the ASR stage");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value() && vlm_submit->request_id == id,
        "the VLM-only request did not run under the public request id");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    check(payload.request.inputs.find("audio") == nullptr,
          "the VLM-only payload carried an audio input");
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 1 &&
              content.as_array()[0].find("text")->as_str() ==
                  "what is two plus two",
          "the VLM lost the string content");
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 1 &&
              modalities.as_array()[0].as_str() == "text",
          "the VLM-only request declared non-text modalities");
    h.stream(vlm_submit->request_id, "vlm", text_chunk("four", false));
    h.complete(vlm_submit->request_id, "vlm", true, text_result("four"));
  }
  check(engine.wait(id, 5.0), "VLM-only string request did not complete");
  const auto events = log.snapshot();
  check(!events.empty() &&
            std::none_of(events.begin(), events.end(),
                         [](const UnifiedEvent& event) {
                           return event.kind ==
                                      UnifiedEvent::Kind::kAsrFinal ||
                                  event.kind ==
                                      UnifiedEvent::Kind::kAsrPartial;
                         }) &&
            events.front().kind == UnifiedEvent::Kind::kVlmDelta &&
            events.back().kind == UnifiedEvent::Kind::kComplete,
        "VLM-only events must start at vlm_delta with no ASR events");
  engine.release(id);
}

// Image + text without audio: the image rides straight to the VLM leg, the
// modalities declare text and image, and no ASR leg is submitted.
void test_vlm_only_image_text() {
  Harness h("vlmimage");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(vlm_only_request(false, true));
  check(h.no_submit(h.asr_recv), "an image+text request reached the ASR stage");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "the image+text request did not reach the VLM");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2 &&
              content.as_array()[0].find("text")->as_str() ==
                  "describe this image" &&
              content.as_array()[1].find("type")->as_str() == "image_url",
          "the VLM prompt is not instruction + image");
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 2 &&
              modalities.as_array()[0].as_str() == "text" &&
              modalities.as_array()[1].as_str() == "image",
          "the image+text request did not declare text and image");
    h.complete(vlm_submit->request_id, "vlm", true,
               text_result("a settings screen"));
  }
  check(engine.wait(id, 5.0), "image+text request did not complete");
  const auto image_events = log.snapshot();
  check(!image_events.empty() &&
            image_events.back().kind == UnifiedEvent::Kind::kComplete,
        "image+text request did not complete successfully");
  engine.release(id);
}

// Session history on VLM-only turns must keep the real user text. There is no
// ASR transcript on these turns, so the recorded history user side is the
// string content or text part, never an empty message.
UnifiedConfig history_config(size_t max_turns, size_t max_chars);
void test_vlm_only_session_history() {
  Harness h("vlmhist");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  auto with_history = [](StagePayload request, const std::string& session) {
    Value::Map inputs = request.request.inputs.as_map();
    inputs["session_id"] = Value(session);
    inputs["session_history"] = Value(true);
    request.request.inputs = Value(std::move(inputs));
    return request;
  };
  // Drives one VLM-only turn and returns its VLM leg payload.
  auto run_turn = [&](StagePayload request, const std::string& answer) {
    const std::string id = engine.submit(std::move(request));
    check(h.no_submit(h.asr_recv), "a history VLM-only turn reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value() && vlm->request_id == id,
          "the VLM-only history leg lost the request id");
    StagePayload payload;
    if (vlm) payload = StagePayload::from_value(vlm->data);
    h.complete(id, "vlm", true, text_result(answer));
    check(engine.wait(id, 5.0), "a VLM-only history turn did not complete");
    engine.release(id);
    return payload;
  };

  // Turn 1: plain string content.
  StagePayload first = run_turn(
      with_history(vlm_only_request(true, false), "vlm-sess"), "four");
  check(prompt_roles(first).size() == 1,
        "the first VLM-only turn already carried history");

  // Turn 2: image+text in the same session. It must see turn 1's real user
  // string and answer, then its own text part (the image is not user text).
  StagePayload second = run_turn(
      with_history(vlm_only_request(false, true), "vlm-sess"),
      "a settings screen");
  const auto roles = prompt_roles(second);
  if (roles.size() != 3) {
    check(false, "the second VLM-only turn did not carry one prior turn");
  } else {
    check(roles[0].first == "user" &&
              roles[0].second == "what is two plus two" &&
              roles[1].first == "assistant" && roles[1].second == "four" &&
              roles[2].first == "user" &&
              roles[2].second == "describe this image",
          "VLM-only history lost the real user text or answer");
  }
}

// Two sequential requests in one session. session_id is a grouping label only:
// ids, sequences and terminals stay per request, and neither request inherits
// the other's transcript or VLM answer.
void test_session_reuse() {
  Harness h("session");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  auto run = [&](const std::string& transcript, const std::string& answer) {
    const std::string id = engine.submit(joint_request("session-1"));
    auto asr = h.next_submit(h.asr_recv);
    check(asr && asr->request_id == id, "ASR leg id does not match the request");
    h.complete(id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm && vlm->request_id == id + ":vlm", "VLM leg id is not a child id");
    if (vlm) {
      // Keep the payload alive: content points into it.
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const Value& content = *payload.request.inputs.find("messages")
                                  ->as_array()
                                  .at(0)
                                  .find("content");
      check(content.is_array() && content.as_array().size() == 2 &&
                content.as_array()[1].find("text")->as_str() == transcript,
            "the VLM leg did not receive this request's transcript");
    }
    h.stream(id + ":vlm", "vlm", text_chunk(answer, false));
    h.complete(id + ":vlm", "vlm", true, text_result(answer));
    check(engine.wait(id, 5.0), "session request did not complete");
    return id;
  };

  const std::string first = run("first transcript", "first answer");
  const std::string second = run("second transcript", "second answer");
  check(first != second, "two session requests shared a request id");

  for (const std::string& id : {first, second}) {
    int64_t next_seq = 0;
    int terminals = 0;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-1", "event lost the session label");
      check(event.seq == next_seq++, "event sequence is not gap-free per request");
      if (event.is_final) ++terminals;
    }
    check(next_seq == 3, "unexpected event count for one session request");
    check(terminals == 2, "a session request did not emit asr_final and complete");
  }
  engine.release(first);
  engine.release(second);
}

// Concurrent requests in the same session interleave. Each event keeps its own
// request id and sequence; no delta or terminal crosses over.
void test_session_reuse_concurrent() {
  Harness h("sessionconc");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string a = engine.submit(joint_request("session-2"));
  const std::string b = engine.submit(joint_request("session-2"));
  auto asr_a = h.next_submit(h.asr_recv);
  auto asr_b = h.next_submit(h.asr_recv);
  check(asr_a && asr_b && asr_a->request_id != asr_b->request_id,
        "concurrent session requests did not reach the ASR stage separately");

  h.complete(a, "asr", true, text_result("alpha"));
  h.complete(b, "asr", true, text_result("beta"));
  auto vlm_a = h.next_submit(h.vlm_recv);
  auto vlm_b = h.next_submit(h.vlm_recv);
  check(vlm_a && vlm_b, "both concurrent VLM legs did not start");

  h.stream(a + ":vlm", "vlm", text_chunk("A", false));
  h.stream(b + ":vlm", "vlm", text_chunk("B", false));
  h.complete(b + ":vlm", "vlm", true, text_result("B"));
  h.complete(a + ":vlm", "vlm", true, text_result("A"));
  check(engine.wait(a, 5.0) && engine.wait(b, 5.0),
        "concurrent session requests did not settle");

  for (const std::string& id : {a, b}) {
    const std::string expected = id == a ? "A" : "B";
    int64_t next_seq = 0;
    std::string delta, terminal;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-2", "event lost the session label");
      check(event.seq == next_seq++, "concurrent event sequence is not per request");
      if (event.kind == UnifiedEvent::Kind::kVlmDelta) delta = event.text;
      if (event.kind == UnifiedEvent::Kind::kComplete) terminal = event.text;
    }
    check(delta == expected, "a VLM delta crossed between session requests");
    check(terminal == expected, "a terminal crossed between session requests");
  }
  engine.release(a);
  engine.release(b);
}

// Cancel before the queued ASR completion is pumped. With no pump thread the
// Coordinator still marks the request aborted synchronously inside cancel(),
// then the stale success completion must be ignored: one kCancelled, no VLM.
void test_cancel_before_pumped_completion() {
  Harness h("stale", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "ASR stage received no submit");

  // A sentinel request whose terminal we can observe. Both completions go on
  // the one PUSH -> PULL completion socket, which is FIFO, so seeing the
  // sentinel terminal proves the stale completion ahead of it was consumed.
  const std::string sentinel = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "sentinel ASR stage received no submit");

  // Queue the ASR success for the cancelled request, then the sentinel's
  // failure (a failure terminates without starting a VLM leg). Nobody drains
  // either message until the pump starts.
  h.complete(id, "asr", true, text_result("transcript"));
  h.complete(sentinel, "asr", false, Value{}, std::string("sentinel failure"));
  check(engine.cancel(id), "cancel rejected an active request");
  check(log.terminal_for(id).has_value() &&
            log.terminal_for(id)->kind == UnifiedEvent::Kind::kCancelled,
        "cancel before pumping did not settle as kCancelled");

  // Drain the queue. The sentinel terminal is a deterministic processing
  // barrier: FIFO delivery means the stale completion was already dequeued.
  h.start_pump();
  wait_until([&] { return log.has_terminal(sentinel); },
             "the sentinel completion was not processed");
  check(log.terminal_for(sentinel) &&
            log.terminal_for(sentinel)->kind == UnifiedEvent::Kind::kError,
        "the sentinel did not terminate as expected");

  // The stale success reached a terminal record, so it neither resurrected the
  // request nor started the VLM leg, and produced no second terminal.
  check(log.terminal_count(id) == 1, "the stale completion added a terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after cancellation");
  engine.release(id);
  engine.release(sentinel);
}

// Both interleavings of the ASR->VLM handoff, repeated. Every request must end
// as exactly one kCancelled; any VLM leg that did start must be aborted.
void test_cancel_handoff_race_stress() {
  Harness h("racestress");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  constexpr int kIterations = 40;
  for (int i = 0; i < kIterations; ++i) {
    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    if ((i % 2) == 0) {
      // Completion queued first; cancel may win or lose the pump race.
      h.complete(id, "asr", true, text_result("t"));
      engine.cancel(id);
    } else {
      engine.cancel(id);
      h.complete(id, "asr", true, text_result("t"));
    }
    check(engine.wait(id, 10.0), "a stress iteration did not settle");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
          "a handoff race did not settle as kCancelled");
    check(log.terminal_count(id) == 1,
          "a handoff race produced a duplicate terminal");

    // Every VLM submit that escaped must have a matching abort.
    std::set<std::string> vlm_submits;
    while (true) {
      auto msg = h.vlm_recv.recv_timeout(20);
      if (!msg) break;
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        vlm_submits.insert(submit->request_id);
      } else if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        h.record_abort(abort->request_id);
      }
    }
    for (const auto& child : vlm_submits) {
      check(h.saw_abort(child), "a started VLM leg was not aborted");
    }
    engine.release(id);
  }
}

// Deterministic reproduction of the ASR->VLM handoff race:
//   1. the Coordinator's ASR record is already "completed", but the engine has
//      not observed that completion (its record is still in phase kAsr);
//   2. cancel() then aborts an already-terminal ASR id, which is a no-op;
//   3. the success completion is handed to the engine afterwards.
// The engine must settle as exactly one kCancelled and must not start the VLM
// leg. This fails on the pre-fix engine, which started the VLM leg instead.
void test_cancel_after_asr_completed_swallows_vlm() {
  Harness h("swallow", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  // Save the engine sink, then install a dropping sink that resolves a promise
  // when the ASR completion is consumed. The pump only ever reads this dropping
  // sink; it is installed before start_pump and never reassigned, so there is
  // no concurrent read/write of coordinator.on_complete. The saved engine sink
  // is a separate copy invoked only on this test thread.
  auto engine_sink = h.coordinator.on_complete;
  std::promise<void> dropped;
  std::future<void> dropped_future = dropped.get_future();
  h.coordinator.on_complete = [&dropped, id](const RequestInfo& info) {
    if (info.request_id == id) dropped.set_value();
  };
  h.start_pump();
  h.complete(id, "asr", true, text_result("transcript"));

  // The future synchronizes with the pump thread: once wait() returns, the
  // dropping callback has run past set_value, the Coordinator ASR record is
  // "completed", and the engine has provably never seen that completion.
  dropped_future.wait();
  {
    auto info = h.coordinator.info(id);
    check(info && info->state == "completed",
          "the ASR coordinator record did not settle as completed");
  }

  // ASR is terminal in the Coordinator: abort(id) is now a no-op, exactly the
  // reviewer's window. cancel must still record the intent.
  check(engine.cancel(id), "cancel rejected the handoff request");

  // Deliver the success the engine has not seen yet, on this thread.
  RequestInfo asr_done;
  asr_done.request_id = id;
  asr_done.state = "completed";
  asr_done.current_stage = "asr";
  asr_done.result = text_result("transcript");
  engine_sink(asr_done);

  const auto terminal = log.terminal_for(id);
  check(terminal.has_value() &&
            terminal->kind == UnifiedEvent::Kind::kCancelled,
        "a cancel whose abort lost to ASR completion did not yield kCancelled");
  check(log.terminal_count(id) == 1, "the handoff produced a duplicate terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after a handoff cancellation");
  engine.release(id);
}

// An out-of-order VLM chunk takes the forwarding path. The previous sink
// reenters info() on the pump thread, which deadlocks if forwarding holds mu_.
void test_previous_sink_reentry_does_not_deadlock() {
  Harness h("sinkreentry");
  UnifiedEngine* engine_ptr = nullptr;
  std::promise<std::optional<RequestInfo>> reentered;
  auto reentered_future = reentered.get_future();
  h.coordinator.on_stream = [&](const StreamMessage& msg) {
    reentered.set_value(engine_ptr->info(msg.request_id));
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  engine_ptr = &engine;
  h.start_pump();

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "vlm", text_chunk("early", false));
  const auto status = reentered_future.wait_for(std::chrono::seconds(1));
  check(status == std::future_status::ready,
        "previous sink reentry deadlocked against engine mu_");
  if (status != std::future_status::ready) std::abort();
  const auto info = reentered_future.get();
  check(info && info->request_id == id && info->state == "running" &&
            info->current_stage == kAsrStageId,
        "previous sink reentry returned the wrong unified state");
  engine.release(id);
}

// Hold the partial callback while cancel() runs on another thread. The terminal
// must queue behind the already-sequenced partial, including callback reentry.
void test_cancel_cannot_overtake_partial_dispatch() {
  Harness h("partialcancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  std::promise<void> partial_done;
  auto partial_done_future = partial_done.get_future();

  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      partial_entered.set_value();
      release_partial_future.wait();
    }
    log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) partial_done.set_value();
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  partial_entered_future.wait();
  check(engine.cancel(id), "cancel rejected the partial-dispatch request");
  check(log.size() == 0, "cancel terminal overtook the blocked partial callback");
  release_partial.set_value();
  partial_done_future.wait();
  check(engine.wait(id, 5.0), "cancel during partial dispatch did not settle");
  wait_until([&] { return log.terminal_count(id) == 1; },
             "cancel during partial dispatch did not emit its terminal");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "single-drainer dispatch did not preserve partial then cancelled order");
  engine.release(id);

  // Same-thread reentry: cancel from inside on_event must only enqueue the
  // terminal. The outer drainer emits it after the partial callback returns.
  EventLog reentrant_log;
  std::string reentrant_id;
  std::promise<void> reentrant_partial_done;
  auto reentrant_partial_future = reentrant_partial_done.get_future();
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      check(engine.cancel(reentrant_id), "reentrant cancel rejected the request");
    }
    reentrant_log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      reentrant_partial_done.set_value();
    }
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { reentrant_log.finish(event); };
  reentrant_id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no reentrant submit");
  h.stream(reentrant_id, "asr", text_chunk("reentrant", false));
  reentrant_partial_future.wait();
  check(engine.wait(reentrant_id, 5.0), "reentrant cancel did not settle");
  wait_until([&] { return reentrant_log.terminal_count(reentrant_id) == 1; },
             "reentrant cancel did not emit its terminal");
  const auto reentrant_events = reentrant_log.snapshot();
  check(reentrant_events.size() == 2 &&
            reentrant_events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            reentrant_events[1].kind == UnifiedEvent::Kind::kCancelled,
        "reentrant cancel recursively overtook the partial callback");
  engine.release(reentrant_id);
}

// Forwarded callbacks share the unified event order. Cancel either from another
// thread or from inside on_event; neither may complete the old sink before the
// earlier stream callback has returned.
void test_forwarded_sink_cancel_order(bool reentrant) {
  Harness h(reentrant ? "fwdreenter" : "fwdthread");
  std::mutex forwarded_mu;
  std::vector<std::string> forwarded;
  bool stream_active = false;
  bool complete_overlapped_stream = false;
  std::promise<void> stream_entered;
  auto stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  auto release_stream_future = release_stream.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    {
      std::lock_guard<std::mutex> lock(forwarded_mu);
      forwarded.push_back("stream");
      stream_active = true;
    }
    stream_entered.set_value();
    release_stream_future.wait();
    std::lock_guard<std::mutex> lock(forwarded_mu);
    stream_active = false;
  };
  h.coordinator.on_complete = [&](const RequestInfo&) {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    complete_overlapped_stream |= stream_active;
    forwarded.push_back("complete");
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();

  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      if (reentrant) {
        check(engine.cancel(id), "reentrant cancel rejected the forwarding request");
      } else {
        partial_entered.set_value();
        release_partial_future.wait();
      }
    }
    log.add(event);
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no forwarding submit");
  h.stream(id, "asr", text_chunk("partial", false));
  if (!reentrant) {
    partial_entered_future.wait();
    check(engine.cancel(id), "cross-thread cancel rejected the forwarding request");
    release_partial.set_value();
  }
  check(engine.wait(id, 5.0), "cancel during forwarding did not settle");
  stream_entered_future.wait();
  release_stream.set_value();
  wait_until([&] {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    return forwarded.size() == 2 && !stream_active &&
           log.terminal_count(id) == 1;
  }, "forwarded stream/completion or unified terminal did not arrive");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "forwarding changed unified partial/cancelled order");
  check(log.terminal_count(id) == 1, "forwarding produced a duplicate terminal");

  std::vector<std::string> order;
  bool overlapped = false;
  {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    order = forwarded;
    overlapped = complete_overlapped_stream;
  }
  check(!overlapped, "forwarded completion reentered the active stream sink");
  std::cerr << (reentrant ? "FORWARD_ORDER_REENTRANT=" : "FORWARD_ORDER_THREAD=");
  for (const auto& call : order) std::cerr << call << ",";
  std::cerr << "\n";
  check(order == std::vector<std::string>{"stream", "complete"},
        "forwarded completion overtook the earlier stream callback");
  engine.release(id);
}

// A previous sink blocked on the pump proves run() is active. Construction must
// reject the exchange without modifying that sink instead of racing it.
void test_construction_after_pump_start_rejected() {
  Harness h("lateconstruct");
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::promise<void> previous_done;
  auto previous_done_future = previous_done.get_future();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    entered.set_value();
    release_future.wait();
    previous_done.set_value();
  };
  h.start_pump();

  const std::string id = h.coordinator.submit(audio_request(), "asr");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  entered_future.wait();

  bool rejected = false;
  try {
    UnifiedEngine late(h.coordinator, UnifiedConfig{});
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "construction exchanged sinks after the pump had started");
  release.set_value();
  previous_done_future.wait();
}

// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
// The destructor cannot return while a pump callback still uses the engine.
// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
void test_destructor_synchronizes_inflight_callback() {
  Harness h("destruct");
  std::atomic<int> forwarded_calls{0};
  std::promise<void> entered;
  std::future<void> entered_future = entered.get_future();
  std::promise<void> release;
  std::shared_future<void> release_future = release.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    if (forwarded_calls.fetch_add(1) == 0) {
      entered.set_value();
      release_future.wait();
    }
  };

  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine->on_event = [&](const UnifiedEvent& event) { log.add(event); };
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  entered_future.wait();
  const auto copied_sink = h.coordinator.on_stream;

  std::promise<void> destroy_started;
  std::future<void> destroy_started_future = destroy_started.get_future();
  std::promise<void> destroyed;
  std::future<void> destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    destroy_started.set_value();
    engine.reset();
    destroyed.set_value();
  });
  destroy_started_future.wait();
  const std::future_status early = destroyed_future.wait_for(std::chrono::seconds(1));
  check(early == std::future_status::timeout,
        "destructor returned while an inflight callback still used the engine");
  release.set_value();
  destroyed_future.wait();
  destroyer.join();
  check(log.size() == 1, "the inflight callback did not finish before destruction");

  StreamMessage after;
  after.request_id = "unknown";
  after.from_stage = "asr";
  after.chunk = text_chunk("after", false);
  copied_sink(after);
  check(forwarded_calls.load() == 2,
        "a copied sink did not forward safely after engine destruction");
}

struct RetiredSinkProbe {
  std::mutex mu;
  std::vector<std::string> calls;
  bool stream_active = false;
  bool overlapped = false;
  bool nested = false;
  std::thread::id stream_thread;
  bool throw_after_stream = false;
  bool throw_after_complete = false;
  int stream_calls = 0;
  std::function<void()> stream_action;
  std::promise<void> stream_entered;
  std::future<void> stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  std::shared_future<void> release_stream_future = release_stream.get_future().share();

  void stream(const StreamMessage&) {
    bool first = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      calls.push_back("stream");
      stream_active = true;
      stream_thread = std::this_thread::get_id();
      first = ++stream_calls == 1;
    }
    if (first) stream_entered.set_value();
    if (stream_action) stream_action();
    release_stream_future.wait();
    {
      std::lock_guard<std::mutex> lock(mu);
      stream_active = false;
    }
    if (throw_after_stream) throw std::runtime_error("previous stream failed");
  }

  void complete(const RequestInfo&) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (stream_active) {
        overlapped = true;
        if (std::this_thread::get_id() == stream_thread) nested = true;
      }
      calls.push_back("complete");
    }
    if (throw_after_complete) throw std::runtime_error("previous complete failed");
  }

  size_t call_count() {
    std::lock_guard<std::mutex> lock(mu);
    return calls.size();
  }

  void release() { release_stream.set_value(); }

  void check_order(const char* tag) {
    check_order_with_exceptions(tag, {"stream", "complete"});
  }

  void check_order_with_exceptions(
      const char* tag, const std::vector<std::string>& expected) {
    std::vector<std::string> order;
    bool saw_overlap = false;
    bool saw_nested = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      order = calls;
      saw_overlap = overlapped;
      saw_nested = nested;
    }
    std::cerr << tag << "=";
    for (const auto& call : order) std::cerr << call << ",";
    std::cerr << " OVERLAP=" << saw_overlap << " NESTED=" << saw_nested << "\n";
    check(order == expected, "forwarded callback order or drain is wrong");
    check(!saw_overlap, "forwarded completion overlapped the active stream callback");
    check(!saw_nested, "forwarded completion nested inside the stream callback");
  }
};

StreamMessage retired_stream_message() {
  StreamMessage msg;
  msg.request_id = "retired-stream";
  msg.from_stage = "asr";
  msg.chunk = text_chunk("after", false);
  return msg;
}

RequestInfo retired_complete_info() {
  RequestInfo info;
  info.request_id = "retired-complete";
  info.state = "aborted";
  return info;
}

// The copied completion arrives after retire() has nulled the engine but while
// the active previous_stream callback is still blocked.
void test_retire_wait_fallback_order() {
  RetiredSinkProbe probe;
  Harness h("retirewait");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  h.start_pump();
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "retire-wait ASR submit missing");
  h.stream(id, "asr", text_chunk("partial", false));
  probe.stream_entered_future.wait();
  const auto copied_complete = h.coordinator.on_complete;

  std::promise<void> destroyer_started;
  auto destroyer_started_future = destroyer_started.get_future();
  std::promise<void> destroyed;
  auto destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    destroyer_started.set_value();
    engine.reset();
    destroyed.set_value();
  });
  destroyer_started_future.wait();
  const bool retiring =
      destroyed_future.wait_for(std::chrono::milliseconds(25)) ==
      std::future_status::timeout;
  check(retiring, "retire did not wait for the active forwarded stream");
  if (!retiring) {
    probe.release();
    destroyed_future.wait();
    destroyer.join();
    return;
  }

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread fallback([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  complete_returned_future.wait();
  check(probe.call_count() == 1,
        "fallback completion ran before the active stream returned");
  probe.release();
  destroyed_future.wait();
  fallback.join();
  destroyer.join();
  probe.check_order("RETIRE_WAIT_ORDER");
}

// Two copied sinks race after destruction. The completion caller may return
// after enqueueing, but its old sink must not run until stream is released.
void test_retired_cross_thread_order() {
  RetiredSinkProbe probe;
  Harness h("retiredthreads");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  probe.stream_entered_future.wait();
  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  complete_returned_future.wait();
  check(probe.call_count() == 1,
        "post-retirement completion overlapped the blocked stream");
  probe.release();
  stream_thread.join();
  complete_thread.join();
  probe.check_order("RETIRED_THREAD_ORDER");
}

// A retired stream sink reenters completion on the same thread. Reentry must
// enqueue behind the active callback rather than recursively invoke the old sink.
void test_retired_same_thread_reentry() {
  RetiredSinkProbe probe;
  Harness h("retiredreenter");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.stream_action = [&] {
    copied_complete(retired_complete_info());
    check(probe.call_count() == 1,
          "retired completion recursively entered the old sink");
  };
  copied_stream(retired_stream_message());
  probe.check_order("RETIRED_REENTRANT_ORDER");
}

// A copied sink reports an exception only after every callback queued ahead of
// the error has drained. A later copied sink must still be usable.
void test_retired_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("retiredthrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "retired copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// The active engine drain has the same reentry contract before retirement.
void test_active_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("activethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "active copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// retire() is a destructor boundary: old sink exceptions are suppressed there,
// but the queued completion still runs and later copied sinks remain callable.
void test_destructor_suppresses_previous_sink_exception() {
  RetiredSinkProbe probe;
  Harness h("retirethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  probe.stream_entered_future.wait();
  std::promise<void> destroyer_started;
  auto destroyer_started_future = destroyer_started.get_future();
  std::promise<void> destroyed;
  auto destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    destroyer_started.set_value();
    engine.reset();
    destroyed.set_value();
  });
  destroyer_started_future.wait();
  check(destroyed_future.wait_for(std::chrono::milliseconds(25)) ==
            std::future_status::timeout,
        "retire did not wait for the blocked previous stream");

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  // The stream is still blocked, so this return proves the completion is in the
  // fallback queue and cannot run until retire() drains it.
  complete_returned_future.wait();
  probe.throw_after_complete = true;
  probe.release();
  stream_thread.join();
  complete_thread.join();
  destroyer.join();
  probe.check_order("DESTRUCTOR_EXCEPTION_DRAIN");

  probe.throw_after_complete = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("DESTRUCTOR_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// Real HTTP/SSE and Coordinator, with the same fake stage sockets as the other
// contract tests. This checks public wiring only; it is not a model result.
void test_public_http_unified_sse() {
  Harness h("publichttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public unified server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  nlohmann::json body = {{"model", "qwen3-unified"},
                         {"messages", nlohmann::json::array({message})},
                         {"stream", true},
                         {"max_tokens", 37},
                         {"temperature", 0.25}};
  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public HTTP did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "public WAV was not converted to the ASR PCM-only contract");
    check(!payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.0 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "ASR parameters were not fixed without losing max_new_tokens");
    h.complete(asr_submit->request_id, "asr", true, text_result("real transcript"));
  }

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public HTTP did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "real transcript" &&
              payload.request.inputs.find("audio") == nullptr,
          "VLM did not receive exactly the ASR transcript");
    check(payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.25 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "VLM did not preserve the public sampling parameters");
    h.stream(vlm_submit->request_id, "vlm", text_chunk("answer", false));
    auto result = text_result("answer");
    result["finish_reason"] = Value("stop");
    h.complete(vlm_submit->request_id, "vlm", true, Value(std::move(result)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public unified SSE returned an HTTP error");
  const size_t asr_pos = response.body.find("\"kind\":\"asr_final\"");
  const size_t vlm_pos = response.body.find("\"kind\":\"vlm_delta\"");
  const size_t complete_pos = response.body.find("\"kind\":\"complete\"");
  check(asr_pos != std::string::npos && vlm_pos != std::string::npos &&
            complete_pos != std::string::npos && asr_pos < vlm_pos &&
            vlm_pos < complete_pos,
        "public SSE order is not asr_final, vlm_delta, complete");
  check(response.body.find("\"text\":\"real transcript\"") != std::string::npos &&
            response.body.find("\"content\":\"answer\"") != std::string::npos,
        "public SSE lost the transcript or answer");
  check(response.body.find("\"kind\":\"complete\"", complete_pos + 1) ==
            std::string::npos,
        "public SSE emitted more than one successful terminal");
  wait_until([&] { return engine.in_flight() == 0; },
             "public unified request stayed in flight after completion");
}

// Public joint request: one instruction, one image, one WAV. The ASR leg sees
// PCM only; the VLM leg sees instruction, image, transcript in that order and
// every SSE unified event carries the public session label.
void test_public_http_joint_input() {
  Harness h("publicjoint");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public joint server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json instruction = {{"type", "text"}, {"text", "describe this image"}};
  nlohmann::json image = {{"type", "image_url"},
                          {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "public-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"},
           {"content", nlohmann::json::array({instruction, image, audio_part})}}})},
      {"stream", true}};
  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public joint request did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "the ASR leg received more than PCM");
    h.complete(asr_submit->request_id, "asr", true, text_result("spoken"));
  }
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public joint request did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 3,
          "the VLM prompt is not instruction + image + transcript");
    if (content.is_array() && content.as_array().size() == 3) {
      check(content.as_array()[0].find("text")->as_str() == "describe this image" &&
                content.as_array()[1].find("type")->as_str() == "image_url" &&
                content.as_array()[2].find("text")->as_str() == "spoken",
            "the VLM part order or content is wrong");
    }
    h.complete(vlm_submit->request_id, "vlm", true, text_result("answer"));
  }
  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public joint SSE returned an HTTP error");
  check(response.body.find("\"session_id\":\"public-session\"") !=
            std::string::npos,
        "public SSE lost the session label");
}

// Text-only and image+text requests without an input_audio part are valid
// unified turns: the HTTP boundary accepts both a plain string content and a
// content part array, no ASR leg is submitted, and the VLM answers directly.
void test_public_http_vlm_only() {
  Harness h("publicvlmonly");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public VLM-only server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  auto post = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    test_http::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    return test_http::perform(request);
  };
  auto body_with = [](nlohmann::json content) {
    return nlohmann::json{{"messages", nlohmann::json::array({nlohmann::json{
                               {"role", "user"}, {"content", std::move(content)}}})}};
  };

  // GSM8K shape: a plain string content.
  auto string_future = std::async(std::launch::async, [&] {
    return post(body_with("what is two plus two"));
  });
  check(h.no_submit(h.asr_recv), "string content submitted an ASR leg");
  auto string_vlm = h.next_submit(h.vlm_recv);
  check(string_vlm.has_value(), "string content did not reach the VLM");
  if (string_vlm) {
    StagePayload payload = StagePayload::from_value(string_vlm->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() &&
              content.as_array()[0].find("text")->as_str() ==
                  "what is two plus two",
          "the VLM lost the string content text");
    h.complete(string_vlm->request_id, "vlm", true, text_result("four"));
  }
  const auto string_response = string_future.get();
  check(string_response.status == 200 &&
            string_response.body.find("\"content\":\"four\"") !=
                std::string::npos,
        "the string-content turn did not return the VLM answer");

  // VQA shape: text part plus image part, no audio.
  const nlohmann::json image = {{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  const nlohmann::json instruction = {{"type", "text"},
                                      {"text", "describe this image"}};
  auto image_future = std::async(std::launch::async, [&] {
    return post(body_with(nlohmann::json::array({instruction, image})));
  });
  check(h.no_submit(h.asr_recv), "image+text content submitted an ASR leg");
  auto image_vlm = h.next_submit(h.vlm_recv);
  check(image_vlm.has_value(), "image+text content did not reach the VLM");
  if (image_vlm) {
    StagePayload payload = StagePayload::from_value(image_vlm->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2 &&
              content.as_array()[1].find("type")->as_str() == "image_url",
          "the VLM lost the image part on the audio-free turn");
    h.complete(image_vlm->request_id, "vlm", true,
               text_result("a settings screen"));
  }
  const auto image_response = image_future.get();
  check(image_response.status == 200 &&
            image_response.body.find("a settings screen") != std::string::npos,
        "the image+text turn did not return the VLM answer");

  // Two audio parts still fail the at-most-one-audio contract before submit.
  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)},
                       {"format", "wav"}}}};
  bool two_audio_rejected = false;
  try {
    post(body_with(nlohmann::json::array({audio_part, audio_part})));
  } catch (const client::Error& exc) {
    two_audio_rejected =
        std::string(exc.what()).find("HTTP 400") != std::string::npos;
  }
  check(two_audio_rejected, "two input_audio parts were accepted");
  check(h.no_submit(h.asr_recv),
        "a rejected two-audio request reached the ASR stage");

  // An audio-free request with no text or image is an empty prompt and must be
  // rejected before any stage sees it. Both an empty content array and an empty
  // string content are covered.
  auto rejected_with_400 = [&](nlohmann::json content) {
    try {
      post(body_with(std::move(content)));
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  check(rejected_with_400(nlohmann::json::array()),
        "an empty content array was accepted as an empty prompt");
  check(rejected_with_400(""),
        "an empty string content was accepted as an empty prompt");
  check(h.no_submit(h.vlm_recv),
        "a rejected empty-prompt request reached the VLM stage");

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "VLM-only requests stayed in flight after completion");
}

// The streaming VLM-only path must keep the pre-allocated HTTP request id on
// the single VLM leg. A generated id orphans the SSE sink: the client then
// sees only the role frame and hangs until the stream timeout.
void test_public_http_vlm_only_stream() {
  Harness h("publicvlmstream");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public VLM-only stream server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"}, {"content", "what is two plus two"}}})},
      {"stream", true},
      {"max_tokens", 16}};
  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  });

  check(h.no_submit(h.asr_recv), "a streaming VLM-only request reached ASR");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "the streaming VLM-only request missed the VLM");
  if (vlm_submit) {
    // The leg must run under the HTTP id, not a Coordinator-generated one.
    const std::string leg_id = vlm_submit->request_id;
    check(leg_id.rfind("http-", 0) == 0,
          "the VLM-only leg lost the public HTTP request id");
    h.stream(leg_id, "vlm", text_chunk("four", false));
    auto result = text_result("four");
    result["finish_reason"] = Value("stop");
    h.complete(leg_id, "vlm", true, Value(std::move(result)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200,
        "the streaming VLM-only turn returned an HTTP error");
  check(response.body.find("\"kind\":\"vlm_delta\"") != std::string::npos &&
            response.body.find("\"content\":\"four\"") != std::string::npos,
        "the streaming VLM-only turn delivered no vlm_delta content");
  check(response.body.find("\"kind\":\"complete\"") != std::string::npos &&
            response.body.find("data: [DONE]") != std::string::npos,
        "the streaming VLM-only turn lost its terminal or [DONE]");
  wait_until([&] { return engine.in_flight() == 0; },
             "the streaming VLM-only request stayed in flight");
}

// Bad unified media fields are rejected before any stage sees the work.
void test_public_http_unified_errors() {
  Harness h("publicbad");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public error-path server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  const nlohmann::json image = {{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  // The public client throws on a non-2xx response; a 400 is the expected result.
  auto rejected = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  auto body_with = [](nlohmann::json content) {
    return nlohmann::json{{"messages", nlohmann::json::array({nlohmann::json{
                               {"role", "user"}, {"content", std::move(content)}}})}};
  };

  auto two_images = body_with(nlohmann::json::array({image, image, audio_part}));
  check(rejected(two_images), "two image parts were accepted");
  auto two_texts = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", "a"}},
       nlohmann::json{{"type", "text"}, {"text", "b"}}, image, audio_part}));
  check(rejected(two_texts), "two text parts were accepted");
  auto empty_text = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", ""}}, image, audio_part}));
  check(rejected(empty_text), "an empty text part was accepted");
  auto empty_session = body_with(nlohmann::json::array({image, audio_part}));
  empty_session["session_id"] = "";
  check(rejected(empty_session), "an empty session_id was accepted");
  check(h.no_submit(h.asr_recv), "an invalid unified request reached the ASR stage");
  api.stop();
  server_thread.join();
}

size_t occurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

// Public chat body with one WAV input_audio part. The fake ASR stage receives
// the decoded PCM; no model runs in this file.
nlohmann::json public_audio_body(int max_tokens = 37) {
  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  return {{"model", "qwen3-unified"},
          {"messages", nlohmann::json::array({message})},
          {"stream", true},
          {"max_tokens", max_tokens},
          {"temperature", 0.0}};
}

// Public HTTP abort while the VLM leg is already streaming. The deltas that
// were sequenced before the abort arrive, then exactly one cancelled terminal.
// A late delta and a late success completion are dropped, the VLM stage sees the
// abort, in_flight returns to zero, and the next request completes. A
// cancellation is not a generation failure, so the stream carries no error
// frame.
void test_public_http_cancel_during_vlm() {
  Harness h("httpcancelvlm");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "cancel server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  std::mutex body_mu;
  std::string body;
  std::string stream_request_id;
  std::promise<void> saw_two_deltas;
  auto saw_two_deltas_future = saw_two_deltas.get_future();
  bool signaled = false;
  auto stream_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    request.on_body = [&](std::string_view bytes) {
      std::lock_guard<std::mutex> lock(body_mu);
      body.append(bytes);
      if (stream_request_id.empty()) {
        const size_t id_pos = body.find("\"id\":\"");
        if (id_pos != std::string::npos) {
          const size_t begin = id_pos + 6;
          const size_t end = body.find('"', begin);
          if (end != std::string::npos) {
            stream_request_id = body.substr(begin, end - begin);
          }
        }
      }
      if (!signaled && occurrences(body, "\"kind\":\"vlm_delta\"") >= 2) {
        signaled = true;
        saw_two_deltas.set_value();
      }
      return true;
    };
    return test_http::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "cancel request did not reach the ASR stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", true, text_result("transcript"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "cancel request did not reach the VLM stage");
  const std::string child = vlm_submit ? vlm_submit->request_id : id + ":vlm";
  h.stream(child, "vlm", text_chunk("first ", false));
  h.stream(child, "vlm", text_chunk("second", false));

  check(saw_two_deltas_future.wait_for(std::chrono::seconds(5)) ==
            std::future_status::ready,
        "two VLM deltas never reached the public stream");
  check(!stream_request_id.empty() && stream_request_id == id,
        "the public stream id did not match the submitted request");

  test_http::HttpRequest abort_request;
  abort_request.url = base + "/abort";
  abort_request.body =
      nlohmann::json{{"request_id", stream_request_id.empty() ? id
                                                              : stream_request_id}}
          .dump();
  abort_request.headers.emplace("Content-Type", "application/json");
  abort_request.timeout = client::Milliseconds{5000};
  const auto abort_response = test_http::perform(abort_request);
  check(abort_response.status == 200,
        "public abort did not accept the active request");

  // Tokens already in flight at the real worker can still arrive, and the
  // worker can still report success. Both are stale after the abort.
  h.stream(child, "vlm", text_chunk("late", false));
  h.complete(child, "vlm", true, text_result("late answer"));

  const auto response = stream_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "cancelled SSE returned an HTTP error");
  check(occurrences(body, "\"kind\":\"cancelled\"") == 1,
        "cancel during VLM did not produce exactly one cancelled terminal");
  check(occurrences(body, "\"kind\":\"complete\"") == 0 &&
            occurrences(body, "\"kind\":\"error\"") == 0,
        "cancel during VLM produced a second terminal");
  check(occurrences(body, "\"kind\":\"vlm_delta\"") == 2,
        "a VLM delta delivered after the abort reached the public stream");
  check(body.find("late") == std::string::npos,
        "late VLM text reached the public stream");
  check(occurrences(body, "[DONE]") == 1,
        "the cancelled stream did not end with exactly one [DONE]");
  const size_t cancelled_pos = body.find("\"kind\":\"cancelled\"");
  const size_t last_delta = body.rfind("\"kind\":\"vlm_delta\"");
  check(cancelled_pos != std::string::npos &&
            (last_delta == std::string::npos || last_delta < cancelled_pos),
        "a vlm_delta was written after the cancelled terminal");
  check(occurrences(body, "\"type\":\"generation_error\"") == 0,
        "a client-requested cancel was reported as a generation error");
  check(log.terminal_count(id) == 1,
        "the engine reported more than one terminal for one cancellation");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after a VLM cancellation");
  check(h.wait_abort(child), "the VLM stage never received the abort");

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after a cancellation did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("second transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after a cancellation did not reach the VLM stage");
  h.complete(next + ":vlm", "vlm", true, text_result("second answer"));
  check(engine.wait(next, 5.0),
        "the request after a cancellation did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after a cancellation did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

// Public HTTP request whose ASR leg fails in the stage, not at request
// validation. The stream carries one error attributed to the ASR stage, no ASR
// final, no VLM delta, no VLM submit, in_flight returns to zero, and the next
// request completes.
void test_public_http_asr_model_failure() {
  Harness h("httpasrfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "ASR failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR failure request did not reach the stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", false, Value{},
             std::string("ASR model backend unavailable"));

  const auto response = response_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "ASR failure SSE returned an HTTP error");
  check(occurrences(response.body, "\"kind\":\"error\"") == 1,
        "the ASR stage failure did not produce exactly one error terminal");
  check(occurrences(response.body, "\"kind\":\"asr_final\"") == 0 &&
            occurrences(response.body, "\"kind\":\"vlm_delta\"") == 0 &&
            occurrences(response.body, "\"kind\":\"complete\"") == 0,
        "the ASR stage failure emitted success or VLM events");
  check(occurrences(response.body, "\"type\":\"generation_error\"") == 1,
        "the ASR stage failure did not produce exactly one error frame");
  check(response.body.find("\"stage\":\"asr\"") != std::string::npos,
        "the error was not attributed to the ASR stage");
  check(response.body.find("ASR model backend unavailable") !=
            std::string::npos,
        "the stage failure reason was not propagated");
  check(occurrences(response.body, "[DONE]") == 1,
        "the failed stream did not end with exactly one [DONE]");
  check(log.terminal_count(id) == 1, "the ASR failure produced duplicate terminals");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->stage_id == kAsrStageId,
        "the ASR failure terminal did not name the ASR stage");
  check(h.no_submit(h.vlm_recv),
        "the VLM leg started after an ASR stage failure");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after an ASR stage failure");
  engine.release(id);

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after an ASR failure did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("recovered transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after an ASR failure did not reach the VLM stage");
  h.stream(next + ":vlm", "vlm", text_chunk("recovered", false));
  h.complete(next + ":vlm", "vlm", true, text_result("recovered answer"));
  check(engine.wait(next, 5.0),
        "the request after an ASR failure did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after an ASR failure did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

UnifiedConfig history_config(size_t max_turns, size_t max_chars) {
  UnifiedConfig config;
  config.session_history.enabled = true;
  config.session_history.max_turns = max_turns;
  config.session_history.max_chars = max_chars;
  return config;
}

// Opt-in history: the second turn's VLM prompt prepends the first completed
// turn as user/assistant messages, in order, ahead of the current user turn.
void test_session_history_order() {
  Harness h("historder");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  StagePayload first = complete_turn(h, engine, history_request("sess"),
                                     "first transcript", "first answer");
  check(prompt_roles(first).size() == 1,
        "the first history turn already carried prior context");

  StagePayload second = complete_turn(h, engine, history_request("sess"),
                                      "second transcript", "second answer");
  const auto roles = prompt_roles(second);
  if (roles.size() != 3) {
    check(false, "the second turn did not carry one prior turn plus itself");
  } else {
    check(roles[0].first == "user" && roles[0].second == "first transcript" &&
              roles[1].first == "assistant" && roles[1].second == "first answer" &&
              roles[2].first == "user" && roles[2].second == "second transcript",
          "history order or content is wrong");
  }
}

// Default off at two levels: an engine without the feature rejects the opt-in,
// and an enabled engine leaves a plain session_id request without history.
void test_session_history_default_off() {
  {
    Harness h("histoff");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    bool rejected = false;
    try {
      engine.submit(history_request("sess"));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, "a default engine accepted session_history");
    check(h.no_submit(h.asr_recv), "a rejected history request reached ASR");
  }
  {
    Harness h("histlabel");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    // A request carrying only the label: no opt-in, no history on the next
    // turn even though the engine supports it.
    complete_turn(h, engine, joint_request("sess"), "t1", "a1");
    StagePayload next = complete_turn(h, engine, joint_request("sess"), "t2", "a2");
    check(prompt_roles(next).size() == 1,
          "a label-only request inherited history without opt-in");
  }
}

// Sessions never share history, and an opt-in without an explicit session_id
// is rejected rather than silently attached to a generated label.
void test_session_history_isolation() {
  Harness h("histiso");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  complete_turn(h, engine, history_request("sess-a"), "a transcript", "a answer");
  StagePayload other =
      complete_turn(h, engine, history_request("sess-b"), "b transcript", "b answer");
  check(prompt_roles(other).size() == 1,
        "history leaked between two sessions");

  StagePayload no_label = history_request("sess-x");
  no_label.request.inputs.as_map_mut().erase("session_id");
  bool rejected = false;
  try {
    engine.submit(no_label);
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "session_history without session_id was accepted");
}

// Cancelled and failed requests never enter history: the next turn sees only
// the successful turns, never a failed transcript or answer.
void test_session_history_cancel_error_excluded() {
  Harness h("histfail");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  // A cancelled turn.
  const std::string cancelled = engine.submit(history_request("sess"));
  check(h.next_submit(h.asr_recv).has_value(), "cancel turn: no ASR submit");
  check(engine.cancel(cancelled), "cancel rejected an active history request");
  h.complete(cancelled, "asr", false, Value{}, std::string("aborted"));
  check(engine.wait(cancelled, 5.0), "cancelled turn did not settle");
  check(log.terminal_for(cancelled)->kind == UnifiedEvent::Kind::kCancelled,
        "the cancelled turn did not report kCancelled");
  engine.release(cancelled);

  // A failed turn: ASR stage failure, no VLM leg.
  const std::string failed = engine.submit(history_request("sess"));
  check(h.next_submit(h.asr_recv).has_value(), "fail turn: no ASR submit");
  h.complete(failed, "asr", false, Value{}, std::string("backend reset"));
  check(engine.wait(failed, 5.0), "failed turn did not settle");
  check(log.terminal_for(failed)->kind == UnifiedEvent::Kind::kError,
        "the failed turn did not report kError");
  check(h.no_submit(h.vlm_recv), "a failed turn started the VLM leg");
  engine.release(failed);

  StagePayload good =
      complete_turn(h, engine, history_request("sess"), "good transcript", "good answer");
  check(prompt_roles(good).size() == 1,
        "a cancelled or failed turn entered session history");
}

// Turn-count bound: only the newest max_turns completed turns stay. Eviction
// runs when a turn is recorded, so a turn still sees the turns present at its
// own start; the oldest turn disappears one turn later.
void test_session_history_turn_eviction() {
  Harness h("histturns");
  UnifiedEngine engine(h.coordinator, history_config(2, 16384));
  h.start_pump();

  complete_turn(h, engine, history_request("sess"), "t1", "a1");
  complete_turn(h, engine, history_request("sess"), "t2", "a2");
  // The third turn starts while both prior turns are still retained.
  StagePayload third =
      complete_turn(h, engine, history_request("sess"), "t3", "a3");
  auto roles = prompt_roles(third);
  check(roles.size() == 5, "third turn prompt has the wrong size");
  if (roles.size() == 5) {
    check(roles[0].second == "t1" && roles[2].second == "t2" &&
              roles[4].second == "t3",
          "the third turn did not see both retained turns in order");
  }

  // Recording the third turn evicted the first; the fourth sees t2 and t3.
  StagePayload fourth =
      complete_turn(h, engine, history_request("sess"), "t4", "a4");
  roles = prompt_roles(fourth);
  check(roles.size() == 5 && roles[0].second == "t2" && roles[4].second == "t4",
        "turn eviction order is not FIFO");
}

// Character bound: a small budget evicts the oldest turns when a new turn is
// recorded, and one oversized single turn evicts itself rather than staying
// unbounded.
void test_session_history_char_eviction() {
  Harness h("histchars");
  UnifiedEngine engine(h.coordinator, history_config(8, 20));
  h.start_pump();

  complete_turn(h, engine, history_request("sess"), "0123456789", "0123456789");
  // The second turn starts while the 20-char first turn is still retained.
  StagePayload second =
      complete_turn(h, engine, history_request("sess"), "new", "new");
  auto roles = prompt_roles(second);
  check(roles.size() == 3 && roles[0].second == "0123456789" &&
            roles[2].second == "new",
        "the second turn did not see the retained first turn");

  // Recording the second turn overflowed the 20-char budget and evicted the
  // first. One turn larger than the whole budget is then evicted on recording,
  // so the next turn starts with empty history.
  complete_turn(h, engine, history_request("sess"),
                std::string(30, 'x'), std::string(30, 'y'));
  StagePayload after =
      complete_turn(h, engine, history_request("sess"), "z", "z");
  check(prompt_roles(after).size() == 1,
        "an oversized turn survived the character budget");
}

// Global session cap: with max_sessions=2, recording a third session evicts
// the session with the oldest last turn, even though each session holds only
// one turn.
void test_session_history_session_eviction() {
  UnifiedConfig config = history_config(8, 16384);
  config.session_history.max_sessions = 2;
  Harness h("histsessions");
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  complete_turn(h, engine, history_request("s1"), "t1", "a1");
  complete_turn(h, engine, history_request("s2"), "t2", "a2");
  // A new turn on s1 makes s2 the least recently used session.
  complete_turn(h, engine, history_request("s1"), "t1b", "a1b");
  complete_turn(h, engine, history_request("s3"), "t3", "a3");

  // Check the retained session first: recording a fresh s2 turn below would
  // itself evict the current oldest session.
  StagePayload s1_later =
      complete_turn(h, engine, history_request("s1"), "t1c", "a1c");
  check(prompt_roles(s1_later).size() == 5,
        "a retained session lost its turns after another session was evicted");
  StagePayload s2_later =
      complete_turn(h, engine, history_request("s2"), "t2b", "a2b");
  check(prompt_roles(s2_later).size() == 1,
        "the evicted session retained its history");
}

// Concurrent requests in one history-enabled session keep independent request
// ids, per-request seq from zero, and one terminal each. A turn is visible to
// later turns only after its own successful terminal.
void test_session_history_concurrent() {
  Harness h("histconc");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string a = engine.submit(history_request("sess"));
  const std::string b = engine.submit(history_request("sess"));
  check(a != b, "concurrent turns shared a request id");
  auto asr_a = h.next_submit(h.asr_recv);
  auto asr_b = h.next_submit(h.asr_recv);
  check(asr_a && asr_b, "concurrent turns did not both reach ASR");

  // Neither turn sees the other: both were submitted before either completed.
  h.complete(a, "asr", true, text_result("ta"));
  h.complete(b, "asr", true, text_result("tb"));
  auto vlm_a = h.next_submit(h.vlm_recv);
  auto vlm_b = h.next_submit(h.vlm_recv);
  check(vlm_a && vlm_b, "concurrent VLM legs did not start");
  for (const auto* submit : {&vlm_a, &vlm_b}) {
    StagePayload payload = StagePayload::from_value((*submit)->data);
    check(payload.request.inputs.find("messages")->as_array().size() == 1,
          "an in-flight turn observed another turn's uncommitted history");
  }

  h.complete(a + ":vlm", "vlm", true, text_result("aa"));
  h.complete(b + ":vlm", "vlm", true, text_result("ab"));
  check(engine.wait(a, 5.0) && engine.wait(b, 5.0),
        "concurrent turns did not settle");
  for (const std::string& id : {a, b}) {
    int64_t next_seq = 0;
    int terminals = 0;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.seq == next_seq++, "concurrent history seq is not per request");
      if (event.is_final) ++terminals;
    }
    check(terminals == 2, "a concurrent turn lost its asr_final or terminal");
    engine.release(id);
  }

  // A later turn sees exactly the two committed turns.
  StagePayload later =
      complete_turn(h, engine, history_request("sess"), "tc", "ac");
  const auto roles = prompt_roles(later);
  check(roles.size() == 5, "the later turn did not see both committed turns");
}

// History is process-local: an engine on a fresh Coordinator starts with empty
// history. The store has no persistence path.
void test_session_history_restart_drops() {
  StagePayload later;
  {
    Harness h("histrestart");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    complete_turn(h, engine, history_request("sess"), "old transcript", "old answer");
  }
  {
    Harness h("histrestart2");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    later = complete_turn(h, engine, history_request("sess"),
                          "new transcript", "new answer");
  }
  check(prompt_roles(later).size() == 1,
        "history survived an engine restart");
}

// Public HTTP contract for the session_history opt-in: rejected outside
// unified mode and on a default engine, accepted and carried into the VLM
// prompt only when the server enabled it alongside an explicit session_id.
void test_public_http_session_history() {
  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  auto body = [&](nlohmann::json extra) {
    extra["model"] = "qwen3-unified";
    extra["stream"] = false;
    extra["messages"] = nlohmann::json::array(
        {nlohmann::json{{"role", "user"},
                        {"content", nlohmann::json::array({audio_part})}}});
    return extra;
  };
  auto post = [](OpenAIApi& api, const nlohmann::json& payload) {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = payload.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what());
    }
    return std::string();
  };

  // Non-unified server: 400, never accepted and dropped.
  {
    Harness h("histhttpplain");
    ApiConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model = "qwen3-unified";
    config.request_timeout_s = 5.0;
    OpenAIApi api(h.coordinator, config);
    check(api.bind(), "plain history server did not bind");
    h.start_pump();
    std::thread server_thread([&] { api.listen(); });
    const std::string error = post(api, body({{"session_history", true}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("requires unified mode") != std::string::npos,
          "non-unified server accepted session_history");
    check(h.no_submit(h.asr_recv),
          "a rejected session_history request reached the ASR stage");
    api.stop();
    server_thread.join();
  }

  // Unified engine started without the switch: the opt-in is refused.
  {
    Harness h("histhttpdisabled");
    ApiConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model = "qwen3-unified";
    config.request_timeout_s = 5.0;
    OpenAIApi api(h.coordinator, config);
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    api.set_unified_engine(engine, "vlm");
    check(api.bind(), "disabled-history server did not bind");
    h.start_pump();
    std::thread server_thread([&] { api.listen(); });
    std::string error =
        post(api, body({{"session_history", true}, {"session_id", "sess"}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("not enabled") != std::string::npos,
          "a default engine accepted the history opt-in over HTTP");
    error = post(api, body({{"session_history", "yes"}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("must be a boolean") != std::string::npos,
          "a non-boolean session_history was accepted");
    api.stop();
    server_thread.join();
  }

  // Enabled engine: missing session_id is 400; with the label the second turn
  // carries the first turn in its VLM prompt.
  Harness h("histhttpok");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "history-enabled server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  std::string error = post(api, body({{"session_history", true}}));
  check(error.find("HTTP 400") != std::string::npos &&
            error.find("requires a nonempty session_id") != std::string::npos,
        "session_history without session_id was accepted");

  auto run_turn = [&](const std::string& transcript, const std::string& answer) {
    auto response_future = std::async(std::launch::async, [&] {
      return post(api, body({{"session_history", true},
                             {"session_id", "http-session"}}));
    });
    auto asr = h.next_submit(h.asr_recv);
    check(asr.has_value(), "HTTP history turn: no ASR submit");
    const std::string id = asr ? asr->request_id : std::string();
    h.complete(id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "HTTP history turn: no VLM submit");
    StagePayload vlm_payload;
    if (vlm) vlm_payload = StagePayload::from_value(vlm->data);
    h.complete(id + ":vlm", "vlm", true, text_result(answer));
    check(response_future.get().empty(), "a successful history turn returned an HTTP error");
    wait_until([&] { return engine.in_flight() == 0; },
               "HTTP history turn stayed in flight");
    return vlm_payload;
  };

  StagePayload first = run_turn("first transcript", "first answer");
  check(prompt_roles(first).size() == 1,
        "the first HTTP history turn already carried context");
  StagePayload second = run_turn("second transcript", "second answer");
  const auto roles = prompt_roles(second);
  check(roles.size() == 3 && roles[0].second == "first transcript" &&
            roles[1].second == "first answer" &&
            roles[2].second == "second transcript",
        "the second HTTP turn did not receive the prior turn as history");

  api.stop();
  server_thread.join();
}

// Without a unified engine, session_id must be rejected, not accepted and
// dropped. This keeps the server-wide unknown-field contract intact.
void test_public_http_session_id_requires_unified() {
  Harness h("plainsession");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  check(api.bind(), "plain server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "plain-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"}, {"content", "hello"}}})}};
  test_http::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                "/v1/chat/completions";
  request.body = body.dump();
  request.headers.emplace("Content-Type", "application/json");
  request.timeout = client::Milliseconds{5000};
  bool rejected_with_400 = false;
  try {
    test_http::perform(request);
  } catch (const client::Error& exc) {
    rejected_with_400 =
        std::string(exc.what()).find("HTTP 400") != std::string::npos &&
        std::string(exc.what()).find("requires unified mode") != std::string::npos;
  }
  check(rejected_with_400,
        "a non-unified request accepted session_id instead of rejecting it");
  check(h.no_submit(h.asr_recv),
        "a non-unified request with session_id reached the ASR stage");
  api.stop();
  server_thread.join();
}

// ---- Mixed output (image + tts) contract tests ---------------------------
//
// The image and tts "stages" are the same bound PULL/PUSH fakes as ASR/VLM.
// Their Bin results are contract fixtures, not real model artifacts.

// Drives a VLM-only mixed request to the output barrier and returns the two
// leg submit payloads in arrival order. A VLM-only turn keeps the unified id
// on its single VLM leg, so the completion uses the observed submit id.
std::vector<StagePayload> reach_output_barrier(
    Harness& h, UnifiedEngine& engine, const std::string& id,
    const std::string& answer, bool expect_tts = true) {
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "mixed request did not reach the VLM");
  h.complete(vlm ? vlm->request_id : id, "vlm", true,
             text_result(answer));
  std::vector<StagePayload> legs;
  // Only the legs the request asked for are awaited: a leg that was never
  // requested would wait out the whole budget and then report a missing
  // submit the contract never promised.
  auto image = h.next_submit(h.image_recv);
  check(image.has_value(), "image leg was not submitted");
  if (image) legs.push_back(StagePayload::from_value(image->data));
  if (!expect_tts) return legs;
  auto tts = h.next_submit(h.tts_recv);
  check(tts.has_value(), "tts leg was not submitted");
  if (tts) legs.push_back(StagePayload::from_value(tts->data));
  return legs;
}

void test_mixed_fanout_join_and_prompt_rules() {
  Harness h("mixedjoin");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(mixed_request(true, true, "explicit prompt"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "no VLM submit");
  h.complete(vlm_submit->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  check(image_submit && image_submit->request_id == id + ":image",
        "image leg id is not the derived :image child id");
  check(tts_submit && tts_submit->request_id == id + ":tts",
        "tts leg id is not the derived :tts child id");
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "explicit prompt",
          "explicit image prompt did not win over the VLM answer");
    check(p.request.inputs.find("messages") == nullptr,
          "image leg received VLM-only messages");
  }
  if (tts_submit) {
    StagePayload p = StagePayload::from_value(tts_submit->data);
    check(p.request.inputs.find("text")->as_str() == "the answer",
          "tts input is not the aggregated VLM answer");
    check(p.request.inputs.find("voice")->as_str() == "Tina" &&
              p.request.inputs.find("sample_rate")->as_int() == 24000,
          "tts options were not forwarded");
  }

  const std::string image_bytes(10, '\x89');
  const std::string audio_bytes(20, '\x52');
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", image_bytes));
  h.complete(id + ":tts", "tts", true,
             artifact_result("audio", "audio/wav", audio_bytes));
  check(engine.wait(id, 5.0), "mixed request did not complete");

  const auto events = log.snapshot();
  int64_t expected_seq = 0;
  size_t artifacts = 0;
  for (const UnifiedEvent& e : events) {
    check(e.seq == expected_seq++, "mixed event seq has a gap");
    if (e.kind == UnifiedEvent::Kind::kArtifact) {
      ++artifacts;
      check(e.artifact_bytes != nullptr &&
                e.artifact_bytes->substr(e.artifact_offset,
                                         e.artifact_length)
                        .size() == e.artifact_length,
            "artifact chunk does not cover its declared slice");
      check(e.artifact.find("artifact_id") != nullptr,
            "artifact descriptor missing artifact_id");
    }
  }
  check(artifacts == 2, "expected one artifact event per small artifact");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            terminal->legs.is_array() &&
            terminal->legs.as_array().size() == 2,
        "mixed terminal is not one kComplete with a two-entry legs array");
  if (terminal) {
    for (const Value& leg : terminal->legs.as_array()) {
      check(leg.find("state")->as_str() == "completed",
            "a successful leg is not marked completed");
    }
  }
  check(engine.in_flight() == 0, "mixed request stayed in flight");
  engine.release(id);
}

void test_mixed_image_prompt_falls_back_to_answer() {
  Harness h("mixedprompt");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, false));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  h.complete(vlm_submit->request_id, "vlm", true,
             text_result("fallback answer"));
  auto image_submit = h.next_submit(h.image_recv);
  check(image_submit.has_value(), "image leg did not start");
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "fallback answer",
          "image leg did not fall back to the VLM answer");
  }
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(5, 'x')));
  check(engine.wait(id, 5.0), "image-only mixed request did not complete");
  engine.release(id);
}

void test_mixed_partial_leg_failure_keeps_artifacts() {
  Harness h("mixedpartial");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(7, 'p')));
  h.complete(id + ":tts", "tts", false, Value(),
             std::string("tts worker down"));
  check(engine.wait(id, 5.0), "partial-failure request did not settle");

  const auto events = log.snapshot();
  size_t leg_errors = 0;
  size_t artifacts = 0;
  for (const UnifiedEvent& e : events) {
    if (e.kind == UnifiedEvent::Kind::kLegError) {
      ++leg_errors;
      check(e.stage_id == kTtsStageId && e.error == "tts worker down",
            "leg_error does not name the failed tts leg");
    }
    if (e.kind == UnifiedEvent::Kind::kArtifact) ++artifacts;
  }
  check(leg_errors == 1 && artifacts == 1,
        "partial failure did not keep the delivered image artifact");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "a leg failure produced the wrong terminal or more than one");
  if (terminal) {
    const auto& legs = terminal->legs.as_array();
    check(legs.size() == 2, "terminal legs array does not cover both legs");
    bool tts_failed = false;
    for (const Value& leg : legs) {
      if (leg.find("stage")->as_str() == kTtsStageId) {
        tts_failed = leg.find("state")->as_str() == "failed" &&
                     leg.find("error") != nullptr;
      }
    }
    check(tts_failed, "terminal does not report the failed tts leg");
  }
  engine.release(id);
}

void test_mixed_cancel_aborts_every_submitted_leg() {
  Harness h("mixedcancel");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  check(engine.cancel(id), "cancel rejected an active mixed request");
  check(h.wait_abort(id + ":image"), "image leg never received abort");
  check(h.wait_abort(id + ":tts"), "tts leg never received abort");
  // Abort completions arrive through the same pump; both legs must settle
  // before the single terminal.
  h.complete(id + ":image", "image", false, Value(),
             std::string("request aborted"));
  h.complete(id + ":tts", "tts", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled mixed request did not settle");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
            log.terminal_count(id) == 1,
        "mixed cancel did not produce exactly one cancelled terminal");
  if (terminal && terminal->legs.is_array()) {
    bool image_aborted = false;
    bool tts_aborted = false;
    for (const Value& leg : terminal->legs.as_array()) {
      const Value* stage = leg.find("stage");
      const Value* state = leg.find("state");
      if (stage != nullptr && state != nullptr && stage->is_str() && state->is_str()) {
        if (stage->as_str() == "image" && state->as_str() == "aborted") {
          image_aborted = true;
        }
        if (stage->as_str() == "tts" && state->as_str() == "aborted") {
          tts_aborted = true;
        }
      }
    }
    check(image_aborted, "cancelled mixed terminal did not report image leg as aborted");
    check(tts_aborted, "cancelled mixed terminal did not report tts leg as aborted");
  } else {
    check(false, "cancelled mixed terminal missing legs array");
  }
  engine.release(id);
  check(engine.in_flight() == 0, "in_flight nonzero after cancel and release");
}

void test_streamed_tts_submit_failure_becomes_leg_error() {
  Harness h("streamedttsfail");
  UnifiedConfig cfg = h.mixed_config();
  // Point tts_stage to a stage name that is not registered in Coordinator so
  // Coordinator::submit throws during handle_stream -> drain_sentences_locked.
  cfg.tts_stage = "unregistered_tts";
  UnifiedEngine engine(h.coordinator, cfg);
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  StagePayload req = mixed_request(false, true);
  Value::Map inputs = req.request.inputs.as_map();
  Value::Map audio_opts = inputs["audio_options"].as_map();
  audio_opts["stream"] = Value(true);
  inputs["audio_options"] = Value(std::move(audio_opts));
  req.request.inputs = Value(std::move(inputs));

  const std::string id = engine.submit(std::move(req));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "VLM submit missing for streamed TTS failure test");
  if (!vlm) return;

  // Sentence-ending punctuation triggers drain_sentences_locked inside handle_stream.
  Value::Map delta;
  delta["text"] = Value("Hello world.");
  h.stream(vlm->request_id, "vlm", Value(std::move(delta)));
  h.complete(vlm->request_id, "vlm", true, text_result("Hello world."));

  check(engine.wait(id, 5.0),
        "streamed TTS submit failure must not wedge or kill the coordinator pump");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "VLM request must still complete when streamed TTS submit fails");
  bool saw_leg_error = false;
  bool saw_vlm_delta = false;
  {
    std::lock_guard<std::mutex> lock(log.mu);
    for (const auto& e : log.events) {
      if (e.request_id != id) continue;
      if (e.kind == UnifiedEvent::Kind::kLegError && e.stage_id == "tts") {
        saw_leg_error = true;
      }
      if (e.kind == UnifiedEvent::Kind::kVlmDelta && e.text == "Hello world.") {
        saw_vlm_delta = true;
      }
    }
  }
  check(saw_leg_error, "streamed TTS submit failure did not emit kLegError");
  check(saw_vlm_delta, "VLM delta was lost after streamed TTS submit failure");
  bool tts_failed = false;
  if (terminal && terminal->legs.is_array()) {
    for (const Value& leg : terminal->legs.as_array()) {
      const Value* stage = leg.find("stage");
      const Value* state = leg.find("state");
      if (stage != nullptr && state != nullptr && stage->is_str() &&
          state->is_str() && stage->as_str() == "tts" &&
          state->as_str() == "failed") {
        tts_failed = true;
      }
    }
  }
  check(tts_failed, "terminal legs did not report failed tts leg");
  engine.release(id);
}

void test_mixed_cancel_before_fanout_blocks_legs() {
  Harness h("mixednofanout");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(engine.cancel(id), "cancel rejected the VLM-phase request");
  check(h.wait_abort(vlm ? vlm->request_id : id),
        "the VLM leg never received abort");
  h.complete(vlm->request_id, "vlm", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "pre-fanout cancel did not settle");
  check(h.no_submit(h.image_recv), "an image leg started after cancel");
  check(h.no_submit(h.tts_recv), "a tts leg started after cancel");
  engine.release(id);
}

void test_mixed_join_release_then_late_complete() {
  Harness h("mixedrelease");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  // Release while legs are still running, then let both complete late. The
  // orphan completions must be ignored without a crash and the engine stays
  // usable with in_flight at zero.
  check(engine.release(id), "release rejected the in-flight record");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(4, 'q')));
  h.complete(id + ":tts", "tts", true,
             artifact_result("audio", "audio/wav", std::string(6, 'w')));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  check(engine.in_flight() == 0, "late completions resurrected a record");
  const std::string next = engine.submit(mixed_request(false, false));
  auto next_vlm = h.next_submit(h.vlm_recv);
  h.complete(next_vlm->request_id, "vlm", true, text_result("recovered"));
  check(engine.wait(next, 5.0), "the request after release did not settle");
  check(engine.in_flight() == 0, "in_flight nonzero after recovery");
  engine.release(next);
}

void test_mixed_artifact_limits_become_leg_errors() {
  Harness h("mixedlimits");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", false);
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png",
                             std::string(kMaxArtifactImageBytes + 1, '\x01')));
  check(engine.wait(id, 5.0), "oversized request did not settle");
  bool saw_limit_error = false;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kLegError &&
        e.error.find("limit") != std::string::npos) {
      saw_limit_error = true;
    }
  }
  check(saw_limit_error, "an oversized artifact was not rejected");
  engine.release(id);
}

void test_mixed_chunk_seq_reservation() {
  Harness h("mixedchunks");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", false);
  // 5 MiB of fake bytes must split into three 2 MiB-target chunks.
  const std::string bytes(5u << 20, '\xab');
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", bytes));
  check(engine.wait(id, 5.0), "chunked request did not complete");
  std::vector<UnifiedEvent> chunks;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kArtifact) chunks.push_back(e);
  }
  check(chunks.size() == 3, "5 MiB did not produce three artifact chunks");
  if (chunks.size() == 3) {
    check(chunks[0].seq + 1 == chunks[1].seq &&
              chunks[1].seq + 1 == chunks[2].seq,
          "artifact chunk seq values are not contiguous");
    size_t reassembled = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      check(chunks[i].artifact.find("index")->as_int() ==
                    static_cast<int64_t>(i) &&
                chunks[i].artifact.find("total")->as_int() == 3,
            "artifact chunk index/total is wrong");
      reassembled += chunks[i].artifact_length;
    }
    check(reassembled == bytes.size(),
          "artifact chunks do not reassemble to the full byte count");
  }
  engine.release(id);
}

void test_mixed_previous_stream_tolerates_bin_chunk() {
  Harness h("mixedbin");
  std::promise<const StreamMessage*> forwarded;
  auto future = forwarded.get_future();
  // Attached before engine construction, exactly like the realtime listener:
  // the engine exchanges sinks and this probe becomes its previous_stream.
  h.coordinator.on_stream = [&](const StreamMessage& msg) {
    forwarded.set_value(&msg);
  };
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", false);
  h.stream(id + ":image", "image", Value::bin(std::string(12, '\x7f')));
  check(future.wait_for(std::chrono::seconds(2)) ==
            std::future_status::ready,
        "a Bin chunk did not pass through the previous_stream chain");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(3, 'z')));
  check(engine.wait(id, 5.0), "bin-tolerance request did not complete");
  engine.release(id);
}

// Splits an SSE response body into its data payload strings.
using nlohmann::json;

std::vector<std::string> sse_data_payloads(const std::string& body) {
  std::vector<std::string> payloads;
  size_t pos = 0;
  while (pos < body.size()) {
    const size_t line_end = body.find('\n', pos);
    std::string line = body.substr(
        pos, line_end == std::string::npos ? std::string::npos
                                          : line_end - pos);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("data: ", 0) == 0) {
      std::string data = line.substr(6);
      if (data != "[DONE]") payloads.push_back(std::move(data));
    }
    if (line_end == std::string::npos) break;
    pos = line_end + 1;
  }
  return payloads;
}

// Full HTTP path: real OpenAIApi plus the fake image/tts PULL stages. The raw
// SSE body is parsed to prove chunking, base64, sha256, gap-free seq and the
// single terminal end to end.
void test_public_http_mixed_artifact_stream() {
  Harness h("publicmixed");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "mixed HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  // 5 MiB image forces three chunks; the audio stays in one small chunk.
  const std::string image_bytes(5u << 20, '\x89');
  const std::string audio_bytes(4096, '\x52');
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image", "audio"})},
      {"image", {{"output_format", "png"}, {"size", "1024x1024"}}},
      {"audio", {{"voice", "Tina"}, {"format", "wav"},
                 {"sample_rate", 24000}}}};

  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "mixed HTTP request did not reach the VLM");
  h.complete(vlm->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "the answer",
          "HTTP image leg did not use the VLM answer as its prompt");
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", image_bytes));
  }
  if (tts_submit) {
    StagePayload p = StagePayload::from_value(tts_submit->data);
    check(p.request.inputs.find("text")->as_str() == "the answer",
          "HTTP tts leg did not receive the aggregated VLM answer");
    Value::Map extra;
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    h.complete(tts_submit->request_id, "tts", true,
               artifact_result("audio", "audio/wav", audio_bytes,
                               std::move(extra)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "mixed SSE returned an HTTP error");
  check(response.body.rfind("data: [DONE]") != std::string::npos &&
            occurrences(response.body, "data: [DONE]") == 1,
        "mixed SSE does not end with exactly one [DONE]");

  std::map<std::string, std::string> reassembled;
  std::map<std::string, std::string> digests;
  std::map<std::string, int64_t> expected_bytes;
  int64_t last_seq = -1;
  size_t terminals = 0;
  size_t artifact_frames = 0;
  for (const std::string& payload : sse_data_payloads(response.body)) {
    const json frame = json::parse(payload);
    if (!frame.contains("unified")) continue;
    const json& unified = frame["unified"];
    const int64_t seq = unified["seq"].get<int64_t>();
    check(seq == last_seq + 1, "HTTP unified seq is not gap-free");
    last_seq = seq;
    const std::string kind = unified["kind"].get<std::string>();
    if (kind == "artifact") {
      ++artifact_frames;
      const json& a = unified["artifact"];
      const std::string id = a["artifact_id"].get<std::string>();
      const std::string data = a["data"].get<std::string>();
      check(data.size() % 4 == 0, "an artifact chunk is not aligned base64");
      reassembled[id] += util::base64_decode(
          data, kArtifactChunkTargetBytes + 16);
      if (a.contains("sha256")) {
        digests[id] = a["sha256"].get<std::string>();
        expected_bytes[id] = a["bytes"].get<int64_t>();
      }
    }
    if (kind == "complete" || kind == "error" || kind == "cancelled") {
      ++terminals;
      check(unified.contains("legs") &&
                unified["legs"].size() == 2,
            "the mixed terminal does not carry two legs");
    }
  }
  check(terminals == 1, "mixed SSE has not exactly one terminal");
  check(artifact_frames == 4, "expected 3 image chunks plus 1 audio chunk");
  check(reassembled.size() == 2, "expected two artifacts");
  for (auto& [id, bytes] : reassembled) {
    check(digests.count(id) == 1 &&
              digests[id] == util::sha256_hex(bytes),
          "an artifact sha256 does not match its reassembled bytes");
    check(expected_bytes[id] == static_cast<int64_t>(bytes.size()),
          "an artifact byte count does not match");
  }
  check(reassembled.begin()->second.size() == image_bytes.size() ||
            reassembled.rbegin()->second.size() == image_bytes.size(),
        "the 5 MiB image was not reassembled intact");
  wait_until([&] { return engine.in_flight() == 0; },
             "mixed HTTP request stayed in flight");
}

// HTTP edge for partial failure: the image artifact is delivered while the tts
// leg fails. The SSE stream must carry leg_error, keep the artifact, and end
// with one complete terminal naming both leg states.
void test_public_http_mixed_partial_failure() {
  Harness h("publicmixedfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "partial-failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image", "audio"})},
      {"image", nlohmann::json::object()},
      {"audio", {{"voice", "Tina"}, {"format", "wav"}}}};
  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  h.complete(vlm->request_id, "vlm", true, text_result("answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  if (image_submit) {
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", std::string(40, '\x90')));
  }
  if (tts_submit) {
    h.complete(tts_submit->request_id, "tts", false, Value(),
               std::string("tts unavailable"));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "partial-failure SSE returned an HTTP error");
  check(response.body.find("\"kind\":\"leg_error\"") != std::string::npos &&
            response.body.find("tts unavailable") != std::string::npos,
        "the tts leg_error did not reach the public SSE stream");
  check(response.body.find("\"kind\":\"artifact\"") != std::string::npos,
        "the successful image artifact was dropped on partial failure");
  check(occurrences(response.body, "\"kind\":\"complete\"") == 1 &&
            occurrences(response.body, "data: [DONE]") == 1,
        "partial failure did not end with one complete and one [DONE]");
  check(response.body.find("\"state\":\"failed\"") != std::string::npos &&
            response.body.find("\"state\":\"completed\"") != std::string::npos,
        "the terminal legs do not name both states");
  wait_until([&] { return engine.in_flight() == 0; },
             "partial-failure request stayed in flight");
}

// One non-streaming chat.completion body carrying text plus every optional
// modality. Each artifact arrives whole: one base64 blob, its decoded byte
// count and its sha256, with no SSE chunk fields.
void test_public_http_nonstream_mixed_all_modalities() {
  Harness h("nonstreammixed");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "non-streaming mixed server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  // 3 MiB forces two chunks for the image; audio and video stay single-chunk.
  const std::string image_bytes(3u << 20, '\x11');
  const std::string audio_bytes(1024, '\x22');
  const std::string video_bytes(2048, '\x33');
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", false},
      {"session_id", "nonstream-sess"},
      {"modalities", nlohmann::json::array({"text", "image", "audio", "video"})},
      {"image", {{"output_format", "png"}, {"size", "1024x1024"}}},
      {"audio", {{"voice", "Tina"}, {"format", "wav"}, {"sample_rate", 24000}}},
      {"video", {{"size", "1280x720"}, {"seconds", 4}, {"fps", 24}}}};

  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "non-streaming mixed request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  auto video_submit = h.next_submit(h.video_recv);
  check(image_submit.has_value() && tts_submit.has_value() &&
            video_submit.has_value(),
        "a non-streaming mixed leg did not start");
  if (image_submit) {
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", image_bytes));
  }
  if (tts_submit) {
    Value::Map extra;
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    h.complete(tts_submit->request_id, "tts", true,
               artifact_result("audio", "audio/wav", audio_bytes,
                               std::move(extra)));
  }
  if (video_submit) {
    Value::Map extra;
    extra["format"] = Value("mp4");
    extra["size"] = Value("1280x720");
    h.complete(video_submit->request_id, "video", true,
               artifact_result("video", "video/mp4", video_bytes,
                               std::move(extra)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200,
        "non-streaming mixed request returned an HTTP error");
  check(response.body.find("data: ") == std::string::npos,
        "non-streaming mixed response is not a single JSON body");

  const json out = json::parse(response.body);
  check(out["object"] == "chat.completion" && out["id"].is_string() &&
            out["created"].is_number_integer() &&
            out["model"] == "qwen3-unified",
        "non-streaming mixed response lost a standard OpenAI field");
  check(out["session_id"] == "nonstream-sess",
        "non-streaming mixed response lost its session_id");
  check(out["choices"][0]["message"]["role"] == "assistant" &&
            out["choices"][0]["message"]["content"] == "the answer",
        "non-streaming mixed response lost the VLM text");

  const std::map<std::string, std::string> expected = {
      {"image", image_bytes}, {"audio", audio_bytes}, {"video", video_bytes}};
  std::map<std::string, std::string> decoded;
  check(out["artifacts"].is_array() && out["artifacts"].size() == 3,
        "expected one whole artifact per output modality");
  for (const json& artifact : out["artifacts"]) {
    const std::string media = artifact["media_type"].get<std::string>();
    check(artifact["encoding"] == "base64" &&
              !artifact.contains("index") && !artifact.contains("total"),
          "a whole artifact kept SSE chunk fields or lost its encoding");
    const std::string bytes =
        util::base64_decode(artifact["data"].get<std::string>(),
                            kArtifactChunkTargetBytes * 4);
    check(artifact["bytes"].get<int64_t>() ==
              static_cast<int64_t>(bytes.size()),
          "artifact byte count does not match its decoded data");
    check(artifact["sha256"] == util::sha256_hex(bytes),
          "artifact sha256 does not match its decoded data");
    check(expected.count(media) == 1 && expected.at(media) == bytes,
          "artifact bytes do not match the stage output");
    decoded[media] = bytes;
  }
  check(decoded.size() == 3, "an output modality produced no artifact");

  std::map<std::string, std::string> leg_states;
  check(out["legs"].is_array() && out["legs"].size() == 3,
        "expected one terminal leg per output modality");
  for (const json& leg : out["legs"]) {
    leg_states[leg["stage"].get<std::string>()] = leg["state"].get<std::string>();
  }
  check(leg_states["image"] == "completed" && leg_states["tts"] == "completed" &&
            leg_states["video"] == "completed",
        "a non-streaming mixed leg did not report completed");
  wait_until([&] { return engine.in_flight() == 0; },
             "non-streaming mixed request stayed in flight");
}

// A failed optional leg keeps HTTP 200: the VLM text and the other artifacts
// stay valid, and the legs array names the failure.
void test_public_http_nonstream_mixed_partial_failure() {
  Harness h("nonstreamfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "non-streaming partial-failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", false},
      {"modalities", nlohmann::json::array({"text", "image", "video"})},
      {"image", nlohmann::json::object()},
      {"video", nlohmann::json::object()}};
  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "partial-failure request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto video_submit = h.next_submit(h.video_recv);
  if (image_submit) {
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", std::string(40, '\x90')));
  }
  if (video_submit) {
    h.complete(video_submit->request_id, "video", false, Value(),
               std::string("video unavailable"));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200,
        "a failed optional leg dropped the non-streaming HTTP 200");
  const json out = json::parse(response.body);
  check(out["choices"][0]["message"]["content"] == "answer",
        "partial failure dropped the VLM text");
  check(out["artifacts"].size() == 1 &&
            out["artifacts"][0]["media_type"] == "image",
        "partial failure dropped the succeeded artifact");
  std::map<std::string, std::string> leg_states;
  for (const json& leg : out["legs"]) {
    leg_states[leg["stage"].get<std::string>()] = leg["state"].get<std::string>();
    if (leg["stage"] == "video") {
      check(leg.contains("error") &&
                leg["error"].get<std::string>() == "video unavailable",
            "the failed leg did not report its error");
    }
  }
  check(leg_states["image"] == "completed" && leg_states["video"] == "failed",
        "the legs array does not name both final states");
  wait_until([&] { return engine.in_flight() == 0; },
             "partial-failure non-streaming request stayed in flight");
}

// Streamed PCM chunks have no place in one JSON body, so audio.stream=true
// still requires stream=true. Unregistered output stages still answer 400.
void test_public_http_nonstream_mixed_rejections() {
  Harness h("nonstreamreject");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "non-streaming rejection server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  auto post = [&](const nlohmann::json& payload) {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = payload.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what());
    }
    return std::string();
  };
  auto base = [](nlohmann::json extra) {
    extra["model"] = "qwen3-unified";
    extra["stream"] = false;
    extra["messages"] = nlohmann::json::array(
        {nlohmann::json{{"role", "user"}, {"content", "hi"}}});
    return extra;
  };

  std::string error = post(base({{"modalities", nlohmann::json::array({"text", "audio"})},
                                 {"audio", {{"voice", "Tina"}, {"format", "wav"},
                                            {"stream", true}}}}));
  check(error.find("HTTP 400") != std::string::npos &&
            error.find("audio.stream=true requires stream=true") !=
                std::string::npos,
        "non-streaming audio.stream=true was not rejected with 400");

  api.stop();
  server_thread.join();

  // An engine with no optional output stage registered rejects the modality.
  Harness h2("nonstreamnostage");
  ApiConfig config2;
  config2.host = "127.0.0.1";
  config2.port = 0;
  config2.model = "qwen3-unified";
  config2.request_timeout_s = 5.0;
  OpenAIApi api2(h2.coordinator, config2);
  UnifiedEngine engine2(h2.coordinator, UnifiedConfig{});
  api2.set_unified_engine(engine2, "vlm");
  check(api2.bind(), "no-stage non-streaming server did not bind");
  h2.start_pump();
  std::thread server2([&] { api2.listen(); });

  auto post2 = [&](const nlohmann::json& payload) {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api2.bound_port()) +
                  "/v1/chat/completions";
    request.body = payload.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what());
    }
    return std::string();
  };
  for (const auto& [key, value, message] :
       std::vector<std::tuple<std::string, nlohmann::json, std::string>>{
           {"image", nlohmann::json::object(), "image output is not enabled on this server"},
           {"audio", nlohmann::json{{"voice", "Tina"}, {"format", "wav"}}, "audio output is not enabled on this server"},
           {"video", nlohmann::json::object(), "video output is not enabled on this server"}}) {
    std::string rejected =
        post2(base({{"modalities", nlohmann::json::array({"text", key})},
                    {key, value}}));
    const std::string detail =
        "an unregistered " + key + " stage was not rejected with 400";
    check(rejected.find("HTTP 400") != std::string::npos &&
              rejected.find(message) != std::string::npos,
          detail.c_str());
  }
  api2.stop();
  server2.join();
}

// A text-only non-streaming request keeps the exact previous JSON shape: no
// artifacts, no legs, no session_id unless the caller supplied one.
void test_public_http_nonstream_text_only_compatible() {
  Harness h("nonstreamtext");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "text-only non-streaming server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = nlohmann::json{
                       {"model", "qwen3-unified"},
                       {"messages", nlohmann::json::array(
                                        {nlohmann::json{{"role", "user"},
                                                        {"content", "hi"}}})},
                       {"stream", false}}
                       .dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  });
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "text-only non-streaming request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("plain answer"));
  const auto response = response_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "text-only non-streaming request failed");
  const json out = json::parse(response.body);
  check(out["choices"][0]["message"]["content"] == "plain answer",
        "text-only non-streaming text changed");
  check(!out.contains("artifacts") && !out.contains("legs") &&
            !out.contains("session_id"),
        "the text-only non-streaming response gained a mixed-output field");
  wait_until([&] { return engine.in_flight() == 0; },
             "text-only non-streaming request stayed in flight");
}

// top_p and seed pass through the public HTTP boundary into the VLM leg's
// params. The knobs are accepted only in (0, 1] and as signed integers, they
// stay absent when omitted or null, and the ASR leg never sees them.
void test_public_http_top_p_seed_passthrough() {
  Harness h("toppseed");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "top_p/seed server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url =
      "http://127.0.0.1:" + std::to_string(api.bound_port()) +
      "/v1/chat/completions";

  auto make_request = [&](const nlohmann::json& body) {
    test_http::HttpRequest request;
    request.url = url;
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    return request;
  };
  auto text_body = [](bool stream) {
    return nlohmann::json{
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"},
                                         {"content", "hello"}}})},
        {"stream", stream}};
  };
  auto has_param = [](const StagePayload& payload, const char* key) {
    return payload.request.params.find(key) != payload.request.params.end();
  };

  // Streaming: top_p 0.9 and seed 42 reach the VLM leg.
  {
    nlohmann::json body = text_body(true);
    body["top_p"] = 0.9;
    body["seed"] = 42;
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "streaming top_p/seed request did not reach the VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(has_param(payload, "top_p") &&
                payload.request.params.at("top_p").is_float() &&
                payload.request.params.at("top_p").as_double() == 0.9 &&
                has_param(payload, "seed") &&
                payload.request.params.at("seed").is_int() &&
                payload.request.params.at("seed").as_int() == 42,
            "streaming top_p/seed did not reach the VLM params");
      h.stream(vlm->request_id, "vlm", text_chunk("hi", false));
      auto result = text_result("hi");
      result["finish_reason"] = Value("stop");
      h.complete(vlm->request_id, "vlm", true, Value(std::move(result)));
    }
    const auto response = future.get();
    check(response.status == 200,
          "streaming top_p/seed request returned an HTTP error");
  }

  // Non-streaming: top_p 1.0 and seed 0 reach the VLM leg.
  {
    nlohmann::json body = text_body(false);
    body["top_p"] = 1.0;
    body["seed"] = 0;
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(),
          "non-streaming top_p/seed request did not reach the VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(has_param(payload, "top_p") &&
                payload.request.params.at("top_p").as_double() == 1.0 &&
                has_param(payload, "seed") &&
                payload.request.params.at("seed").as_int() == 0,
            "non-streaming top_p/seed did not reach the VLM params");
      h.complete(vlm->request_id, "vlm", true, text_result("plain answer"));
    }
    const auto response = future.get();
    check(response.status == 200 &&
              response.body.find("plain answer") != std::string::npos,
          "non-streaming top_p/seed request returned the wrong response");
  }

  // A negative seed is a valid signed int64 and reaches the VLM leg.
  {
    nlohmann::json body = text_body(false);
    body["seed"] = -7;
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "negative seed request did not reach the VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(has_param(payload, "seed") &&
                payload.request.params.at("seed").as_int() == -7,
            "a negative seed was not forwarded");
      h.complete(vlm->request_id, "vlm", true, text_result("neg"));
    }
    const auto response = future.get();
    check(response.status == 200, "a negative seed request was rejected");
  }

  // Omitted and null knobs stay absent from the VLM params.
  for (bool null_knobs : {false, true}) {
    nlohmann::json body = text_body(false);
    if (null_knobs) {
      body["top_p"] = nullptr;
      body["seed"] = nullptr;
    }
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "absent-knob request did not reach the VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(!has_param(payload, "top_p") && !has_param(payload, "seed"),
            "an omitted or null knob was added to the VLM params");
      h.complete(vlm->request_id, "vlm", true, text_result("compat"));
    }
    const auto response = future.get();
    check(response.status == 200,
          "an omitted or null knob request was rejected");
  }

  // Audio: the ASR leg is pinned and never sees top_p/seed; the VLM leg does.
  {
    nlohmann::json body = public_audio_body();
    body["top_p"] = 0.5;
    body["seed"] = 7;
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto asr = h.next_submit(h.asr_recv);
    check(asr.has_value(), "audio top_p/seed request did not reach the ASR leg");
    if (asr) {
      StagePayload payload = StagePayload::from_value(asr->data);
      check(payload.request.params.at("temperature").as_double() == 0.0 &&
                !has_param(payload, "top_p") && !has_param(payload, "seed"),
            "the ASR leg received VLM sampling knobs");
      h.complete(asr->request_id, "asr", true, text_result("spoken"));
    }
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "audio top_p/seed request did not reach the VLM leg");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(has_param(payload, "top_p") &&
                payload.request.params.at("top_p").as_double() == 0.5 &&
                has_param(payload, "seed") &&
                payload.request.params.at("seed").as_int() == 7,
            "the VLM leg lost top_p/seed on the audio path");
      auto result = text_result("answer");
      result["finish_reason"] = Value("stop");
      h.complete(vlm->request_id, "vlm", true, Value(std::move(result)));
    }
    const auto response = future.get();
    check(response.status == 200,
          "audio top_p/seed request returned an HTTP error");
  }

  // Invalid knobs are rejected at the HTTP boundary before any stage submits.
  auto rejected_with_400 = [&](nlohmann::json body) {
    try {
      test_http::perform(make_request(body));
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  const std::vector<std::pair<const char*, nlohmann::json>> bad_top_p = {
      {"top_p", 0},   {"top_p", -0.1}, {"top_p", 1.01},
      {"top_p", "0.9"}, {"top_p", true}};
  for (const auto& [key, value] : bad_top_p) {
    nlohmann::json body = text_body(false);
    body[key] = value;
    check(rejected_with_400(body), "an invalid top_p was accepted");
    check(h.no_submit(h.vlm_recv), "an invalid top_p reached the VLM stage");
  }
  const std::vector<std::pair<const char*, nlohmann::json>> bad_seed = {
      {"seed", "42"},
      {"seed", 1.5},
      {"seed", true},
      {"seed", nlohmann::json::parse("99999999999999999999")},
      {"seed", std::numeric_limits<uint64_t>::max()}};
  for (const auto& [key, value] : bad_seed) {
    nlohmann::json body = text_body(false);
    body[key] = value;
    check(rejected_with_400(body), "an invalid seed was accepted");
    check(h.no_submit(h.vlm_recv), "an invalid seed reached the VLM stage");
  }

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "top_p/seed requests stayed in flight");
}

// Non-unified mode forwards top_p/seed to the single entry stage and applies
// the same range checks.
void test_public_http_nonunified_top_p_seed_passthrough() {
  Harness h("nonuniftopp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "legacy-chat";
  config.entry_stage = "vlm";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  check(api.bind(), "non-unified top_p/seed server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url =
      "http://127.0.0.1:" + std::to_string(api.bound_port()) +
      "/v1/chat/completions";

  auto make_request = [&](const nlohmann::json& body) {
    test_http::HttpRequest request;
    request.url = url;
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    return request;
  };
  auto text_body = [](bool stream) {
    return nlohmann::json{
        {"model", "legacy-chat"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"},
                                         {"content", "hello"}}})},
        {"stream", stream}};
  };

  for (bool stream : {false, true}) {
    nlohmann::json body = text_body(stream);
    body["top_p"] = 0.9;
    body["seed"] = 42;
    auto future = std::async(std::launch::async,
                             [&] { return test_http::perform(make_request(body)); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(),
          "non-unified top_p/seed request did not reach the entry stage");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      const Value::Map& params = payload.request.params;
      check(params.find("top_p") != params.end() &&
                params.at("top_p").as_double() == 0.9 &&
                params.find("seed") != params.end() &&
                params.at("seed").as_int() == 42,
            "non-unified mode did not forward top_p/seed");
      auto result = text_result("entry answer");
      result["finish_reason"] = Value("stop");
      if (stream) h.stream(vlm->request_id, "vlm", text_chunk("entry", false));
      h.complete(vlm->request_id, "vlm", true, Value(std::move(result)));
    }
    const auto response = future.get();
    check(response.status == 200,
          "non-unified top_p/seed request returned an HTTP error");
  }

  for (const auto& [key, value] :
       std::vector<std::pair<const char*, nlohmann::json>>{
           {"top_p", 0}, {"top_p", 1.5}, {"seed", 1.5}, {"seed", "42"}}) {
    nlohmann::json body = text_body(false);
    body[key] = value;
    bool rejected = false;
    try {
      test_http::perform(make_request(body));
    } catch (const client::Error& exc) {
      rejected = std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    check(rejected, "an invalid knob was accepted in non-unified mode");
    check(h.no_submit(h.vlm_recv),
          "an invalid non-unified request reached the entry stage");
  }

  api.stop();
  server_thread.join();
}

// The C++ client decodes and verifies a non-streaming mixed body end to end.
void test_openai_client_nonstream_mixed_roundtrip() {
  Harness h("clientnonstream");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "client non-streaming mixed server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  client::OpenAIConfig cfg;
  cfg.base_url = "http://127.0.0.1:" + std::to_string(api.bound_port()) + "/v1";
  cfg.model = "qwen3-unified";
  cfg.max_attempts = 1;
  client::OpenAIClient client(cfg);

  const std::string image_bytes(256 * 1024, '\x41');
  const std::string audio_bytes(4096, '\x42');
  client::ChatRequest req;
  req.messages = {client::Message{"user", "draw and speak", {}}};
  req.modalities = {"text", "image", "audio"};
  req.stream = false;
  req.image = client::ImageOutput{};
  client::AudioOutput voice;
  voice.voice = "Tina";
  req.audio = voice;

  auto resp_future = std::async(std::launch::async, [&] {
    try {
      return std::pair<std::optional<client::ChatResponse>, std::string>{
          client.chat(req), ""};
    } catch (const std::exception& err) {
      return std::pair<std::optional<client::ChatResponse>, std::string>{
          std::nullopt, err.what()};
    }
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "client non-streaming request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("mixed answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  check(image_submit.has_value() && tts_submit.has_value(),
        "a client non-streaming leg did not start");
  if (image_submit) {
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", image_bytes));
  }
  if (tts_submit) {
    h.complete(tts_submit->request_id, "tts", true,
               artifact_result("audio", "audio/wav", audio_bytes));
  }

  auto [response, error] = resp_future.get();
  api.stop();
  server_thread.join();
  const std::string failure = "OpenAIClient rejected a non-streaming mixed response: " + error;
  check(error.empty() && response.has_value(), failure.c_str());
  if (response) {
    check(response->text == "mixed answer",
          "OpenAIClient lost the non-streaming mixed text");
    std::map<std::string, std::string> decoded;
    for (const client::Artifact& artifact : response->artifacts) {
      decoded[artifact.media_type] = artifact.data;
      check(artifact.sha256 == util::sha256_hex(artifact.data),
            "OpenAIClient kept an unverified artifact digest");
    }
    check(decoded.size() == 2 && decoded["image"] == image_bytes &&
              decoded["audio"] == audio_bytes,
          "OpenAIClient did not decode the non-streaming artifacts");
    std::map<std::string, std::string> leg_states;
    for (const json& leg : response->legs) {
      leg_states[leg["stage"].get<std::string>()] =
          leg["state"].get<std::string>();
    }
    check(leg_states["image"] == "completed" && leg_states["tts"] == "completed",
          "OpenAIClient lost the non-streaming legs array");
  }
  wait_until([&] { return engine.in_flight() == 0; },
             "client non-streaming mixed request stayed in flight");
}

UnifiedConfig realtime_history_config(const Harness& h);

void run_realtime_spoken_audio_format_and_leg_error_subtest(
    Harness& h, UnifiedEngine& engine, RealtimeApi& realtime) {
  std::atomic<bool> cancel{false};
  const std::string ws_url =
      "ws://127.0.0.1:" + std::to_string(realtime.bound_port()) +
      "/v1/realtime?intent=transcription&model=qwen3-unified";
  test_http::WebSocket ws(ws_url, client::Milliseconds{5000}, &cancel);

  std::mutex ws_mu;
  std::condition_variable ws_cv;
  std::deque<json> ws_events;
  std::thread ws_reader([&] {
    try {
      ws.run([&](std::string_view msg) {
        json parsed = json::parse(msg, nullptr, false);
        if (parsed.is_discarded()) return;
        {
          std::lock_guard<std::mutex> lock(ws_mu);
          ws_events.push_back(std::move(parsed));
        }
        ws_cv.notify_all();
      });
    } catch (...) {
    }
  });

  auto next_ws_event = [&](double timeout_s = 5.0) -> std::optional<json> {
    std::unique_lock<std::mutex> lock(ws_mu);
    if (!ws_cv.wait_for(lock, std::chrono::duration<double>(timeout_s),
                        [&] { return !ws_events.empty(); })) {
      return std::nullopt;
    }
    json ev = std::move(ws_events.front());
    ws_events.pop_front();
    return ev;
  };

  ws.send_text(json{{"type", "session.update"},
                    {"session",
                     {{"input_audio_format", "pcm16"},
                      {"turn_detection",
                       {{"type", "server_vad"}, {"silence_ms", 100}}},
                      {"audio_output", {{"voice", "vivian"}}}}}}
                   .dump());
  auto updated = next_ws_event();
  if (updated.has_value() &&
      (*updated).value("type", "") == "transcription_session.created") {
    updated = next_ws_event();
  }
  check(updated.has_value() &&
            (*updated)["type"] == "transcription_session.updated",
        "realtime session.update did not echo transcription_session.updated");
  check(updated.has_value() &&
            !(*updated)["session"]["turn_detection"].contains("echo_suppression"),
        "default echo protection changed the session.updated wire shape");

  // 120 ms voiced 440 Hz PCM16 at 16 kHz (3840 bytes, < 6400 preview window)
  // followed by 150 ms digital silence (4800 bytes) to commit one VAD turn.
  std::string voiced(3840, '\0');
  for (int i = 0; i < 1920; ++i) {
    const auto sample = static_cast<int16_t>(
        6000.0 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * i / 16000.0));
    voiced[static_cast<size_t>(i) * 2] = static_cast<char>(sample & 0xff);
    voiced[static_cast<size_t>(i) * 2 + 1] =
        static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xff);
  }
  const std::string quiet(4800, '\0');

  auto send_utterance = [&] {
    ws.send_text(json{{"type", "input_audio_buffer.append"},
                      {"audio", util::base64_encode(voiced)}}
                     .dump());
    ws.send_text(json{{"type", "input_audio_buffer.append"},
                      {"audio", util::base64_encode(quiet)}}
                     .dump());
    while (auto req = h.next_submit(h.asr_recv)) {
      if (req->request_id.rfind("realtime-preview-", 0) == 0) {
        h.complete(req->request_id, "asr", true, text_result("he"));
        continue;
      }
      h.complete(req->request_id, "asr", true, text_result("hello"));
      break;
    }
    auto vlm_req = h.next_submit(h.vlm_recv);
    check(vlm_req.has_value(), "realtime spoken turn did not submit to VLM");
    if (vlm_req) {
      h.complete(vlm_req->request_id, "vlm", true, text_result("world"));
    }
    return h.next_submit(h.tts_recv);
  };

  // Turn 0: non-streaming TTS stage returns a RIFF WAV without streamed=true.
  // RealtimeApi must unwrap the RIFF header and emit genuine headerless PCM16.
  if (auto tts_req = send_utterance()) {
    const std::string raw_pcm = "\x11\x22\x33\x44\x55\x66\x77\x08";
    const std::string wav = audio::wav_from_pcm16(raw_pcm, 24000);
    Value::Map extra;
    extra["format"] = Value("wav");
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    h.complete(tts_req->request_id, "tts", true,
               artifact_result("audio", "audio/wav", wav, std::move(extra)));

    std::string collected_pcm;
    bool saw_audio_done = false;
    bool saw_completed = false;
    while (auto ev = next_ws_event()) {
      const std::string type = ev->value("type", "");
      if (type == "audio.delta") {
        check(ev->value("format", "") == "pcm16",
              "fallback WAV turn did not label audio.delta as pcm16");
        check(ev->value("sample_rate", 0) == 24000 &&
                  ev->value("channels", 0) == 1,
              "fallback WAV turn did not report WAV sample_rate/channels");
        collected_pcm +=
            util::base64_decode(ev->at("audio").get<std::string>(), 1u << 20);
      } else if (type == "audio.done") {
        saw_audio_done = true;
      } else if (type == "response.completed") {
        saw_completed = true;
        break;
      } else if (type == "response.failed") {
        check(false, "fallback WAV turn unexpectedly failed");
        break;
      }
    }
    check(collected_pcm == raw_pcm && collected_pcm.rfind("RIFF", 0) != 0,
          "fallback WAV turn leaked RIFF header bytes into pcm16 audio.delta");
    check(saw_audio_done && saw_completed,
          "fallback WAV turn did not emit audio.done and response.completed");
  } else {
    check(false, "turn 0 did not submit to TTS");
  }

  // Turn 1: streaming TTS stage emits PCM16 chunks and sets streamed=true on
  // its terminal WAV. RealtimeApi must emit the streamed chunks once and skip
  // the terminal WAV.
  if (auto tts_req = send_utterance()) {
    const std::string c0("\x01\x00\x02\x00", 4);
    const std::string c1("\x03\x00\x04\x00", 4);
    auto pcm_chunk = [](const std::string& bytes, int64_t idx) {
      Value::Map m;
      m["audio"] = Value::bin(bytes);
      m["index"] = Value(idx);
      m["sample_rate"] = Value(int64_t{24000});
      m["channels"] = Value(int64_t{1});
      m["format"] = Value("pcm16");
      return Value(std::move(m));
    };
    h.stream(tts_req->request_id, "tts", pcm_chunk(c0, 0));
    h.stream(tts_req->request_id, "tts", pcm_chunk(c1, 1));
    Value::Map extra;
    extra["format"] = Value("wav");
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    extra["streamed"] = Value(true);
    h.complete(
        tts_req->request_id, "tts", true,
        artifact_result("audio", "audio/wav",
                        audio::wav_from_pcm16(c0 + c1, 24000),
                        std::move(extra)));

    std::string collected_pcm;
    bool saw_completed = false;
    while (auto ev = next_ws_event()) {
      const std::string type = ev->value("type", "");
      if (type == "audio.delta") {
        collected_pcm +=
            util::base64_decode(ev->at("audio").get<std::string>(), 1u << 20);
      } else if (type == "response.completed") {
        saw_completed = true;
        break;
      }
    }
    check(saw_completed && collected_pcm == c0 + c1,
          "streamed TTS turn duplicated or altered PCM16 audio");
  } else {
    check(false, "turn 1 did not submit to TTS");
  }

  // Turn 2: TTS leg fails after VLM succeeds (kLegError). RealtimeApi must
  // report response.failed instead of claiming audio.done/response.completed.
  if (auto tts_req = send_utterance()) {
    h.complete(tts_req->request_id, "tts", false, Value(),
               std::string("tts worker crashed"));
    bool saw_audio_done = false;
    bool saw_completed = false;
    std::string failed_error;
    while (auto ev = next_ws_event()) {
      const std::string type = ev->value("type", "");
      if (type == "audio.done") {
        saw_audio_done = true;
      } else if (type == "response.completed") {
        saw_completed = true;
        break;
      } else if (type == "response.failed") {
        failed_error = ev->value("error", "");
        break;
      }
    }
    check(!saw_audio_done && !saw_completed &&
              failed_error == "tts worker crashed",
          "TTS leg failure did not surface as realtime response.failed");
  } else {
    check(false, "turn 2 did not submit to TTS");
  }

  // Turn 3: actual audio.delta emission seeds this session's far-end ring.
  // Echoed assistant audio alone must not cancel the response; adding an
  // independent near-end signal during playback must cancel it immediately.
  if (auto tts_req = send_utterance()) {
    constexpr size_t kFrames = 50;
    constexpr size_t kSamples = kFrames * 160;
    auto bytes_of = [](const std::vector<int16_t>& samples) {
      std::string bytes(samples.size() * 2, '\0');
      for (size_t i = 0; i < samples.size(); ++i) {
        const auto value = static_cast<uint16_t>(samples[i]);
        bytes[i * 2] = static_cast<char>(value & 0xff);
        bytes[i * 2 + 1] = static_cast<char>((value >> 8) & 0xff);
      }
      return bytes;
    };
    auto append_pcm = [&](const std::string& pcm) {
      ws.send_text(json{{"type", "input_audio_buffer.append"},
                        {"audio", util::base64_encode(pcm)}}
                       .dump());
    };
    auto send_reference = [&](const std::vector<int16_t>& pcm, int64_t index) {
      Value::Map chunk;
      chunk["audio"] = Value::bin(bytes_of(pcm));
      chunk["index"] = Value(index);
      chunk["sample_rate"] = Value(int64_t{16000});
      chunk["channels"] = Value(int64_t{1});
      chunk["format"] = Value("pcm16");
      h.stream(tts_req->request_id, "tts", Value(std::move(chunk)));
      bool delivered = false;
      while (auto event = next_ws_event()) {
        if (event->value("type", "") == "audio.delta") {
          delivered = true;
          break;
        }
      }
      check(delivered, "synthetic far-end PCM was not emitted as audio.delta");
    };

    std::vector<int16_t> reference(kSamples);
    std::vector<int16_t> echo(kSamples, 0);
    constexpr size_t kDelay = 800;  // 50 ms, deliberately not frame aligned.
    for (size_t i = 0; i < kSamples; ++i) {
      reference[i] = static_cast<int16_t>(
          7000.0 * std::sin(2.0 * 3.14159265358979323846 * 397.0 * i / 16000.0) +
          4000.0 * std::sin(2.0 * 3.14159265358979323846 * 1153.0 * i / 16000.0));
      if (i >= kDelay) echo[i] = static_cast<int16_t>(0.6 * reference[i - kDelay]);
    }
    send_reference(reference, 0);
    const std::string echo_bytes = bytes_of(echo);
    for (size_t frame = 0; frame < kFrames; ++frame) {
      append_pcm(echo_bytes.substr(frame * 320, 320));
    }
    bool false_barge_in = false;
    while (auto event = next_ws_event(0.25)) {
      const std::string type = event->value("type", "");
      if (type == "input_audio_buffer.speech_started" ||
          type == "response.cancelled") {
        false_barge_in = true;
      }
    }
    check(!false_barge_in,
          "wire-level assistant echo falsely interrupted the active response");

    std::vector<int16_t> double_talk(kSamples);
    for (size_t i = 0; i < kSamples; ++i) {
      const int16_t near = static_cast<int16_t>(
          6500.0 * std::sin(2.0 * 3.14159265358979323846 * 733.0 * i / 16000.0) +
          3500.0 * std::sin(2.0 * 3.14159265358979323846 * 1297.0 * i / 16000.0));
      double_talk[i] = static_cast<int16_t>(std::clamp(
          static_cast<int>(0.6 * reference[i] + near), -32768, 32767));
    }
    send_reference(reference, 1);
    const std::string double_bytes = bytes_of(double_talk);
    append_pcm(double_bytes.substr(0, 320));
    bool saw_speech_start = false;
    bool saw_response_cancel = false;
    for (int i = 0; i < 16 && !(saw_speech_start && saw_response_cancel); ++i) {
      auto event = next_ws_event();
      if (!event) break;
      saw_speech_start |=
          event->value("type", "") == "input_audio_buffer.speech_started";
      saw_response_cancel |= event->value("type", "") == "response.cancelled";
    }
    check(saw_speech_start && saw_response_cancel,
          "wire-level double-talk did not promptly cancel the response");
    append_pcm(quiet);
    bool drained_final = false;
    while (auto req = h.next_submit(h.asr_recv)) {
      if (req->request_id.rfind("realtime-preview-", 0) == 0) {
        h.complete(req->request_id, "asr", true, text_result(""));
        continue;
      }
      h.complete(req->request_id, "asr", true, text_result(""));
      drained_final = true;
      break;
    }
    check(drained_final,
          "double-talk cleanup did not drain its preview and final ASR requests");
    std::cout << "REALTIME_ECHO_WIRE echo_only_false_barge_ins=0/50 "
                 "double_talk_cancel=PASS\n";
  } else {
    check(false, "turn 3 echo-robustness turn did not submit to TTS");
  }

  // Turn 4: the real-model A/B harness toggles only echo_suppression while an
  // acknowledged response remains active. Prove that this narrow update keeps
  // the response alive, reports the effective baseline, and replays exactly the
  // same PCM into the old energy-VAD behavior. Default suppression stayed on
  // for the first half and no other active-session setting is changed.
  if (auto tts_req = send_utterance()) {
    auto bytes_of = [](const std::vector<int16_t>& samples) {
      std::string bytes(samples.size() * 2, '\0');
      for (size_t i = 0; i < samples.size(); ++i) {
        const auto value = static_cast<uint16_t>(samples[i]);
        bytes[i * 2] = static_cast<char>(value & 0xff);
        bytes[i * 2 + 1] = static_cast<char>((value >> 8) & 0xff);
      }
      return bytes;
    };
    auto append_pcm = [&](const std::string& pcm) {
      ws.send_text(json{{"type", "input_audio_buffer.append"},
                        {"audio", util::base64_encode(pcm)}}
                       .dump());
    };
    constexpr size_t kFrames = 50;
    constexpr size_t kDelay = 800;
    std::vector<int16_t> reference(kFrames * 160);
    std::vector<int16_t> echo(kFrames * 160, 0);
    for (size_t i = 0; i < reference.size(); ++i) {
      reference[i] = static_cast<int16_t>(
          7000.0 * std::sin(2.0 * 3.14159265358979323846 * 397.0 * i / 16000.0) +
          4000.0 * std::sin(2.0 * 3.14159265358979323846 * 1153.0 * i / 16000.0));
      if (i >= kDelay) echo[i] = static_cast<int16_t>(0.6 * reference[i - kDelay]);
    }
    Value::Map chunk;
    chunk["audio"] = Value::bin(bytes_of(reference));
    chunk["index"] = Value(int64_t{0});
    chunk["sample_rate"] = Value(int64_t{16000});
    chunk["channels"] = Value(int64_t{1});
    chunk["format"] = Value("pcm16");
    h.stream(tts_req->request_id, "tts", Value(std::move(chunk)));
    bool delivered = false;
    while (auto event = next_ws_event()) {
      if (event->value("type", "") == "audio.delta") {
        delivered = true;
        break;
      }
    }
    check(delivered, "A/B real-response fixture did not emit audio.delta");
    const std::string echo_bytes = bytes_of(echo);
    for (size_t frame = 0; frame < kFrames; ++frame) {
      append_pcm(echo_bytes.substr(frame * 320, 320));
    }
    bool protected_cutin = false;
    while (auto event = next_ws_event(0.25)) {
      const std::string type = event->value("type", "");
      protected_cutin |= type == "input_audio_buffer.speech_started" ||
                         type == "response.cancelled";
    }
    check(!protected_cutin, "protected A/B echo cut into its own response");

    ws.send_text(json{{"type", "session.update"},
                      {"session",
                       {{"turn_detection",
                         {{"type", "server_vad"}, {"silence_ms", 100},
                          {"echo_suppression", false}}}}}}
                     .dump());
    std::optional<json> baseline_update;
    while (auto event = next_ws_event()) {
      if (event->value("type", "") == "transcription_session.updated") {
        baseline_update = *event;
        break;
      }
    }
    check(baseline_update.has_value() &&
              (*baseline_update)["session"]["turn_detection"].value(
                  "echo_suppression", true) == false,
          "active echo-only A/B update was not applied and reported");

    for (size_t frame = 0; frame < kFrames; ++frame) {
      append_pcm(echo_bytes.substr(frame * 320, 320));
    }
    bool baseline_start = false;
    bool baseline_cancel = false;
    for (int i = 0; i < 16 && !(baseline_start && baseline_cancel); ++i) {
      auto event = next_ws_event();
      if (!event) break;
      baseline_start |=
          event->value("type", "") == "input_audio_buffer.speech_started";
      baseline_cancel |= event->value("type", "") == "response.cancelled";
    }
    check(baseline_start && baseline_cancel,
          "identical PCM did not exercise the echo-disabled A/B baseline");
    append_pcm(quiet);
    bool drained_final = false;
    while (auto req = h.next_submit(h.asr_recv)) {
      h.complete(req->request_id, "asr", true, text_result(""));
      if (req->request_id.rfind("realtime-preview-", 0) != 0) {
        drained_final = true;
        break;
      }
    }
    check(drained_final, "echo A/B baseline did not endpoint and drain ASR");
    std::cout << "REALTIME_ECHO_ACTIVE_AB same_pcm=1 protected_cutins=0 "
                 "baseline_cutins=1 default_unchanged=1\n";
  } else {
    check(false, "turn 4 active echo A/B turn did not submit to TTS");
  }

  wait_until([&] { return engine.in_flight() == 0; },
             "realtime spoken requests stayed in flight");
  cancel.store(true, std::memory_order_relaxed);
  ws.close();
  if (ws_reader.joinable()) ws_reader.join();
}

void run_realtime_barge_in_playback_ack_wire_subtest(
    Harness& h, UnifiedEngine& engine, RealtimeApi& realtime) {

  struct Client {
    std::atomic<bool> cancel{false};
    test_http::WebSocket ws;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> events;
    std::vector<json> observed;
    std::thread reader;

    explicit Client(const std::string& url)
        : ws(url, client::Milliseconds{5000}, &cancel),
          reader([this] {
            try {
              ws.run([this](std::string_view message) {
                json event = json::parse(message, nullptr, false);
                if (event.is_discarded()) return;
                {
                  std::lock_guard<std::mutex> lock(mu);
                  events.push_back(std::move(event));
                }
                cv.notify_all();
              });
            } catch (...) {
            }
          }) {}
    ~Client() {
      cancel.store(true, std::memory_order_relaxed);
      ws.close();
      if (reader.joinable()) reader.join();
    }
    void send(const json& event) { ws.send_text(event.dump()); }
    std::optional<json> next() {
      std::unique_lock<std::mutex> lock(mu);
      if (!cv.wait_for(lock, std::chrono::seconds(5),
                       [&] { return !events.empty(); })) {
        return std::nullopt;
      }
      json event = std::move(events.front());
      events.pop_front();
      lock.unlock();
      observed.push_back(event);
      return event;
    }
  };

  auto wait_event = [](Client& client, const std::string& type,
                       const std::function<bool(const json&)>& predicate =
                           [](const json&) { return true; }) {
    for (int i = 0; i < 128; ++i) {
      auto event = client.next();
      if (!event) return std::optional<json>{};
      if (event->value("type", "") == type && predicate(*event)) return event;
    }
    return std::optional<json>{};
  };

  const std::string ws_url =
      "ws://127.0.0.1:" + std::to_string(realtime.bound_port()) +
      "/v1/realtime?intent=transcription&model=qwen3-unified";
  std::string voiced(3840, '\0');
  for (int i = 0; i < 1920; ++i) {
    const auto sample = static_cast<int16_t>(
        6000.0 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * i / 16000.0));
    voiced[static_cast<size_t>(i) * 2] = static_cast<char>(sample & 0xff);
    voiced[static_cast<size_t>(i) * 2 + 1] =
        static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xff);
  }
  const std::string quiet(4800, '\0');
  auto append = [](Client& client, const std::string& pcm) {
    client.send({{"type", "input_audio_buffer.append"},
                 {"audio", util::base64_encode(pcm)}});
  };
  auto configure = [&](Client& client) {
    client.send({{"type", "session.update"},
                 {"session",
                  {{"input_audio_format", "pcm16"},
                   {"turn_detection",
                    {{"type", "server_vad"}, {"silence_ms", 100}}},
                   {"audio_output",
                    {{"voice", "vivian"}, {"playback_ack", true}}}}}});
    auto updated = wait_event(client, "transcription_session.updated");
    check(updated.has_value() &&
              (*updated)["session"]["audio_output"]["playback_ack"] == true,
          "realtime playback acknowledgement mode was not negotiated");
  };
  auto start_turn = [&](Client& client, const std::string& transcript)
      -> std::optional<SubmitMessage> {
    append(client, voiced);
    append(client, quiet);
    auto asr = h.next_submit(h.asr_recv);
    check(asr.has_value(), "barge-in turn did not submit final ASR");
    if (asr) h.complete(asr->request_id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "barge-in turn did not submit VLM");
    return vlm;
  };
  auto pcm_chunk = [](const std::string& bytes, int64_t index) {
    Value::Map chunk;
    chunk["audio"] = Value::bin(bytes);
    chunk["index"] = Value(index);
    chunk["sample_rate"] = Value(int64_t{24000});
    chunk["channels"] = Value(int64_t{1});
    chunk["format"] = Value("pcm16");
    return Value(std::move(chunk));
  };
  auto ack = [](Client& client, const std::string& response_id, size_t cursor) {
    client.send({{"type", "output_audio_buffer.playback_ack"},
                 {"response_id", response_id},
                 {"text_cursor", cursor}});
  };

  {
    Client a(ws_url);
    configure(a);

    // Partial TTS: the second sentence's audio carries the end of the first
    // sentence as its safe acknowledgement cursor. The generated suffix is
    // deliberately longer than anything the client acknowledged.
    auto first_vlm = start_turn(a, "first question");
    StagePayload first_payload;
    if (first_vlm) first_payload = StagePayload::from_value(first_vlm->data);
    check(prompt_roles(first_payload).size() == 1,
          "the first realtime prompt unexpectedly contained history");
    const std::string first_id = first_vlm ? first_vlm->request_id : "";
    const std::string answer =
        "First sentence. Second sentence. unplayed suffix";
    const std::string acknowledged = "First sentence.";
    h.stream(first_id, "vlm", text_chunk(answer, false));
    auto tts0 = h.next_submit(h.tts_recv);
    auto tts1 = h.next_submit(h.tts_recv);
    check(tts0.has_value() && tts1.has_value(),
          "partial TTS did not create the first two sentence legs");
    if (tts0) {
      h.stream(tts0->request_id, "tts",
               pcm_chunk(std::string("\x01\x00\x02\x00", 4), 0));
    }
    if (tts1) {
      h.stream(tts1->request_id, "tts",
               pcm_chunk(std::string("\x03\x00\x04\x00", 4), 0));
    }
    auto audio = wait_event(a, "audio.delta", [&](const json& event) {
      return event.value("response_id", "") == first_id &&
             event.value("text_cursor", size_t{0}) == acknowledged.size();
    });
    check(audio.has_value() &&
              audio->value("text_end_cursor", size_t{0}) >
                  acknowledged.size(),
          "audio.delta did not carry a conservative playback text cursor");
    ack(a, first_id, acknowledged.size());
    auto acked = wait_event(a, "response.playback_ack", [&](const json& event) {
      return event.value("response_id", "") == first_id &&
             event.value("text_cursor", size_t{0}) == acknowledged.size() &&
             !event.value("final", true);
    });
    check(acked.has_value(), "the server did not accept the playback cursor");

    append(a, voiced);  // barge-in while VLM and partial TTS are active
    auto cancelled = wait_event(a, "response.cancelled", [&](const json& event) {
      return event.value("response_id", "") == first_id;
    });
    check(cancelled.has_value() &&
              cancelled->value("acknowledged_text", "") == acknowledged &&
              cancelled->value("text_cursor", size_t{0}) == acknowledged.size() &&
              cancelled->value("history_recorded", false),
          "barge-in did not report exactly the acknowledged history prefix");
    const size_t after_cancel = a.observed.size();

    // Late VLM/TTS events and a late ack belong to the released response and
    // must not reach the next turn or change its acknowledged prefix.
    h.stream(first_id, "vlm", text_chunk("LATE UNPLAYED", false));
    if (tts1) {
      h.stream(tts1->request_id, "tts",
               pcm_chunk(std::string("\x05\x00\x06\x00", 4), 1));
    }
    h.complete(first_id, "vlm", false, Value{}, std::string("aborted"));
    if (tts0) h.complete(tts0->request_id, "tts", false, Value{}, "aborted");
    if (tts1) h.complete(tts1->request_id, "tts", false, Value{}, "aborted");
    ack(a, first_id, answer.size());

    append(a, quiet);
    auto second_asr = h.next_submit(h.asr_recv);
    check(second_asr.has_value(), "the turn after barge-in did not reach ASR");
    if (second_asr) {
      h.complete(second_asr->request_id, "asr", true, text_result("second question"));
    }
    auto second_vlm = h.next_submit(h.vlm_recv);
    check(second_vlm.has_value(), "the turn after barge-in did not reach VLM");
    StagePayload second_payload;
    if (second_vlm) second_payload = StagePayload::from_value(second_vlm->data);
    auto roles = prompt_roles(second_payload);
    check(roles.size() == 3 && roles[0].second == "first question" &&
              roles[1].second == acknowledged && roles[2].second == "second question",
          "the next VLM prompt did not preserve exactly the acknowledged prefix");
    check(std::none_of(roles.begin(), roles.end(), [](const auto& role) {
            return role.second.find("Second sentence") != std::string::npos ||
                   role.second.find("unplayed suffix") != std::string::npos ||
                   role.second.find("LATE UNPLAYED") != std::string::npos;
          }),
          "the next VLM prompt leaked unplayed or late assistant text");
    for (size_t i = after_cancel; i < a.observed.size(); ++i) {
      const json& event = a.observed[i];
      check(!(event.value("response_id", "") == first_id &&
              (event.value("type", "") == "audio.delta" ||
               event.value("type", "") == "response.text.delta" ||
               event.value("type", "") == "audio.done" ||
               event.value("type", "") == "response.completed")),
            "stale response audio or text arrived after response.cancelled");
    }

    // Barge-in before any VLM delta: retain the second user turn once, with an
    // empty assistant prefix rather than the whole or a fabricated answer.
    const std::string second_id = second_vlm ? second_vlm->request_id : "";
    append(a, voiced);
    auto second_cancel = wait_event(a, "response.cancelled", [&](const json& event) {
      return event.value("response_id", "") == second_id;
    });
    check(second_cancel.has_value() &&
              second_cancel->value("acknowledged_text", "nonempty").empty() &&
              second_cancel->value("history_recorded", false),
          "pre-VLM barge-in did not retain an empty assistant prefix");
    append(a, quiet);
    auto third_asr = h.next_submit(h.asr_recv);
    if (third_asr) {
      h.complete(third_asr->request_id, "asr", true, text_result("third question"));
    }
    auto third_vlm = h.next_submit(h.vlm_recv);
    check(third_vlm.has_value(), "the third turn did not reach VLM");
    StagePayload third_payload;
    if (third_vlm) third_payload = StagePayload::from_value(third_vlm->data);
    roles = prompt_roles(third_payload);
    check(roles.size() == 5 && roles[1].second == acknowledged &&
              roles[2].second == "second question" && roles[3].second.empty() &&
              roles[4].second == "third question",
          "repeated barge-in duplicated or reordered interrupted turns");

    // Complete the third response, then acknowledge its audio.done cursor.
    // This releases the retained terminal record and commits the full answer.
    const std::string third_id = third_vlm ? third_vlm->request_id : "";
    const std::string third_answer = "fully played answer";
    h.complete(third_id, "vlm", true, text_result(third_answer));
    auto third_tts = h.next_submit(h.tts_recv);
    check(third_tts.has_value(), "the completed response had no TTS tail");
    Value::Map extra;
    extra["streamed"] = Value(true);
    if (third_tts) {
      h.complete(third_tts->request_id, "tts", true,
                 artifact_result("audio", "audio/wav",
                                 audio::wav_from_pcm16(std::string(4, '\x01'), 24000),
                                 extra));
    }
    auto done = wait_event(a, "audio.done", [&](const json& event) {
      return event.value("response_id", "") == third_id;
    });
    check(done.has_value() &&
              done->value("text_cursor", size_t{0}) == third_answer.size(),
          "audio.done did not carry the final playback cursor");
    wait_until([&] { return engine.in_flight() == 0; },
               "generation stayed in flight after response.completed");
    ack(a, third_id, third_answer.size());
    auto final_ack = wait_event(a, "response.playback_ack", [&](const json& event) {
      return event.value("response_id", "") == third_id &&
             event.value("final", false);
    });
    check(final_ack.has_value(), "final playback acknowledgement did not finish");

    // A separate WebSocket/session must not see either interrupted turn or the
    // completed one, even though all requests share one UnifiedEngine store.
    Client b(ws_url);
    configure(b);
    auto other_vlm = start_turn(b, "other session question");
    StagePayload other_payload;
    if (other_vlm) other_payload = StagePayload::from_value(other_vlm->data);
    check(prompt_roles(other_payload).size() == 1,
          "realtime playback history leaked across WebSocket sessions");
    ack(b, third_id, third_answer.size());  // stale cross-session ack: ignored
  }
}

// Transport idle validity and application response stalls are different
// lifecycles. These clients intentionally do not run a receive loop (and so do
// not answer PING with PONG) until the assertions require the queued wire data.
void run_realtime_transport_validity_lifecycle_subtest(
    Harness& h, UnifiedEngine& engine, RealtimeApi& realtime) {
  std::atomic<bool> cancel{false};
  const std::string ws_url =
      "ws://127.0.0.1:" + std::to_string(realtime.bound_port()) +
      "/v1/realtime?intent=transcription&model=qwen3-unified";
  struct WireLog {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> events;
    size_t next = 0;
    bool done = false;
  };
  auto start_reader = [](test_http::WebSocket& ws, WireLog& log) {
    return std::thread([&] {
      ws.run([&](std::string_view message) {
        json event = json::parse(message, nullptr, false);
        if (event.is_discarded()) return;
        {
          std::lock_guard<std::mutex> lock(log.mu);
          log.events.push_back(std::move(event));
        }
        log.cv.notify_all();
      });
      {
        std::lock_guard<std::mutex> lock(log.mu);
        log.done = true;
      }
      log.cv.notify_all();
    });
  };
  auto wait_type = [](WireLog& log, const std::string& type,
                      double timeout_s = 5.0) -> std::optional<json> {
    std::unique_lock<std::mutex> lock(log.mu);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(timeout_s));
    while (true) {
      for (size_t i = log.next; i < log.events.size(); ++i) {
        if (log.events[i].value("type", "") != type) continue;
        json found = log.events[i];
        log.next = i + 1;
        return found;
      }
      if (log.done || log.cv.wait_until(lock, deadline) == std::cv_status::timeout) {
        return std::nullopt;
      }
    }
  };
  auto wait_done = [](WireLog& log, double timeout_s) {
    std::unique_lock<std::mutex> lock(log.mu);
    return log.cv.wait_for(lock, std::chrono::duration<double>(timeout_s),
                           [&] { return log.done; });
  };
  auto count_type = [](WireLog& log, const std::string& type) {
    std::lock_guard<std::mutex> lock(log.mu);
    return static_cast<size_t>(std::count_if(
        log.events.begin(), log.events.end(), [&](const json& event) {
          return event.value("type", "") == type;
        }));
  };

  std::string voiced(3840, '\0');
  for (int i = 0; i < 1920; ++i) {
    const auto sample = static_cast<int16_t>(
        6000.0 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * i / 16000.0));
    voiced[static_cast<size_t>(i) * 2] = static_cast<char>(sample & 0xff);
    voiced[static_cast<size_t>(i) * 2 + 1] =
        static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xff);
  }
  const std::string quiet(4800, '\0');
  auto drive_turn = [&](test_http::WebSocket& ws) {
    ws.send_text(json{{"type", "session.update"},
                      {"session",
                       {{"input_audio_format", "pcm16"},
                        {"turn_detection",
                         {{"type", "server_vad"}, {"silence_ms", 100}}},
                        {"audio_output",
                         {{"voice", "vivian"}, {"playback_ack", true}}}}}}
                     .dump());
    ws.send_text(json{{"type", "input_audio_buffer.append"},
                      {"audio", util::base64_encode(voiced)}}
                     .dump());
    ws.send_text(json{{"type", "input_audio_buffer.append"},
                      {"audio", util::base64_encode(quiet)}}
                     .dump());
    while (auto request = h.next_submit(h.asr_recv)) {
      if (request->request_id.rfind("realtime-preview-", 0) == 0) {
        h.complete(request->request_id, "asr", true, text_result("he"));
        continue;
      }
      h.complete(request->request_id, "asr", true, text_result("hello"));
      break;
    }
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "transport-validity turn did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm", text_chunk("world", false));
      h.complete(vlm->request_id, "vlm", true, text_result("world"));
    }
    return h.next_submit(h.tts_recv);
  };
  auto pcm_chunk = [](const std::string& bytes, int64_t index) {
    Value::Map chunk;
    chunk["audio"] = Value::bin(bytes);
    chunk["index"] = Value(index);
    chunk["sample_rate"] = Value(int64_t{24000});
    chunk["channels"] = Value(int64_t{1});
    chunk["format"] = Value("pcm16");
    return Value(std::move(chunk));
  };

  // Active response work must survive beyond PING=1s / hangup=2s even when the
  // consumer intentionally sends no PONG. request_timeout_s remains 10s.
  test_http::WebSocket paused(ws_url, client::Milliseconds{120000}, &cancel);
  auto paused_tts = drive_turn(paused);
  std::this_thread::sleep_for(std::chrono::milliseconds(3200));
  check(engine.in_flight() == 1,
        "transport idle expiry released an active response before terminal");
  const std::string c0("\x01\x00\x02\x00", 4);
  const std::string c1("\x03\x00\x04\x00", 4);
  // One delayed long narration exceeds the 1,024-frame per-socket high-water
  // boundary. Backpressure must retain and later drain every event, not reset.
  constexpr size_t kDelayedAudioChunks = 1025;
  std::string delayed_pcm;
  if (paused_tts && engine.in_flight() != 0) {
    for (size_t index = 0; index < kDelayedAudioChunks; ++index) {
      std::string bytes(8192, static_cast<char>(index));
      const uint16_t first = static_cast<uint16_t>(index);
      const uint16_t second = static_cast<uint16_t>(index ^ 0x5a5a);
      bytes[0] = static_cast<char>(first & 0xff);
      bytes[1] = static_cast<char>(first >> 8);
      bytes[2] = static_cast<char>(second & 0xff);
      bytes[3] = static_cast<char>(second >> 8);
      delayed_pcm += bytes;
      h.stream(paused_tts->request_id, "tts", pcm_chunk(bytes, index));
    }
    Value::Map extra;
    extra["streamed"] = Value(true);
    h.complete(paused_tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16(delayed_pcm, 24000), extra));
  }
  WireLog paused_log;
  auto paused_reader = start_reader(paused, paused_log);
  auto completed = wait_type(paused_log, "response.completed", 60.0);
  check(completed.has_value(),
        "active response lost response.completed after delayed reads");
  if (completed) {
    check(completed->value("text_cursor", size_t{0}) == 5,
          "active response terminal carried the wrong final cursor");
    paused.send_text(json{{"type", "output_audio_buffer.playback_ack"},
                          {"response_id", completed->at("response_id")},
                          {"text_cursor", 5}}
                         .dump());
    auto final_ack = wait_type(paused_log, "response.playback_ack", 2.0);
    check(final_ack && final_ack->value("final", false) &&
              final_ack->value("text_cursor", size_t{0}) == 5,
          "active response did not finish with one final playback ACK");
    if (final_ack) {
      paused.send_text(json{{"type", "output_audio_buffer.playback_ack"},
                            {"response_id", completed->at("response_id")},
                            {"text_cursor", 5}}
                           .dump());
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::string joined_pcm;
  std::string joined_text;
  size_t previous_cursor = 0;
  {
    std::lock_guard<std::mutex> lock(paused_log.mu);
    for (const json& event : paused_log.events) {
      const std::string type = event.value("type", "");
      if (type == "audio.delta") {
        joined_pcm += util::base64_decode(event.at("audio").get<std::string>(),
                                         1u << 20);
        const size_t cursor = event.value("text_cursor", size_t{0});
        check(cursor >= previous_cursor, "realtime PCM cursors decreased");
        previous_cursor = cursor;
      } else if (type == "response.text.delta") {
        joined_text += event.value("text", "");
      }
    }
  }
  if (joined_pcm != delayed_pcm || joined_text != "world") {
    std::cerr << "DIAG: transport reconstruction pcm_bytes=" << joined_pcm.size()
              << " expected_bytes=" << delayed_pcm.size()
              << " text=" << json(joined_text).dump() << "\n";
  }
  check(joined_pcm == delayed_pcm && joined_text == "world",
        "active response changed or lost native text/PCM after idle hold");
  check(count_type(paused_log, "audio.delta") == kDelayedAudioChunks,
        "delayed long narration did not drain all 1,025 public audio events");
  check(count_type(paused_log, "response.completed") == 1 &&
            count_type(paused_log, "response.failed") == 0 &&
            count_type(paused_log, "response.playback_ack") == 1,
        "active response emitted duplicate or failed terminal events");
  paused.close();
  paused_reader.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "active response stayed in flight after final ACK");

  // With no response activity, the unchanged 10s application deadline must
  // still win; the transport hold must not refresh response_deadline.
  test_http::WebSocket stalled(ws_url, client::Milliseconds{5000}, &cancel);
  auto stalled_tts = drive_turn(stalled);
  const auto stall_started = std::chrono::steady_clock::now();
  wait_until([&] { return engine.in_flight() == 0; },
             "stalled response did not leave flight at its application deadline");
  const double stall_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - stall_started).count();
  check(stall_s >= 8.0 && stall_s <= 20.0,
        "transport lifecycle changed the 10s response-stall deadline");
  WireLog stalled_log;
  auto stalled_reader = start_reader(stalled, stalled_log);
  auto failed = wait_type(stalled_log, "response.failed", 2.0);
  check(failed && failed->value("error", "") == "spoken response timed out",
        "stalled response did not emit its application timeout terminal");
  if (stalled_tts) {
    h.complete(stalled_tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16(c0, 24000)));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  check(count_type(stalled_log, "response.completed") == 0 &&
            count_type(stalled_log, "audio.delta") == 0 &&
            count_type(stalled_log, "response.failed") == 1,
        "stalled response emitted stale audio or duplicate terminals");
  stalled.close();
  stalled_reader.join();

  // Once no request/response work is owned, ordinary idle expiry must return.
  test_http::WebSocket idle(ws_url, client::Milliseconds{5000}, &cancel);
  std::this_thread::sleep_for(std::chrono::milliseconds(3200));
  WireLog idle_log;
  auto idle_reader = start_reader(idle, idle_log);
  check(wait_done(idle_log, 5.0),
        "idle session remained open after active-work ownership ended");
  idle.close();
  idle_reader.join();
  check(count_type(idle_log, "response.completed") == 0 &&
            count_type(idle_log, "response.failed") == 0,
        "idle validity cleanup emitted a synthetic response terminal");
}

// Runs both realtime WebSocket wire subtests (ordinary spoken-audio/leg-error
// and opt-in playback_ack barge-in history) on one RealtimeApi/lws_context so
// we exercise both modes side-by-side without triggering Homebrew libwebsockets
// 4.5.7's macOS second-context teardown fd bug.
void test_realtime_spoken_audio_format_and_leg_error() {
  Harness h("rtspoken");
  ApiConfig rt_config;
  rt_config.host = "127.0.0.1";
  rt_config.port = 0;
  rt_config.model = "qwen3-unified";
  rt_config.entry_stage = "asr";
  rt_config.request_timeout_s = 10.0;
  rt_config.realtime_validity_ping_s = 1;
  rt_config.realtime_validity_hangup_s = 2;
  UnifiedEngine engine(h.coordinator, realtime_history_config(h));
  RealtimeApi realtime(h.coordinator, rt_config);
  realtime.attach_unified(engine);
  engine.on_event = [&](const UnifiedEvent& event) {
    realtime.forward_event(event);
  };
  realtime.install_stream_sink();
  check(realtime.bind(), "realtime server did not bind");
  h.start_pump();
  std::thread rt_thread([&] { realtime.listen(); });

  run_realtime_spoken_audio_format_and_leg_error_subtest(h, engine, realtime);
  run_realtime_barge_in_playback_ack_wire_subtest(h, engine, realtime);
  run_realtime_transport_validity_lifecycle_subtest(h, engine, realtime);

  realtime.stop();
  rt_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "realtime wire tests ended with requests in flight");
}

void test_public_http_finish_reason_length_and_router_reset() {
  Harness h("httplength");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.served_model_name = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  h.start_pump();
  check(api.bind(), "httplength OpenAIApi failed to bind");
  std::thread server_thread([&] { api.listen(); });
  for (int i = 0; i < 100; ++i) {
    httplib::Client probe("127.0.0.1", api.bound_port());
    if (auto r = probe.Get("/health"); r && r->status == 200) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const std::string url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                          "/v1/chat/completions";

  {
    auto fut = std::async(std::launch::async, [&] {
      test_http::HttpRequest req;
      req.url = url;
      req.body = R"({"model":"qwen3-unified","max_tokens":4,"messages":[{"role":"user","content":"long"}]})";
      req.headers.emplace("Content-Type", "application/json");
      return test_http::perform(req);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "httplength non-stream VLM submit missing");
    if (vlm) {
      Value::Map res;
      res["content"] = Value("truncated answer");
      res["finish_reason"] = Value("length");
      h.complete(vlm->request_id, "vlm", true, Value(std::move(res)));
    }
    const auto resp = fut.get();
    const auto parsed = json::parse(resp.body);
    check(parsed["choices"][0]["finish_reason"] == "length",
          "non-streaming chat did not propagate finish_reason=length");
  }

  {
    auto fut = std::async(std::launch::async, [&] {
      test_http::HttpRequest req;
      req.url = url;
      req.body = R"({"model":"qwen3-unified","stream":true,"max_tokens":4,"messages":[{"role":"user","content":"long"}]})";
      req.headers.emplace("Content-Type", "application/json");
      return test_http::perform(req);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "httplength stream VLM submit missing");
    if (vlm) {
      h.stream(vlm->request_id, "vlm", text_chunk("truncated", false));
      Value::Map res;
      res["content"] = Value("truncated");
      res["finish_reason"] = Value("length");
      h.complete(vlm->request_id, "vlm", true, Value(std::move(res)));
    }
    const auto resp = fut.get();
    check(resp.body.find("\"finish_reason\":\"length\"") != std::string::npos,
          "streaming chat SSE did not propagate finish_reason=length");
  }

  api.stop();
  server_thread.join();

  CommRouter router("asr", std::nullopt, std::nullopt, {}, {"vlm"},
                    {{"vlm", {0}}}, {});
  const auto [kind1, r1] = router.relay_for("vlm");
  check(kind1 == TransportKind::Shm && r1 != nullptr,
        "CommRouter initial outbound SHM selection failed");
  router.reset_target("vlm");
  const auto [kind2, r2] = router.relay_for("vlm");
  check(kind2 == TransportKind::Shm && r2 != nullptr,
        "CommRouter outbound selection after reset_target failed");
  router.close();
}

// ---- Standalone speech synthesis ------------------------------------------

// One text goes to the registered TTS stage; the WAV the stage returns is
// served as audio/wav, or stripped to raw PCM16 for response_format=pcm.
void test_public_http_audio_speech() {
  Harness h("httpspeech");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "speech server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url = "http://127.0.0.1:" +
                          std::to_string(api.bound_port()) + "/v1/audio/speech";

  auto post = [&](const json& body) {
    test_http::HttpRequest request;
    request.url = url;
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  };
  auto post_status = [&](const json& body) -> long {
    try {
      return post(body).status;
    } catch (const client::Error& error) {
      return static_cast<long>(error.status());
    }
  };
  auto content_type = [](const test_http::HttpResponse& response) {
    for (const auto& [key, value] : response.headers) {
      std::string lower = key;
      std::transform(lower.begin(), lower.end(), lower.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (lower == "content-type") return value;
    }
    return std::string();
  };

  const std::string raw_pcm("\x10\x20\x30\x40\x50\x60", 6);
  const std::string wav = audio::wav_from_pcm16(raw_pcm, 24000);
  Value::Map extra;
  extra["sample_rate"] = Value(int64_t{24000});
  extra["channels"] = Value(int64_t{1});

  auto serve = [&](const std::string& format) {
    auto future = std::async(std::launch::async, [&] {
      return post(json{{"model", "qwen3-unified"},
                       {"input", "hello there"},
                       {"voice", "vivian"},
                       {"response_format", format}});
    });
    auto submit = h.next_submit(h.tts_recv);
    check(submit.has_value(), "speech request did not reach the TTS stage");
    if (submit) {
      StagePayload p = StagePayload::from_value(submit->data);
      check(p.request.inputs.find("text")->as_str() == "hello there",
            "speech request did not forward the input text");
      check(p.request.inputs.find("voice")->as_str() == "vivian",
            "speech request did not forward the voice");
      h.complete(submit->request_id, "tts", true,
                 artifact_result("audio", "audio/wav", wav, extra));
    }
    return future.get();
  };

  const auto wav_response = serve("wav");
  check(wav_response.status == 200, "wav speech did not answer 200");
  check(wav_response.body == wav,
        "wav speech did not return the stage bytes unchanged");
  check(content_type(wav_response).rfind("audio/wav", 0) == 0,
        "wav speech did not answer audio/wav");

  const auto pcm_response = serve("pcm");
  check(pcm_response.status == 200, "pcm speech did not answer 200");
  check(pcm_response.body == raw_pcm,
        "pcm speech did not strip the RIFF container");
  check(content_type(pcm_response).rfind("audio/pcm", 0) == 0,
        "pcm speech did not answer audio/pcm");

  // Contract errors answer 400 before any stage submit.
  check(post_status(json::object()) == 400,
        "a missing input did not answer 400");
  check(post_status(json{{"input", ""}}) == 400,
        "an empty input did not answer 400");
  check(post_status(json{{"input", "hi"}, {"response_format", "mp3"}}) == 400,
        "an unsupported response_format did not answer 400");
  check(post_status(json{{"input", "hi"}, {"speed", 1.5}}) == 400,
        "an unsupported field did not answer 400");
  check(post_status(json{{"input", "hi"}, {"model", "other"}}) == 400,
        "an unserved model did not answer 400");
  check(h.no_submit(h.tts_recv),
        "a rejected speech request still submitted to the TTS stage");

  api.stop();
  server_thread.join();
}

// A process without a registered TTS stage answers 501, not a 200 with no
// audio.
void test_public_http_audio_speech_without_tts() {
  Harness h("httpspeechnone");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "no-tts speech server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  test_http::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                "/v1/audio/speech";
  request.body = json{{"input", "hello"}}.dump();
  request.headers.emplace("Content-Type", "application/json");
  request.timeout = client::Milliseconds{5000};
  long status = 0;
  std::string body;
  try {
    const auto response = test_http::perform(request);
    status = response.status;
    body = response.body;
  } catch (const client::Error& error) {
    status = static_cast<long>(error.status());
    body = error.what();
  }
  check(status == 501, "speech without a TTS stage did not answer 501");
  check(body.find("TTS stage") != std::string::npos,
        "the 501 body does not name the missing TTS stage");
  api.stop();
  server_thread.join();
}

// ---- Session-scoped cancellation -----------------------------------------

// A request carrying session_id may only be aborted by that session. The
// engine reports kSessionMismatch, the HTTP route answers 403, and the request
// keeps running for its owner.
void test_cancel_for_session_rejects_foreign_session() {
  Harness h("cancelsession");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "session-cancel server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  StagePayload request = audio_request();
  Value::Map inputs = request.request.inputs.as_map();
  inputs["session_id"] = Value("tenant-a");
  request.request.inputs = Value(std::move(inputs));

  const std::string id = engine.submit(request);
  auto asr = h.next_submit(h.asr_recv);
  check(asr && asr->request_id == id,
        "the session request did not reach the ASR stage under its own id");

  check(engine.cancel_for_session(id, "tenant-b") ==
            UnifiedEngine::CancelStatus::kSessionMismatch,
        "cancel_for_session accepted a foreign session id");

  test_http::HttpRequest abort_request;
  abort_request.url = "http://127.0.0.1:" +
                      std::to_string(api.bound_port()) + "/abort";
  abort_request.body =
      json{{"request_id", id}, {"session_id", "tenant-b"}}.dump();
  abort_request.headers.emplace("Content-Type", "application/json");
  abort_request.timeout = client::Milliseconds{5000};
  long abort_status = 0;
  try {
    abort_status = test_http::perform(abort_request).status;
  } catch (const client::Error& error) {
    abort_status = static_cast<long>(error.status());
  }
  check(abort_status == 403, "a cross-session abort did not answer 403");
  check(!h.saw_abort(id), "a cross-session abort reached the request");

  h.complete(id, "asr", true, text_result("transcript"));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "the owned request did not reach the VLM stage");
  h.complete(vlm ? vlm->request_id : id + ":vlm", "vlm", true,
             text_result("answer"));
  check(engine.wait(id, 5.0), "the owned request did not settle");
  const auto info = engine.info(id);
  check(info && info->state == "completed",
        "a foreign cancel changed the owned request's outcome");
  engine.release(id);

  // The owning session still cancels its own request.
  const std::string second = engine.submit(request);
  auto second_asr = h.next_submit(h.asr_recv);
  check(second_asr.has_value(), "the second request did not reach ASR");
  check(engine.cancel_for_session(second, "tenant-a") ==
            UnifiedEngine::CancelStatus::kCancelled,
        "cancel_for_session rejected the owning session");
  check(h.wait_abort(second), "the owning cancel did not abort the leg");
  h.complete(second, "asr", false, Value(), std::string("request aborted"));
  check(engine.wait(second, 5.0), "the owned cancel did not settle");
  engine.release(second);

  api.stop();
  server_thread.join();
}

// ---- Streamed TTS fan-out ------------------------------------------------

// A VLM-only request that asks for streamed audio output.
StagePayload streamed_audio_request() {
  StagePayload payload = mixed_request(false, true);
  Value::Map inputs = payload.request.inputs.as_map();
  Value::Map options = inputs.at("audio_options").as_map();
  options["stream"] = Value(true);
  inputs["audio_options"] = Value(std::move(options));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

StagePayload realtime_playback_request(const std::string& session_id) {
  StagePayload payload = streamed_audio_request();
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["session_id"] = Value(session_id);
  inputs["session_history"] = Value(true);
  inputs["realtime_playback_ack"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

StagePayload realtime_audio_playback_request(const std::string& session_id) {
  StagePayload payload = audio_leg_request(true);
  Value::Map inputs = payload.request.inputs.as_map();
  Value::Array outputs{Value("text"), Value("audio")};
  inputs["output_modalities"] = Value(std::move(outputs));
  Value::Map audio;
  audio["voice"] = Value("vivian");
  audio["stream"] = Value(true);
  inputs["audio_options"] = Value(std::move(audio));
  inputs["session_id"] = Value(session_id);
  inputs["session_history"] = Value(true);
  inputs["realtime_playback_ack"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

std::vector<size_t> realtime_stress_permutation(uint64_t seed, size_t count) {
  std::vector<size_t> order(count);
  for (size_t i = 0; i < count; ++i) order[i] = i;
  auto next = [&seed] {
    uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  for (size_t i = count; i > 1; --i) {
    std::swap(order[i - 1], order[static_cast<size_t>(next() % i)]);
  }
  return order;
}

StagePayload inspect_history_probe(Harness& h, UnifiedEngine& engine,
                                   const std::string& session_id) {
  StagePayload request = vlm_only_request(true, false);
  Value::Map inputs = request.request.inputs.as_map();
  inputs["session_id"] = Value(session_id);
  inputs["session_history"] = Value(true);
  request.request.inputs = Value(std::move(inputs));
  const std::string id = engine.submit(std::move(request));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "history probe did not reach the VLM");
  StagePayload payload;
  if (vlm) payload = StagePayload::from_value(vlm->data);
  check(engine.cancel(id), "history probe cancellation was rejected");
  h.complete(id, "vlm", false, Value{}, std::string("aborted"));
  check(engine.wait(id, 5.0), "history probe did not settle");
  engine.release(id);
  return payload;
}

UnifiedConfig realtime_history_config(const Harness& h) {
  UnifiedConfig config = h.mixed_config();
  config.session_history.enabled = true;
  return config;
}

// Barge-in before the first VLM delta keeps the interrupted user turn with an
// empty acknowledged assistant prefix. After VLM output starts, only the exact
// client cursor is retained; repeated cancellation cannot append or extend it.
void test_realtime_barge_in_history_before_and_after_vlm() {
  Harness h("rt-history-engine");
  UnifiedEngine engine(h.coordinator, realtime_history_config(h));
  h.start_pump();

  const std::string before = engine.submit(realtime_playback_request("rt-before"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "pre-VLM barge-in request did not reach the VLM inbox");
  auto first = engine.cancel_realtime_barge_in(before, "rt-before", 0);
  check(first.accepted && first.generation_cancelled && first.history_recorded &&
            first.assistant_prefix.empty(),
        "pre-VLM barge-in did not retain an empty acknowledged prefix");
  auto repeated = engine.cancel_realtime_barge_in(before, "rt-before", 0);
  check(repeated.accepted && !repeated.history_recorded,
        "repeated pre-VLM cancellation recorded a duplicate turn");
  h.complete(before, "vlm", false, Value{}, std::string("aborted"));
  check(engine.wait(before, 5.0), "pre-VLM barge-in did not settle");
  engine.release(before);

  StagePayload probe = inspect_history_probe(h, engine, "rt-before");
  auto roles = prompt_roles(probe);
  check(roles.size() == 3 && roles[0].first == "user" &&
            roles[0].second == "describe the scene" &&
            roles[1].first == "assistant" && roles[1].second.empty() &&
            roles[2].first == "user",
        "pre-VLM barge-in history is missing, duplicated, or out of order");

  const std::string after = engine.submit(realtime_playback_request("rt-after"));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "post-VLM barge-in request did not reach the VLM");
  const std::string answer = "Played prefix. unplayed suffix";
  const std::string acknowledged = "Played prefix. ";
  h.stream(after, "vlm", text_chunk(answer, false));
  check(h.next_submit(h.tts_recv).has_value(),
        "post-VLM barge-in did not observe the streamed answer");
  auto cancelled = engine.cancel_realtime_barge_in(
      after, "rt-after", acknowledged.size());
  check(cancelled.accepted && cancelled.history_recorded &&
            cancelled.assistant_prefix == acknowledged,
        "post-VLM barge-in did not retain the exact acknowledged prefix");
  auto cancelled_again = engine.cancel_realtime_barge_in(
      after, "rt-after", answer.size());
  check(!cancelled_again.history_recorded &&
            cancelled_again.assistant_prefix == acknowledged,
        "repeated cancellation extended the already acknowledged prefix");
  h.complete(after, "vlm", false, Value{}, std::string("aborted"));
  check(engine.wait(after, 5.0), "post-VLM barge-in did not settle");
  engine.release(after);

  probe = inspect_history_probe(h, engine, "rt-after");
  roles = prompt_roles(probe);
  check(roles.size() == 3 && roles[0].second == "describe the scene" &&
            roles[1].second == acknowledged && roles[2].first == "user",
        "the next prompt does not contain exactly one acknowledged prefix");
  check(std::none_of(roles.begin(), roles.end(), [](const auto& role) {
          return role.second.find("unplayed suffix") != std::string::npos;
        }),
        "the next prompt leaked the unplayed assistant suffix");
}

// A terminal response is emitted, not necessarily played. History remains
// deferred until the final cursor arrives; normal cancel/error and foreign
// session cancellation keep their original exclusion behavior.
void test_realtime_playback_completion_and_default_exclusion() {
  Harness h("rt-playback-engine");
  UnifiedEngine engine(h.coordinator, realtime_history_config(h));
  h.start_pump();

  const std::string id = engine.submit(realtime_playback_request("rt-complete"));
  check(h.next_submit(h.vlm_recv).has_value(), "completion request missing VLM");
  const std::string answer = "complete but not yet played";
  h.complete(id, "vlm", true, text_result(answer));
  auto tts = h.next_submit(h.tts_recv);
  check(tts.has_value(), "completion request missing its TTS tail");
  Value::Map extra;
  extra["streamed"] = Value(true);
  if (tts) {
    h.complete(tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16(std::string(4, '\x01'), 24000),
                               extra));
  }
  check(engine.wait(id, 5.0), "completion request did not settle");
  check(!engine.complete_realtime_playback(id, "rt-complete", 4),
        "a partial playback cursor committed a complete answer");

  StagePayload probe = inspect_history_probe(h, engine, "rt-complete");
  check(prompt_roles(probe).size() == 1,
        "server emission entered history before playback acknowledgement");
  check(engine.complete_realtime_playback(id, "rt-complete", answer.size()),
        "the final playback cursor did not commit the completed turn");
  check(engine.complete_realtime_playback(id, "rt-complete", answer.size()),
        "repeating a final playback ack was not idempotent");
  engine.release(id);
  probe = inspect_history_probe(h, engine, "rt-complete");
  auto roles = prompt_roles(probe);
  check(roles.size() == 3 && roles[1].second == answer,
        "final playback acknowledgement did not record exactly one full turn");

  const std::string ordinary =
      engine.submit(realtime_playback_request("rt-default-cancel"));
  check(h.next_submit(h.vlm_recv).has_value(), "ordinary cancel missing VLM");
  h.stream(ordinary, "vlm", text_chunk("must not be retained", false));
  check(engine.cancel(ordinary), "ordinary cancellation was rejected");
  h.complete(ordinary, "vlm", false, Value{}, std::string("aborted"));
  check(engine.wait(ordinary, 5.0), "ordinary cancellation did not settle");
  engine.release(ordinary);
  probe = inspect_history_probe(h, engine, "rt-default-cancel");
  check(prompt_roles(probe).size() == 1,
        "ordinary cancel globally enabled cancelled-request history");

  const std::string foreign =
      engine.submit(realtime_playback_request("rt-session-a"));
  check(h.next_submit(h.vlm_recv).has_value(), "foreign-session request missing VLM");
  auto wrong = engine.cancel_realtime_barge_in(foreign, "rt-session-b", 0);
  check(!wrong.accepted && !wrong.history_recorded,
        "a foreign session accepted a realtime barge-in cancellation");
  check(engine.cancel(foreign), "cleanup cancellation failed");
  h.complete(foreign, "vlm", false, Value{}, std::string("aborted"));
  check(engine.wait(foreign, 5.0), "foreign-session request did not settle");
  engine.release(foreign);
  probe = inspect_history_probe(h, engine, "rt-session-b");
  check(prompt_roles(probe).size() == 1,
        "realtime history leaked across session ids");
}

// Fixed-seed multi-session ASR -> VLM -> streamed-TTS lifecycle stress. The
// permutation is scheduler input, not timing-dependent thread randomness: each
// backend message is sent in the listed order and each prompt is checked against
// a separately maintained per-session history model.
void test_realtime_deterministic_coordinator_history_stress() {
  constexpr size_t kSessions = 6;
  constexpr size_t kRounds = 3;
  constexpr uint64_t kSeed = 0x38c001dULL;
  Harness h("rt-history-stress");
  UnifiedConfig config = realtime_history_config(h);
  config.max_in_flight = kSessions;
  config.session_history.max_sessions = 16;
  config.session_history.max_turns = 64;
  config.session_history.max_chars = 65536;
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  struct ExpectedTurn {
    std::string user;
    std::string assistant;
  };
  struct Turn {
    size_t session = 0;
    size_t round = 0;
    std::string id;
    std::string vlm_id;
    std::string transcript;
    std::string answer;
    size_t acknowledged = 0;
    bool cancel = false;
    std::vector<std::string> tts_ids;
  };
  std::vector<std::vector<ExpectedTurn>> expected(kSessions);
  std::vector<std::pair<std::string, size_t>> owners;

  for (size_t round = 0; round < kRounds; ++round) {
    std::vector<Turn> turns(kSessions);
    const auto submit_order =
        realtime_stress_permutation(kSeed + round * 101 + 1, kSessions);
    for (const size_t session : submit_order) {
      Turn& turn = turns[session];
      turn.session = session;
      turn.round = round;
      turn.cancel = ((session + round) % 2) == 0;
      const std::string tag = "s" + std::to_string(session) + "r" +
                              std::to_string(round);
      turn.transcript = tag + " microphone transcript";
      turn.answer = tag + " played. " + tag + " unplayed suffix.";
      turn.acknowledged = turn.answer.find(tag + " unplayed");
      turn.id = engine.submit(realtime_audio_playback_request("stress-s" +
          std::to_string(session)));
      owners.emplace_back(turn.id, session);
    }

    if (round == 0) {
      bool full = false;
      try {
        (void)engine.submit(realtime_audio_playback_request("stress-overflow"));
      } catch (const QueueFullError&) {
        full = true;
      }
      check(full, "saturated unified admission did not fail closed");
    }

    std::vector<SubmitMessage> asr_submits;
    for (size_t i = 0; i < kSessions; ++i) {
      auto submit = h.next_submit(h.asr_recv);
      check(submit.has_value(), "stress ASR submit was not received");
      if (!submit) return;
      asr_submits.push_back(*submit);
    }
    if (round == 0) {
      check(h.no_submit(h.asr_recv),
            "an over-capacity realtime request reached the ASR backend");
    }
    const auto asr_finish_order =
        realtime_stress_permutation(kSeed + round * 101 + 2, kSessions);
    for (const size_t index : asr_finish_order) {
      const SubmitMessage& submit = asr_submits[index];
      // Locate by public request id rather than relying on completion order.
      auto owner = std::find_if(turns.begin(), turns.end(), [&](const Turn& turn) {
        return turn.id == submit.request_id;
      });
      check(owner != turns.end(), "ASR submit has a foreign request id");
      if (owner == turns.end()) return;
      h.stream(submit.request_id, "asr", text_chunk(owner->transcript, false));
      h.complete(submit.request_id, "asr", true, text_result(owner->transcript));
    }

    // Completion messages ride one ordered PUSH socket. Consume each resulting
    // VLM submit, but permute which ASR final was sent above.
    for (size_t index = 0; index < kSessions; ++index) {
      auto submit = h.next_submit(h.vlm_recv);
      check(submit.has_value(), "stress VLM submit was not received");
      if (!submit) return;
      auto owner = std::find_if(turns.begin(), turns.end(), [&](const Turn& turn) {
        return turn.id + ":vlm" == submit->request_id;
      });
      check(owner != turns.end(), "VLM submit has a foreign or duplicate request id");
      if (owner == turns.end()) return;
      owner->vlm_id = submit->request_id;
      StagePayload payload = StagePayload::from_value(submit->data);
      const auto roles = prompt_roles(payload);
      const auto& history = expected[owner->session];
      check(roles.size() == history.size() * 2 + 1,
            "realtime history has a missing or duplicate turn");
      if (roles.size() != history.size() * 2 + 1) return;
      for (size_t item = 0; item < history.size(); ++item) {
        check(roles[item * 2].first == "user" &&
                  roles[item * 2].second == history[item].user &&
                  roles[item * 2 + 1].first == "assistant" &&
                  roles[item * 2 + 1].second == history[item].assistant,
              "realtime history order/content differs from its acknowledged model");
        check(roles[item * 2].second.find("stress-s") == std::string::npos ||
                  roles[item * 2].second.find(
                      "s" + std::to_string(owner->session) + "r") == 0,
              "history text crossed realtime sessions");
      }
      check(roles.back().first == "user" &&
                roles.back().second == owner->transcript,
            "ASR final did not become exactly the current VLM user turn");
    }

    const auto stream_order =
        realtime_stress_permutation(kSeed + round * 101 + 4, kSessions);
    for (const size_t session : stream_order) {
      Turn& turn = turns[session];
      h.stream(turn.vlm_id, "vlm", text_chunk(turn.answer, false));
      for (size_t sentence = 0; sentence < 2; ++sentence) {
        auto tts = h.next_submit(h.tts_recv);
        check(tts.has_value(), "streamed response did not create both TTS legs");
        if (!tts) return;
        check(tts->request_id.rfind(turn.id + ":tts:", 0) == 0,
              "TTS leg crossed request ownership");
        turn.tts_ids.push_back(tts->request_id);
        Value::Map chunk;
        const std::string bytes = "PCM-" + std::to_string(session) + "-" +
                                  std::to_string(round) + "-" +
                                  std::to_string(sentence);
        chunk["audio"] = Value::bin(bytes);
        chunk["index"] = Value(int64_t{0});
        chunk["sample_rate"] = Value(int64_t{16000});
        chunk["channels"] = Value(int64_t{1});
        chunk["format"] = Value("pcm16");
        h.stream(tts->request_id, "tts", Value(std::move(chunk)));
      }
    }

    const auto settle_order =
        realtime_stress_permutation(kSeed + round * 101 + 5, kSessions);
    for (const size_t session : settle_order) {
      Turn& turn = turns[session];
      const std::string session_id = "stress-s" + std::to_string(session);
      if (turn.cancel) {
        auto foreign = engine.cancel_realtime_barge_in(
            turn.id, "stress-s-other", turn.acknowledged);
        check(!foreign.accepted && !foreign.history_recorded,
              "foreign realtime session cancelled or recorded history");
        auto cancelled = engine.cancel_realtime_barge_in(
            turn.id, session_id, turn.acknowledged);
        check(cancelled.accepted && cancelled.history_recorded &&
                  cancelled.assistant_prefix ==
                      turn.answer.substr(0, turn.acknowledged),
              "barge-in did not retain the exact acknowledged prefix");
        auto repeated = engine.cancel_realtime_barge_in(
            turn.id, session_id, turn.answer.size());
        check(repeated.accepted && !repeated.history_recorded &&
                  repeated.assistant_prefix == cancelled.assistant_prefix,
              "delayed duplicate barge-in extended or duplicated history");
        expected[session].push_back(
            ExpectedTurn{turn.transcript, cancelled.assistant_prefix});
        // RealtimeApi releases immediately after sending response.cancelled.
        // Backend work is deliberately still in flight at this point.
        engine.release(turn.id);
        h.stream(turn.vlm_id, "vlm", text_chunk("LATE-TEXT-" + session_id, false));
        for (const std::string& tts : turn.tts_ids) {
          Value::Map late;
          late["audio"] = Value::bin("LATE-PCM-" + session_id);
          late["index"] = Value(int64_t{1});
          late["sample_rate"] = Value(int64_t{16000});
          late["channels"] = Value(int64_t{1});
          late["format"] = Value("pcm16");
          h.stream(tts, "tts", Value(std::move(late)));
          h.complete(tts, "tts", false, Value{}, "aborted");
        }
        h.complete(turn.vlm_id, "vlm", false, Value{}, "aborted");
        h.complete(turn.vlm_id, "vlm", true, text_result("LATE-COMPLETE"));
      } else {
        h.complete(turn.vlm_id, "vlm", true, text_result(turn.answer));
        h.complete(turn.vlm_id, "vlm", true, text_result("DUPLICATE-COMPLETE"));
        for (const std::string& tts : turn.tts_ids) {
          Value::Map extra;
          extra["streamed"] = Value(true);
          h.complete(tts, "tts", true,
                     artifact_result("audio", "audio/wav",
                                     audio::wav_from_pcm16(std::string(4, '\x01'), 16000),
                                     extra));
          h.complete(tts, "tts", true, Value{});
        }
        check(engine.wait(turn.id, 5.0), "completed stress response did not settle");
        check(!engine.complete_realtime_playback(
                  turn.id, session_id, turn.acknowledged),
              "partial playback cursor committed a complete answer");
        check(engine.complete_realtime_playback(turn.id, session_id, turn.answer.size()),
              "monotonic final playback cursor was rejected");
        check(engine.complete_realtime_playback(turn.id, session_id, turn.answer.size()),
              "delayed duplicate final playback cursor was not idempotent");
        expected[session].push_back(ExpectedTurn{turn.transcript, turn.answer});
        engine.release(turn.id);
      }
    }
  }

  wait_until([&] { return engine.in_flight() == 0; },
             "realtime history stress left unified requests in flight");
  check(h.coordinator.in_flight() == 0,
        "realtime history stress left Coordinator requests in flight");
  std::set<std::string> counted;
  for (const auto& [id, session] : owners) {
    check(log.terminal_count(id) <= 1,
          "duplicate backend terminal produced double completion");
    counted.insert(id + ":" + std::to_string(session));
  }
  check(counted.size() == kSessions * kRounds,
        "request ids were reused or crossed sessions");
  for (const UnifiedEvent& event : log.snapshot()) {
    const auto owner = std::find_if(owners.begin(), owners.end(), [&](const auto& item) {
      return item.first == event.request_id;
    });
    check(owner != owners.end(), "an event has no realtime session owner");
    if (owner == owners.end()) continue;
    check(event.session_id == "stress-s" + std::to_string(owner->second),
          "an event crossed realtime session ids");
    check(event.text.find("LATE-") == std::string::npos &&
              event.text.find("DUPLICATE-COMPLETE") == std::string::npos,
          "stale or duplicate backend text was emitted after cancellation/terminal");
    if (event.artifact_bytes != nullptr) {
      check(event.artifact_bytes->find("LATE-PCM-") == std::string::npos,
            "stale TTS PCM was emitted after cancellation/release");
    }
  }
  std::cout << "REALTIME_COORDINATOR_STRESS PASS seed=" << kSeed
            << " sessions=" << kSessions << " rounds=" << kRounds
            << " requests=" << owners.size()
            << " history_exact=1 terminal_double=0 stale_after_release=0\n";
}

// Each VLM delta that completes a sentence starts its own :tts:N leg in
// sentence order.
void test_streamed_tts_sentence_fanout() {
  Harness h("ttsfanout");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(streamed_audio_request());
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "the streamed request did not reach the VLM");
  const std::string vlm_id = vlm ? vlm->request_id : id;

  h.stream(vlm_id, "vlm", text_chunk("Hello there. ", false));
  h.stream(vlm_id, "vlm", text_chunk("How are you? ", false));
  h.stream(vlm_id, "vlm", text_chunk("Fine.", false));

  const std::vector<std::string> expected = {"Hello there.", "How are you?",
                                             "Fine."};
  std::vector<std::string> leg_ids;
  std::vector<std::string> texts;
  for (size_t i = 0; i < expected.size(); ++i) {
    auto leg = h.next_submit(h.tts_recv);
    check(leg.has_value(), "a streamed sentence leg was not submitted");
    if (!leg) break;
    leg_ids.push_back(leg->request_id);
    texts.push_back(StagePayload::from_value(leg->data)
                        .request.inputs.find("text")
                        ->as_str());
  }
  check(texts == expected,
        "streamed TTS legs do not carry the sentences in order");
  for (size_t i = 0; i < leg_ids.size(); ++i) {
    check(leg_ids[i] == id + ":tts:" + std::to_string(i),
          "a streamed TTS leg id is not the ordered :tts:N child id");
  }

  h.complete(vlm_id, "vlm", true, text_result("Hello there. How are you? Fine."));
  Value::Map extra;
  extra["sample_rate"] = Value(int64_t{24000});
  extra["channels"] = Value(int64_t{1});
  for (const std::string& leg_id : leg_ids) {
    h.complete(leg_id, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16(std::string(4, '\x01'),
                                                     24000),
                               extra));
  }
  check(engine.wait(id, 5.0), "the streamed request did not complete");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
        "the streamed request did not end in one complete terminal");
  check(h.no_submit(h.tts_recv),
        "a fourth TTS leg started after the stream was flushed");
  engine.release(id);
}

// Markdown presentation stays in public text but is not fed to the native
// Talker. Skipped markup-only spans still advance the original UTF-8 cursors.
void test_streamed_tts_markdown_spans_and_cursors() {
  Harness h("ttsmarkdown");
  UnifiedEngine engine(h.coordinator, realtime_history_config(h));
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::vector<std::string> spans = {
      "### 1. 黑洞吸积盘：accretion disk\n",
      "**示意图：**\n",
      "---\n",
      "`event horizon` 与 **gravitational lensing**。\n",
      "*重点。*\n",
      "_下划线_。\n",
      "计算 2 * 3 * 4，2*3*4。\n",
  };
  const std::vector<std::string> expected_speech = {
      "1. 黑洞吸积盘：accretion disk",
      "示意图：",
      "event horizon 与 gravitational lensing。",
      "重点。",
      "下划线。",
      "计算 2 * 3 * 4，2*3*4。",
  };
  std::string full;
  for (const std::string& span : spans) full += span;
  const std::string id = engine.submit(realtime_playback_request("markdown-spans"));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "Markdown request did not reach VLM");
  const std::string vlm_id = vlm ? vlm->request_id : id + ":vlm";
  for (const std::string& span : spans) {
    h.stream(vlm_id, "vlm", text_chunk(span, false));
  }

  std::vector<std::string> leg_ids;
  std::vector<std::string> actual_speech;
  for (size_t i = 0; i < expected_speech.size(); ++i) {
    auto submit = h.next_submit(h.tts_recv);
    check(submit.has_value(), "a speakable Markdown span did not reach TTS");
    if (!submit) break;
    leg_ids.push_back(submit->request_id);
    actual_speech.push_back(StagePayload::from_value(submit->data)
                                .request.inputs.find("text")->as_str());
  }
  check(actual_speech == expected_speech,
        "native TTS received Markdown presentation tokens or dropped prose");
  check(h.no_submit(h.tts_recv),
        "the horizontal-rule-only span incorrectly started a native TTS leg");
  h.complete(vlm_id, "vlm", true, text_result(full));

  for (size_t i = 0; i < leg_ids.size(); ++i) {
    const std::string pcm(8, static_cast<char>(i + 1));
    Value::Map chunk;
    chunk["audio"] = Value::bin(pcm);
    chunk["index"] = Value(int64_t{0});
    chunk["sample_rate"] = Value(int64_t{24000});
    chunk["channels"] = Value(int64_t{1});
    h.stream(leg_ids[i], "tts", Value(std::move(chunk)));
    Value::Map extra;
    extra["streamed"] = Value(true);
    h.complete(leg_ids[i], "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16(pcm, 24000), extra));
  }
  check(engine.wait(id, 5.0), "Markdown stream did not complete");

  size_t raw_begin = 0;
  size_t speech_index = 0;
  size_t audio_events = 0;
  std::string public_text;
  for (const UnifiedEvent& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) public_text += event.text;
    if (event.kind != UnifiedEvent::Kind::kArtifact) continue;
    ++audio_events;
    while (speech_index + 1 < spans.size() && spans[speech_index] == "---\n") {
      raw_begin += spans[speech_index].size();
      ++speech_index;
    }
    check(speech_index < spans.size(), "audio has no source Markdown span");
    if (speech_index >= spans.size()) break;
    const Value* begin = event.artifact.find("text_cursor");
    const Value* end = event.artifact.find("text_end_cursor");
    const size_t span_end = raw_begin + spans[speech_index].size();
    check(begin != nullptr && end != nullptr &&
              begin->as_int() == static_cast<int64_t>(raw_begin) &&
              end->as_int() > static_cast<int64_t>(raw_begin) &&
              end->as_int() <= static_cast<int64_t>(span_end) &&
              (speech_index >= 4 ||
               end->as_int() == static_cast<int64_t>(span_end)),
          "speech normalization corrupted original playback ACK cursors");
    raw_begin = span_end;
    ++speech_index;
  }
  check(public_text == full, "Markdown was removed from the public prose stream");
  check(audio_events == expected_speech.size() && raw_begin == full.size(),
        "a Markdown span or its source range was dropped");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            terminal->text == full,
        "Markdown request terminal changed public text");
  check(engine.complete_realtime_playback(id, "markdown-spans", full.size()),
        "final playback ACK did not cover the complete original text");
  engine.release(id);

  // A markup-only response has no intended narration leg and must still finish.
  const std::string markup_id = engine.submit(streamed_audio_request());
  auto markup_vlm = h.next_submit(h.vlm_recv);
  check(markup_vlm.has_value(), "markup-only request did not reach VLM");
  h.stream(markup_vlm->request_id, "vlm", text_chunk("---\n", false));
  h.complete(markup_vlm->request_id, "vlm", true, text_result("---\n"));
  check(h.no_submit(h.tts_recv), "markup-only response started native TTS");
  check(engine.wait(markup_id, 5.0),
        "markup-only response did not reach a terminal without native TTS");
  engine.release(markup_id);
}

// Streamed PCM chunks are consumed independently under bounded transport
// backpressure; they are not one terminal aggregate artifact. Deliver 62 MiB
// losslessly while preserving the existing per-chunk and nonstreamed caps.
void test_streamed_tts_long_stream_drains_past_aggregate_cap() {
  Harness h("ttslongstream");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(streamed_audio_request());
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "long stream did not reach VLM");
  const std::string vlm_id = vlm ? vlm->request_id : id + ":vlm";
  h.stream(vlm_id, "vlm", text_chunk("Long narration. ", false));
  auto leg = h.next_submit(h.tts_recv);
  check(leg.has_value(), "long stream did not start TTS");
  const std::string leg_id = leg ? leg->request_id : id + ":tts:0";

  constexpr size_t kChunks = 31;
  constexpr size_t kChunkBytes = 2u << 20;
  std::string expected;
  expected.reserve(kChunks * kChunkBytes);
  for (size_t index = 0; index < kChunks; ++index) {
    const std::string pcm(kChunkBytes, static_cast<char>(index));
    expected += pcm;
    Value::Map chunk;
    chunk["audio"] = Value::bin(pcm);
    chunk["index"] = Value(static_cast<int64_t>(index));
    chunk["sample_rate"] = Value(int64_t{24000});
    chunk["channels"] = Value(int64_t{1});
    h.stream(leg_id, "tts", Value(std::move(chunk)));
  }
  check(!h.saw_abort(leg_id),
        "independent streamed chunks incorrectly exhausted the terminal aggregate cap");
  h.complete(vlm_id, "vlm", true, text_result("Long narration. "));
  Value::Map extra;
  extra["streamed"] = Value(true);
  h.complete(leg_id, "tts", true,
             artifact_result("audio", "audio/wav", audio::wav_from_pcm16("PCMM", 24000),
                             extra));
  check(engine.wait(id, 10.0), "long lossless stream did not settle");

  std::string actual;
  actual.reserve(expected.size());
  size_t artifacts = 0;
  size_t errors = 0;
  for (const UnifiedEvent& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kLegError) ++errors;
    if (event.kind != UnifiedEvent::Kind::kArtifact) continue;
    ++artifacts;
    check(event.artifact_bytes && event.artifact_offset == 0 &&
              event.artifact_length == kChunkBytes,
          "long-stream chunk was sliced, dropped, or coalesced");
    actual.append(event.artifact_bytes->data(), event.artifact_length);
  }
  check(errors == 0 && artifacts == kChunks && actual == expected &&
            util::sha256_hex(actual) == util::sha256_hex(expected),
        "long streamed PCM did not drain byte-for-byte");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
        "long streamed PCM did not end complete");
  engine.release(id);
}

// The 32-leg bound limits active native children, not total narration. Completed
// children release capacity and every later sentence remains a separate leg.
void test_streamed_tts_active_bound_drains_all_sentences() {
  Harness h("ttsactivebound");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  constexpr size_t kSentences = 40;
  std::vector<std::string> expected;
  std::string full;
  for (size_t index = 0; index < kSentences; ++index) {
    expected.push_back("第" + std::to_string(index) + "段。");
    full += expected.back() + "\n";
  }
  const std::string id = engine.submit(streamed_audio_request());
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "active-bound request did not reach VLM");
  h.stream(vlm->request_id, "vlm", text_chunk(full, false));

  std::vector<std::string> pending;
  std::vector<std::string> actual;
  auto collect = [&] {
    while (auto message = h.tts_recv.recv_timeout(20)) {
      if (!std::holds_alternative<SubmitMessage>(*message)) continue;
      const StagePayload payload = StagePayload::from_value(
          std::get<SubmitMessage>(*message).data);
      pending.push_back(std::get<SubmitMessage>(*message).request_id);
      actual.push_back(payload.request.inputs.find("text")->as_str());
    }
  };
  collect();
  check(pending.size() == kMaxStreamedTtsLegs,
        "streamed TTS did not respect its 32-active-child high-water mark");
  h.complete(vlm->request_id, "vlm", true, text_result(full));
  collect();
  check(actual.size() == kMaxStreamedTtsLegs,
        "VLM EOF bypassed the active bound or coalesced a giant final tail");

  size_t settled = 0;
  size_t high_water = pending.size();
  while (!pending.empty()) {
    const std::string child = pending.front();
    pending.erase(pending.begin());
    Value::Map extra;
    extra["streamed"] = Value(true);
    h.complete(child, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16("PCMM", 24000), extra));
    ++settled;
    collect();
    high_water = std::max(high_water, pending.size());
    check(high_water <= kMaxStreamedTtsLegs,
          "completed TTS children did not release bounded capacity");
  }
  check(settled == kSentences && actual == expected,
        "long narration dropped, reordered, or coalesced a sentence leg");
  check(engine.wait(id, 5.0), "bounded long narration did not complete");
  engine.release(id);
}

// A streamed chunk past the per-request audio limit aborts the TTS legs and
// fails the request instead of forwarding the bytes as an artifact.
void test_streamed_tts_chunk_limit_aborts() {
  Harness h("ttslimit");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(streamed_audio_request());
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "the limited request did not reach the VLM");
  const std::string vlm_id = vlm ? vlm->request_id : id;
  h.stream(vlm_id, "vlm", text_chunk("Hello there. ", false));
  auto leg = h.next_submit(h.tts_recv);
  check(leg.has_value(), "the first streamed leg was not submitted");
  const std::string leg_id = leg ? leg->request_id : id + ":tts:0";

  Value::Map chunk;
  chunk["audio"] = Value::bin(std::string(kMaxArtifactAudioBytes + 1, '\x05'));
  chunk["index"] = Value(int64_t{0});
  h.stream(leg_id, "tts", Value(std::move(chunk)));
  check(h.wait_abort_on(h.tts_recv, leg_id),
        "an oversized streamed chunk did not abort the TTS leg");

  h.complete(vlm_id, "vlm", true, text_result("Hello there. "));
  h.complete(leg_id, "tts", false, Value(), std::string("request aborted"));
  check(engine.wait(id, 6.0), "the oversized stream did not settle");

  bool saw_limit = false;
  size_t artifacts = 0;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kLegError &&
        e.error.find("artifact limit") != std::string::npos) {
      saw_limit = true;
    }
    if (e.kind == UnifiedEvent::Kind::kArtifact) ++artifacts;
  }
  check(saw_limit, "the oversized stream did not report an artifact-limit error");
  check(artifacts == 0, "an oversized streamed chunk was forwarded as an artifact");
  const auto terminal = log.terminal_for(id);
  bool tts_failed = false;
  if (terminal && terminal->legs.is_array()) {
    for (const Value& reported : terminal->legs.as_array()) {
      if (reported.find("stage")->as_str() == kTtsStageId) {
        tts_failed = reported.find("state")->as_str() == "failed";
      }
    }
  }
  check(tts_failed, "the terminal does not report the failed TTS leg");
  engine.release(id);
}

// ---- Scheduler batch collection and abort tombstones ---------------------

// A request whose abort arrived before the request is dropped by the tombstone
// and produces no result; the requests behind it still collect into one batch.
void test_simple_scheduler_batch_and_abort_tombstone() {
  std::mutex mu;
  std::vector<std::vector<std::string>> batches;
  auto compute = [](const Value&) { return Value(std::string("single")); };
  auto batch_fn = [&](const std::vector<Value>& inputs) {
    std::vector<std::string> ids;
    std::vector<Value> results;
    for (const Value& input : inputs) {
      ids.push_back(input.as_str());
      results.push_back(Value("b:" + input.as_str()));
    }
    std::lock_guard<std::mutex> lock(mu);
    batches.push_back(std::move(ids));
    return results;
  };
  SimpleScheduler scheduler(compute, batch_fn, /*max_batch_size=*/4,
                            /*max_batch_wait_ms=*/200,
                            /*batch_wait_when_idle=*/true);
  std::thread runner([&] { scheduler.start(); });

  scheduler.abort("gone");
  scheduler.enqueue(
      IncomingMessage{"gone", "new_request", Value(std::string("gone"))});
  for (const char* id : {"a", "b", "c"}) {
    scheduler.enqueue(
        IncomingMessage{id, "new_request", Value(std::string(id))});
  }

  std::map<std::string, std::string> results;
  const auto deadline = std::chrono::steady_clock::now() + kStateWait;
  while (results.size() < 3 && std::chrono::steady_clock::now() < deadline) {
    auto msg = scheduler.outbox().get(0.1);
    if (!msg) continue;
    results[msg->request_id] =
        msg->data.is_str() ? msg->data.as_str() : std::string();
  }
  check(results.size() == 3, "the queued requests did not all answer");
  check(results.count("gone") == 0, "an aborted request produced a result");
  check(results["a"] == "b:a" && results["b"] == "b:b" &&
            results["c"] == "b:c",
        "batch results did not map back to their request ids");
  {
    std::lock_guard<std::mutex> lock(mu);
    check(batches.size() == 1 && batches[0].size() == 3,
          "the waiting requests were not collected into one batch");
  }
  scheduler.stop();
  runner.join();
}

const LegSnapshot* find_leg(const std::vector<LegSnapshot>& legs,
                            const char* stage_id) {
  for (const LegSnapshot& leg : legs) {
    if (leg.stage_id == stage_id) return &leg;
  }
  return nullptr;
}

// Unified 4-modality lifecycle (text + image + audio + video) in one request
// and session: verifies pending -> running -> completed transitions via
// legs_snapshot, prompt rules, artifact slicing, gap-free seq, and cleanup.
void test_mixed_video_lifecycle_and_prompt_rules() {
  Harness h("videolife");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(
      true, true, "explicit image prompt", "Tina", true,
      "explicit video prompt", "sess-all-4"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "no VLM submit for 4-modality request");
  {
    const auto legs = engine.legs_snapshot(id);
    const auto* vlm_leg = find_leg(legs, kVlmStageId);
    const auto* image_leg = find_leg(legs, kImageStageId);
    const auto* tts_leg = find_leg(legs, kTtsStageId);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(vlm_leg && vlm_leg->modality == UnifiedModality::kText &&
              vlm_leg->state == LegState::kRunning,
          "VLM leg should be running before fan-out");
    check(image_leg && image_leg->modality == UnifiedModality::kImage &&
              image_leg->state == LegState::kPending,
          "image leg should be pending before fan-out");
    check(tts_leg && tts_leg->modality == UnifiedModality::kAudio &&
              tts_leg->state == LegState::kPending,
          "tts leg should be pending before fan-out");
    check(video_leg && video_leg->modality == UnifiedModality::kVideo &&
              video_leg->state == LegState::kPending,
          "video leg should be pending before fan-out");
  }

  h.complete(vlm_submit->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  auto video_submit = h.next_submit(h.video_recv);
  check(image_submit && image_submit->request_id == id + ":image",
        "image leg did not get :image child id");
  check(tts_submit && tts_submit->request_id == id + ":tts",
        "tts leg did not get :tts child id");
  check(video_submit && video_submit->request_id == id + ":video",
        "video leg did not get :video child id");
  if (video_submit) {
    StagePayload p = StagePayload::from_value(video_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "explicit video prompt",
          "explicit video prompt did not win over VLM answer");
    check(p.request.inputs.find("size")->as_str() == "1280x720" &&
              p.request.inputs.find("seconds")->as_int() == 4 &&
              p.request.inputs.find("fps")->as_int() == 24 &&
              p.request.inputs.find("output_format")->as_str() == "mp4",
          "video options were not forwarded to video stage");
    check(p.request.inputs.find("messages") == nullptr,
          "video leg received VLM-only messages");
  }
  {
    const auto legs = engine.legs_snapshot(id);
    const auto* vlm_leg = find_leg(legs, kVlmStageId);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(vlm_leg && vlm_leg->state == LegState::kCompleted,
          "VLM leg should be completed after fan-out");
    check(video_leg && video_leg->state == LegState::kRunning,
          "video leg should be running after fan-out");
  }

  // A stray StreamMessage from the video stage must not leak into ASR handling.
  h.stream(id + ":video", "video", text_chunk("progress", false));

  Value::Map video_meta;
  video_meta["format"] = Value("mp4");
  video_meta["size"] = Value("1280x720");
  video_meta["fps"] = Value(int64_t{24});
  video_meta["seconds"] = Value(int64_t{4});
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(10, '\x89')));
  h.complete(id + ":tts", "tts", true,
             artifact_result("audio", "audio/wav", std::string(20, '\x52')));
  h.complete(id + ":video", "video", true,
             artifact_result("video", "video/mp4", std::string(32, '\x66'),
                             std::move(video_meta)));
  check(engine.wait(id, 5.0), "4-modality request did not complete");

  const auto events = log.snapshot();
  int64_t expected_seq = 0;
  size_t artifacts = 0;
  bool saw_video_artifact = false;
  for (const UnifiedEvent& e : events) {
    check(e.seq == expected_seq++, "4-modality event seq has a gap");
    check(e.session_id == "sess-all-4",
          "4-modality event lost explicit session_id");
    check(e.kind != UnifiedEvent::Kind::kAsrPartial &&
              e.kind != UnifiedEvent::Kind::kAsrFinal,
          "video stage stream leaked into ASR event");
    if (e.kind == UnifiedEvent::Kind::kArtifact) {
      ++artifacts;
      if (e.stage_id == kVideoStageId) {
        saw_video_artifact = true;
        check(e.artifact.find("media_type")->as_str() == "video" &&
                  e.artifact.find("mime_type")->as_str() == "video/mp4",
              "video artifact descriptor mismatch");
      }
    }
  }
  check(artifacts == 3 && saw_video_artifact,
        "expected image, audio and video artifacts");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            terminal->legs.is_array() &&
            terminal->legs.as_array().size() == 3,
        "4-modality terminal missing 3-entry legs array");
  {
    const auto legs = engine.legs_snapshot(id);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(video_leg && video_leg->state == LegState::kCompleted,
          "video leg snapshot not completed at terminal");
  }
  check(engine.in_flight() == 0, "4-modality request stayed in flight");
  check(engine.release(id), "release failed for completed video request");
  check(!h.coordinator.info(id + ":video").has_value(),
        "video child record was not released from coordinator");

  // Fallback prompt rule: omitted video.prompt uses aggregated VLM answer.
  const std::string id2 = engine.submit(
      mixed_request(false, false, "", "Tina", true, "", "sess-all-4"));
  auto vlm2 = h.next_submit(h.vlm_recv);
  check(vlm2.has_value(), "no VLM submit for video fallback prompt");
  h.complete(vlm2->request_id, "vlm", true, text_result("fallback video text"));
  auto vid2 = h.next_submit(h.video_recv);
  check(vid2.has_value(), "no video submit for fallback prompt");
  if (vid2) {
    StagePayload p = StagePayload::from_value(vid2->data);
    check(p.request.inputs.find("prompt")->as_str() == "fallback video text",
          "video leg did not fall back to VLM answer");
  }
  h.complete(id2 + ":video", "video", true,
             artifact_result("video", "video/mp4", std::string(16, '\x66')));
  check(engine.wait(id2, 5.0), "fallback video request did not complete");
  engine.release(id2);
}

// Video stage failure and artifact size limit both turn into kLegError while
// preserving delivered artifacts and settling with a single terminal.
void test_mixed_video_failure_and_artifact_limit() {
  Harness h("videofail");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(mixed_request(true, false, "", "Tina", true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "no VLM submit in video failure test");
  h.complete(vlm->request_id, "vlm", true, text_result("scene"));
  check(h.next_submit(h.image_recv).has_value(), "no image submit");
  check(h.next_submit(h.video_recv).has_value(), "no video submit");

  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(8, '\x89')));
  h.complete(id + ":video", "video", false, Value(),
             std::string("video worker crashed"));
  check(engine.wait(id, 5.0), "partial video failure did not settle");

  bool saw_leg_error = false;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kLegError && e.stage_id == kVideoStageId) {
      saw_leg_error = (e.error == "video worker crashed");
    }
  }
  check(saw_leg_error, "missing kLegError for failed video leg");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "video leg failure produced wrong terminal");
  if (terminal && terminal->legs.is_array()) {
    bool video_failed = false;
    bool image_completed = false;
    for (const Value& leg : terminal->legs.as_array()) {
      const std::string stage = leg.find("stage")->as_str();
      const std::string state = leg.find("state")->as_str();
      if (stage == kVideoStageId && state == "failed" &&
          leg.find("error") != nullptr) {
        video_failed = true;
      }
      if (stage == kImageStageId && state == "completed") {
        image_completed = true;
      }
    }
    check(video_failed && image_completed,
          "terminal legs did not report video=failed and image=completed");
  }
  {
    const auto legs = engine.legs_snapshot(id);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(video_leg && video_leg->state == LegState::kFailed,
          "video leg snapshot not failed");
  }
  engine.release(id);

  // Oversized video artifact (> 48 MiB) must fail as kLegError without crashing.
  const std::string id2 =
      engine.submit(mixed_request(false, false, "", "Tina", true));
  auto vlm2 = h.next_submit(h.vlm_recv);
  h.complete(vlm2->request_id, "vlm", true, text_result("big"));
  check(h.next_submit(h.video_recv).has_value(), "no video submit for cap");
  h.complete(id2 + ":video", "video", true,
             artifact_result("video", "video/mp4",
                             std::string(kMaxArtifactVideoBytes + 1, '\x00')));
  check(engine.wait(id2, 5.0), "oversized video request did not settle");
  const auto terminal2 = log.terminal_for(id2);
  check(terminal2 && terminal2->kind == UnifiedEvent::Kind::kComplete &&
            terminal2->legs.is_array() &&
            terminal2->legs.as_array()[0].find("state")->as_str() == "failed",
        "oversized video artifact did not become failed leg");
  engine.release(id2);
}

// Video cancellation enforces session ownership, aborts :video, reports
// aborted state, blocks pending video before fan-out, and cleans resources.
void test_mixed_video_cancel_and_session_ownership() {
  Harness h("videocancel");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(
      mixed_request(false, false, "", "Tina", true, "", "sess-video"));
  auto vlm = h.next_submit(h.vlm_recv);
  h.complete(vlm->request_id, "vlm", true, text_result("clip"));
  check(h.next_submit(h.video_recv).has_value(), "video leg did not start");

  check(engine.cancel_for_session(id, "foreign-session") ==
            UnifiedEngine::CancelStatus::kSessionMismatch,
        "foreign session was allowed to cancel video request");
  check(!h.saw_abort(id + ":video"),
        "foreign session cancel aborted video leg");

  check(engine.cancel_for_session(id, "sess-video") ==
            UnifiedEngine::CancelStatus::kCancelled,
        "owner session cancel failed");
  check(h.wait_abort_on(h.video_recv, id + ":video"),
        "video leg never received abort");
  h.complete(id + ":video", "video", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled video request did not settle");

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
            log.terminal_count(id) == 1,
        "cancelled video request did not produce single kCancelled terminal");
  if (terminal && terminal->legs.is_array()) {
    check(terminal->legs.as_array().size() == 1 &&
              terminal->legs.as_array()[0].find("stage")->as_str() == "video" &&
              terminal->legs.as_array()[0].find("state")->as_str() == "aborted",
          "cancelled terminal did not report video leg as aborted");
  }
  {
    const auto legs = engine.legs_snapshot(id);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(video_leg && video_leg->state == LegState::kAborted,
          "video leg snapshot not aborted");
  }
  check(engine.release(id), "release failed after video cancel");
  check(engine.in_flight() == 0, "in_flight nonzero after video cancel");
  check(!h.coordinator.info(id + ":video").has_value(),
        "video child record leaked after cancel release");

  // Cancel before fan-out: video leg stays unsubmitted and settles as aborted.
  const std::string id2 = engine.submit(
      mixed_request(false, false, "", "Tina", true, "", "sess-video"));
  auto vlm2 = h.next_submit(h.vlm_recv);
  check(vlm2.has_value(), "no VLM submit for pre-fanout cancel");
  check(engine.cancel_for_session(id2, "sess-video") ==
            UnifiedEngine::CancelStatus::kCancelled,
        "pre-fanout cancel failed");
  h.complete(vlm2->request_id, "vlm", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id2, 5.0), "pre-fanout cancelled video did not settle");
  {
    const auto legs = engine.legs_snapshot(id2);
    const auto* video_leg = find_leg(legs, kVideoStageId);
    check(video_leg && video_leg->state == LegState::kAborted,
          "pending video leg was not marked aborted on pre-fanout cancel");
  }
  bool saw_unexpected_video_submit = false;
  while (auto msg = h.video_recv.recv_timeout(100)) {
    if (std::holds_alternative<SubmitMessage>(*msg)) {
      saw_unexpected_video_submit = true;
      break;
    }
  }
  check(!saw_unexpected_video_submit,
        "video leg was submitted despite pre-fanout cancel");
  engine.release(id2);
  check(engine.in_flight() == 0, "in_flight nonzero after pre-fanout cancel");
}

// Public HTTP SSE path for unified video output: chunked base64 + sha256
// reassembly, 400 gates (stream=false / unregistered video_stage), and
// /abort session ownership enforcement.
void test_public_http_mixed_video_stream_and_abort() {
  Harness h("publicvideo");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "video HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string base_url =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  // 1) Streaming 5 MiB fake MP4 video artifact -> 3 base64 chunks + sha256.
  const std::string video_bytes(5u << 20, '\x66');
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", true},
      {"session_id", "http-video-sess"},
      {"modalities", nlohmann::json::array({"text", "video"})},
      {"video", {{"size", "1280x720"}, {"seconds", 4}, {"fps", 24},
                 {"output_format", "mp4"}}}};

  auto response_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = base_url + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "HTTP video request did not reach VLM");
  h.complete(vlm->request_id, "vlm", true, text_result("the video prompt"));
  auto video_submit = h.next_submit(h.video_recv);
  check(video_submit.has_value(), "HTTP video leg did not start");
  if (video_submit) {
    StagePayload p = StagePayload::from_value(video_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "the video prompt",
          "HTTP video leg did not use VLM answer as fallback prompt");
    Value::Map extra;
    extra["format"] = Value("mp4");
    extra["size"] = Value("1280x720");
    extra["fps"] = Value(int64_t{24});
    h.complete(video_submit->request_id, "video", true,
               artifact_result("video", "video/mp4", video_bytes,
                               std::move(extra)));
  }

  const auto response = response_future.get();
  check(response.status == 200, "HTTP video SSE returned error");
  check(occurrences(response.body, "data: [DONE]") == 1,
        "HTTP video SSE missing single [DONE]");
  std::string reassembled;
  std::string digest;
  int64_t reported_bytes = 0;
  int64_t last_seq = -1;
  size_t artifact_frames = 0;
  bool terminal_video_completed = false;
  for (const std::string& payload : sse_data_payloads(response.body)) {
    const json frame = json::parse(payload);
    if (!frame.contains("unified")) continue;
    const json& unified = frame["unified"];
    const int64_t seq = unified["seq"].get<int64_t>();
    check(seq == last_seq + 1, "HTTP video unified seq is not gap-free");
    last_seq = seq;
    check(unified["session_id"].get<std::string>() == "http-video-sess",
          "HTTP video event session_id mismatch");
    const std::string kind = unified["kind"].get<std::string>();
    if (kind == "artifact") {
      ++artifact_frames;
      const json& a = unified["artifact"];
      check(a["media_type"].get<std::string>() == "video" &&
                a["mime_type"].get<std::string>() == "video/mp4",
            "HTTP video artifact descriptor mismatch");
      reassembled += util::base64_decode(a["data"].get<std::string>(),
                                         kArtifactChunkTargetBytes + 16);
      if (a.contains("sha256")) {
        digest = a["sha256"].get<std::string>();
        reported_bytes = a["bytes"].get<int64_t>();
      }
    } else if (kind == "complete") {
      if (unified.contains("legs") && unified["legs"].size() == 1 &&
          unified["legs"][0]["stage"] == "video" &&
          unified["legs"][0]["state"] == "completed") {
        terminal_video_completed = true;
      }
    }
  }
  check(artifact_frames == 3, "expected 3 chunks for 5 MiB video artifact");
  check(reassembled == video_bytes &&
            digest == util::sha256_hex(video_bytes) &&
            reported_bytes == static_cast<int64_t>(video_bytes.size()),
        "reassembled HTTP video artifact did not match original bytes/sha256");
  check(terminal_video_completed,
        "HTTP video terminal did not report video leg completed");
  wait_until([&] { return engine.in_flight() == 0; },
             "HTTP video request stayed in flight");

  // 3) /abort with wrong session_id -> 403; matching session_id -> 200 aborts :video.
  auto abort_future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = base_url + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });
  auto vlm_abort = h.next_submit(h.vlm_recv);
  check(vlm_abort.has_value(), "HTTP abort video request did not reach VLM");
  const std::string http_req_id = vlm_abort->request_id;
  h.complete(http_req_id, "vlm", true, text_result("cancel me"));
  auto vid_abort = h.next_submit(h.video_recv);
  check(vid_abort.has_value(), "HTTP abort video leg did not start");

  {
    test_http::HttpRequest bad_abort;
    bad_abort.url = base_url + "/abort";
    bad_abort.body = nlohmann::json{{"request_id", http_req_id},
                                    {"session_id", "wrong-sess"}}
                         .dump();
    bad_abort.headers.emplace("Content-Type", "application/json");
    bool got_403 = false;
    try {
      test_http::perform(bad_abort);
    } catch (const client::Error& err) {
      got_403 = (err.status() == 403);
    }
    check(got_403, "HTTP /abort with wrong session_id did not return 403");
  }
  {
    test_http::HttpRequest good_abort;
    good_abort.url = base_url + "/abort";
    good_abort.body = nlohmann::json{{"request_id", http_req_id},
                                     {"session_id", "http-video-sess"}}
                          .dump();
    good_abort.headers.emplace("Content-Type", "application/json");
    const auto abort_res = test_http::perform(good_abort);
    check(abort_res.status == 200, "HTTP /abort with owner session failed");
  }
  check(h.wait_abort_on(h.video_recv, http_req_id + ":video"),
        "HTTP /abort did not propagate to :video stage");
  h.complete(http_req_id + ":video", "video", false, Value(),
             std::string("request aborted"));
  const auto aborted_resp = abort_future.get();
  check(aborted_resp.status == 200, "aborted HTTP video SSE failed");
  wait_until([&] { return engine.in_flight() == 0; },
             "aborted HTTP video request stayed in flight");

  api.stop();
  server_thread.join();

  // 4) Server without video_stage registered rejects video output with HTTP 400.
  Harness h2("novideo");
  ApiConfig config2;
  config2.host = "127.0.0.1";
  config2.port = 0;
  config2.model = "qwen3-unified";
  OpenAIApi api2(h2.coordinator, config2);
  UnifiedEngine engine2(h2.coordinator, UnifiedConfig{});
  api2.set_unified_engine(engine2, "vlm");
  check(api2.bind(), "no-video HTTP server did not bind");
  h2.start_pump();
  std::thread server2([&] { api2.listen(); });
  {
    test_http::HttpRequest req;
    req.url = "http://127.0.0.1:" + std::to_string(api2.bound_port()) +
              "/v1/chat/completions";
    req.body = body.dump();
    req.headers.emplace("Content-Type", "application/json");
    bool got_400 = false;
    try {
      test_http::perform(req);
    } catch (const client::Error& err) {
      got_400 = (err.status() == 400);
    }
    check(got_400, "unregistered video_stage did not reject video with 400");
  }
  api2.stop();
  server2.join();
}

// 1) C++ OpenAIClient::chat end-to-end video request, VideoOutput validation,
// and 48 MiB MP4 (64 MiB base64 + SSE envelope > 64 MiB text cap) wire boundary.
void test_openai_client_video_and_wire_limit_boundary() {
  Harness h("clientvideo");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 15.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "client video server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  client::OpenAIConfig cfg;
  cfg.base_url = "http://127.0.0.1:" + std::to_string(api.bound_port()) + "/v1";
  cfg.model = "qwen3-unified";
  cfg.max_attempts = 1;
  client::OpenAIClient client(cfg);

  // Client-side validation checks.
  {
    client::ChatRequest bad;
    bad.messages = {client::Message{"user", "clip", {}}};
    bad.modalities = {"text"};
    bad.stream = true;
    bad.video = client::VideoOutput{};
    bool rejected = false;
    try {
      client.chat(bad);
    } catch (const client::Error& err) {
      rejected = (err.code() == client::ErrorCode::kConfig);
    }
    check(rejected, "OpenAIClient accepted video options without video modality");
  }

  // End-to-end OpenAIClient::chat -> OpenAIApi -> UnifiedEngine -> :video.
  const std::string small_mp4(64 * 1024, '\x66');
  client::ChatRequest req;
  req.messages = {client::Message{"user", "make a clip", {}}};
  req.modalities = {"text", "video"};
  req.stream = true;
  client::VideoOutput vopt;
  vopt.prompt = "sunset over sea";
  vopt.size = "1280x720";
  vopt.seconds = 4;
  vopt.fps = 24;
  vopt.seed = 7;
  vopt.output_format = "mp4";
  req.video = vopt;

  auto resp_future = std::async(std::launch::async, [&] {
    try {
      return std::pair<std::optional<client::ChatResponse>, std::string>{
          client.chat(req), ""};
    } catch (const std::exception& err) {
      return std::pair<std::optional<client::ChatResponse>, std::string>{
          std::nullopt, err.what()};
    }
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "OpenAIClient video request did not reach VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("vlm answer"));
  auto vid = vlm ? h.next_submit(h.video_recv) : std::nullopt;
  check(vid.has_value(), "OpenAIClient video request did not reach video stage");
  if (vid) {
    StagePayload p = StagePayload::from_value(vid->data);
    check(p.request.inputs.find("prompt")->as_str() == "sunset over sea" &&
              p.request.inputs.find("size")->as_str() == "1280x720" &&
              p.request.inputs.find("seconds")->as_int() == 4 &&
              p.request.inputs.find("fps")->as_int() == 24 &&
              p.request.inputs.find("seed")->as_int() == 7 &&
              p.request.inputs.find("output_format")->as_str() == "mp4",
          "OpenAIClient did not forward VideoOutput options to video stage");
    Value::Map extra;
    extra["format"] = Value("mp4");
    extra["size"] = Value("1280x720");
    h.complete(vid->request_id, "video", true,
               artifact_result("video", "video/mp4", small_mp4,
                               std::move(extra)));
  }
  const auto [resp_opt, resp_err] = resp_future.get();
  check(resp_opt.has_value(),
        ("OpenAIClient::chat failed: " + resp_err).c_str());
  if (resp_opt) {
    const client::ChatResponse& resp = *resp_opt;
    check(resp.artifacts.size() == 1 &&
              resp.artifacts[0].media_type == "video" &&
              resp.artifacts[0].mime_type == "video/mp4" &&
              resp.artifacts[0].data == small_mp4 &&
              resp.artifacts[0].sha256 == util::sha256_hex(small_mp4),
          "OpenAIClient::chat did not decode and verify video artifact");
    check(resp.legs.is_array() && resp.legs.size() == 1 &&
              resp.legs[0]["stage"] == "video" &&
              resp.legs[0]["state"] == "completed",
          "OpenAIClient::chat terminal legs missing completed video");
  }
  wait_until([&] { return engine.in_flight() == 0; },
             "OpenAIClient video request stayed in flight");

  api.stop();
  server_thread.join();

  // 48 MiB MP4 wire-limit boundary: 48 MiB decoded == 64 MiB base64, so with
  // SSE/JSON envelopes the wire stream exceeds the 64 MiB text-only cap and
  // requires mixed_output to select kMixedStreamWireBytes (96 MiB).
  httplib::Server mock_server;
  constexpr size_t kChunkBytes = 2u << 20;
  constexpr size_t kChunkCount = 24;  // 24 * 2 MiB == 48 MiB exact
  const std::string one_chunk_raw(kChunkBytes, '\x5a');
  const std::string one_chunk_b64 = util::base64_encode(one_chunk_raw);
  const std::string full_48m_sha =
      util::sha256_hex(std::string(kChunkBytes * kChunkCount, '\x5a'));

  mock_server.Post(
      "/v1/chat/completions",
      [&](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider(
            "text/event-stream",
            [&](size_t, httplib::DataSink& sink) {
              auto write_frame = [&](const json& j) {
                const std::string line = "data: " + j.dump() + "\n\n";
                return sink.write(line.data(), line.size());
              };
              int64_t seq = 0;
              for (size_t i = 0; i < kChunkCount; ++i) {
                json art = {{"artifact_id", "art-0"},
                            {"media_type", "video"},
                            {"mime_type", "video/mp4"},
                            {"encoding", "base64"},
                            {"data", one_chunk_b64},
                            {"index", static_cast<int64_t>(i)},
                            {"total", static_cast<int64_t>(kChunkCount)},
                            {"format", "mp4"}};
                if (i + 1 == kChunkCount) {
                  art["bytes"] =
                      static_cast<int64_t>(kChunkBytes * kChunkCount);
                  art["sha256"] = full_48m_sha;
                }
                json frame = {
                    {"id", "mock-48m"},
                    {"object", "chat.completion.chunk"},
                    {"created", 0},
                    {"model", "qwen3-unified"},
                    {"choices",
                     {{{"index", 0},
                       {"delta", json::object()},
                       {"finish_reason", nullptr}}}},
                    {"unified",
                     {{"kind", "artifact"},
                      {"stage", "video"},
                      {"seq", seq++},
                      {"session_id", "s-48m"},
                      {"is_final", true},
                      {"text", ""},
                      {"artifact", std::move(art)}}}};
                if (!write_frame(frame)) return false;
              }
              json done = {
                  {"id", "mock-48m"},
                  {"object", "chat.completion.chunk"},
                  {"created", 0},
                  {"model", "qwen3-unified"},
                  {"choices",
                   {{{"index", 0},
                     {"delta", json::object()},
                     {"finish_reason", "stop"}}}},
                  {"unified",
                   {{"kind", "complete"},
                    {"stage", "vlm"},
                    {"seq", seq++},
                    {"session_id", "s-48m"},
                    {"is_final", true},
                    {"text", ""},
                    {"legs",
                     {{{"stage", "video"}, {"state", "completed"}}}}}}};
              if (!write_frame(done)) return false;
              static const std::string kDone = "data: [DONE]\n\n";
              sink.write(kDone.data(), kDone.size());
              sink.done();
              return true;
            });
      });
  const int mock_port = mock_server.bind_to_any_port("127.0.0.1");
  check(mock_port > 0, "mock 48 MiB server failed to bind");
  std::thread mock_thread([&] { mock_server.listen_after_bind(); });

  client::OpenAIConfig cfg48;
  cfg48.base_url = "http://127.0.0.1:" + std::to_string(mock_port) + "/v1";
  cfg48.model = "qwen3-unified";
  cfg48.max_attempts = 1;
  client::OpenAIClient client48(cfg48);
  client::ChatRequest req48;
  req48.messages = {client::Message{"user", "max video", {}}};
  req48.modalities = {"text", "video"};
  req48.stream = true;
  std::optional<client::ChatResponse> resp48;
  std::string err48;
  try {
    resp48 = client48.chat(req48);
  } catch (const std::exception& err) {
    err48 = err.what();
  }
  mock_server.stop();
  mock_thread.join();

  check(resp48.has_value() && resp48->artifacts.size() == 1 &&
            resp48->artifacts[0].media_type == "video" &&
            resp48->artifacts[0].data.size() == kChunkBytes * kChunkCount &&
            resp48->artifacts[0].sha256 == full_48m_sha,
        ("OpenAIClient failed 48 MiB base64+envelope video wire boundary: " +
         err48)
            .c_str());
}

// 2) Non-unified mode keeps backwards compatibility for null mixed fields
// (e.g. "video": null), while still rejecting non-null mixed fields with 400.
void test_non_unified_null_video_field_compatible() {
  Harness h("nonunifnull");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "legacy-chat";
  config.entry_stage = "vlm";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  check(api.bind(), "non-unified server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string url =
      "http://127.0.0.1:" + std::to_string(api.bound_port()) +
      "/v1/chat/completions";

  // "video": null (plus null modalities/image/audio) must be accepted.
  auto future = std::async(std::launch::async, [&] {
    test_http::HttpRequest req;
    req.url = url;
    req.body = nlohmann::json{
                   {"model", "legacy-chat"},
                   {"messages", nlohmann::json::array(
                                    {nlohmann::json{{"role", "user"},
                                                    {"content", "hello"}}})},
                   {"stream", false},
                   {"modalities", nullptr},
                   {"image", nullptr},
                   {"audio", nullptr},
                   {"video", nullptr}}
                   .dump();
    req.headers.emplace("Content-Type", "application/json");
    try {
      return std::pair<int, std::string>{test_http::perform(req).status, ""};
    } catch (const client::Error& err) {
      return std::pair<int, std::string>{err.status(), err.what()};
    }
  });
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "non-unified request with video:null did not submit");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("ok"));
  const auto [status_null, err_null] = future.get();
  check(status_null == 200,
        ("non-unified request with video:null returned " +
         std::to_string(status_null) + ": " + err_null)
            .c_str());

  // Non-null "video": {} must still be rejected with 400 in non-unified mode.
  {
    test_http::HttpRequest req;
    req.url = url;
    req.body = nlohmann::json{
                   {"model", "legacy-chat"},
                   {"messages", nlohmann::json::array(
                                    {nlohmann::json{{"role", "user"},
                                                    {"content", "hello"}}})},
                   {"stream", true},
                   {"video", nlohmann::json::object()}}
                   .dump();
    req.headers.emplace("Content-Type", "application/json");
    bool got_400 = false;
    try {
      test_http::perform(req);
    } catch (const client::Error& err) {
      got_400 = (err.status() == 400);
    }
    check(got_400, "non-unified request with non-null video was not rejected");
  }

  api.stop();
  server_thread.join();
}

// 3) Direct UnifiedEngine::submit with inputs.modalities=["video"] must be
// explicitly rejected even when config.video_stage is configured (never
// silently degrade to text-only); output_modalities=["video"] selects output.
void test_direct_engine_video_input_modality_rejected_even_with_video_stage() {
  Harness h("directvidin");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();

  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value("describe video");
  Value::Map inputs;
  inputs["messages"] = Value(Value::Array{Value(std::move(message))});
  inputs["modalities"] = Value(Value::Array{Value("video")});
  payload.request.inputs = Value(std::move(inputs));

  bool rejected = false;
  try {
    engine.submit(std::move(payload));
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected,
        "direct submit with inputs.modalities=[video] silently degraded to text");
  check(engine.in_flight() == 0, "rejected direct submit leaked in_flight");
}

// 4) 60 MiB aggregate boundary: exact 60 MiB succeeds; exceeding 60 MiB fails
// only the offending leg without emitting partial artifacts or aborting others.
void test_mixed_artifact_aggregate_60mib_boundary() {
  Harness h("agg60m");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  // Exact 60 MiB: 40 MiB image + 20 MiB video -> both succeed.
  {
    const std::string id =
        engine.submit(mixed_request(true, false, "", "Tina", true));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "no VLM submit for exact 60 MiB test");
    h.complete(vlm->request_id, "vlm", true, text_result("exact 60"));
    check(h.next_submit(h.image_recv).has_value(), "no image submit");
    check(h.next_submit(h.video_recv).has_value(), "no video submit");

    h.complete(id + ":image", "image", true,
               artifact_result("image", "image/png",
                               std::string(40u << 20, '\x89')));
    h.complete(id + ":video", "video", true,
               artifact_result("video", "video/mp4",
                               std::string(20u << 20, '\x66')));
    check(engine.wait(id, 10.0), "exact 60 MiB request did not settle");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->legs.is_array() && terminal->legs.as_array().size() == 2,
          "exact 60 MiB request did not complete with two legs");
    if (terminal && terminal->legs.is_array()) {
      for (const Value& leg : terminal->legs.as_array()) {
        check(leg.find("state")->as_str() == "completed",
              "exact 60 MiB leg was not completed");
      }
    }
    engine.release(id);
  }

  // Over 60 MiB: 40 MiB image succeeds, 20 MiB + 1 video fails without emitting
  // any partial video artifact and without aborting the concurrent TTS leg.
  {
    const std::string id =
        engine.submit(mixed_request(true, true, "", "Tina", true));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "no VLM submit for >60 MiB test");
    h.complete(vlm->request_id, "vlm", true, text_result("over 60"));
    check(h.next_submit(h.image_recv).has_value(), "no image submit");
    check(h.next_submit(h.tts_recv).has_value(), "no tts submit");
    check(h.next_submit(h.video_recv).has_value(), "no video submit");

    h.complete(id + ":image", "image", true,
               artifact_result("image", "image/png",
                               std::string(40u << 20, '\x89')));
    h.complete(id + ":video", "video", true,
               artifact_result("video", "video/mp4",
                               std::string((20u << 20) + 1, '\x66')));
    // TTS must NOT be aborted by video exceeding the aggregate cap.
    check(!h.saw_abort(id + ":tts"),
          "video aggregate cap overflow aborted concurrent TTS leg");
    h.complete(id + ":tts", "tts", true,
               artifact_result("audio", "audio/wav", std::string(1024, '\x52')));
    check(engine.wait(id, 10.0), "over-60 MiB request did not settle");

    size_t video_artifacts = 0;
    size_t video_leg_errors = 0;
    size_t tts_artifacts = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.request_id != id) continue;
      if (e.kind == UnifiedEvent::Kind::kArtifact && e.stage_id == kVideoStageId) {
        ++video_artifacts;
      }
      if (e.kind == UnifiedEvent::Kind::kLegError && e.stage_id == kVideoStageId) {
        ++video_leg_errors;
      }
      if (e.kind == UnifiedEvent::Kind::kArtifact && e.stage_id == kTtsStageId) {
        ++tts_artifacts;
      }
    }
    check(video_artifacts == 0,
          "failed >60 MiB video leg emitted partial artifact chunks");
    check(video_leg_errors == 1,
          "failed >60 MiB video leg did not emit kLegError");
    check(tts_artifacts == 1,
          "concurrent TTS leg did not deliver its artifact");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->legs.is_array() && terminal->legs.as_array().size() == 3,
          "over-60 MiB terminal missing 3-entry legs array");
    if (terminal && terminal->legs.is_array()) {
      std::map<std::string, std::string> states;
      for (const Value& leg : terminal->legs.as_array()) {
        states[leg.find("stage")->as_str()] = leg.find("state")->as_str();
      }
      check(states["image"] == "completed" && states["tts"] == "completed" &&
                states["video"] == "failed",
            "over-60 MiB terminal leg states mismatch");
    }
    engine.release(id);
    check(engine.in_flight() == 0, "over-60 MiB request stayed in flight");
  }
}

// ---- Section 1/2 offline contracts --------------------------------------
// Control-plane parsing, the stage submit path, the energy VAD, the fixed
// upstream media proxy and the two unified metric gaps. None of these need a
// model, a GPU or a real upstream.

// Every control message the stage and coordinator exchange parses from its
// wire map; a foreign protocol version, an unknown type and a missing type are
// rejected instead of guessed.
void test_control_message_parse_contract() {
  const auto parse = [](Value::Map fields) {
    return parse_message(Value(std::move(fields)));
  };

  {
    Value::Map ref;
    ref["object_id"] = Value("obj-1");
    Value::Map m;
    m["type"] = Value("data_ready");
    m["request_id"] = Value("req-dr");
    m["from_stage"] = Value("asr");
    m["to_stage"] = Value("vlm");
    m["data_ref"] = Value(std::move(ref));
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<DataReadyMessage>(msg),
          "data_ready did not parse");
    check(std::get<DataReadyMessage>(msg).request_id == "req-dr" &&
              std::get<DataReadyMessage>(msg).to_stage == "vlm",
          "data_ready lost its routing fields");
  }
  {
    Value::Map chunk;
    chunk["text"] = Value("hello");
    Value::Map m;
    m["type"] = Value("stream");
    m["request_id"] = Value("req-stream");
    m["from_stage"] = Value("vlm");
    m["chunk"] = Value(std::move(chunk));
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<StreamMessage>(msg), "stream did not parse");
    check(std::get<StreamMessage>(msg).chunk.find("text")->as_str() == "hello",
          "stream lost its chunk");
  }
  {
    Value::Map m;
    m["type"] = Value("complete");
    m["request_id"] = Value("req-complete");
    m["from_stage"] = Value("vlm");
    m["success"] = Value(true);
    m["result"] = Value(text_result("answer"));
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<CompleteMessage>(msg),
          "complete did not parse");
    check(std::get<CompleteMessage>(msg).success,
          "complete lost its success flag");
  }
  {
    Value::Map data;
    data["request_id"] = Value("req-submit");
    Value::Map m;
    m["type"] = Value("submit");
    m["request_id"] = Value("req-submit");
    m["data"] = Value(std::move(data));
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<SubmitMessage>(msg), "submit did not parse");
    check(std::get<SubmitMessage>(msg).request_id == "req-submit",
          "submit lost its request_id");
  }
  {
    Value::Map m;
    m["type"] = Value("abort");
    m["request_id"] = Value("req-abort");
    check(std::holds_alternative<AbortMessage>(parse(std::move(m))),
          "abort did not parse");
  }
  {
    Value::Map m;
    m["type"] = Value("stage_ready");
    m["from_stage"] = Value("tts");
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<StageReadyMessage>(msg),
          "stage_ready did not parse");
    check(std::get<StageReadyMessage>(msg).from_stage == "tts",
          "stage_ready lost its stage name");
    check(!std::get<StageReadyMessage>(msg).to_value().find(
              "native_identity_sha256"),
          "default stage_ready gained an identity field");
  }
  {
    StageReadyMessage ready{"vlm", std::string(64, 'a')};
    const Message msg = parse(ready.to_value().as_map());
    check(std::holds_alternative<StageReadyMessage>(msg) &&
              std::get<StageReadyMessage>(msg).native_identity_sha256 ==
                  std::string(64, 'a'),
          "stage_ready lost its native identity digest");
  }
  {
    Value::Map operation;
    operation["op_id"] = Value("op-1");
    operation["action"] = Value("model_info");
    Value::Map m;
    m["type"] = Value("admin");
    m["operation"] = Value(std::move(operation));
    const Message msg = parse(std::move(m));
    check(std::holds_alternative<AdminMessage>(msg), "admin did not parse");
    check(std::get<AdminMessage>(msg).operation.action == "model_info",
          "admin lost its action");
  }

  const auto rejects = [&](Value::Map m, const char* why) {
    bool threw = false;
    try {
      (void)parse(std::move(m));
    } catch (const ProtocolError&) {
      threw = true;
    }
    check(threw, why);
  };
  Value::Map bad_version;
  bad_version["type"] = Value("abort");
  bad_version["request_id"] = Value("r");
  bad_version["protocol_version"] = Value(int64_t{2});
  rejects(std::move(bad_version), "protocol_version 2 was accepted");
  Value::Map string_version;
  string_version["type"] = Value("abort");
  string_version["request_id"] = Value("r");
  string_version["protocol_version"] = Value("1");
  rejects(std::move(string_version), "a string protocol_version was accepted");
  Value::Map unknown;
  unknown["type"] = Value("nonesuch");
  rejects(std::move(unknown), "an unknown message type was accepted");
  Value::Map missing;
  missing["request_id"] = Value("r");
  rejects(std::move(missing), "a message without a type was accepted");
}

std::string make_pcm_frame(int16_t sample, size_t samples) {
  std::string bytes(samples * 2, '\0');
  for (size_t i = 0; i < samples; ++i) {
    const auto value = static_cast<uint16_t>(sample);
    bytes[i * 2] = static_cast<char>(value & 0xFF);
    bytes[i * 2 + 1] = static_cast<char>((value >> 8) & 0xFF);
  }
  return bytes;
}

// Energy VAD: silence stays silent, a loud frame opens the turn, and exactly
// the configured continuous-silence window closes it. All frames are
// synthetic 10 ms PCM16 at 16 kHz.
void test_energy_vad_speech_transitions() {
  constexpr size_t kFrameSamples = 160;
  constexpr size_t kFrameBytes = kFrameSamples * 2;
  serve_detail::VadConfig config;
  config.frame_ms = 10;
  config.silence_ms = 100;  // ten frames
  serve_detail::EnergyVad vad(config);
  check(vad.frame_bytes() == kFrameBytes, "a 10 ms frame is not 320 bytes");

  const std::string silence = make_pcm_frame(0, kFrameSamples);
  const std::string speech = make_pcm_frame(8000, kFrameSamples);

  std::string warmup;
  for (int i = 0; i < 20; ++i) warmup += silence;
  check(vad.push(warmup.data(), warmup.size()).empty(),
        "silence before speech reported an event");
  check(!vad.speech(), "the VAD started in speech");

  const auto started = vad.push(speech.data(), speech.size());
  check(started.size() == 1 &&
            started[0].event == serve_detail::EnergyVad::Event::kSpeechStarted,
        "a loud frame did not start speech");
  check(!started.empty() && started[0].bytes_consumed == kFrameBytes,
        "speech start did not report its frame boundary");
  check(vad.speech(), "the VAD did not enter speech");

  for (int i = 0; i < 5; ++i) {
    check(vad.push(speech.data(), speech.size()).empty(),
          "a continuing loud frame reported an event");
  }

  // One frame short of the window must not close the turn.
  for (int i = 0; i < 9; ++i) {
    check(vad.push(silence.data(), silence.size()).empty(),
          "an early silent frame ended the turn");
  }
  check(vad.speech(), "the turn ended before the silence window elapsed");

  const auto stopped = vad.push(silence.data(), silence.size());
  check(stopped.size() == 1 &&
            stopped[0].event == serve_detail::EnergyVad::Event::kSpeechStopped,
        "the silence window did not end the turn");
  check(!stopped.empty() && stopped[0].bytes_consumed == kFrameBytes,
        "speech stop was not reported at the window boundary");
  check(!vad.speech(), "the VAD stayed in speech after the stop");
}

// Echo-robust VAD fixtures use deterministic broadband far-end audio and an
// independent two-tone near-end signal. No acoustic or learned-quality claim
// is made: these tests exercise alignment, residual energy and event timing.
std::vector<int16_t> make_echo_reference(size_t samples, uint32_t seed) {
  std::vector<int16_t> out(samples);
  uint32_t state = seed;
  int32_t filtered = 0;
  for (size_t i = 0; i < samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const int32_t noise = static_cast<int32_t>((state >> 16) & 0xffffu) - 32768;
    filtered = (filtered * 3 + noise) / 4;
    const double tone = 2500.0 * std::sin(2.0 * 3.14159265358979323846 *
                                         431.0 * i / 16000.0);
    out[i] = static_cast<int16_t>(std::clamp(
        static_cast<int>(filtered / 3 + static_cast<int>(tone)), -14000, 14000));
  }
  return out;
}

std::vector<int16_t> make_near_end_user(size_t samples) {
  std::vector<int16_t> out(samples);
  for (size_t i = 0; i < samples; ++i) {
    out[i] = static_cast<int16_t>(
        6500.0 * std::sin(2.0 * 3.14159265358979323846 * 733.0 * i / 16000.0) +
        3500.0 * std::sin(2.0 * 3.14159265358979323846 * 1297.0 * i / 16000.0));
  }
  return out;
}

std::string echo_pcm_slice(const std::vector<int16_t>& samples, size_t begin,
                           size_t count) {
  std::string bytes(count * 2, '\0');
  for (size_t i = 0; i < count; ++i) {
    const auto value = static_cast<uint16_t>(samples[begin + i]);
    bytes[i * 2] = static_cast<char>(value & 0xff);
    bytes[i * 2 + 1] = static_cast<char>((value >> 8) & 0xff);
  }
  return bytes;
}

std::vector<int16_t> make_synthetic_near(
    const std::vector<int16_t>& reference,
    const std::vector<int16_t>& user,
    const std::function<size_t(size_t)>& delay_for_frame, double echo_gain,
    int clip_limit = 0, size_t user_first_frame = std::numeric_limits<size_t>::max(),
    size_t user_last_frame = 0) {
  std::vector<int16_t> near(reference.size(), 0);
  for (size_t i = 0; i < reference.size(); ++i) {
    const size_t frame = i / 160;
    const size_t delay = delay_for_frame(frame);
    double value = 0.0;
    if (i >= delay) value += echo_gain * reference[i - delay];
    if (frame >= user_first_frame && frame < user_last_frame) value += user[i];
    if (clip_limit > 0) {
      value = std::clamp(value, static_cast<double>(-clip_limit),
                         static_cast<double>(clip_limit));
    }
    near[i] = static_cast<int16_t>(std::clamp(
        static_cast<int>(std::lround(value)), -32768, 32767));
  }
  return near;
}

struct SyntheticVadResult {
  std::vector<size_t> starts;
  std::vector<size_t> stops;
  size_t max_candidates = 0;
  size_t max_reference_samples = 0;
};

SyntheticVadResult run_synthetic_vad(serve_detail::EchoRobustVad& vad,
                                     const std::vector<int16_t>& reference,
                                     const std::vector<int16_t>& near,
                                     int reference_rate = 16000) {
  check(reference.empty() || reference.size() == near.size() ||
            (reference_rate == 24000 && reference.size() == near.size() * 3 / 2),
        "synthetic VAD input lengths do not match");
  constexpr size_t kNearFrame = 160;
  const size_t reference_frame = static_cast<size_t>(reference_rate / 100);
  SyntheticVadResult result;
  for (size_t frame = 0; frame < near.size() / kNearFrame; ++frame) {
    if (!reference.empty()) {
      const std::string bytes =
          echo_pcm_slice(reference, frame * reference_frame, reference_frame);
      check(vad.push_reference(bytes.data(), bytes.size(), reference_rate),
            "valid synthetic playback reference was rejected");
    }
    const std::string bytes = echo_pcm_slice(near, frame * kNearFrame, kNearFrame);
    for (const auto& event : vad.push(bytes.data(), bytes.size())) {
      if (event.event == serve_detail::EnergyVad::Event::kSpeechStarted) {
        result.starts.push_back(frame);
        check(event.bytes_consumed == bytes.size(),
              "echo VAD speech start did not map to the current input frame");
      } else if (event.event == serve_detail::EnergyVad::Event::kSpeechStopped) {
        result.stops.push_back(frame);
      }
    }
    result.max_candidates =
        std::max(result.max_candidates, vad.last_alignment().candidate_evaluations);
    result.max_reference_samples =
        std::max(result.max_reference_samples, vad.reference_size());
  }
  return result;
}

void test_echo_robust_vad_echo_and_double_talk() {
  serve_detail::VadConfig vad_config;
  vad_config.silence_ms = 100;
  serve_detail::EchoVadConfig echo_config;
  serve_detail::EchoRobustVad vad(vad_config, echo_config);
  check(vad.reference_capacity() == 16000 && vad.max_delay_samples() == 8000,
        "echo VAD defaults do not provide the documented 1 s / 500 ms bounds");

  constexpr size_t kFrames = 100;
  const auto reference = make_echo_reference(kFrames * 160, 0x12345678);
  const auto user = make_near_end_user(kFrames * 160);
  const auto echo = make_synthetic_near(
      reference, user, [](size_t) { return size_t{1168}; }, 0.55);

  // No reference is a strict compatibility fallback: the same echo is speech
  // to the old energy detector, which proves the echo-only fixture is active.
  serve_detail::EchoRobustVad no_reference(vad_config, echo_config);
  const auto fallback =
      run_synthetic_vad(no_reference, {}, echo);
  check(fallback.starts.size() == 1 && fallback.starts[0] == 7,
        "missing playback reference did not preserve energy-VAD behavior");

  const auto echo_only = run_synthetic_vad(vad, reference, echo);
  check(echo_only.starts.empty() && echo_only.stops.empty(),
        "delayed assistant echo caused a false barge-in");
  check(echo_only.max_candidates <= 8050,
        "echo alignment exceeded its per-frame candidate bound");
  check(echo_only.max_reference_samples == 16000,
        "echo reference ring did not reach and stay at its bounded capacity");

  serve_detail::EchoRobustVad double_talk_vad(vad_config, echo_config);
  const auto double_talk = make_synthetic_near(
      reference, user, [](size_t) { return size_t{1457}; }, 0.65, 0, 30, 60);
  const auto simultaneous =
      run_synthetic_vad(double_talk_vad, reference, double_talk);
  check(simultaneous.starts.size() == 1 && simultaneous.starts[0] == 30,
        "genuine double-talk did not interrupt on its first 10 ms frame");
  check(simultaneous.stops.size() == 1 && simultaneous.stops[0] == 69,
        "echo after near-end speech prevented the 100 ms endpoint");

  serve_detail::EchoRobustVad near_only_vad(vad_config, echo_config);
  const auto near_only = make_synthetic_near(
      reference, user, [](size_t) { return size_t{0}; }, 0.0, 0, 30, 60);
  const auto near_result = run_synthetic_vad(near_only_vad, reference, near_only);
  check(near_result.starts.size() == 1 && near_result.starts[0] == 30,
        "unrelated playback suppressed near-end-only speech");

  serve_detail::EchoRobustVad silence_vad(vad_config, echo_config);
  const std::vector<int16_t> silence(kFrames * 160, 0);
  const auto silent = run_synthetic_vad(silence_vad, reference, silence);
  check(silent.starts.empty() && silent.stops.empty(),
        "playback plus digital microphone silence produced a VAD event");

  for (const auto& test_case : std::vector<std::pair<double, int>>{
           {0.08, 0}, {0.35, 0}, {1.4, 0}, {1.0, 4500}}) {
    serve_detail::EchoRobustVad scaled_vad(vad_config, echo_config);
    const auto scaled = make_synthetic_near(
        reference, user, [](size_t) { return size_t{2027}; }, test_case.first,
        test_case.second);
    const auto result = run_synthetic_vad(scaled_vad, reference, scaled);
    check(result.starts.empty(),
          "attenuated, amplified, or moderately clipped echo started a turn");
  }

  // A completed frame split over two calls reports an offset in the second
  // call, preserving RealtimeApi's turn-cut contract for arbitrary chunking.
  serve_detail::EchoRobustVad chunked(vad_config, echo_config);
  const std::string speech = echo_pcm_slice(user, 0, 160);
  check(chunked.push(speech.data(), 100).empty(),
        "a partial near-end frame emitted an early event");
  const auto chunk_events = chunked.push(speech.data() + 100, speech.size() - 100);
  check(chunk_events.size() == 1 && chunk_events[0].bytes_consumed == 220,
        "split-frame speech start offset was not relative to the current push");

  std::cout << "ECHO_VAD_SYNTHETIC echo_only_false_starts=0/100 "
               "double_talk_detection_ms=10 near_only_detection_ms=10 "
               "double_talk_endpoint_ms=100 attenuation_clipping_cases=4/4\n";
}

void test_echo_robust_vad_delay_bounds_and_session_isolation() {
  serve_detail::VadConfig vad_config;
  vad_config.silence_ms = 100;
  serve_detail::EchoVadConfig echo_config;
  constexpr size_t kFrames = 1000;
  const auto reference = make_echo_reference(kFrames * 160, 0x87654321);
  const auto user = make_near_end_user(kFrames * 160);
  const auto delays = std::vector<size_t>{208, 1872, 3984, 7984, 1441};
  const auto jittered = make_synthetic_near(
      reference, user,
      [&](size_t frame) { return delays[(frame / 20) % delays.size()]; }, 0.45);
  serve_detail::EchoRobustVad jitter_vad(vad_config, echo_config);
  const auto begin = std::chrono::steady_clock::now();
  const auto jitter = run_synthetic_vad(jitter_vad, reference, jittered);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
  check(jitter.starts.empty() && jitter.stops.empty(),
        "delay jumps inside the 500 ms search window caused false speech");
  check(jitter.max_candidates <= 8050,
        "jitter search exceeded the bounded candidate count");

  // Beyond the explicit alignment window the mechanism deliberately falls
  // back to energy VAD instead of claiming an uncertain echo match.
  const auto too_late = make_synthetic_near(
      reference, user, [](size_t) { return size_t{8800}; }, 0.6);
  serve_detail::EchoRobustVad outside_vad(vad_config, echo_config);
  const auto outside = run_synthetic_vad(outside_vad, reference, too_late);
  check(outside.starts.size() == 1 && outside.starts[0] == 55,
        "insufficient alignment did not gracefully use the energy fallback");

  // Three independent responses can each be interrupted once. Resetting the
  // reference at each response boundary neither duplicates events nor leaks
  // the preceding response's waveform into the next alignment search.
  serve_detail::EchoRobustVad repeated_vad(vad_config, echo_config);
  size_t repeated_starts = 0;
  size_t repeated_stops = 0;
  for (size_t cycle = 0; cycle < 3; ++cycle) {
    repeated_vad.reset_reference();
    check(repeated_vad.reference_size() == 0,
          "response reset retained stale playback samples");
    const auto response = make_echo_reference(30 * 160, 0x1000 + cycle);
    const auto response_user = make_near_end_user(30 * 160);
    const auto near = make_synthetic_near(
        response, response_user, [](size_t) { return size_t{319}; }, 0.5, 0, 10, 20);
    const auto result = run_synthetic_vad(repeated_vad, response, near);
    check(result.starts == std::vector<size_t>{10} &&
              result.stops == std::vector<size_t>{29},
          "repeated barge-in did not start and stop exactly once per response");
    repeated_starts += result.starts.size();
    repeated_stops += result.stops.size();
  }

  // Old-response audio is intentionally not suppressed after a response reset.
  const auto old_response = make_echo_reference(80 * 160, 0xaaa);
  const auto new_response = make_echo_reference(80 * 160, 0xbbb);
  const auto response_user = make_near_end_user(80 * 160);
  const auto old_echo = make_synthetic_near(
      old_response, response_user, [](size_t) { return size_t{640}; }, 0.5);
  repeated_vad.reset_reference();
  const auto stale = run_synthetic_vad(repeated_vad, new_response, old_echo);
  check(stale.starts.size() == 1 && stale.starts[0] == 4,
        "stale playback reference leaked across response generations");

  serve_detail::EchoRobustVad session_a(vad_config, echo_config);
  serve_detail::EchoRobustVad session_b(vad_config, echo_config);
  const auto isolated_a = run_synthetic_vad(session_a, reference, jittered);
  const auto isolated_b = run_synthetic_vad(session_b, {}, jittered);
  check(isolated_a.starts.empty() && !isolated_b.starts.empty(),
        "playback reference or suppression state leaked across sessions");

  // Native TTS commonly emits 24 kHz. Analytic bandlimited tones provide an
  // independent 16 kHz near-end signal while the wrapper resamples 24 kHz PCM.
  constexpr size_t kResampleFrames = 100;
  std::vector<int16_t> reference_24k(kResampleFrames * 240);
  std::vector<int16_t> near_16k(kResampleFrames * 160, 0);
  const auto analytic = [](double seconds) {
    return 7000.0 * std::sin(2.0 * 3.14159265358979323846 * 397.0 * seconds) +
           4000.0 * std::sin(2.0 * 3.14159265358979323846 * 1153.0 * seconds);
  };
  for (size_t i = 0; i < reference_24k.size(); ++i) {
    reference_24k[i] = static_cast<int16_t>(analytic(i / 24000.0));
  }
  constexpr size_t kDelay = 1168;
  for (size_t i = kDelay; i < near_16k.size(); ++i) {
    near_16k[i] = static_cast<int16_t>(0.6 * analytic((i - kDelay) / 16000.0));
  }
  serve_detail::EchoRobustVad resampled(vad_config, echo_config);
  const auto resample_result =
      run_synthetic_vad(resampled, reference_24k, near_16k, 24000);
  check(resample_result.starts.empty() && resample_result.stops.empty(),
        "24 kHz playback reference did not suppress its 16 kHz echo");
  check(resampled.last_alignment().aligned,
        "24 kHz reference did not produce a final alignment");

  std::cout << "ECHO_VAD_SYNTHETIC delay_jitter_false_starts=0/1000 "
               "jitter_audio_ms=10000 processing_ms="
            << elapsed_ms
            << " max_candidates=" << jitter.max_candidates
            << " out_of_window_fallback_detection_ms=10 repeated_barge_ins="
            << repeated_starts << "/" << repeated_stops
            << " session_isolation=PASS response_reset=PASS resample_24k=PASS\n";
}

// Stage::on_submit and Stage::fail_request over the real control sockets: a
// submit without payload data computes on request.inputs, and a payload whose
// request_id disagrees with its submit fails that request with a non-success
// completion.
void test_stage_submit_fallback_and_failure_completion() {
  IpcDir dir = IpcDir::create("stagesubmit");
  PubSocket aborts(dir.endpoint("abort"));
  aborts.bind();
  PullSocket completions(dir.endpoint("done"), /*bind=*/true);
  completions.start();

  std::mutex mu;
  std::vector<Value> work;
  auto compute = [&](const Value& input) {
    std::lock_guard<std::mutex> lock(mu);
    work.push_back(input);
    return input;
  };
  Stage::Config config;
  config.name = "entry";
  config.recv_endpoint = dir.endpoint("asr");
  config.coordinator_endpoint = dir.endpoint("done");
  config.abort_endpoint = dir.endpoint("abort");
  Stage stage(config, ComputeFn(compute));
  std::thread stage_thread([&] { stage.run(); });
  PushSocket submits(config.recv_endpoint);
  submits.connect();

  Value::Map fallback_inputs;
  fallback_inputs["probe"] = Value("inputs-fallback");
  StagePayload fallback;
  fallback.request_id = "stage-fallback";
  fallback.request.inputs = Value(std::move(fallback_inputs));
  SubmitMessage first;
  first.request_id = "stage-fallback";
  first.data = fallback.to_value();
  submits.send(Message{first});

  StagePayload mismatched;
  mismatched.request_id = "other-id";
  SubmitMessage second;
  second.request_id = "stage-mismatch";
  second.data = mismatched.to_value();
  submits.send(Message{second});

  std::vector<CompleteMessage> done;
  const auto deadline = std::chrono::steady_clock::now() + kStateWait;
  while (done.size() < 2 && std::chrono::steady_clock::now() < deadline) {
    auto msg = completions.recv_timeout(50);
    if (!msg) continue;
    if (auto* complete = std::get_if<CompleteMessage>(&*msg)) {
      done.push_back(*complete);
    }
  }
  check(done.size() == 2, "the stage did not complete both submits");
  if (done.size() == 2) {
    // The id mismatch fails on the receive thread while the valid request runs
    // on the scheduler. Either terminal may therefore arrive first.
    const auto fallback_done = std::find_if(
        done.begin(), done.end(), [](const CompleteMessage& complete) {
          return complete.request_id == "stage-fallback";
        });
    const auto mismatch_done = std::find_if(
        done.begin(), done.end(), [](const CompleteMessage& complete) {
          return complete.request_id == "stage-mismatch";
        });
    check(fallback_done != done.end() && fallback_done->success,
          "the fallback submit did not complete successfully");
    if (fallback_done != done.end()) {
      const Value* probe = fallback_done->result.find("probe");
      check(probe != nullptr && probe->as_str() == "inputs-fallback",
            "a submit without payload.data did not compute on request.inputs");
    }
    check(work.size() == 1 && work[0].find("probe") != nullptr,
          "the scheduler received the wrong fallback input");
    check(mismatch_done != done.end() && !mismatch_done->success,
          "a payload request_id mismatch did not fail its request");
    check(mismatch_done != done.end() && mismatch_done->error &&
              mismatch_done->error->find("differs") != std::string::npos,
          "the failure completion lost its reason");
  }
  stage.stop();
  stage_thread.join();
  submits.close();
  completions.close();
  aborts.close();
}

// The reliable PUSH and low-latency PUB abort channels can both deliver the
// same request id. A Stage must emit one native abort terminal and remain
// usable; the Coordinator's stale-event filtering is not a substitute.
void test_stage_dual_channel_abort_emits_one_terminal() {
  IpcDir dir = IpcDir::create("stagedualabort");
  PubSocket aborts(dir.endpoint("abort"));
  aborts.bind();
  PullSocket completions(dir.endpoint("done"), /*bind=*/true);
  completions.start();
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  CancellableComputeFn compute = [&](const Value& input, const AbortCheck& abort) {
    if (input.as_str() == "block") {
      entered.set_value();
      while (!abort || !abort()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      throw RequestAborted("deterministic dual-channel abort");
    }
    return input;
  };
  Stage::Config config;
  config.name = "video";
  config.recv_endpoint = dir.endpoint("video");
  config.coordinator_endpoint = dir.endpoint("done");
  config.abort_endpoint = dir.endpoint("abort");
  Stage stage(config, std::move(compute));
  std::thread stage_thread([&] { stage.run(); });
  PushSocket controls(config.recv_endpoint);
  controls.connect();
  auto submit = [&](const std::string& id, const std::string& input) {
    StagePayload payload;
    payload.request_id = id;
    payload.request.inputs = Value(input);
    SubmitMessage message;
    message.request_id = id;
    message.data = payload.to_value();
    controls.send(Message{message});
  };
  submit("dual-abort", "block");
  check(entered_future.wait_for(kStateWait) == std::future_status::ready,
        "cancellable Stage request did not enter compute");
  // Let the SUB subscription propagate, then exercise both channels repeatedly.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  for (int index = 0; index < 3; ++index) {
    controls.send(Message{AbortMessage{"dual-abort"}});
    aborts.publish(AbortMessage{"dual-abort"});
  }

  std::vector<CompleteMessage> done;
  auto receive_until = [&](const std::string& id) {
    const auto deadline = std::chrono::steady_clock::now() + kStateWait;
    while (std::chrono::steady_clock::now() < deadline) {
      auto message = completions.recv_timeout(50);
      if (!message) continue;
      if (auto* complete = std::get_if<CompleteMessage>(&*message)) {
        done.push_back(*complete);
        if (complete->request_id == id) return;
      }
    }
    check(false, ("Stage did not settle " + id).c_str());
  };
  receive_until("dual-abort");
  submit("abort-recovery", "recovered");
  receive_until("abort-recovery");
  // Drain delayed PUB deliveries before asserting the raw terminal count.
  const auto quiet_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
  while (std::chrono::steady_clock::now() < quiet_deadline) {
    auto message = completions.recv_timeout(20);
    if (message && std::holds_alternative<CompleteMessage>(*message)) {
      done.push_back(std::get<CompleteMessage>(*message));
    }
  }
  stage.stop();
  stage_thread.join();
  controls.close();
  completions.close();
  aborts.close();

  const size_t abort_terminals = static_cast<size_t>(std::count_if(
      done.begin(), done.end(), [](const CompleteMessage& complete) {
        return complete.request_id == "dual-abort" && !complete.success;
      }));
  check(abort_terminals == 1,
        ("dual-channel abort emitted " + std::to_string(abort_terminals) +
         " native terminals instead of one").c_str());
  const auto recovery = std::find_if(
      done.begin(), done.end(), [](const CompleteMessage& complete) {
        return complete.request_id == "abort-recovery";
      });
  check(recovery != done.end() && recovery->success &&
            recovery->result.as_str() == "recovered",
        "Stage was not usable after duplicate abort delivery");
}

// A running Stage re-registers with a brand-new Coordinator on the same IPC
// endpoints through its periodic stage_ready heartbeat. The gateway (~10 MB)
// restarts; the stage, which in production holds 120+ GiB of GPU weights, keeps
// running and serves the new gateway without a reload.
void test_stage_heartbeat_reregisters_after_coordinator_restart() {
  IpcDir dir = IpcDir::create("stageheartbeat");
  auto compute = [](const Value& input) { return input; };
  Stage::Config config;
  config.name = "entry";
  config.recv_endpoint = dir.endpoint("asr");
  config.coordinator_endpoint = dir.endpoint("done");
  config.abort_endpoint = dir.endpoint("abort");
  config.announce_ready = true;
  config.ready_heartbeat_ms = 50;
  Stage stage(config, ComputeFn(compute));
  std::thread stage_thread([&] { stage.run(); });

  auto run_request = [](Coordinator& coordinator) {
    StagePayload payload;
    Value::Map inputs;
    inputs["probe"] = Value("heartbeat");
    payload.request.inputs = Value(std::move(inputs));
    const std::string id = coordinator.submit(std::move(payload), "entry");
    if (!coordinator.wait(id, static_cast<double>(kStateWait.count()))) {
      return false;
    }
    const auto info = coordinator.info(id);
    const bool completed = info && info->state == "completed";
    coordinator.release(id);
    return completed;
  };

  {
    Coordinator first(dir.endpoint("done"), dir.endpoint("abort"));
    first.register_stage("entry", dir.endpoint("asr"), /*require_ready=*/true);
    first.start();
    std::thread pump([&] { first.run(); });
    wait_until([&] { return first.stage_ready("entry"); },
               "coordinator #1 never saw the stage ready");
    check(run_request(first), "request through coordinator #1 did not complete");
    first.stop();
    pump.join();
  }  // The destructor closes and unbinds the completion and abort endpoints.

  // Same endpoints, new process identity: the stage is still running.
  Coordinator second(dir.endpoint("done"), dir.endpoint("abort"));
  second.register_stage("entry", dir.endpoint("asr"), /*require_ready=*/true);
  second.start();
  std::thread pump([&] { second.run(); });
  wait_until([&] { return second.stage_ready("entry"); },
             "the running stage did not re-register with coordinator #2");
  check(run_request(second), "request through coordinator #2 did not complete");
  second.stop();
  pump.join();

  stage.stop();
  stage_thread.join();
}

// Fixed-upstream media proxy: happy-path forwarding keeps status and body and
// substitutes the upstream model; an upstream 4xx reaches the caller unchanged.
void test_upstream_media_proxy_routes() {
  httplib::Server upstream;
  std::mutex mu;
  std::vector<std::string> seen;
  std::string edit_model;
  auto note = [&](const char* route) {
    std::lock_guard<std::mutex> lock(mu);
    seen.push_back(route);
  };
  const std::string mp4 = std::string("\0\0\0\x18", 4) + "ftypmp42video";
  upstream.Post("/v1/images/generations",
                [&](const httplib::Request& req, httplib::Response& res) {
                  const json body = json::parse(req.body, nullptr, false);
                  note("POST /v1/images/generations");
                  if (body.is_discarded() || !body.is_object() ||
                      body.value("prompt", "") == "teapot") {
                    res.status = 418;
                    res.set_content(
                        R"({"error":{"message":"upstream teapot"}})",
                        "application/json");
                    return;
                  }
                  if (body.value("model", "") != "upstream-image") {
                    res.status = 400;
                    res.set_content(
                        R"({"error":{"message":"model was not replaced"}})",
                        "application/json");
                    return;
                  }
                  res.set_content(R"({"data":[{"b64_json":"aW1n"}]})",
                                  "application/json");
                });
  upstream.Post("/v1/images/edits",
                [&](const httplib::Request& req, httplib::Response& res) {
                  {
                    std::lock_guard<std::mutex> lock(mu);
                    seen.push_back("POST /v1/images/edits");
                    edit_model = req.form.get_field("model");
                  }
                  if (!req.is_multipart_form_data() ||
                      !req.form.has_file("image")) {
                    res.status = 400;
                    res.set_content(
                        R"({"error":{"message":"missing multipart image"}})",
                        "application/json");
                    return;
                  }
                  res.set_content(R"({"data":[{"b64_json":"ZWRpdA=="}]})",
                                  "application/json");
                });
  upstream.Post("/v1/videos",
                [&](const httplib::Request&, httplib::Response& res) {
                  note("POST /v1/videos");
                  res.set_content(
                      R"({"id":"vid-1","object":"video","status":"queued"})",
                      "application/json");
                });
  upstream.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
               [&](const httplib::Request&, httplib::Response& res) {
                 note("GET /v1/videos/{id}");
                 res.set_content(
                     R"({"id":"vid-1","object":"video","status":"completed"})",
                     "application/json");
               });
  upstream.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128})/content)",
               [&](const httplib::Request&, httplib::Response& res) {
                 note("GET /v1/videos/{id}/content");
                 res.set_content(mp4, "video/mp4");
               });
  upstream.Delete(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                  [&](const httplib::Request&, httplib::Response& res) {
                    note("DELETE /v1/videos/{id}");
                    res.set_content(R"({"id":"vid-1","deleted":true})",
                                    "application/json");
                  });

  const int upstream_port = upstream.bind_to_any_port("127.0.0.1");
  check(upstream_port > 0, "the mock upstream did not bind");
  std::thread upstream_thread([&] { upstream.listen_after_bind(); });

  Harness h("upstreamproxy");
  const std::string upstream_base =
      "http://127.0.0.1:" + std::to_string(upstream_port);

  ApiConfig image_config;
  image_config.host = "127.0.0.1";
  image_config.port = 0;
  image_config.model = "local-image";
  image_config.upstream_kind = "sglang";
  image_config.upstream_base_url = upstream_base;
  image_config.upstream_model = "upstream-image";
  image_config.upstream_variant = "image";
  OpenAIApi image_api(h.coordinator, image_config);
  check(image_api.bind(), "the image proxy did not bind");
  std::thread image_thread([&] { image_api.listen(); });

  httplib::Client image_client("127.0.0.1", image_api.bound_port());
  image_client.set_read_timeout(10, 0);
  const auto generated = image_client.Post(
      "/v1/images/generations",
      json{{"model", "local-image"}, {"prompt", "a cat"}}.dump(),
      "application/json");
  check(generated && generated->status == 200,
        "image generation did not forward the upstream 200");
  check(generated && generated->body.find("b64_json") != std::string::npos,
        "image generation lost the upstream body");

  httplib::UploadFormDataItems edit_items = {
      {"image", std::string("img-bytes"), "a.png", "image/png"},
      {"prompt", "edit it", "", "text/plain"},
      {"model", "local-image", "", "text/plain"},
  };
  const auto edited = image_client.Post("/v1/images/edits", edit_items);
  check(edited && edited->status == 200,
        "image edit did not forward the upstream 200");
  {
    std::lock_guard<std::mutex> lock(mu);
    check(edit_model == "upstream-image",
          "image edit did not replace the model with the upstream model");
  }

  const auto teapot = image_client.Post(
      "/v1/images/generations", json{{"prompt", "teapot"}}.dump(),
      "application/json");
  check(teapot && teapot->status == 418,
        "an upstream 4xx did not reach the caller unchanged");
  check(teapot && teapot->body.find("\"type\":\"upstream_error\"") !=
                      std::string::npos,
        "an upstream error lost its error type");

  ApiConfig video_config = image_config;
  video_config.model = "local-video";
  video_config.upstream_model = "upstream-video";
  video_config.upstream_variant = "video";
  OpenAIApi video_api(h.coordinator, video_config);
  check(video_api.bind(), "the video proxy did not bind");
  std::thread video_thread([&] { video_api.listen(); });

  httplib::Client video_client("127.0.0.1", video_api.bound_port());
  video_client.set_read_timeout(10, 0);
  const auto created = video_client.Post(
      "/v1/videos", json{{"model", "local-video"}, {"prompt", "a river"}}.dump(),
      "application/json");
  check(created && created->status == 200 &&
            created->body.find("\"id\":\"vid-1\"") != std::string::npos,
        "video create did not forward the upstream job");
  const auto video_status = video_client.Get("/v1/videos/vid-1");
  check(video_status && video_status->status == 200 &&
            video_status->body.find("completed") != std::string::npos,
        "video status did not forward the upstream job state");
  const auto content = video_client.Get("/v1/videos/vid-1/content");
  check(content && content->status == 200 &&
            content->body.find("ftypmp42") != std::string::npos,
        "video content did not stream the upstream bytes");
  check(content &&
            content->get_header_value("Content-Type").rfind("video/mp4", 0) == 0,
        "video content lost its media type");
  const auto removed = video_client.Delete("/v1/videos/vid-1");
  check(removed && removed->status == 200 &&
            removed->body.find("\"deleted\":true") != std::string::npos,
        "video delete did not forward the upstream response");

  {
    std::lock_guard<std::mutex> lock(mu);
    for (const char* route :
         {"POST /v1/images/generations", "POST /v1/images/edits",
          "POST /v1/videos", "GET /v1/videos/{id}",
          "GET /v1/videos/{id}/content", "DELETE /v1/videos/{id}"}) {
      bool found = false;
      for (const std::string& entry : seen) found = found || entry == route;
      check(found, "a proxy route never reached the upstream");
    }
  }

  image_api.stop();
  video_api.stop();
  image_thread.join();
  video_thread.join();
  upstream.stop();
  upstream_thread.join();
}

// Reads a rendered Prometheus counter or gauge value. Returns 0 when the
// series is absent, which is the correct reading for a counter never touched
// and for a gauge the server never set.
size_t metric_value(const std::string& text, const std::string& prefix) {
  const size_t at = text.find(prefix);
  if (at == std::string::npos) return 0;
  return static_cast<size_t>(
      std::strtoull(text.c_str() + at + prefix.size(), nullptr, 10));
}

// Metrics gaps: a completion that races a cancel counts the primary leg as
// aborted, and releasing a non-terminal record drops the admission gauge.
void test_unified_metrics_cancel_race_and_release_gauge() {
  const std::string leg_prefix =
      "umm_leg_requests_total{leg=\"vlm\",request_type=\"chat\",status=\"aborted\"} ";
  const std::string gauge_prefix =
      "umm_queue_depth{queue=\"unified_admission\"} ";
  const size_t aborted_before =
      metric_value(obs::Metrics::instance().render(), leg_prefix);

  Harness h("metrics");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(image_part_request());
  check(h.next_submit(h.vlm_recv).has_value(),
        "the metrics request never reached the VLM stage");
  check(metric_value(obs::Metrics::instance().render(), gauge_prefix) == 1,
        "admission was not published at submit");

  check(engine.cancel(id), "cancel rejected the active request");
  // The stage completes after the cancel: the terminal is still kCancelled and
  // the primary leg must be recorded as aborted.
  h.complete(id, "vlm", true, text_result("late answer"));
  wait_until([&] { return log.has_terminal(id); },
             "the cancel-race completion produced no terminal");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
        "a completion that raced a cancel was not aborted");

  engine.release(id);
  wait_until([&] { return engine.in_flight() == 0; },
             "the released request kept its admission slot");
  check(metric_value(obs::Metrics::instance().render(), gauge_prefix) == 0,
        "release did not refresh the admission gauge");
  const size_t aborted_after =
      metric_value(obs::Metrics::instance().render(), leg_prefix);
  check(aborted_after == aborted_before + 1,
        "the cancel-race completion did not record the primary leg as aborted");
}

// Wave 7 HTTP last-mile passthrough + dynamic TTS chunk metadata:
//   - image.steps & image.image data URL reach the image stage;
//   - video.num_frames, video.steps & video.guidance_scale reach the video stage;
//   - audio.ref_audio & /v1/audio/speech ref_audio decode WAV into binary inputs;
//   - streamed TTS chunk sample_rate/channels propagate dynamically.
void test_public_http_wave7_media_passthrough_and_tts_chunk_meta() {
  Harness h("wave7media");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "wave7 server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  auto post_json = [&](const std::string& path, const nlohmann::json& body) {
    test_http::HttpRequest req;
    req.url = base + path;
    req.body = body.dump();
    req.headers.emplace("Content-Type", "application/json");
    req.timeout = client::Milliseconds{8000};
    try {
      return test_http::perform(req);
    } catch (const client::Error& err) {
      test_http::HttpResponse out;
      out.status = err.status();
      out.body = err.what();
      return out;
    }
  };

  const std::string sample_pcm(64, '\x11');
  const std::string sample_wav = audio::wav_from_pcm16(sample_pcm, 24000);
  const std::string wav_b64 = util::base64_encode(sample_wav);
  const std::string wav_data_url = "data:audio/wav;base64," + wav_b64;
  const std::string img_data_url =
      "data:image/png;base64," + util::base64_encode("PNGDATA");
  const std::string vid_img_data_url =
      "data:image/jpeg;base64," + util::base64_encode("VIDFRAME");

  // 1. Mixed chat with image.steps/image.image, video.num_frames/steps/guidance_scale,
  //    and audio.ref_audio forwards every field intact.
  {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"},
                                         {"content", "remix"}}})},
        {"stream", false},
        {"modalities", nlohmann::json::array({"text", "image", "audio", "video"})},
        {"image",
         {{"prompt", "lantern"},
          {"size", "1024x1024"},
          {"steps", 4},
          {"seed", 42},
          {"image", img_data_url},
          {"output_format", "png"}}},
        {"audio",
         {{"voice", "clone"},
          {"format", "wav"},
          {"ref_audio", wav_data_url}}},
        {"video",
         {{"prompt", "waves"},
          {"size", "704x480"},
          {"seconds", 2},
          {"fps", 16},
          {"num_frames", 17},
          {"steps", 15},
          {"guidance_scale", 4.5},
          {"seed", 7},
          {"image", vid_img_data_url},
          {"output_format", "mp4"}}}};
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/chat/completions", body);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "wave7 mixed request did not reach VLM");
    if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("ok"));

    auto img = h.next_submit(h.image_recv);
    check(img.has_value(), "wave7 image leg not submitted");
    if (img) {
      StagePayload p = StagePayload::from_value(img->data);
      const Value& in = p.request.inputs;
      check(in.find("steps") && in.find("steps")->as_int() == 4,
            "image.steps did not reach image stage");
      check(in.find("image") && in.find("image")->as_str() == img_data_url,
            "image.image data URL did not reach image stage");
      h.complete(img->request_id, "image", true,
                 artifact_result("image", "image/png", "PNG1"));
    }

    auto tts = h.next_submit(h.tts_recv);
    check(tts.has_value(), "wave7 tts leg not submitted");
    if (tts) {
      StagePayload p = StagePayload::from_value(tts->data);
      const Value& in = p.request.inputs;
      check(in.find("ref_audio") && in.find("ref_audio")->is_bin() &&
                in.find("ref_audio")->as_bin() == sample_wav,
            "audio.ref_audio binary WAV did not reach tts stage");
      h.complete(tts->request_id, "tts", true,
                 artifact_result("audio", "audio/wav", sample_wav));
    }

    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "wave7 video leg not submitted");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      const Value& in = p.request.inputs;
      check(in.find("num_frames") && in.find("num_frames")->as_int() == 17,
            "video.num_frames did not reach video stage");
      check(in.find("steps") && in.find("steps")->as_int() == 15,
            "video.steps did not reach video stage");
      check(in.find("guidance_scale") &&
                in.find("guidance_scale")->as_double() == 4.5,
            "video.guidance_scale did not reach video stage");
      check(in.find("image") && in.find("image")->as_str() == vid_img_data_url,
            "video.image data URL did not reach video stage");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4DATA"));
    }

    const auto resp = fut.get();
    check(resp.status == 200, "wave7 mixed request did not return 200");
  }

  // 2. POST /v1/audio/speech accepts ref_audio (raw base64 WAV) and rejects bad WAV.
  {
    const auto bad = post_json("/v1/audio/speech",
                               {{"model", "qwen3-unified"},
                                {"input", "hello"},
                                {"ref_audio", util::base64_encode("not-a-wav")}});
    check(bad.status == 400, "speech must reject invalid ref_audio WAV");

    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/audio/speech",
                       {{"model", "qwen3-unified"},
                        {"input", "clone me"},
                        {"voice", "clone"},
                        {"ref_audio", wav_b64}});
    });
    auto tts = h.next_submit(h.tts_recv);
    check(tts.has_value(), "speech ref_audio did not reach tts stage");
    if (tts) {
      StagePayload p = StagePayload::from_value(tts->data);
      const Value& in = p.request.inputs;
      check(in.find("ref_audio") && in.find("ref_audio")->is_bin() &&
                in.find("ref_audio")->as_bin() == sample_wav,
            "speech ref_audio binary WAV did not reach tts stage");
      h.complete(tts->request_id, "tts", true,
                 artifact_result("audio", "audio/wav", sample_wav));
    }
    const auto resp = fut.get();
    check(resp.status == 200 && resp.body == sample_wav,
          "speech with ref_audio did not return WAV");
  }

  // 3. Invalid image.steps / image.image / video.steps / video.guidance_scale -> 400.
  {
    auto bad_chat = [&](nlohmann::json patch) {
      nlohmann::json body = {
          {"model", "qwen3-unified"},
          {"messages", nlohmann::json::array(
                           {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
          {"stream", false}};
      for (auto it = patch.begin(); it != patch.end(); ++it) {
        body[it.key()] = it.value();
      }
      return post_json("/v1/chat/completions", body).status;
    };
    check(bad_chat({{"image", {{"steps", 0}}}}) == 400,
          "image.steps=0 must return 400");
    check(bad_chat({{"image", {{"image", "http://example.com/a.png"}}}}) == 400,
          "image.image http URL must return 400");
    check(bad_chat({{"video", {{"num_frames", 0}}}}) == 400,
          "video.num_frames=0 must return 400");
    check(bad_chat({{"video", {{"steps", 101}}}}) == 400,
          "video.steps=101 must return 400");
    check(bad_chat({{"video", {{"guidance_scale", -1.0}}}}) == 400,
          "negative video.guidance_scale must return 400");
    check(bad_chat({{"video", {{"image", "http://example.com/f.png"}}}}) == 400,
          "video.image http URL must return 400");
    check(bad_chat({{"video", {{"image", 42}}}}) == 400,
          "video.image non-string must return 400");
  }

  // 4. Streamed TTS chunk dynamically propagates sample_rate and channels.
  {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
        {"stream", true},
        {"modalities", nlohmann::json::array({"text", "audio"})},
        {"audio", {{"voice", "vivian"}, {"format", "wav"}, {"stream", true}}}};
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/chat/completions", body);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "streamed tts meta test did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm", text_chunk("Hello world.", false));
      h.complete(vlm->request_id, "vlm", true, text_result("Hello world."));
    }
    auto tts = h.next_submit(h.tts_recv);
    check(tts.has_value(), "streamed tts meta test did not reach TTS");
    if (tts) {
      Value::Map chunk;
      chunk["audio"] = Value::bin(std::string(32, '\x01'));
      chunk["index"] = Value(int64_t{0});
      chunk["sample_rate"] = Value(int64_t{48000});
      chunk["channels"] = Value(int64_t{2});
      h.stream(tts->request_id, "tts", Value(std::move(chunk)));
      Value::Map extra;
      extra["streamed"] = Value(true);
      h.complete(tts->request_id, "tts", true,
                 artifact_result("audio", "audio/wav", sample_wav,
                                 std::move(extra)));
    }
    const auto resp = fut.get();
    check(resp.status == 200 &&
              resp.body.find("\"sample_rate\":48000") != std::string::npos &&
              resp.body.find("\"channels\":2") != std::string::npos,
          "streamed TTS chunk did not propagate dynamic sample_rate/channels");
  }

  // 5. C++ OpenAIClient::speech and OpenAIClient::chat Wave 9 struct parity.
  {
    client::OpenAIConfig cfg;
    cfg.base_url = base + "/v1";
    cfg.model = "qwen3-unified";
    cfg.timeout = client::Milliseconds{8000};
    client::OpenAIClient cli(cfg);

    // OpenAIClient::speech with ref_audio
    {
      client::SpeechRequest sreq;
      sreq.input = "hello clone";
      sreq.voice = "clone";
      sreq.response_format = "wav";
      sreq.ref_audio = wav_data_url;
      auto fut = std::async(std::launch::async, [&] { return cli.speech(sreq); });
      auto tts = h.next_submit(h.tts_recv);
      check(tts.has_value(), "OpenAIClient::speech did not reach tts stage");
      if (tts) {
        StagePayload p = StagePayload::from_value(tts->data);
        check(p.request.inputs.find("ref_audio") &&
                  p.request.inputs.find("ref_audio")->as_bin() == sample_wav,
              "OpenAIClient::speech ref_audio binary WAV mismatch");
        h.complete(tts->request_id, "tts", true,
                   artifact_result("audio", "audio/wav", sample_wav));
      }
      check(fut.get() == sample_wav, "OpenAIClient::speech WAV bytes mismatch");
    }

    // OpenAIClient::chat with ImageOutput{steps,image}, AudioOutput{ref_audio},
    // VideoOutput{num_frames,steps,guidance_scale}
    {
      client::ChatRequest creq;
      creq.messages.push_back({"user", "all"});
      creq.modalities = {"text", "image", "audio", "video"};
      client::ImageOutput io;
      io.prompt = "lantern";
      io.size = "512x512";
      io.steps = 4;
      io.image = img_data_url;
      creq.image = io;
      client::AudioOutput ao;
      ao.voice = "clone";
      ao.ref_audio = wav_b64;
      creq.audio = ao;
      client::VideoOutput vo;
      vo.prompt = "waves";
      vo.size = "480x320";
      vo.num_frames = 13;
      vo.steps = 15;
      vo.guidance_scale = 5.0;
      creq.video = vo;
      auto fut = std::async(std::launch::async, [&] { return cli.chat(creq); });
      if (auto vlm = h.next_submit(h.vlm_recv)) {
        h.complete(vlm->request_id, "vlm", true, text_result("ok"));
      }
      if (auto img = h.next_submit(h.image_recv)) {
        StagePayload p = StagePayload::from_value(img->data);
        check(p.request.inputs.find("steps")->as_int() == 4 &&
                  p.request.inputs.find("image")->as_str() == img_data_url,
              "OpenAIClient::chat image fields mismatch");
        h.complete(img->request_id, "image", true,
                   artifact_result("image", "image/png", "PNG1"));
      }
      if (auto tts = h.next_submit(h.tts_recv)) {
        StagePayload p = StagePayload::from_value(tts->data);
        check(p.request.inputs.find("ref_audio")->as_bin() == sample_wav,
              "OpenAIClient::chat audio.ref_audio mismatch");
        h.complete(tts->request_id, "tts", true,
                   artifact_result("audio", "audio/wav", sample_wav));
      }
      if (auto vid = h.next_submit(h.video_recv)) {
        StagePayload p = StagePayload::from_value(vid->data);
        check(p.request.inputs.find("num_frames")->as_int() == 13 &&
                  p.request.inputs.find("steps")->as_int() == 15 &&
                  p.request.inputs.find("guidance_scale")->as_double() == 5.0,
              "OpenAIClient::chat video fields mismatch");
        h.complete(vid->request_id, "video", true,
                   artifact_result("video", "video/mp4", "MP4DATA"));
      }
      const client::ChatResponse cresp = fut.get();
      check(cresp.artifacts.size() == 3, "OpenAIClient::chat expected 3 artifacts");
    }
  }

  api.stop();
  server_thread.join();
  wait_until([&]() -> bool { return engine.in_flight() == 0; },
             "wave7 requests stayed in flight");
}
// ---- Interleaved narrative ------------------------------------------------

// A VLM-only request with the narrative object attached.
StagePayload narrative_request(bool want_image, bool want_video,
                               bool want_audio,
                               const std::string& visual_mode, int max_slots,
                               bool audio_stream = false) {
  StagePayload payload =
      mixed_request(want_image, want_audio, "", "Tina", want_video);
  Value::Map inputs = payload.request.inputs.as_map();
  if (want_audio && audio_stream) {
    Value::Map options = inputs.at("audio_options").as_map();
    options["stream"] = Value(true);
    inputs["audio_options"] = Value(std::move(options));
  }
  Value::Map narrative;
  narrative["enabled"] = Value(true);
  narrative["visual_mode"] = Value(visual_mode);
  narrative["max_slots"] = Value(static_cast<int64_t>(max_slots));
  inputs["narrative"] = Value(std::move(narrative));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

size_t count_kind(EventLog& log, UnifiedEvent::Kind kind) {
  size_t n = 0;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == kind) ++n;
  }
  return n;
}

// Parser unit contract: split tokens, stray brackets, unknown tags, UTF-8,
// empty prompts and EOF flush.
void test_narrative_parser_unit() {
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    std::string prose = parser.feed("Hello [[ima", out);
    check(prose == "Hello " && out.empty(),
          "parser did not hold a split directive prefix");
    prose = parser.feed("ge: a red fox]] world", out);
    check(prose == " world", "parser prose around a split directive is wrong");
    check(out.size() == 1 && out[0].kind == "image" &&
              out[0].prompt == "a red fox" && out[0].prose_offset == 6,
          "split image directive was not parsed");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    check(parser.feed("a [b] c", out) == "a [b] c",
          "a single stray bracket changed the prose");
    check(out.empty(), "a single bracket produced a directive");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    const std::string prose = parser.feed("[[[image: cat]]]", out);
    check(prose == "[]" && out.size() == 1 && out[0].prompt == "cat",
          "overlapping nested brackets were not handled");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    check(parser.feed("[[note: x]] done", out) == "[[note: x]] done" &&
              out.empty(),
          "an unknown tag was treated as a directive");
    check(parser.feed("[[IMAGE: upper]]", out) == "" && out.size() == 1 &&
              out[0].kind == "image" && out[0].prompt == "upper",
          "tag case insensitivity is wrong");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    check(parser.feed("[[image:   ]]", out).empty() && out.empty(),
          "an empty directive prompt was not dropped");
    check(parser.feed("[[image: unclosed", out).empty() && out.empty(),
          "an unclosed directive leaked into prose before EOF");
    const std::string tail = parser.flush(out);
    check(tail.empty() && out.empty(),
          "EOF leaked an unterminated reserved directive into prose/TTS");
    check(parser.removed_spans().size() == 2 &&
              parser.removed_spans()[1].begin == std::string("[[image:   ]]").size() &&
              parser.removed_spans()[1].end ==
                  std::string("[[image:   ]]").size() +
                      std::string("[[image: unclosed").size(),
          "EOF did not bind the removed directive span after the earlier empty tag");
  }
  {
    // UTF-8 prose split inside a multibyte sequence plus English prompts.
    const std::string chinese = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C";  // 你好，世界
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    std::string prose = parser.feed(chinese.substr(0, 4), out);
    prose += parser.feed(chinese.substr(4), out);
    check(prose == chinese && out.empty(),
          "UTF-8 prose was corrupted by the parser");
    prose = parser.feed(" [[visual: a red fox in snow]] />", out);
    // 15 UTF-8 bytes of Chinese prose plus the space before the directive.
    check(prose == "  />" && out.size() == 1 && out[0].kind == "visual" &&
              out[0].prompt == "a red fox in snow" && out[0].prose_offset == 16,
          "visual directive after UTF-8 prose is wrong");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    const std::string prose =
        parser.feed("A[[visual: one]]B[[video: two]][[image: three]]C", out);
    check(prose == "ABC" && out.size() == 3 &&
              out[0].kind == "visual" && out[1].kind == "video" &&
              out[2].kind == "image" && out[0].prose_offset == 1 &&
              out[1].prose_offset == 2 && out[2].prose_offset == 2,
          "multiple directives did not keep prose order and offsets");
    check(parser.flush(out).empty() && out.size() == 3,
          "flush after a complete stream produced extra output");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    const std::string prose = parser.feed(
        "Before [[image: still]] middle "
        "[[video: camera rises | source=slot-0]] after",
        out);
    check(prose == "Before  middle  after" && out.size() == 2 &&
              out[1].prompt == "camera rises" &&
              out[1].source_slot == "slot-0" &&
              out[1].reference_error.empty(),
          "explicit image-slot reference was not parsed");
  }
  {
    NarrativeStreamParser parser;
    std::vector<NarrativeDirective> out;
    parser.feed("[[video: one | source=slot-0 | source=slot-0]]"
                "[[video: two | source=bogus]]"
                "[[video: three | source slot-0]]"
                "[[video: a literal | pipe remains]]",
                out);
    check(out.size() == 4 && out[0].reference_error == "duplicate source reference" &&
              out[1].reference_error == "source reference must name slot-N" &&
              out[2].reference_error == "source reference must use source=slot-N" &&
              out[3].source_slot.empty() && out[3].reference_error.empty() &&
              out[3].prompt == "a literal | pipe remains",
          "malformed or duplicate source references did not fail closed");
  }
}

void test_narrative_streaming_opening_director_prompt() {
  const std::string prompt =
      unified_detail::narrative_director_prompt("auto", 2, "i2v_chain");
  check(prompt.find("explicitly requests both a still image/diagram") ==
                std::string::npos &&
            prompt.find("one concise opening sentence") == std::string::npos &&
            prompt.find("Do not shorten or outline the remaining answer") ==
                std::string::npos,
        "the NOT_PROMOTED Task #87 opening candidate remains in default runtime");
  check(prompt.find("pure math, code, logic proofs") != std::string::npos &&
            prompt.find("Only emit visual markers when a visual genuinely") !=
                std::string::npos &&
            prompt.find("source=slot-N") != std::string::npos,
        "restoring the director removed accepted routing or source guardrails");
}

// A concise opening reaches TTS before fragmented visual directives close. The
// image admits immediately; the explicit I2V slot remains pending until exact
// source bytes settle, without leaking either directive into prose or speech.
void test_narrative_streaming_opening_slots_and_tts() {
  Harness h("narrativeopening");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  StagePayload request = narrative_request(true, true, true, "auto", 2, true);
  request.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
  const std::string id = engine.submit(std::move(request));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "opening request did not reach the VLM");
  if (!vlm) return;

  h.stream(vlm->request_id, "vlm", text_chunk("Opening sentence.", false));
  auto tts = h.next_submit(h.tts_recv);
  check(tts.has_value(), "the concise opening was not submitted immediately to TTS");
  if (tts) {
    StagePayload payload = StagePayload::from_value(tts->data);
    check(payload.request.inputs.find("text")->as_str() == "Opening sentence.",
          "the first TTS leg is not exactly the concise opening");
  }

  h.stream(vlm->request_id, "vlm", text_chunk("\n[[im", false));
  check(count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 0,
        "a split image prefix admitted a false visual slot");
  h.stream(vlm->request_id, "vlm",
           text_chunk("age: labeled accretion disk structure]]", false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
             "the opening image did not admit mid-stream");
  auto image = h.next_submit(h.image_recv);
  check(image.has_value() && image->request_id == id + ":slot:0",
        "the opening image slot was not submitted immediately");

  h.stream(vlm->request_id, "vlm",
           text_chunk("\n[[video: animate accretion flow | source=slot-0]]\n", false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 2; },
             "the anchored dynamic slot did not admit mid-stream");
  check(h.no_submit(h.video_recv),
        "I2V started before its exact source image completed");

  bool opening_before_slots = false;
  bool clean = true;
  int64_t opening_seq = -1;
  int64_t image_seq = -1;
  int64_t video_seq = -1;
  for (const UnifiedEvent& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
      if (event.text.find("Opening sentence.") != std::string::npos) {
        opening_seq = event.seq;
      }
      clean = clean && event.text.find("[[") == std::string::npos &&
              event.text.find("source=slot") == std::string::npos;
    } else if (event.kind == UnifiedEvent::Kind::kVisualSlot) {
      if (event.slot_id == "slot-0") image_seq = event.seq;
      if (event.slot_id == "slot-1") {
        video_seq = event.seq;
        clean = clean && event.anchor_slot_index == 0 && event.anchor_explicit;
      }
    }
  }
  opening_before_slots = opening_seq >= 0 && opening_seq < image_seq &&
                         image_seq < video_seq;
  check(opening_before_slots && clean,
        "opening/slot ordering, anchor metadata, or clean text regressed");

  const std::string png = "exact-source-png";
  h.complete(id + ":slot:0", "image", true,
             artifact_result("image", "image/png", png));
  auto video = h.next_submit(h.video_recv);
  check(video.has_value() && video->request_id == id + ":slot:1",
        "the deferred I2V slot did not start after its source completed");
  if (video) {
    StagePayload payload = StagePayload::from_value(video->data);
    check(payload.request.inputs.find("image")->as_bin() == png &&
              payload.request.inputs.find("anchor_explicit")->as_bool() &&
              payload.request.inputs.find("anchor_slot_index")->as_int() == 0 &&
              payload.request.inputs.find("anchor_sha256")->as_str() ==
                  util::sha256_hex(png),
          "I2V did not bind the exact slot-0 bytes and digest");
  }

  h.complete(vlm->request_id, "vlm", true, text_result("Opening sentence."));
  h.complete(id + ":slot:1", "video", true,
             artifact_result("video", "video/mp4", "VIDBYTES"));
  if (tts) {
    Value::Map extra;
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    extra["streamed"] = Value(true);
    h.complete(tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav", std::string(8, '\x02'), extra));
  }
  check(engine.wait(id, 5.0), "the streaming opening request did not complete");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "streaming opening produced a false or duplicate terminal");
  engine.release(id);
}

// Interleaved narrative with one image and one video slot, plus global
// streaming TTS. Prose stays clean; slots dispatch mid-stream.
void test_narrative_interleaved_slots_and_streaming_tts() {
  Harness h("narrativeslots");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(narrative_request(true, true, true, "auto", 3, true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "narrative request did not reach the VLM");
  if (!vlm) return;
  {
    StagePayload p = StagePayload::from_value(vlm->data);
    const Value* messages = p.request.inputs.find("messages");
    check(messages != nullptr && messages->is_array() &&
              messages->as_array().size() >= 2 &&
              messages->as_array().front().find("role")->as_str() == "system" &&
              messages->as_array().front()
                      .find("content")
                      ->as_str()
                      .find("[[image:") != std::string::npos,
          "the narrative director system prompt was not prepended");
  }
  const std::string vlm_id = vlm->request_id;

  h.stream(vlm_id, "vlm",
           text_chunk(
               "Lanterns rise over the lake. "
               "[[image: a glowing red lantern over a misty lake]]",
               false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
             "the first visual slot was not emitted mid-stream");
  auto image_slot = h.next_submit(h.image_recv);
  check(image_slot.has_value() &&
            image_slot->request_id == id + ":slot:0",
        "the image slot child id is not <id>:slot:0");
  if (image_slot) {
    StagePayload p = StagePayload::from_value(image_slot->data);
    check(p.request.inputs.find("prompt")->as_str() ==
              "a glowing red lantern over a misty lake",
          "the slot child did not carry the directive prompt");
  }
  check(log.terminal_for(id) == std::nullopt,
        "a terminal appeared before the VLM completed");

  h.stream(vlm_id, "vlm",
           text_chunk(
               "The camera pulls back. "
               "[[video: slow camera pull back over the lake]]",
               false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 2; },
             "the second visual slot was not emitted mid-stream");
  auto video_slot = h.next_submit(h.video_recv);
  check(video_slot.has_value() &&
            video_slot->request_id == id + ":slot:1",
        "the video slot child id is not <id>:slot:1");

  auto tts0 = h.next_submit(h.tts_recv);
  auto tts1 = h.next_submit(h.tts_recv);
  check(tts0.has_value() && tts1.has_value(),
        "global streaming TTS did not start during the narrative");
  std::vector<std::string> tts_ids;
  for (auto* submit : {&tts0, &tts1}) {
    if (!*submit) continue;
    tts_ids.push_back((*submit)->request_id);
    StagePayload p = StagePayload::from_value((*submit)->data);
    const std::string text = p.request.inputs.find("text")->as_str();
    check(text.find("[[") == std::string::npos,
          "a narrative directive leaked into the streaming TTS text");
  }

  // A streamed PCM chunk proves voice runs concurrently with the slots.
  if (!tts_ids.empty()) {
    Value::Map chunk;
    chunk["audio"] = Value::bin(std::string(16, '\x01'));
    chunk["index"] = Value(int64_t{0});
    chunk["sample_rate"] = Value(int64_t{48000});
    chunk["channels"] = Value(int64_t{2});
    h.stream(tts_ids.front(), "tts", Value(std::move(chunk)));
    wait_until(
        [&] {
          for (const UnifiedEvent& e : log.snapshot()) {
            if (e.kind == UnifiedEvent::Kind::kArtifact &&
                e.stage_id == std::string(kTtsStageId) &&
                e.artifact.find("mime_type") != nullptr &&
                e.artifact.find("mime_type")->as_str() == "audio/pcm") {
              return true;
            }
          }
          return false;
        },
        "no streamed audio/pcm artifact was emitted during the narrative");
  }

  {
    size_t deltas = 0;
    size_t slots = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVlmDelta) {
        ++deltas;
        check(e.text.find("[[") == std::string::npos,
              "a directive leaked into vlm_delta");
      }
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) {
        ++slots;
        check(e.text_offset >= 0 && !e.prompt.empty(),
              "a visual_slot event lacks prompt/text_offset");
      }
    }
    check(deltas == 2 && slots == 2,
          "unexpected clean delta or slot count during the narrative");
  }

  h.complete(vlm_id, "vlm", true,
             text_result(
                 "Lanterns rise over the lake. The camera pulls back."));
  h.complete(id + ":slot:0", "image", true,
             artifact_result("image", "image/png", "IMGBYTES"));
  h.complete(id + ":slot:1", "video", true,
             artifact_result("video", "video/mp4", "VIDBYTES"));
  Value::Map tts_extra;
  tts_extra["sample_rate"] = Value(int64_t{24000});
  tts_extra["channels"] = Value(int64_t{1});
  tts_extra["streamed"] = Value(true);
  for (const std::string& leg_id : tts_ids) {
    h.complete(leg_id, "tts", true,
               artifact_result("audio", "audio/wav", std::string(8, '\x02'),
                               tts_extra));
  }
  check(engine.wait(id, 5.0), "the narrative request did not complete");

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "the narrative request did not end in one complete terminal");
  bool image_leg = false;
  bool video_leg = false;
  bool tts_leg = false;
  if (terminal && terminal->legs.is_array()) {
    const auto& legs = terminal->legs.as_array();
    check(legs.size() == 3,
          "the narrative terminal legs array does not cover slots plus tts");
    for (const Value& leg : legs) {
      const std::string stage = leg.find("stage")->as_str();
      if (stage == "image") {
        image_leg = leg.find("slot_id")->as_str() == "slot-0" &&
                    leg.find("slot_index")->as_int() == 0 &&
                    leg.find("state")->as_str() == "completed";
      } else if (stage == "video") {
        video_leg = leg.find("slot_id")->as_str() == "slot-1" &&
                    leg.find("slot_index")->as_int() == 1 &&
                    leg.find("state")->as_str() == "completed";
      } else if (stage == "tts") {
        tts_leg = leg.find("state")->as_str() == "completed";
      }
    }
  }
  check(image_leg && video_leg && tts_leg,
        "the terminal legs array lacks a slot or tts entry");

  bool image_artifact_tagged = false;
  bool video_artifact_tagged = false;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind != UnifiedEvent::Kind::kArtifact) continue;
    if (e.stage_id == std::string(kImageStageId)) {
      image_artifact_tagged =
          e.artifact.find("slot_id") != nullptr &&
          e.artifact.find("slot_id")->as_str() == "slot-0" &&
          e.artifact.find("slot_index")->as_int() == 0;
    } else if (e.stage_id == std::string(kVideoStageId)) {
      video_artifact_tagged =
          e.artifact.find("slot_id") != nullptr &&
          e.artifact.find("slot_id")->as_str() == "slot-1" &&
          e.artifact.find("slot_index")->as_int() == 1;
    }
  }
  check(image_artifact_tagged && video_artifact_tagged,
        "slot artifacts do not carry slot_id and slot_index");

  bool snapshot_has_slot = false;
  for (const LegSnapshot& leg : engine.legs_snapshot(id)) {
    if (leg.slot_id == "slot-0" && leg.stage_id == kImageStageId) {
      snapshot_has_slot = true;
    }
  }
  check(snapshot_has_slot, "legs_snapshot does not track a narrative slot leg");
  engine.release(id);
}

// max_slots drops extra directives (still stripped from prose) and a
// zero-marker narrative falls back to exactly one slot.
void test_narrative_max_slots_and_zero_marker_fallback() {
  {
    Harness h("narrativemax");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    const std::string id =
        engine.submit(narrative_request(true, false, false, "auto", 1));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "max_slots request did not reach the VLM");
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("A [[image: cat]] B [[image: dog]] C", false));
    wait_until(
        [&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
        "the first slot beyond max_slots was not dispatched");
    auto slot = h.next_submit(h.image_recv);
    check(slot.has_value() && slot->request_id == id + ":slot:0",
          "the allowed slot child id is wrong");
    check(h.no_submit(h.image_recv),
          "max_slots=1 still dispatched a second visual slot");
    size_t slots = 0;
    size_t prose_bytes = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) ++slots;
      if (e.kind == UnifiedEvent::Kind::kVlmDelta) {
        prose_bytes += e.text.size();
        check(e.text.find("[[") == std::string::npos,
              "a dropped directive leaked into prose");
      }
    }
    check(slots == 1 && prose_bytes == std::string("A  B  C").size(),
          "max_slots stripping or prose output is wrong");
    h.complete(vlm->request_id, "vlm", true, text_result("A B C"));
    h.complete(id + ":slot:0", "image", true,
               artifact_result("image", "image/png", "X"));
    check(engine.wait(id, 5.0), "max_slots request did not complete");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->legs.is_array() &&
              terminal->legs.as_array().size() == 1,
          "max_slots terminal legs are wrong");
    engine.release(id);
  }
  {
    Harness h("narrativefallback");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    const std::string id =
        engine.submit(narrative_request(true, false, false, "auto", 3));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "fallback request did not reach the VLM");
    if (!vlm) return;
    h.complete(vlm->request_id, "vlm", true,
               text_result("A quiet harbor at dawn."));
    auto slot = h.next_submit(h.image_recv);
    check(slot.has_value() && slot->request_id == id + ":slot:0",
          "the zero-marker fallback slot was not dispatched");
    if (slot) {
      StagePayload p = StagePayload::from_value(slot->data);
      check(p.request.inputs.find("prompt")->as_str() ==
                "A quiet harbor at dawn.",
            "the fallback slot does not use the clean VLM text");
    }
    bool fallback_event = false;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot &&
          e.slot_id == "slot-0" &&
          e.prompt == "A quiet harbor at dawn.") {
        fallback_event = true;
      }
    }
    check(fallback_event, "the fallback visual_slot event is missing");
    h.complete(id + ":slot:0", "image", true,
               artifact_result("image", "image/png", "FB"));
    check(engine.wait(id, 5.0), "fallback request did not complete");
    engine.release(id);
  }
  {
    // Autonomous text-only routing: with allow_text_only=true in auto mode, a
    // zero-marker VLM answer finishes cleanly without dispatching any visual.
    Harness h("narrativeallowtextonly");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    StagePayload req = narrative_request(true, true, false, "auto", 2);
    req.request.inputs["narrative"]["allow_text_only"] = Value(true);
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "allow_text_only request did not reach the VLM");
    if (!vlm) return;
    h.complete(vlm->request_id, "vlm", true,
               text_result("sqrt(2) is irrational by integer parity."));
    check(engine.wait(id, 5.0),
          "allow_text_only zero-marker request did not complete immediately");
    size_t visual_slots = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) ++visual_slots;
    }
    check(visual_slots == 0,
          "allow_text_only=true must not dispatch a fallback visual slot");
    const auto terminal = log.terminal_for(id);
    check(terminal.has_value() && terminal->kind == UnifiedEvent::Kind::kComplete,
          "allow_text_only zero-marker request did not terminate with complete");
    engine.release(id);
  }
}

// Cancel mid-narrative broadcasts to every slot child and to streaming TTS.
void test_narrative_cancel_aborts_slots_and_tts() {
  Harness h("narrativecancel");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(narrative_request(true, false, true, "auto", 3, true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "cancel request did not reach the VLM");
  if (!vlm) return;
  h.stream(vlm->request_id, "vlm",
           text_chunk("A lantern. [[image: a red lantern]]", false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
             "cancel request never opened a slot");
  auto slot = h.next_submit(h.image_recv);
  auto tts = h.next_submit(h.tts_recv);
  check(slot.has_value() && tts.has_value(),
        "cancel request did not start its slot and tts legs");

  check(engine.cancel(id), "cancel rejected an active narrative request");
  check(h.wait_abort_on(h.image_recv, id + ":slot:0"),
        "cancel did not abort the narrative slot child");
  if (tts) {
    check(h.wait_abort_on(h.tts_recv, tts->request_id),
          "cancel did not abort the streaming TTS leg");
    h.complete(tts->request_id, "tts", false, Value(),
               std::string("request aborted"));
  }
  if (slot) {
    h.complete(slot->request_id, "image", false, Value(),
               std::string("request aborted"));
  }
  h.complete(vlm->request_id, "vlm", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled narrative request did not settle");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
            log.terminal_count(id) == 1,
        "cancelled narrative request did not produce one cancelled terminal");
  bool cancelled_slot_aborted = false;
  if (terminal && terminal->legs.is_array()) {
    for (const Value& leg : terminal->legs.as_array()) {
      if (leg.find("slot_id") != nullptr &&
          leg.find("slot_id")->as_str() == "slot-0") {
        cancelled_slot_aborted = leg.find("state")->as_str() == "aborted";
      }
    }
  }
  check(cancelled_slot_aborted,
        "cancelled public terminal omitted the finalized aborted slot");
  check(engine.in_flight() == 0,
        "in_flight nonzero after narrative cancel");
  engine.release(id);
}


// A truncated reserved directive is finalized as directive bytes at EOF, not
// released into public prose or submitted as a native narration span.
void test_narrative_unterminated_directive_never_reaches_tts() {
  Harness h("narrativeunterminatedtts");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(narrative_request(true, false, true, "auto", 1, true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "unterminated narrative request did not reach VLM");
  const std::string raw = "开场。\n[[image: unfinished black-hole diagram";
  h.stream(vlm->request_id, "vlm", text_chunk(raw, false));
  h.complete(vlm->request_id, "vlm", true, text_result(raw));
  auto tts = h.next_submit(h.tts_recv);
  check(tts.has_value(), "clean narration before EOF did not reach TTS");
  if (tts) {
    const StagePayload payload = StagePayload::from_value(tts->data);
    check(payload.request.inputs.find("text")->as_str() == "开场。",
          "an unterminated directive leaked into native TTS");
    Value::Map extra;
    extra["streamed"] = Value(true);
    h.complete(tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav",
                               audio::wav_from_pcm16("PCMM", 24000), extra));
  }
  check(h.no_submit(h.tts_recv), "EOF created a TTS leg for a partial directive");
  auto fallback = h.next_submit(h.image_recv);
  check(fallback.has_value(), "clean-prefix narrative fallback did not start");
  if (fallback) {
    h.complete(fallback->request_id, "image", true,
               artifact_result("image", "image/png", "EOF"));
  }
  check(engine.wait(id, 5.0), "unterminated directive did not settle cleanly");
  std::string public_text;
  for (const UnifiedEvent& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) public_text += event.text;
    check(event.text.find("[[image:") == std::string::npos,
          "unterminated directive leaked into a public event");
  }
  const auto terminal = log.terminal_for(id);
  check(public_text == "开场。\n" && terminal &&
            terminal->kind == UnifiedEvent::Kind::kComplete &&
            terminal->text == public_text,
        "EOF did not preserve exactly the clean narration prefix");
  engine.release(id);
}

// EOF release, no duplicate directive dispatch, and a clean terminal result.
void test_narrative_eof_and_terminal_result() {
  {
    Harness h("narrativeeof");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    const std::string id =
        engine.submit(narrative_request(true, false, false, "auto", 3));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "EOF request did not reach the VLM");
    if (!vlm) return;
    const std::string full = "The list ends here [";
    h.stream(vlm->request_id, "vlm", text_chunk(full, false));
    h.complete(vlm->request_id, "vlm", true, text_result(full));
    // No directive was seen, so the contract fallback renders slot-0.
    auto slot = h.next_submit(h.image_recv);
    check(slot.has_value(), "the fallback slot did not start");
    if (slot) {
      h.complete(slot->request_id, "image", true,
                 artifact_result("image", "image/png", "EOF"));
    }
    check(engine.wait(id, 5.0), "EOF request did not complete");
    std::string emitted;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVlmDelta) emitted += e.text;
    }
    check(emitted == full,
          "the parser's held EOF suffix was dropped from the prose stream");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
          "EOF request did not end complete");
    if (terminal) {
      check(terminal->text == full, "terminal text lost the held suffix");
      const Value* text = terminal->result.find("text");
      check(text != nullptr && text->is_str() && text->as_str() == full,
            "terminal result text is not the clean prose");
    }
    engine.release(id);
  }
  {
    Harness h("narrativedironly");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    const std::string id =
        engine.submit(narrative_request(true, false, false, "auto", 3));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "directive-only request did not reach the VLM");
    if (!vlm) return;
    const std::string full = "[[image: a red fox]]";
    h.stream(vlm->request_id, "vlm", text_chunk(full, false));
    wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
               "the directive-only slot was not dispatched");
    auto slot = h.next_submit(h.image_recv);
    check(slot.has_value() && slot->request_id == id + ":slot:0",
          "the directive-only slot child id is wrong");
    h.complete(vlm->request_id, "vlm", true, text_result(full));
    check(h.no_submit(h.image_recv),
          "the completion re-parsed the streamed directive and dispatched twice");
    size_t slots = count_kind(log, UnifiedEvent::Kind::kVisualSlot);
    check(slots == 1, "the streamed directive was dispatched twice");
    if (slot) {
      h.complete(slot->request_id, "image", true,
                 artifact_result("image", "image/png", "DIR"));
    }
    check(engine.wait(id, 5.0), "directive-only request did not complete");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
          "directive-only request did not end complete");
    if (terminal) {
      check(terminal->text.empty(), "terminal text kept a raw directive");
      const Value* text = terminal->result.find("text");
      check(text != nullptr && text->is_str() &&
                text->as_str().find("[[") == std::string::npos,
            "terminal result leaked a raw directive to the wire");
    }
    engine.release(id);
  }
}

// A VLM failure after a slot was dispatched aborts that running child.
void test_narrative_vlm_failure_aborts_slots() {
  Harness h("narrativevlmfail");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(narrative_request(true, false, false, "auto", 3));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "failure request did not reach the VLM");
  if (!vlm) return;
  h.stream(vlm->request_id, "vlm",
           text_chunk("A lantern. [[image: a red lantern]]", false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1; },
             "failure request never opened a slot");
  auto slot = h.next_submit(h.image_recv);
  check(slot.has_value(), "failure request did not start its slot");
  h.complete(vlm->request_id, "vlm", false, Value(), std::string("vlm down"));
  check(h.wait_abort_on(h.image_recv, id + ":slot:0"),
        "a VLM failure did not abort the running slot child");
  if (slot) {
    h.complete(slot->request_id, "image", false, Value(),
               std::string("request aborted"));
  }
  check(engine.wait(id, 5.0), "failed narrative request did not settle");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError,
        "failed narrative request did not end in an error terminal");
  bool error_slot_aborted = false;
  if (terminal && terminal->legs.is_array()) {
    for (const Value& leg : terminal->legs.as_array()) {
      if (leg.find("slot_id") != nullptr &&
          leg.find("slot_id")->as_str() == "slot-0") {
        error_slot_aborted = leg.find("state")->as_str() == "aborted";
      }
    }
  }
  check(error_slot_aborted,
        "failed public terminal omitted the finalized aborted slot");
  check(engine.in_flight() == 0,
        "in_flight nonzero after a failed narrative request");
  const auto failed_legs = engine.legs_snapshot(id);
  bool failed_slot_aborted = false;
  for (const LegSnapshot& leg : failed_legs) {
    if (leg.slot_id == "slot-0") {
      failed_slot_aborted = leg.state == LegState::kAborted;
    }
  }
  check(failed_slot_aborted,
        "VLM failure left a running narrative slot in the finalized request");
  engine.release(id);
}

// ---- Standalone visual render endpoint -----------------------------------

std::string lower_header(const std::map<std::string, std::string>& headers,
                         const std::string& name) {
  for (const auto& [key, value] : headers) {
    std::string lower = key;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (lower == name) return value;
  }
  return {};
}

void test_public_http_visual_render() {
  Harness h("visualrender");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "visual render server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  auto post = [&](const nlohmann::json& body) {
    test_http::HttpRequest request;
    request.url = base + "/v1/visual/render";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return test_http::perform(request);
  };
  auto status_of = [&](const nlohmann::json& body) {
    try {
      post(body);
    } catch (const client::Error& exc) {
      return exc.status();
    }
    return 200;
  };

  {
    auto future = std::async(std::launch::async, [&] {
      return post({{"kind", "image"},
                   {"prompt", "a red fox in snow"},
                   {"steps", 4},
                   {"size", "512x512"}});
    });
    auto submit = h.next_submit(h.image_recv);
    check(submit.has_value(), "image render did not reach the image stage");
    if (submit) {
      StagePayload p = StagePayload::from_value(submit->data);
      check(p.request.inputs.find("prompt")->as_str() ==
                "a red fox in snow" &&
                p.request.inputs.find("steps")->as_int() == 4,
            "image render inputs were not forwarded");
      h.complete(submit->request_id, "image", true,
                 artifact_result("image", "image/png", "PNGBYTES"));
    }
    const auto response = future.get();
    check(response.status == 200 && response.body == "PNGBYTES",
          "image render did not return the raw PNG bytes");
    check(lower_header(response.headers, "content-type") == "image/png" &&
              lower_header(response.headers, "x-umminfer-kind") == "image",
          "image render response headers are wrong");
  }
  {
    auto future = std::async(std::launch::async, [&] {
      return post({{"kind", "video"},
                   {"prompt", "waves at dusk"},
                   {"num_frames", 13},
                   {"fps", 16},
                   {"guidance_scale", 5.0}});
    });
    auto submit = h.next_submit(h.video_recv);
    check(submit.has_value(), "video render did not reach the video stage");
    if (submit) {
      StagePayload p = StagePayload::from_value(submit->data);
      check(p.request.inputs.find("prompt")->as_str() == "waves at dusk" &&
                p.request.inputs.find("num_frames")->as_int() == 13 &&
                p.request.inputs.find("fps")->as_int() == 16 &&
                p.request.inputs.find("guidance_scale")->as_double() == 5.0,
            "video render inputs were not forwarded");
      h.complete(submit->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4BYTES"));
    }
    const auto response = future.get();
    check(response.status == 200 && response.body == "MP4BYTES",
          "video render did not return the raw MP4 bytes");
    check(lower_header(response.headers, "content-type") == "video/mp4" &&
              lower_header(response.headers, "x-umminfer-kind") == "video",
          "video render response headers are wrong");
  }
  check(status_of({{"kind", "image"}}) == 400,
        "visual render without a prompt must return 400");
  check(status_of({{"kind", "image"}, {"prompt", ""}}) == 400,
        "visual render with an empty prompt must return 400");
  check(status_of({{"kind", "audio"}, {"prompt", "x"}}) == 400,
        "visual render with an unknown kind must return 400");
  check(status_of({{"kind", "image"}, {"prompt", "x"}, {"nope", 1}}) == 400,
        "visual render with an unknown field must return 400");
  check(status_of({{"kind", "image"}, {"prompt", "x"}, {"num_frames", 13}}) ==
            400,
        "visual render with a video field on kind image must return 400");
  // Native I2V: a valid image data URL on kind video reaches the video stage
  // with inputs["image"] forwarded.
  {
    const std::string i2v_image =
        "data:image/png;base64," + util::base64_encode("first-frame-png");
    auto future = std::async(std::launch::async, [&] {
      return post({{"kind", "video"},
                   {"prompt", "animate the fox"},
                   {"image", i2v_image}});
    });
    auto submit = h.next_submit(h.video_recv);
    check(submit.has_value(), "I2V video render did not reach the video stage");
    if (submit) {
      StagePayload p = StagePayload::from_value(submit->data);
      check(p.request.inputs.find("image") != nullptr &&
                p.request.inputs.find("image")->as_str() == i2v_image,
            "I2V video render did not forward inputs[\"image\"]");
      h.complete(submit->request_id, "video", true,
                 artifact_result("video", "video/mp4", "I2VMP4"));
    }
    const auto response = future.get();
    check(response.status == 200 && response.body == "I2VMP4" &&
              lower_header(response.headers, "content-type") == "video/mp4",
          "I2V video render did not return video/mp4");
  }
  check(status_of({{"kind", "video"}, {"prompt", "x"},
                   {"image", "data:image/png;base64,AA"}}) == 400,
        "visual render with an invalid image on kind video must return 400");
  check(status_of({{"kind", "video"}, {"prompt", "x"},
                   {"image", "not-a-data-url"}}) == 400,
        "visual render with a non-data-URL image on kind video must return 400");
  check(status_of({{"kind", "image"}, {"prompt", "x"}, {"steps", 0}}) == 400,
        "visual render with steps=0 must return 400");
  check(status_of({{"kind", "video"}, {"prompt", "x"},
                   {"guidance_scale", -1.0}}) == 400,
        "visual render with a negative guidance must return 400");

  api.stop();
  server_thread.join();

  // No registered stage answers 501.
  Harness h2("visualnostage");
  ApiConfig config2;
  config2.host = "127.0.0.1";
  config2.port = 0;
  config2.model = "qwen3-unified";
  OpenAIApi api2(h2.coordinator, config2);
  UnifiedEngine engine2(h2.coordinator, UnifiedConfig{});
  api2.set_unified_engine(engine2, "vlm");
  check(api2.bind(), "no-stage visual server did not bind");
  h2.start_pump();
  std::thread server2([&] { api2.listen(); });
  auto status2 = [&](const nlohmann::json& body) {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api2.bound_port()) +
                  "/v1/visual/render";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return exc.status();
    }
    return 200;
  };
  check(status2({{"kind", "image"}, {"prompt", "x"}}) == 501,
        "image render without an image stage must return 501");
  check(status2({{"kind", "video"}, {"prompt", "x"}}) == 501,
        "video render without a video stage must return 501");
  api2.stop();
  server2.join();
}

// HTTP validation for the narrative object. Every case fails before submit.
void test_public_http_narrative_validation() {
  Harness h("narrativevalidate");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "narrative validation server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  auto status_of = [&](const nlohmann::json& extra) {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "hi"}}})}};
    for (auto it = extra.begin(); it != extra.end(); ++it) {
      body[it.key()] = it.value();
    }
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    try {
      test_http::perform(request);
    } catch (const client::Error& exc) {
      return exc.status();
    }
    return 200;
  };
  const nlohmann::json enabled = {{"enabled", true}};
  check(status_of({{"stream", true},
                   {"narrative", "not-an-object"}}) == 400,
        "a non-object narrative must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", {{"enabled", "yes"}}}}) == 400,
        "a non-boolean narrative.enabled must return 400");
  check(status_of({{"stream", false},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", enabled}}) == 400,
        "narrative without stream=true must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text"})},
                   {"narrative", enabled}}) == 400,
        "narrative without a visual modality must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", {{"enabled", true},
                                  {"visual_mode", "video"}}}}) == 400,
        "narrative visual_mode=video without video must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", {{"enabled", true}, {"visual_mode", "bogus"}}}}) ==
            400,
        "an unknown narrative.visual_mode must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", {{"enabled", true}, {"max_slots", 0}}}}) == 400,
        "narrative.max_slots=0 must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text", "image"})},
                   {"narrative", {{"enabled", true}, {"max_slots", 7}}}}) == 400,
        "narrative.max_slots=7 must return 400");
  check(status_of({{"stream", true},
                   {"modalities", nlohmann::json::array({"text"})},
                   {"image", {{"size", "512x512"}}},
                   {"narrative", enabled}}) == 400,
        "narrative must require image/video in modalities, not just an image object");

  api.stop();
  server_thread.join();
}

// HTTP SSE contract for the narrative wire format.
void test_public_http_narrative_sse() {
  Harness h("narrativesse");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "narrative SSE server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"},
                                       {"content", "tell a story"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image", "audio"})},
      {"audio", {{"voice", "Tina"}, {"format", "wav"}, {"stream", true}}},
      {"narrative", {{"enabled", true},
                     {"visual_mode", "auto"},
                     {"max_slots", 2}}}};
  auto future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return test_http::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "narrative SSE request did not reach the VLM");
  if (!vlm) {
    api.stop();
    server_thread.join();
    return;
  }
  const std::string vlm_id = vlm->request_id;
  h.stream(vlm_id, "vlm",
           text_chunk("A hooded traveler walks north. "
                      "[[image: a hooded traveler under a pale moon]]",
                      false));
  auto slot = h.next_submit(h.image_recv);
  check(slot.has_value() && slot->request_id == vlm_id + ":slot:0",
        "the HTTP narrative slot was not dispatched with <request>:slot:0");
  if (slot) {
    h.complete(slot->request_id, "image", true,
               artifact_result("image", "image/png", "SSEIMG"));
  }
  auto tts = h.next_submit(h.tts_recv);
  check(tts.has_value(), "HTTP narrative did not start streaming TTS");
  if (tts) {
    Value::Map chunk;
    chunk["audio"] = Value::bin(std::string(16, '\x03'));
    chunk["index"] = Value(int64_t{0});
    h.stream(tts->request_id, "tts", Value(std::move(chunk)));
    Value::Map extra;
    extra["streamed"] = Value(true);
    h.complete(tts->request_id, "tts", true,
               artifact_result("audio", "audio/wav", std::string(8, '\x04'),
                               extra));
  }
  h.complete(vlm_id, "vlm", true,
             text_result("A hooded traveler walks north."));
  const auto response = future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200 && occurrences(response.body, "[DONE]") == 1,
        "narrative SSE did not end with one [DONE]");

  size_t visual_slot_frames = 0;
  size_t clean_deltas = 0;
  bool slot_frame_ok = false;
  bool audio_pcm = false;
  bool image_tagged = false;
  bool terminal_ok = false;
  int64_t last_seq = -1;
  for (const std::string& payload : sse_data_payloads(response.body)) {
    const json frame = json::parse(payload);
    if (!frame.contains("unified")) continue;
    const json& unified = frame["unified"];
    const int64_t seq = unified["seq"].get<int64_t>();
    check(seq == last_seq + 1, "HTTP narrative seq is not gap-free");
    last_seq = seq;
    const std::string kind = unified["kind"].get<std::string>();
    if (kind == "vlm_delta") {
      ++clean_deltas;
      check(unified["text"].get<std::string>().find("[[") == std::string::npos,
            "HTTP vlm_delta leaked a directive");
    } else if (kind == "visual_slot") {
      ++visual_slot_frames;
      slot_frame_ok =
          unified["stage"].get<std::string>() == "image" &&
          unified["slot_id"].get<std::string>() == "slot-0" &&
          unified["slot_index"].get<int64_t>() == 0 &&
          unified["media_type"].get<std::string>() == "image" &&
          unified["prompt"].get<std::string>() ==
              "a hooded traveler under a pale moon" &&
          unified.contains("text_offset");
    } else if (kind == "artifact") {
      const json& artifact = unified["artifact"];
      if (artifact.value("mime_type", std::string()) == "audio/pcm") {
        audio_pcm = true;
      }
      if (artifact.value("media_type", std::string()) == "image") {
        image_tagged = artifact.value("slot_id", std::string()) == "slot-0" &&
                       artifact.value("slot_index", int64_t{-1}) == 0;
      }
    } else if (kind == "complete") {
      terminal_ok = unified.contains("legs") &&
                    unified["legs"].size() == 2;
    }
  }
  check(visual_slot_frames == 1 && clean_deltas == 1 && slot_frame_ok,
        "HTTP narrative visual_slot frame does not match the wire contract");
  check(audio_pcm, "HTTP narrative did not stream audio/pcm concurrently");
  check(image_tagged, "HTTP narrative image artifact lacks slot attribution");
  check(terminal_ok, "HTTP narrative terminal legs array is wrong");
  wait_until([&] { return engine.in_flight() == 0; },
             "narrative SSE request stayed in flight");
}

// Wave 13: HTTP video_url content part passthrough to VLM & rejection of
// malformed URLs or combining image_url + video_url in a single turn.
void test_public_http_video_url_input() {
  Harness h("videourl");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "video_url HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                          "/v1/chat/completions";
  const std::string data_vid = "data:video/mp4;base64,AAAAIGZ0eXBpc29t";

  // 1. Valid video_url + text reaches VLM with video_url intact.
  nlohmann::json ok_body = {
      {"model", "qwen3-unified"},
      {"stream", false},
      {"messages", nlohmann::json::array({nlohmann::json{
                       {"role", "user"},
                       {"content",
                        nlohmann::json::array({
                            nlohmann::json{{"type", "video_url"},
                                           {"video_url", {{"url", data_vid}}}},
                            nlohmann::json{{"type", "text"},
                                           {"text", "describe the clip"}},
                        })}}})}};
  auto fut = std::async(std::launch::async, [&] {
    test_http::HttpRequest req;
    req.url = url;
    req.body = ok_body.dump();
    req.headers.emplace("Content-Type", "application/json");
    return test_http::perform(req);
  });
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "video_url request did not reach VLM");
  if (vlm) {
    bool saw_video_part = false;
    const StagePayload payload = StagePayload::from_value(vlm->data);
    const Value* msgs = payload.request.inputs.find("messages");
    if (msgs != nullptr && msgs->is_array()) {
      for (const Value& m : msgs->as_array()) {
        const Value* c = m.find("content");
        if (c != nullptr && c->is_array()) {
          for (const Value& p : c->as_array()) {
            const Value* t = p.find("type");
            const Value* vu = p.find("video_url");
            if (t != nullptr && t->is_str() && t->as_str() == "video_url" &&
                vu != nullptr && vu->is_map() && vu->find("url") != nullptr &&
                vu->find("url")->as_str() == data_vid) {
              saw_video_part = vu->as_map().size() == 1 &&
                               vu->find("frame_selection") == nullptr;
            }
          }
        }
      }
    }
    check(saw_video_part, "VLM submission did not preserve video_url part");
    h.complete(vlm->request_id, "vlm", true, text_result("A glowing lantern."));
  }
  const auto ok_res = fut.get();
  check(ok_res.status == 200 &&
            ok_res.body.find("A glowing lantern.") != std::string::npos,
        "video_url HTTP request did not return VLM answer");

  // Temporal opt-in is preserved end to end; no server-side default option is
  // injected into the legacy request above.
  nlohmann::json temporal_body = ok_body;
  temporal_body["messages"][0]["content"][0]["video_url"]["frame_selection"] = {
      {"policy", "temporal"}, {"motion_threshold", 0.04}, {"scene_threshold", 0.3}};
  auto temporal_fut = std::async(std::launch::async, [&] {
    test_http::HttpRequest req;
    req.url = url;
    req.body = temporal_body.dump();
    req.headers.emplace("Content-Type", "application/json");
    return test_http::perform(req);
  });
  auto temporal_vlm = h.next_submit(h.vlm_recv);
  check(temporal_vlm.has_value(), "temporal video request did not reach VLM");
  if (temporal_vlm) {
    const StagePayload payload = StagePayload::from_value(temporal_vlm->data);
    const Value* msgs = payload.request.inputs.find("messages");
    bool saw_temporal = false;
    if (msgs != nullptr && msgs->is_array() && !msgs->as_array().empty()) {
      const Value* content = msgs->as_array()[0].find("content");
      if (content != nullptr && content->is_array() && !content->as_array().empty()) {
        const Value* video = content->as_array()[0].find("video_url");
        const Value* selection = video != nullptr && video->is_map()
                                     ? video->find("frame_selection")
                                     : nullptr;
        if (selection != nullptr && selection->is_map()) {
          const Value* policy = selection->find("policy");
          const Value* motion = selection->find("motion_threshold");
          const Value* scene = selection->find("scene_threshold");
          saw_temporal = policy != nullptr && policy->as_str() == "temporal" &&
                         motion != nullptr && motion->as_double() == 0.04 &&
                         scene != nullptr && scene->as_double() == 0.3;
        }
      }
    }
    check(saw_temporal, "VLM submission dropped temporal frame_selection options");
    h.complete(temporal_vlm->request_id, "vlm", true, text_result("Scene change."));
  }
  check(temporal_fut.get().status == 200, "temporal video request failed");

  // Explicit 4/8/16 survives both public JSON and SSE and both native policies.
  for (const bool stream : {false, true}) {
    for (const char* policy : {"uniform", "temporal"}) {
      for (const int budget : {4, 8, 16}) {
        nlohmann::json budget_body = ok_body;
        budget_body["stream"] = stream;
        budget_body["messages"][0]["content"][0]["video_url"]["frame_selection"] = {
            {"policy", policy}, {"max_frames", budget}};
        auto budget_fut = std::async(std::launch::async, [&] {
          test_http::HttpRequest req;
          req.url = url;
          req.body = budget_body.dump();
          req.headers.emplace("Content-Type", "application/json");
          return test_http::perform(req);
        });
        auto budget_vlm = h.next_submit(h.vlm_recv);
        check(budget_vlm.has_value(), "budgeted video request did not reach VLM");
        if (budget_vlm) {
          const StagePayload payload = StagePayload::from_value(budget_vlm->data);
          const Value* msgs = payload.request.inputs.find("messages");
          const Value* content = msgs != nullptr && msgs->is_array()
                                     ? msgs->as_array()[0].find("content") : nullptr;
          const Value* video = content != nullptr && content->is_array()
                                   ? content->as_array()[0].find("video_url") : nullptr;
          const Value* selection = video != nullptr && video->is_map()
                                       ? video->find("frame_selection") : nullptr;
          const Value* frames = selection != nullptr && selection->is_map()
                                    ? selection->find("max_frames") : nullptr;
          check(frames != nullptr && frames->is_int() && frames->as_int() == budget,
                "public JSON/SSE dropped or changed the explicit frame budget");
          h.complete(budget_vlm->request_id, "vlm", true, text_result("Budget accepted."));
        }
        const auto response = budget_fut.get();
        check(response.status == 200 &&
                  (!stream || response.body.ends_with("data: [DONE]\n\n")),
              "public JSON/SSE rejected a valid 4/8/16 frame budget");
      }
    }
  }

  // 2. Rejects non-data video_url and image_url + video_url in the same turn.
  const auto expect_400 = [&](const nlohmann::json& bad_body, const char* msg) {
    bool rejected = false;
    try {
      test_http::HttpRequest req;
      req.url = url;
      req.body = bad_body.dump();
      req.headers.emplace("Content-Type", "application/json");
      (void)test_http::perform(req);
    } catch (const client::Error& err) {
      rejected = (err.status() == 400);
    }
    check(rejected, msg);
  };

  expect_400(
      nlohmann::json{
          {"model", "qwen3-unified"},
          {"messages",
           nlohmann::json::array({nlohmann::json{
               {"role", "user"},
               {"content",
                nlohmann::json::array({nlohmann::json{
                    {"type", "video_url"},
                    {"video_url", {{"url", "https://example.com/a.mp4"}}}}})}}})}},
      "non-data video_url was not rejected with 400");

  expect_400(
      nlohmann::json{
          {"model", "qwen3-unified"},
          {"messages",
           nlohmann::json::array({nlohmann::json{
               {"role", "user"},
               {"content",
                nlohmann::json::array({
                    nlohmann::json{
                        {"type", "image_url"},
                        {"image_url", {{"url", "data:image/png;base64,AAAA"}}}},
                    nlohmann::json{{"type", "video_url"},
                                   {"video_url", {{"url", data_vid}}}},
                })}}})}},
      "combining image_url and video_url was not rejected with 400");

  auto selection_body = [&](nlohmann::json selection) {
    nlohmann::json body = ok_body;
    body["messages"][0]["content"][0]["video_url"]["frame_selection"] =
        std::move(selection);
    return body;
  };
  expect_400(selection_body({{"policy", "random"}}),
             "unknown video frame policy was not rejected");
  expect_400(selection_body({{"policy", "temporal"}, {"motion_threshold", -0.01}}),
             "negative motion threshold was not rejected");
  expect_400(selection_body({{"policy", "temporal"}, {"scene_threshold", 1.01}}),
             "scene threshold above one was not rejected");
  expect_400(selection_body({{"policy", "temporal"}, {"max_frames", 1000}}),
             "API frame budget override above sixteen was not rejected");
  for (const nlohmann::json& invalid : std::vector<nlohmann::json>{
           true, 4.0, "4", 0, -4, 3, 17, std::numeric_limits<uint64_t>::max()}) {
    expect_400(selection_body({{"policy", "uniform"}, {"max_frames", invalid}}),
               "non-integer, unsupported, or overflowing frame budget was not rejected");
  }
  expect_400(selection_body({{"policy", "uniform"}, {"motion_threshold", 0.1}}),
             "uniform policy with an ignored threshold was not rejected");
  expect_400(selection_body(nlohmann::json::array()),
             "non-object frame_selection was not rejected");

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "video_url request stayed in flight");
}

// Multi-turn multimodal messages through UnifiedEngine: preserves all turns
// (optional leading system + user/assistant history ending in user), keeps
// image_url/video_url anchored on their original turn for VLM prefix KV-cache
// reuse, extracts only the latest user text for session history recording, and
// preserves all single-message and cross-turn safety invariants.
void test_unified_multiturn_engine_and_schema() {
  Harness h("mtengine");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  const std::string img_url = "data:image/png;base64,AA";
  const std::string vid_url = "data:video/mp4;base64,AAAAIGZ0eXBpc29t";
  auto make_text_part = [](const std::string& text) {
    Value::Map p;
    p["type"] = Value("text");
    p["text"] = Value(text);
    return Value(std::move(p));
  };
  auto make_video_part = [](const std::string& url) {
    Value::Map vu;
    vu["url"] = Value(url);
    Value::Map p;
    p["type"] = Value("video_url");
    p["video_url"] = Value(std::move(vu));
    return Value(std::move(p));
  };
  auto make_msg = [](const std::string& role, Value content) {
    Value::Map m;
    m["role"] = Value(role);
    m["content"] = std::move(content);
    return Value(std::move(m));
  };

  // 1. Multi-turn video conversation: [user(video+text), assistant, user].
  {
    StagePayload req;
    Value::Array msgs;
    msgs.push_back(make_msg(
        "user", Value::Array{make_video_part(vid_url),
                             make_text_part("Describe this video.")}));
    msgs.push_back(make_msg("assistant", Value("A red lantern sways at night.")));
    msgs.push_back(make_msg("user", Value("What color is the lantern?")));
    Value::Map inputs;
    inputs["messages"] = Value(std::move(msgs));
    inputs["session_id"] = Value("mt-engine-sess");
    inputs["session_history"] = Value(true);
    req.request.inputs = Value(std::move(inputs));

    const std::string id = engine.submit(std::move(req));
    check(h.no_submit(h.asr_recv), "multi-turn video turn reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "multi-turn video turn missed VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 3 &&
                roles[0].first == "user" &&
                roles[0].second == "Describe this video." &&
                roles[1].first == "assistant" &&
                roles[1].second == "A red lantern sways at night." &&
                roles[2].first == "user" &&
                roles[2].second == "What color is the lantern?",
            "multi-turn video did not forward all 3 turns intact");
      const Value& turn0_content =
          *payload.request.inputs.find("messages")->as_array()[0].find("content");
      check(turn0_content.is_array() && turn0_content.as_array().size() == 2 &&
                turn0_content.as_array()[0].find("type")->as_str() == "video_url" &&
                turn0_content.as_array()[0].find("video_url")->find("url")->as_str() ==
                    vid_url,
            "multi-turn video lost video_url on turn 0");
      const Value& modalities = *payload.request.inputs.find("modalities");
      check(modalities.as_array().size() == 1 &&
                modalities.as_array()[0].as_str() == "text",
            "multi-turn video declared unexpected VLM modalities");
    }
    h.complete(id, "vlm", true, text_result("Red."));
    check(engine.wait(id, 5.0), "multi-turn video request did not complete");
    engine.release(id);
  }

  // Verify session_store_ recorded ONLY the latest user turn text ("What color
  // is the lantern?"), not concatenated prior turns.
  {
    StagePayload probe;
    Value::Array msgs;
    msgs.push_back(make_msg("system", Value("Be concise.")));
    msgs.push_back(make_msg("user", Value("Next question")));
    Value::Map inputs;
    inputs["messages"] = Value(std::move(msgs));
    inputs["session_id"] = Value("mt-engine-sess");
    inputs["session_history"] = Value(true);
    probe.request.inputs = Value(std::move(inputs));

    const std::string id = engine.submit(std::move(probe));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "session-history follow-up missed VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 4 &&
                roles[0].first == "system" && roles[0].second == "Be concise." &&
                roles[1].first == "user" &&
                roles[1].second == "What color is the lantern?" &&
                roles[2].first == "assistant" && roles[2].second == "Red." &&
                roles[3].first == "user" && roles[3].second == "Next question",
            "session_history failed to keep leading system or recorded wrong user text");
    }
    h.complete(id, "vlm", true, text_result("Done."));
    check(engine.wait(id, 5.0), "session-history follow-up did not complete");
    engine.release(id);
  }

  // 2. Direct schema rejections: invalid multi-turn / media combinations.
  auto rejects = [&](Value::Array msgs, const char* what) {
    StagePayload req;
    req.request.inputs = Value::Map{{"messages", Value(std::move(msgs))}};
    bool threw = false;
    try {
      (void)engine.submit(std::move(req));
    } catch (const ProtocolError&) {
      threw = true;
    }
    check(threw, what);
  };

  rejects(Value::Array{}, "empty messages array was accepted");
  rejects(Value::Array{make_msg("user", Value("hi")),
                       make_msg("assistant", Value("hello"))},
          "non-user final turn was accepted");
  rejects(Value::Array{make_msg("user", Value("hi")),
                       make_msg("system", Value("late")),
                       make_msg("user", Value("next"))},
          "late system message was accepted");
  rejects(Value::Array{make_msg("tool", Value("out")),
                       make_msg("user", Value("hi"))},
          "unsupported role was accepted");
  rejects(Value::Array{make_msg("user", Value::Array{make_text_part("a"),
                                                     make_text_part("b")})},
          "two text parts in the same message were accepted");
  rejects(Value::Array{make_msg("user", Value::Array{image_part(img_url),
                                                     make_text_part("a")}),
                       make_msg("assistant", Value("b")),
                       make_msg("user", Value::Array{make_video_part(vid_url),
                                                     make_text_part("c")})},
          "image_url + video_url across turns was accepted");
  rejects(Value::Array{make_msg("user", Value::Array{image_part(img_url),
                                                     make_text_part("a")}),
                       make_msg("assistant", Value("b")),
                       make_msg("user", Value::Array{image_part(img_url),
                                                     make_text_part("c")})},
          "two image_urls across turns were accepted");
  rejects(Value::Array{make_msg("user", Value::Array{Value(part("input_audio"))}),
                       make_msg("assistant", Value("ok")),
                       make_msg("user", Value("next"))},
          "input_audio on non-final turn was accepted");
  check(h.no_submit(h.asr_recv) && h.no_submit(h.vlm_recv),
        "a rejected multi-turn engine request reached a stage");
}

// Public HTTP contract for OpenAI multi-turn multimodal messages in unified
// mode: [system?, user(text|image_url|video_url), assistant, user(followup)].
void test_public_http_multiturn_multimodal() {
  Harness h("httpmultiturn");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "multi-turn HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string url =
      "http://127.0.0.1:" + std::to_string(api.bound_port()) +
      "/v1/chat/completions";
  const std::string data_img = "data:image/png;base64,AAAA";
  const std::string data_vid = "data:video/mp4;base64,AAAAIGZ0eXBpc29t";
  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json img_part = {{"type", "image_url"},
                                   {"image_url", {{"url", data_img}}}};
  const nlohmann::json vid_part = {{"type", "video_url"},
                                   {"video_url", {{"url", data_vid}}}};
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};

  auto post = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    test_http::HttpRequest req;
    req.url = url;
    req.body = body.dump();
    req.headers.emplace("Content-Type", "application/json");
    req.timeout = client::Milliseconds{10000};
    return test_http::perform(req);
  };
  auto expect_400 = [&](nlohmann::json body, const char* what) {
    bool rejected = false;
    try {
      (void)post(std::move(body));
    } catch (const client::Error& err) {
      rejected = (err.status() == 400);
    }
    check(rejected, what);
  };

  // 1. Multi-turn text: [system, user, assistant, user].
  {
    nlohmann::json body = {
        {"messages",
         nlohmann::json::array({
             nlohmann::json{{"role", "system"}, {"content", "Be concise."}},
             nlohmann::json{{"role", "user"}, {"content", "Hello"}},
             nlohmann::json{{"role", "assistant"}, {"content", "Hi there!"}},
             nlohmann::json{{"role", "user"}, {"content", "What is 2+2?"}},
         })}};
    auto fut = std::async(std::launch::async, [&] { return post(body); });
    check(h.no_submit(h.asr_recv), "multi-turn text reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "multi-turn text did not reach VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 4 &&
                roles[0] == std::pair<std::string, std::string>("system", "Be concise.") &&
                roles[1] == std::pair<std::string, std::string>("user", "Hello") &&
                roles[2] == std::pair<std::string, std::string>("assistant", "Hi there!") &&
                roles[3] == std::pair<std::string, std::string>("user", "What is 2+2?"),
            "multi-turn text did not forward full 4-turn history to VLM");
      h.complete(vlm->request_id, "vlm", true, text_result("4"));
    }
    const auto res = fut.get();
    check(res.status == 200 &&
              res.body.find("\"content\":\"4\"") != std::string::npos,
          "multi-turn text HTTP response failed");
  }

  // 2. Multi-turn image: [user(image_url + text), assistant, user(followup)].
  {
    nlohmann::json body = {
        {"messages",
         nlohmann::json::array({
             nlohmann::json{
                 {"role", "user"},
                 {"content",
                  nlohmann::json::array({
                      img_part,
                      nlohmann::json{{"type", "text"},
                                     {"text", "What is in this photo?"}},
                  })}},
             nlohmann::json{{"role", "assistant"},
                            {"content", "A red lantern hangs outside."}},
             nlohmann::json{{"role", "user"},
                            {"content", "What color is the lantern?"}},
         })}};
    auto fut = std::async(std::launch::async, [&] { return post(body); });
    check(h.no_submit(h.asr_recv), "multi-turn image reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "multi-turn image did not reach VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 3 &&
                roles[0] == std::pair<std::string, std::string>(
                                "user", "What is in this photo?") &&
                roles[1] == std::pair<std::string, std::string>(
                                "assistant", "A red lantern hangs outside.") &&
                roles[2] == std::pair<std::string, std::string>(
                                "user", "What color is the lantern?"),
            "multi-turn image did not forward all 3 turns");
      const Value& msgs = *payload.request.inputs.find("messages");
      const Value& turn0_content = *msgs.as_array()[0].find("content");
      check(turn0_content.is_array() && turn0_content.as_array().size() == 2 &&
                turn0_content.as_array()[0].find("type")->as_str() == "image_url" &&
                turn0_content.as_array()[0].find("image_url")->find("url")->as_str() ==
                    data_img,
            "multi-turn image did not keep image_url on turn 0");
      const Value& modalities = *payload.request.inputs.find("modalities");
      check(modalities.as_array().size() == 2 &&
                modalities.as_array()[0].as_str() == "text" &&
                modalities.as_array()[1].as_str() == "image",
            "multi-turn image did not declare text+image modalities");
      h.complete(vlm->request_id, "vlm", true, text_result("Red."));
    }
    const auto res = fut.get();
    check(res.status == 200 &&
              res.body.find("\"content\":\"Red.\"") != std::string::npos,
          "multi-turn image HTTP response failed");
  }

  // 3. Multi-turn video: [user(video_url + text), assistant, user(followup)].
  {
    nlohmann::json body = {
        {"messages",
         nlohmann::json::array({
             nlohmann::json{
                 {"role", "user"},
                 {"content",
                  nlohmann::json::array({
                      vid_part,
                      nlohmann::json{{"type", "text"},
                                     {"text", "Describe this video."}},
                  })}},
             nlohmann::json{{"role", "assistant"},
                            {"content", "A red lantern sways at night."}},
             nlohmann::json{{"role", "user"},
                            {"content", "What color is the lantern?"}},
         })}};
    auto fut = std::async(std::launch::async, [&] { return post(body); });
    check(h.no_submit(h.asr_recv), "multi-turn video reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "multi-turn video did not reach VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 3 &&
                roles[0] == std::pair<std::string, std::string>(
                                "user", "Describe this video.") &&
                roles[1] == std::pair<std::string, std::string>(
                                "assistant", "A red lantern sways at night.") &&
                roles[2] == std::pair<std::string, std::string>(
                                "user", "What color is the lantern?"),
            "multi-turn video did not forward all 3 turns");
      const Value& msgs = *payload.request.inputs.find("messages");
      const Value& turn0_content = *msgs.as_array()[0].find("content");
      check(turn0_content.is_array() && turn0_content.as_array().size() == 2 &&
                turn0_content.as_array()[0].find("type")->as_str() == "video_url" &&
                turn0_content.as_array()[0].find("video_url")->find("url")->as_str() ==
                    data_vid,
            "multi-turn video did not keep video_url on turn 0");
      h.complete(vlm->request_id, "vlm", true, text_result("Red."));
    }
    const auto res = fut.get();
    check(res.status == 200 &&
              res.body.find("\"content\":\"Red.\"") != std::string::npos,
          "multi-turn video HTTP response failed");
  }

  // 4. Multi-turn with input_audio on the final user turn.
  {
    nlohmann::json body = {
        {"messages",
         nlohmann::json::array({
             nlohmann::json{
                 {"role", "user"},
                 {"content",
                  nlohmann::json::array({
                      img_part,
                      nlohmann::json{{"type", "text"},
                                     {"text", "Look at this cat."}},
                  })}},
             nlohmann::json{{"role", "assistant"},
                            {"content", "I see an orange tabby."}},
             nlohmann::json{{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}},
         })}};
    auto fut = std::async(std::launch::async, [&] { return post(body); });
    auto asr = h.next_submit(h.asr_recv);
    check(asr.has_value(), "multi-turn audio follow-up did not reach ASR");
    if (asr) {
      h.complete(asr->request_id, "asr", true, text_result("Is it sleeping?"));
    }
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "multi-turn audio follow-up did not reach VLM");
    if (vlm) {
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const auto roles = prompt_roles(payload);
      check(roles.size() == 3 &&
                roles[0].second == "Look at this cat." &&
                roles[1].second == "I see an orange tabby." &&
                roles[2].second == "Is it sleeping?",
            "multi-turn audio follow-up did not preserve history + transcript");
      h.complete(vlm->request_id, "vlm", true, text_result("Yes."));
    }
    const auto res = fut.get();
    check(res.status == 200 &&
              res.body.find("\"content\":\"Yes.\"") != std::string::npos,
          "multi-turn audio follow-up HTTP response failed");
  }

  // 5. session_history: true does not duplicate client-supplied multi-turn
  // messages and preserves leading system messages on single-turn requests.
  {
    auto run_hist = [&](nlohmann::json messages, const std::string& answer) {
      nlohmann::json body = {{"session_id", "http-mt-sess"},
                             {"session_history", true},
                             {"messages", std::move(messages)}};
      auto fut = std::async(std::launch::async, [&] { return post(body); });
      auto vlm = h.next_submit(h.vlm_recv);
      check(vlm.has_value(), "HTTP multi-turn history request missed VLM");
      StagePayload payload;
      if (vlm) {
        payload = StagePayload::from_value(vlm->data);
        h.complete(vlm->request_id, "vlm", true, text_result(answer));
      }
      const auto res = fut.get();
      check(res.status == 200, "HTTP multi-turn history turn failed");
      wait_until([&] { return engine.in_flight() == 0; },
                 "HTTP multi-turn history turn stayed in flight");
      return payload;
    };
    (void)run_hist(nlohmann::json::array({nlohmann::json{
                       {"role", "user"}, {"content", "turn one"}}}),
                   "answer one");
    const StagePayload client_mt = run_hist(
        nlohmann::json::array({
            nlohmann::json{{"role", "user"}, {"content", "turn one"}},
            nlohmann::json{{"role", "assistant"}, {"content", "answer one"}},
            nlohmann::json{{"role", "user"}, {"content", "turn two"}},
        }),
        "answer two");
    const auto roles = prompt_roles(client_mt);
    check(roles.size() == 3 &&
              roles[0].second == "turn one" &&
              roles[1].second == "answer one" &&
              roles[2].second == "turn two",
          "session_history duplicated client-supplied multi-turn messages");
  }

  // 6. Invalid cases must still be rejected with HTTP 400 before any stage.
  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({nlohmann::json{
               {"role", "user"},
               {"content",
                nlohmann::json::array({
                    nlohmann::json{{"type", "text"}, {"text", "first"}},
                    nlohmann::json{{"type", "text"}, {"text", "second"}},
                })}}})}},
      "two text parts in the same user message were not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        img_part,
                        nlohmann::json{{"type", "text"}, {"text", "img turn"}},
                    })}},
               nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        vid_part,
                        nlohmann::json{{"type", "text"}, {"text", "vid turn"}},
                    })}},
           })}},
      "image_url + video_url across turns was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        img_part,
                        nlohmann::json{{"type", "text"}, {"text", "img 1"}},
                    })}},
               nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        img_part,
                        nlohmann::json{{"type", "text"}, {"text", "img 2"}},
                    })}},
           })}},
      "two image_urls across turns were not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        vid_part,
                        nlohmann::json{{"type", "text"}, {"text", "vid 1"}},
                    })}},
               nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        vid_part,
                        nlohmann::json{{"type", "text"}, {"text", "vid 2"}},
                    })}},
           })}},
      "two video_urls across turns were not rejected");

  expect_400(nlohmann::json{{"messages", nlohmann::json::array()}},
             "empty messages array was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "user"}, {"content", "hi"}},
               nlohmann::json{{"role", "assistant"}, {"content", "hello"}},
           })}},
      "non-user final turn (assistant) was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "system"}, {"content", "only system"}},
           })}},
      "non-user final turn (system) was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "user"}, {"content", "hi"}},
               nlohmann::json{{"role", "system"}, {"content", "late system"}},
               nlohmann::json{{"role", "user"}, {"content", "next"}},
           })}},
      "non-leading system message was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "tool"}, {"content", "result"}},
               nlohmann::json{{"role", "user"}, {"content", "next"}},
           })}},
      "unsupported role was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "user"},
                              {"content", nlohmann::json::array({audio_part})}},
               nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
               nlohmann::json{{"role", "user"}, {"content", "followup"}},
           })}},
      "input_audio on non-final turn was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{{"role", "user"}, {"content", "hi"}},
               nlohmann::json{{"role", "assistant"},
                              {"content", nlohmann::json::array({img_part})}},
               nlohmann::json{{"role", "user"}, {"content", "next"}},
           })}},
      "image_url on assistant message was not rejected");

  expect_400(
      nlohmann::json{
          {"messages",
           nlohmann::json::array({
               nlohmann::json{
                   {"role", "user"},
                   {"content",
                    nlohmann::json::array({
                        nlohmann::json{
                            {"type", "video_url"},
                            {"video_url", {{"url", "data:image/png;base64,AAAA"}}}},
                        nlohmann::json{{"type", "text"}, {"text", "bad mime"}},
                    })}},
               nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
               nlohmann::json{{"role", "user"}, {"content", "followup"}},
           })}},
      "invalid video_url MIME in multi-turn was not rejected");

  // Oversized video_url (> 32 MiB decoded payload).
  {
    const std::string oversized_b64(((32u << 20) / 3u + 4u) * 4u, 'A');
    expect_400(
        nlohmann::json{
            {"messages",
             nlohmann::json::array({
                 nlohmann::json{
                     {"role", "user"},
                     {"content",
                      nlohmann::json::array({
                          nlohmann::json{
                              {"type", "video_url"},
                              {"video_url",
                               {{"url", "data:video/mp4;base64," + oversized_b64}}}},
                          nlohmann::json{{"type", "text"}, {"text", "too big"}},
                      })}},
                 nlohmann::json{{"role", "assistant"}, {"content", "ok"}},
                 nlohmann::json{{"role", "user"}, {"content", "followup"}},
             })}},
        "oversized video_url (> 32 MiB) in multi-turn was not rejected");
  }

  check(h.no_submit(h.asr_recv) && h.no_submit(h.vlm_recv),
        "a rejected multi-turn HTTP request reached a stage");

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "multi-turn HTTP requests stayed in flight");
}

// Per-request Wan2.2 TeaCache speed control (`video.teacache_thresh` in
// [0.0, 0.25], opt-in and omitted by default):
//   - chat completions (normal video leg + narrative visual_slot video)
//   - /v1/visual/render (kind=video accepts [0.0, 0.25]; kind=image rejects)
//   - HTTP 400 on out-of-range, string, and boolean values
//   - direct UnifiedEngine schema validation rejects invalid values
void test_public_http_video_teacache_thresh() {
  Harness h("teacache");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "teacache server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  auto post_json = [&](const std::string& path, const nlohmann::json& body) {
    test_http::HttpRequest req;
    req.url = base + path;
    req.body = body.dump();
    req.headers.emplace("Content-Type", "application/json");
    req.timeout = client::Milliseconds{8000};
    try {
      return test_http::perform(req);
    } catch (const client::Error& err) {
      test_http::HttpResponse out;
      out.status = err.status();
      out.body = err.what();
      return out;
    }
  };

  // 1. Omitted video.teacache_thresh must NOT send or override on chat.
  {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "waves"}}})},
        {"stream", false},
        {"modalities", nlohmann::json::array({"text", "video"})},
        {"video", {{"prompt", "ocean waves"}, {"size", "704x480"}}}};
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/chat/completions", body);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "teacache default chat did not reach VLM");
    if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("ok"));
    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "teacache default chat did not reach video stage");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      check(p.request.inputs.find("teacache_thresh") == nullptr,
            "omitted video.teacache_thresh must not be injected");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4DEFAULT"));
    }
    check(fut.get().status == 200, "teacache default chat must return 200");
  }

  // 2. Valid video.teacache_thresh values (0.0, 0.05, 0.25) forward intact.
  for (double thresh : {0.0, 0.05, 0.25}) {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "fast"}}})},
        {"stream", false},
        {"modalities", nlohmann::json::array({"text", "video"})},
        {"video",
         {{"prompt", "fast ocean waves"},
          {"size", "704x480"},
          {"teacache_thresh", thresh}}}};
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/chat/completions", body);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "teacache chat did not reach VLM");
    if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("ok"));
    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "teacache chat did not reach video stage");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      const Value* tc = p.request.inputs.find("teacache_thresh");
      check(tc != nullptr && (tc->is_float() || tc->is_int()) &&
                std::fabs(tc->as_double() - thresh) < 1e-9,
            "video.teacache_thresh did not forward exact value on chat");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4FAST"));
    }
    check(fut.get().status == 200, "valid teacache chat must return 200");
  }

  // 3. Narrative visual_slot video dispatch forwards video.teacache_thresh.
  {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "story"}}})},
        {"stream", true},
        {"modalities", nlohmann::json::array({"text", "video"})},
        {"video", {{"size", "704x480"}, {"teacache_thresh", 0.05}}},
        {"narrative",
         {{"enabled", true}, {"visual_mode", "video"}, {"max_slots", 2}}}};
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/chat/completions", body);
    });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "narrative teacache chat did not reach VLM");
    if (vlm) {
      const std::string reply = "Scene one.\n[[video: waves crashing on rocks]]\nDone.";
      h.stream(vlm->request_id, "vlm", text_chunk(reply, false));
      h.complete(vlm->request_id, "vlm", true, text_result(reply));
    }
    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "narrative video slot did not reach video stage");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      const Value* tc = p.request.inputs.find("teacache_thresh");
      check(tc != nullptr && (tc->is_float() || tc->is_int()) &&
                std::fabs(tc->as_double() - 0.05) < 1e-9,
            "narrative video slot did not forward video.teacache_thresh");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4SLOT"));
    }
    check(fut.get().status == 200, "narrative teacache SSE must return 200");
  }

  // 4. POST /v1/visual/render forwards omitted & valid teacache_thresh on video.
  {
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/visual/render",
                       {{"kind", "video"}, {"prompt", "calm lake"}});
    });
    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "visual render default video did not reach stage");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      check(p.request.inputs.find("teacache_thresh") == nullptr,
            "visual render omitted teacache_thresh must not be injected");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4VR0"));
    }
    check(fut.get().status == 200, "visual render default video must return 200");
  }
  for (double thresh : {0.0, 0.05, 0.25}) {
    auto fut = std::async(std::launch::async, [&] {
      return post_json("/v1/visual/render",
                       {{"kind", "video"},
                        {"prompt", "fast lake"},
                        {"teacache_thresh", thresh}});
    });
    auto vid = h.next_submit(h.video_recv);
    check(vid.has_value(), "visual render teacache video did not reach stage");
    if (vid) {
      StagePayload p = StagePayload::from_value(vid->data);
      const Value* tc = p.request.inputs.find("teacache_thresh");
      check(tc != nullptr && (tc->is_float() || tc->is_int()) &&
                std::fabs(tc->as_double() - thresh) < 1e-9,
            "visual render teacache_thresh did not forward exact value");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4VR1"));
    }
    check(fut.get().status == 200, "visual render valid teacache must return 200");
  }

  // 5. Reject invalid video.teacache_thresh on both /v1/chat/completions and
  //    /v1/visual/render with HTTP 400.
  auto bad_chat = [&](nlohmann::json tc_val) {
    nlohmann::json body = {
        {"model", "qwen3-unified"},
        {"messages", nlohmann::json::array(
                         {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
        {"stream", false},
        {"video", {{"prompt", "x"}, {"teacache_thresh", std::move(tc_val)}}}};
    return post_json("/v1/chat/completions", body).status;
  };
  check(bad_chat(-0.01) == 400, "chat video.teacache_thresh=-0.01 must return 400");
  check(bad_chat(0.30) == 400, "chat video.teacache_thresh=0.30 must return 400");
  check(bad_chat("0.05") == 400, "chat video.teacache_thresh string must return 400");
  check(bad_chat(true) == 400, "chat video.teacache_thresh bool must return 400");

  auto bad_vr = [&](nlohmann::json tc_val) {
    return post_json("/v1/visual/render",
                     {{"kind", "video"},
                      {"prompt", "x"},
                      {"teacache_thresh", std::move(tc_val)}})
        .status;
  };
  check(bad_vr(-0.01) == 400, "visual render teacache_thresh=-0.01 must return 400");
  check(bad_vr(0.30) == 400, "visual render teacache_thresh=0.30 must return 400");
  check(bad_vr("0.05") == 400, "visual render teacache_thresh string must return 400");
  check(bad_vr(true) == 400, "visual render teacache_thresh bool must return 400");
  check(post_json("/v1/visual/render",
                  {{"kind", "image"}, {"prompt", "x"}, {"teacache_thresh", 0.05}})
                .status == 400,
        "visual render teacache_thresh on kind image must return 400");

  // 6. Direct UnifiedEngine schema validation rejects invalid teacache_thresh.
  for (Value bad_val : {Value(-0.01), Value(0.30), Value("0.05"), Value(true)}) {
    StagePayload payload;
    Value::Map inputs;
    Value::Map msg;
    msg["role"] = Value("user");
    msg["content"] = Value("hi");
    inputs["messages"] = Value(Value::Array{Value(std::move(msg))});
    inputs["output_modalities"] = Value(Value::Array{Value("video")});
    Value::Map vopts;
    vopts["teacache_thresh"] = std::move(bad_val);
    inputs["video_options"] = Value(std::move(vopts));
    payload.request.inputs = Value(std::move(inputs));
    bool rejected = false;
    try {
      (void)engine.submit(std::move(payload));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, "UnifiedEngine must reject invalid video_options.teacache_thresh");
  }

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "teacache HTTP requests stayed in flight");
}

// Stage B Cross-Slot Visual Continuity:
//   - HTTP validation for narrative.continuity_mode & narrative.style_anchor
//   - Byte-identical default ("off") behavior on prompts & SSE wire events
//   - prompt_anchor with style_anchor + UTF-8-safe Chinese Slot 0 anchor
//   - i2v_chain Case A (no image), Case B (completed image), Case C (slow image)
//   - i2v_chain graceful T2V fallback on anchor failure
//   - i2v_chain clean cancellation while video slot waits on image slot
void test_narrative_continuity_http_and_engine() {
  // Helper to verify valid UTF-8 (no truncated multi-byte sequences).
  auto is_valid_utf8 = [](const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      size_t len = 0;
      if (c < 0x80) {
        len = 1;
      } else if ((c & 0xE0) == 0xC0) {
        len = 2;
      } else if ((c & 0xF0) == 0xE0) {
        len = 3;
      } else if ((c & 0xF8) == 0xF0) {
        len = 4;
      } else {
        return false;
      }
      if (i + len > s.size()) return false;
      for (size_t j = 1; j < len; ++j) {
        if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) return false;
      }
      i += len;
    }
    return true;
  };

  // 1. Direct UnifiedEngine: "off" leaves slot prompts untouched and omits
  //    continuity metadata.
  {
    Harness h("contoff");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    StagePayload req = narrative_request(true, true, false, "auto", 3);
    // Explicit "off" with style_anchor must still disable continuity.
    req.request.inputs["narrative"]["continuity_mode"] = Value("off");
    req.request.inputs["narrative"]["style_anchor"] = Value("watercolor ink");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "continuity=off did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("One. [[image: red fox]] Two. [[video: fox leaps]]",
                          false));
      auto s0 = h.next_submit(h.image_recv);
      auto s1 = h.next_submit(h.video_recv);
      check(s0.has_value() && s1.has_value(),
            "continuity=off did not submit both slots immediately");
      if (s0) {
        StagePayload p = StagePayload::from_value(s0->data);
        check(p.request.inputs.find("prompt")->as_str() == "red fox",
              "continuity=off altered slot 0 prompt");
        h.complete(s0->request_id, "image", true,
                   artifact_result("image", "image/png", "PNG0"));
      }
      if (s1) {
        StagePayload p = StagePayload::from_value(s1->data);
        check(p.request.inputs.find("prompt")->as_str() == "fox leaps",
              "continuity=off altered slot 1 prompt");
        check(p.request.inputs.find("image") == nullptr,
              "continuity=off must not attach I2V image");
        h.complete(s1->request_id, "video", true,
                   artifact_result("video", "video/mp4", "MP41"));
      }
      h.complete(vlm->request_id, "vlm", true, text_result("One. Two."));
    }
    check(engine.wait(id, 5.0), "continuity=off request did not complete");
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) {
        check(e.continuity_mode.empty() && e.anchor_slot_index == -1,
              "continuity=off must leave event continuity fields empty/-1");
      }
    }
    engine.release(id);
  }

  // 2. Direct UnifiedEngine: "prompt_anchor" injects style_anchor + UTF-8-safe
  //    truncated Chinese Slot 0 anchor prompt into Slot 1.
  {
    Harness h("contprompt");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    // 50 Chinese characters = 150 UTF-8 bytes (> 140 bytes); 46 chars = 138 bytes.
    std::string long_zh;
    const std::string zh_char = "\xE7\x81\xAF";  // "灯" (3 UTF-8 bytes)
    for (int i = 0; i < 50; ++i) long_zh += zh_char;
    std::string expected_zh_trunc;
    for (int i = 0; i < 46; ++i) expected_zh_trunc += zh_char;

    StagePayload req = narrative_request(true, true, false, "auto", 3);
    // Omitted continuity_mode + non-empty style_anchor defaults to prompt_anchor.
    req.request.inputs["narrative"]["style_anchor"] =
        Value("  warm Ghibli gouache  ");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "prompt_anchor did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("Scene 1. [[image: " + long_zh +
                              "]] Scene 2. [[video: lantern rises]]",
                          false));
      auto s0 = h.next_submit(h.image_recv);
      auto s1 = h.next_submit(h.video_recv);
      check(s0.has_value() && s1.has_value(),
            "prompt_anchor did not dispatch both slots");
      if (s0) {
        StagePayload p = StagePayload::from_value(s0->data);
        const std::string got = p.request.inputs.find("prompt")->as_str();
        check(got == long_zh + " | Style continuity: warm Ghibli gouache",
              "prompt_anchor slot 0 prompt mismatch");
        h.complete(s0->request_id, "image", true,
                   artifact_result("image", "image/png", "P0"));
      }
      if (s1) {
        StagePayload p = StagePayload::from_value(s1->data);
        const std::string got = p.request.inputs.find("prompt")->as_str();
        const std::string expected =
            "lantern rises | Style continuity: warm Ghibli gouache"
            " | Maintain character and scene continuity with opening visual: " +
            expected_zh_trunc;
        check(is_valid_utf8(got),
              "prompt_anchor UTF-8 truncation split a multi-byte codepoint");
        check(got == expected,
              "prompt_anchor slot 1 prompt did not inject style + truncated Chinese anchor");
        check(p.request.inputs.find("image") == nullptr,
              "prompt_anchor must not attach I2V image");
        h.complete(s1->request_id, "video", true,
                   artifact_result("video", "video/mp4", "V1"));
      }
      h.complete(vlm->request_id, "vlm", true, text_result("Scene 1. Scene 2."));
    }
    check(engine.wait(id, 5.0), "prompt_anchor request did not complete");
    size_t slot_events = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) {
        ++slot_events;
        check(e.continuity_mode == "prompt_anchor" && e.anchor_slot_index == -1,
              "prompt_anchor visual_slot continuity metadata wrong");
      }
    }
    check(slot_events == 2, "prompt_anchor expected 2 visual_slot events");
    engine.release(id);
  }

  // 3. Direct UnifiedEngine: "i2v_chain" Case C (slow image Slot 0 + video Slot 1)
  //    and Case B (completed image Slot 0 + video Slot 2).
  {
    Harness h("conti2v");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    StagePayload req = narrative_request(true, true, false, "auto", 3);
    req.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
    req.request.inputs["narrative"]["style_anchor"] = Value("cinematic 35mm");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "i2v_chain did not reach VLM");
    const std::string fake_png = "\x89PNG\r\n\x1a\nSLOT0BYTES";
    const std::string expected_data_url =
        "data:image/png;base64," + util::base64_encode(fake_png);
    if (vlm) {
      // Emit image Slot 0 and video Slot 1 while Slot 0 is still running.
      h.stream(vlm->request_id, "vlm",
               text_chunk("Opening. [[image: silver fox on snowy bridge]] "
                          "Motion. [[video: fox trots across bridge]]",
                          false));
      wait_until(
          [&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 2; },
          "i2v_chain did not emit both visual_slot events mid-stream");
      auto s0 = h.next_submit(h.image_recv);
      check(s0.has_value() && s0->request_id == id + ":slot:0",
            "i2v_chain image slot 0 did not submit");
      // Video Slot 1 MUST wait until image Slot 0 completes!
      check(h.no_submit(h.video_recv),
            "i2v_chain submitted video slot 1 before anchor image slot 0 finished");

      // Complete slow image Slot 0 -> video Slot 1 must now be released with
      // exact I2V init-image data URL!
      h.complete(id + ":slot:0", "image", true,
                 artifact_result("image", "image/png", fake_png));
      auto s1 = h.next_submit(h.video_recv);
      check(s1.has_value() && s1->request_id == id + ":slot:1",
            "i2v_chain did not release video slot 1 after image slot 0 completed");
      if (s1) {
        StagePayload p = StagePayload::from_value(s1->data);
        const Value* img = p.request.inputs.find("image");
        check(img != nullptr && img->is_str() &&
                  img->as_str() == expected_data_url,
              "i2v_chain video slot 1 did not receive exact Slot 0 PNG data URL");
        const std::string prompt = p.request.inputs.find("prompt")->as_str();
        check(prompt.find("Style continuity: cinematic 35mm") != std::string::npos &&
                  prompt.find("silver fox on snowy bridge") != std::string::npos,
              "i2v_chain video slot 1 missing continuity prompt");
        h.complete(s1->request_id, "video", true,
                   artifact_result("video", "video/mp4", "I2V1"));
      }

      // Now emit video Slot 2 while Slot 0 is ALREADY completed (Case B).
      h.stream(vlm->request_id, "vlm",
               text_chunk(" Finale. [[video: camera orbits the fox]]", false));
      auto s2 = h.next_submit(h.video_recv);
      check(s2.has_value() && s2->request_id == id + ":slot:2",
            "i2v_chain Case B video slot 2 did not submit immediately");
      if (s2) {
        StagePayload p = StagePayload::from_value(s2->data);
        const Value* img = p.request.inputs.find("image");
        check(img != nullptr && img->is_str() &&
                  img->as_str() == expected_data_url,
              "i2v_chain Case B video slot 2 missing Slot 0 PNG data URL");
        h.complete(s2->request_id, "video", true,
                   artifact_result("video", "video/mp4", "I2V2"));
      }
      h.complete(vlm->request_id, "vlm", true,
                 text_result("Opening. Motion. Finale."));
    }
    check(engine.wait(id, 5.0), "i2v_chain request did not complete");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
          "i2v_chain did not end complete");
    int anchored_slots = 0;
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) {
        check(e.continuity_mode == "i2v_chain",
              "i2v_chain visual_slot missing continuity_mode");
        if (e.slot_index == 0) {
          check(e.anchor_slot_index == -1,
                "i2v_chain image slot 0 must have anchor_slot_index == -1");
        } else {
          check(e.anchor_slot_index == 0,
                "i2v_chain chained video slot must emit anchor_slot_index == 0");
          ++anchored_slots;
        }
      }
    }
    check(anchored_slots == 2, "i2v_chain expected 2 anchored video slots");
    check(engine.in_flight() == 0, "i2v_chain left request in flight");
    engine.release(id);
  }

  // 4. Direct UnifiedEngine: "i2v_chain" Case A (no earlier image slot) submits
  //    T2V immediately with anchor_slot_index == -1.
  {
    Harness h("conti2vnoimg");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    StagePayload req = narrative_request(true, true, false, "auto", 2);
    req.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "i2v_chain Case A did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("Start. [[video: waves breaking]]", false));
      auto s0 = h.next_submit(h.video_recv);
      check(s0.has_value() && s0->request_id == id + ":slot:0",
            "i2v_chain Case A video slot 0 did not submit immediately");
      if (s0) {
        StagePayload p = StagePayload::from_value(s0->data);
        check(p.request.inputs.find("image") == nullptr,
              "i2v_chain Case A video slot 0 must not have image input");
        h.complete(s0->request_id, "video", true,
                   artifact_result("video", "video/mp4", "T2V0"));
      }
      h.complete(vlm->request_id, "vlm", true, text_result("Start."));
    }
    check(engine.wait(id, 5.0), "i2v_chain Case A did not complete");
    for (const UnifiedEvent& e : log.snapshot()) {
      if (e.kind == UnifiedEvent::Kind::kVisualSlot) {
        check(e.continuity_mode == "i2v_chain" && e.anchor_slot_index == -1,
              "i2v_chain Case A visual_slot metadata wrong");
      }
    }
    engine.release(id);
  }

  // 5. Direct UnifiedEngine: "i2v_chain" when anchor image Slot 0 fails ->
  //    video Slot 1 gracefully falls back to T2V and finishes cleanly.
  {
    Harness h("conti2vfail");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    StagePayload req = narrative_request(true, true, false, "auto", 2);
    req.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "i2v_chain anchor-fail did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("A. [[image: misty valley]] B. [[video: mist drifts]]",
                          false));
      auto s0 = h.next_submit(h.image_recv);
      check(s0.has_value(), "i2v_chain anchor-fail image slot 0 did not submit");
      check(h.no_submit(h.video_recv),
            "i2v_chain anchor-fail submitted video slot 1 before slot 0 failed");
      h.complete(vlm->request_id, "vlm", true, text_result("A. B."));
      // Fail image Slot 0 -> video Slot 1 must fall back to T2V!
      h.complete(id + ":slot:0", "image", false, Value(),
                 std::string("image OOM"));
      auto s1 = h.next_submit(h.video_recv);
      check(s1.has_value() && s1->request_id == id + ":slot:1",
            "i2v_chain did not fall back to T2V when image slot 0 failed");
      if (s1) {
        StagePayload p = StagePayload::from_value(s1->data);
        check(p.request.inputs.find("image") == nullptr,
              "i2v_chain fallback T2V must not carry image key");
        h.complete(s1->request_id, "video", true,
                   artifact_result("video", "video/mp4", "FALLBACKMP4"));
      }
    }
    check(engine.wait(id, 5.0), "i2v_chain anchor-fail request did not complete");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete,
          "i2v_chain anchor-fail request must still complete");
    check(engine.in_flight() == 0,
          "i2v_chain anchor-fail left request in flight");
    engine.release(id);
  }

  // 6. Direct UnifiedEngine: "i2v_chain" cancelled while video Slot 1 waits on
  //    image Slot 0 (after VLM finishes): zero zombie submit, clean terminal,
  //    in_flight == 0.
  {
    Harness h("conti2vcancel");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

    StagePayload req = narrative_request(true, true, false, "auto", 2);
    req.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
    const std::string id = engine.submit(std::move(req));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "i2v_chain cancel did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("A. [[image: lantern]] B. [[video: lantern glows]]",
                          false));
      wait_until(
          [&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 2; },
          "i2v_chain cancel did not emit both slots");
      auto s0 = h.next_submit(h.image_recv);
      check(s0.has_value(), "i2v_chain cancel image slot 0 did not submit");
      h.complete(vlm->request_id, "vlm", true, text_result("A. B."));
      // Wait briefly for VLM completion to transition request into Phase::kOutputs.
      wait_until(
          [&] {
            for (const LegSnapshot& leg : engine.legs_snapshot(id)) {
              if (leg.stage_id == kVlmStageId &&
                  leg.state == LegState::kCompleted) {
                return true;
              }
            }
            return false;
          },
          "i2v_chain cancel VLM leg did not reach kCompleted");
      check(engine.cancel(id), "i2v_chain cancel rejected active request");
      check(h.wait_abort_on(h.image_recv, id + ":slot:0"),
            "i2v_chain cancel did not abort anchor image slot 0");
      h.complete(id + ":slot:0", "image", false, Value(),
                 std::string("request aborted"));
    }
    check(engine.wait(id, 5.0), "i2v_chain cancelled request did not settle");
    check(h.no_submit(h.video_recv),
          "i2v_chain cancel leaked a zombie video submit");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
              log.terminal_count(id) == 1,
          "i2v_chain cancelled request did not end with one kCancelled terminal");
    check(engine.in_flight() == 0,
          "i2v_chain cancelled request left in_flight nonzero");
    engine.release(id);
  }

  // 7. HTTP validation + SSE metadata check for narrative continuity fields.
  {
    Harness h("conthttp");
    ApiConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model = "qwen3-unified";
    config.request_timeout_s = 10.0;
    OpenAIApi api(h.coordinator, config);
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    api.set_unified_engine(engine, "vlm");
    check(api.bind(), "continuity HTTP server did not bind");
    h.start_pump();
    std::thread server_thread([&] { api.listen(); });
    const std::string base =
        "http://127.0.0.1:" + std::to_string(api.bound_port());

    auto post_chat = [&](const nlohmann::json& narrative_obj) {
      nlohmann::json body = {
          {"model", "qwen3-unified"},
          {"messages", nlohmann::json::array(
                           {nlohmann::json{{"role", "user"}, {"content", "go"}}})},
          {"stream", true},
          {"modalities", nlohmann::json::array({"text", "image", "video"})},
          {"narrative", narrative_obj}};
      test_http::HttpRequest req;
      req.url = base + "/v1/chat/completions";
      req.body = body.dump();
      req.headers.emplace("Content-Type", "application/json");
      req.timeout = client::Milliseconds{10000};
      try {
        return test_http::perform(req);
      } catch (const client::Error& err) {
        test_http::HttpResponse out;
        out.status = err.status();
        out.body = err.what();
        return out;
      }
    };

    // Validation rejections -> HTTP 400 with exact contract messages.
    {
      const auto r1 = post_chat(
          {{"enabled", true}, {"continuity_mode", "bogus"}});
      check(r1.status == 400 &&
                r1.body.find(
                    "narrative.continuity_mode must be 'off', 'prompt_anchor', or 'i2v_chain'") !=
                    std::string::npos,
            "invalid narrative.continuity_mode string must return 400 with contract message");
      const auto r2 = post_chat(
          {{"enabled", true}, {"continuity_mode", 123}});
      check(r2.status == 400 &&
                r2.body.find(
                    "narrative.continuity_mode must be 'off', 'prompt_anchor', or 'i2v_chain'") !=
                    std::string::npos,
            "non-string narrative.continuity_mode must return 400 with contract message");
      const auto r3 = post_chat(
          {{"enabled", true}, {"style_anchor", 42}});
      check(r3.status == 400, "non-string narrative.style_anchor must return 400");
      const auto r4 = post_chat(
          {{"enabled", true}, {"style_anchor", std::string(257, 'a')}});
      check(r4.status == 400,
            "oversized narrative.style_anchor (>256B) must return 400");
    }

    // Default ("off") SSE must omit continuity_mode & anchor_slot_index.
    {
      auto fut = std::async(std::launch::async, [&] {
        return post_chat({{"enabled", true}, {"max_slots", 2}});
      });
      auto vlm = h.next_submit(h.vlm_recv);
      check(vlm.has_value(), "HTTP continuity default did not reach VLM");
      if (vlm) {
        h.stream(vlm->request_id, "vlm",
                 text_chunk("Hi. [[image: a pine tree]]", false));
        auto s0 = h.next_submit(h.image_recv);
        if (s0) {
          h.complete(s0->request_id, "image", true,
                     artifact_result("image", "image/png", "PNGDEF"));
        }
        h.complete(vlm->request_id, "vlm", true, text_result("Hi."));
      }
      const auto res = fut.get();
      check(res.status == 200, "HTTP continuity default SSE failed");
      check(res.body.find("\"continuity_mode\"") == std::string::npos &&
                res.body.find("\"anchor_slot_index\"") == std::string::npos,
            "default continuity_mode=off must omit continuity SSE metadata");
    }

    // i2v_chain SSE must include continuity_mode & anchor_slot_index: 0 on
    // chained video slot, and accept 256-byte style_anchor with outer spaces.
    {
      const std::string padded_256 = "  " + std::string(256, 'x') + "  ";
      auto fut = std::async(std::launch::async, [&] {
        return post_chat({{"enabled", true},
                          {"max_slots", 2},
                          {"continuity_mode", "i2v_chain"},
                          {"style_anchor", padded_256}});
      });
      auto vlm = h.next_submit(h.vlm_recv);
      check(vlm.has_value(), "HTTP i2v_chain did not reach VLM");
      if (vlm) {
        h.stream(vlm->request_id, "vlm",
                 text_chunk("One. [[image: red boat]] Two. [[video: boat sails]]",
                            false));
        auto s0 = h.next_submit(h.image_recv);
        if (s0) {
          h.complete(s0->request_id, "image", true,
                     artifact_result("image", "image/png", "BOATPNG"));
        }
        auto s1 = h.next_submit(h.video_recv);
        if (s1) {
          StagePayload p = StagePayload::from_value(s1->data);
          check(p.request.inputs.find("image") != nullptr &&
                    p.request.inputs.find("image")->as_str() ==
                        "data:image/png;base64," + util::base64_encode("BOATPNG"),
                "HTTP i2v_chain did not pass Slot 0 PNG data URL to video stage");
          h.complete(s1->request_id, "video", true,
                     artifact_result("video", "video/mp4", "BOATMP4"));
        }
        h.complete(vlm->request_id, "vlm", true, text_result("One. Two."));
      }
      const auto res = fut.get();
      check(res.status == 200, "HTTP i2v_chain SSE failed");
      check(res.body.find("\"continuity_mode\":\"i2v_chain\"") != std::string::npos &&
                res.body.find("\"anchor_slot_index\":0") != std::string::npos,
            "HTTP i2v_chain SSE missing continuity_mode or anchor_slot_index");
    }

    api.stop();
    server_thread.join();
    wait_until([&] { return engine.in_flight() == 0; },
               "continuity HTTP requests stayed in flight");
  }
}

// Explicit directive-level anchors extend, rather than replace, i2v_chain.
StagePayload explicit_anchor_request(int max_slots = 6) {
  StagePayload request = narrative_request(true, true, false, "auto", max_slots);
  request.request.inputs["narrative"]["continuity_mode"] = Value("i2v_chain");
  Value::Map inputs = request.request.inputs.as_map();
  Value::Map video = inputs.at("video_options").as_map();
  video["size"] = Value("704x480");
  video["seed"] = Value(int64_t{42});
  video["steps"] = Value(int64_t{7});
  video["num_frames"] = Value(int64_t{13});
  video["guidance_scale"] = Value(4.5);
  inputs["video_options"] = Value(std::move(video));
  request.request.inputs = Value(std::move(inputs));
  return request;
}

void check_explicit_anchor_submit(const std::optional<SubmitMessage>& submit,
                                  const std::string& child_id,
                                  const std::string& source_bytes,
                                  int source_index) {
  check(submit.has_value() && submit->request_id == child_id,
        "explicitly anchored I2V slot was not submitted");
  if (!submit) return;
  const StagePayload payload = StagePayload::from_value(submit->data);
  const std::string digest = util::sha256_hex(source_bytes);
  const Value* image = payload.request.inputs.find("image");
  const Value* anchor = payload.request.inputs.find("anchor_slot_index");
  const Value* sha = payload.request.inputs.find("anchor_sha256");
  const Value* crop = payload.request.inputs.find("anchor_crop");
  const Value* explicit_flag = payload.request.inputs.find("anchor_explicit");
  check(image != nullptr && image->is_bin() && image->as_bin() == source_bytes &&
            util::sha256_hex(image->as_bin()) == digest,
        "explicit I2V input does not bind exact settled source bytes");
  check(anchor != nullptr && anchor->as_int() == source_index &&
            sha != nullptr && sha->as_str() == digest &&
            crop != nullptr && crop->as_str() == "center_cover" &&
            explicit_flag != nullptr && explicit_flag->as_bool(),
        "explicit I2V input lacks anchor/SHA-256/crop metadata");
  const Value* prompt = payload.request.inputs.find("prompt");
  const Value* size = payload.request.inputs.find("size");
  const Value* seed = payload.request.inputs.find("seed");
  const Value* steps = payload.request.inputs.find("steps");
  const Value* frames = payload.request.inputs.find("num_frames");
  const Value* guidance = payload.request.inputs.find("guidance_scale");
  check(prompt != nullptr && prompt->as_str().find("camera orbits") == 0 &&
            size != nullptr && size->as_str() == "704x480" &&
            seed != nullptr && seed->as_int() == 42 &&
            steps != nullptr && steps->as_int() == 7 &&
            frames != nullptr && frames->as_int() == 13 &&
            guidance != nullptr && guidance->as_double() == 4.5,
        "explicit I2V dispatch dropped prompt/crop-size/seed/settings");
}

void run_explicit_anchor_completion(bool immediate) {
  Harness h(immediate ? "explicitanchorimmediate" : "explicitanchordelayed");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
  const std::string id = engine.submit(explicit_anchor_request());
  auto vlm = h.next_submit(h.vlm_recv);
  if (!vlm) return;
  const std::string first_bytes = "EXPLICIT-SLOT-0";
  const std::string latest_bytes = "EXPLICIT-SLOT-1-NOT-THE-ANCHOR";
  h.stream(vlm->request_id, "vlm", text_chunk("[[image: opening]]", false));
  auto first = h.next_submit(h.image_recv);
  if (immediate && first) {
    h.complete(first->request_id, "image", true,
               artifact_result("image", "image/png", first_bytes));
  }
  h.stream(vlm->request_id, "vlm", text_chunk("[[image: alternate]]", false));
  auto latest = h.next_submit(h.image_recv);
  if (latest) {
    h.complete(latest->request_id, "image", true,
               artifact_result("image", "image/png", latest_bytes));
  }
  h.stream(vlm->request_id, "vlm",
           text_chunk("[[video: camera orbits | source=slot-0]]", false));
  wait_until([&] { return count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 3; },
             "explicit source visual_slot was not emitted");
  if (!immediate) {
    check(h.no_submit(h.video_recv),
          "explicit I2V dispatched before its named source completed");
    bool pending = false;
    for (const LegSnapshot& leg : engine.legs_snapshot(id)) {
      if (leg.slot_id == "slot-2") pending = leg.state == LegState::kPending;
    }
    check(pending, "explicit dependent slot is not pending on slot-0");
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    check(log.terminal_for(id) == std::nullopt,
          "request terminated with an explicit dependent still pending");
    if (first) {
      h.complete(first->request_id, "image", true,
                 artifact_result("image", "image/png", first_bytes));
    }
  }
  auto video = h.next_submit(h.video_recv);
  check_explicit_anchor_submit(video, id + ":slot:2", first_bytes, 0);
  if (immediate) h.complete(vlm->request_id, "vlm", true, text_result("clean"));
  if (video) {
    h.complete(video->request_id, "video", true,
               artifact_result("video", "video/mp4", "EXPLICIT-I2V"));
  }
  check(engine.wait(id, 5.0), "explicitly linked narrative did not complete");
  const auto terminal = log.terminal_for(id);
  bool linked_artifact = false;
  bool linked_leg = false;
  int64_t expected_seq = 0;
  for (const UnifiedEvent& event : log.snapshot()) {
    check(event.seq == expected_seq++, "explicit linkage broke event monotonicity");
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
      check(event.text.find("source=") == std::string::npos,
            "explicit directive leaked into clean prose");
    }
    if (event.kind == UnifiedEvent::Kind::kVisualSlot &&
        event.slot_id == "slot-2") {
      check(event.continuity_mode == "i2v_chain" &&
                event.anchor_slot_index == 0 && event.anchor_explicit &&
                event.anchor_sha256.empty(),
            "explicit visual_slot metadata is wrong or timing-dependent");
    }
    if (event.kind == UnifiedEvent::Kind::kArtifact &&
        event.artifact.find("slot_id") != nullptr &&
        event.artifact.find("slot_id")->as_str() == "slot-2") {
      linked_artifact = event.artifact.find("anchor_slot_index")->as_int() == 0 &&
                        event.artifact.find("anchor_explicit")->as_bool() &&
                        event.artifact.find("anchor_sha256")->as_str() ==
                            util::sha256_hex(first_bytes);
    }
  }
  if (terminal && terminal->legs.is_array()) {
    check(terminal->legs.as_array().size() == 3,
          "multiple explicit-continuity slots lost terminal identity");
    for (const Value& leg : terminal->legs.as_array()) {
      if (leg.find("slot_id")->as_str() == "slot-2") {
        linked_leg = leg.find("state")->as_str() == "completed" &&
                     leg.find("anchor_slot_index")->as_int() == 0 &&
                     leg.find("anchor_explicit")->as_bool() &&
                     leg.find("anchor_sha256")->as_str() ==
                         util::sha256_hex(first_bytes);
      }
    }
  }
  check(linked_artifact && linked_leg,
        "explicit I2V artifact/terminal leg lacks verified anchor linkage");
  engine.release(id);
}

void test_explicit_anchor_completion() {
  run_explicit_anchor_completion(true);
  run_explicit_anchor_completion(false);
}

void check_explicit_image_submit(const std::optional<SubmitMessage>& submit,
                                 const std::string& child_id,
                                 const std::string& source_bytes,
                                 int source_index) {
  check(submit.has_value() && submit->request_id == child_id,
        "explicit image-edit slot was not submitted");
  if (!submit) return;
  const StagePayload payload = StagePayload::from_value(submit->data);
  const std::string digest = util::sha256_hex(source_bytes);
  const Value* image = payload.request.inputs.find("image");
  const Value* anchor = payload.request.inputs.find("anchor_slot_index");
  const Value* sha = payload.request.inputs.find("anchor_sha256");
  const Value* explicit_flag = payload.request.inputs.find("anchor_explicit");
  check(image != nullptr && image->is_bin() && image->as_bin() == source_bytes &&
            util::sha256_hex(image->as_bin()) == digest,
        "explicit image edit did not bind exact settled source bytes");
  check(anchor != nullptr && anchor->as_int() == source_index &&
            sha != nullptr && sha->as_str() == digest &&
            explicit_flag != nullptr && explicit_flag->as_bool() &&
            payload.request.inputs.find("anchor_crop") == nullptr,
        "explicit image edit has wrong source identity or Wan-only crop metadata");
  const Value* prompt = payload.request.inputs.find("prompt");
  check(prompt != nullptr && prompt->is_str() &&
            prompt->as_str().find("source=") == std::string::npos,
        "explicit image-edit reference leaked into its render prompt");
}

void test_explicit_image_reference_chain() {
  {
    Harness h("explicitimagechain");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(3));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    const std::string source0 = "CHARACTER-STYLE-SLOT-0";
    const std::string source1 = "DEPENDENT-EDIT-SLOT-1";
    h.stream(vlm->request_id, "vlm", text_chunk("[[image: establish]]", false));
    auto first = h.next_submit(h.image_recv);
    if (first) {
      h.complete(first->request_id, "image", true,
                 artifact_result("image", "image/png", source0));
    }
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[image: same character at night | source=slot-0]]", false));
    auto second = h.next_submit(h.image_recv);
    check_explicit_image_submit(second, id + ":slot:1", source0, 0);
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[video: camera orbits | source=slot-1]]", false));
    check(h.no_submit(h.video_recv),
          "I2V dependent dispatched before the selected edited image completed");
    if (second) {
      h.complete(second->request_id, "image", true,
                 artifact_result("image", "image/png", source1));
    }
    auto video = h.next_submit(h.video_recv);
    check_explicit_anchor_submit(video, id + ":slot:2", source1, 1);
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    if (video) {
      h.complete(video->request_id, "video", true,
                 artifact_result("video", "video/mp4", "CHAINED-I2V"));
    }
    check(engine.wait(id, 5.0), "image-edit -> I2V reference chain did not settle");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              log.terminal_count(id) == 1,
          "image-edit reference chain did not produce exactly one complete terminal");
    if (terminal && terminal->legs.is_array()) {
      check(terminal->legs.as_array().size() == 3,
            "three-slot reference chain lost terminal legs");
      for (const Value& leg : terminal->legs.as_array()) {
        check(leg.find("state")->as_str() == "completed",
              "three-slot reference chain contains a non-completed leg");
        if (leg.find("slot_id")->as_str() == "slot-1") {
          const Value* anchor = leg.find("anchor_slot_index");
          const Value* sha = leg.find("anchor_sha256");
          check(anchor != nullptr && anchor->as_int() == 0 && sha != nullptr &&
                    sha->as_str() == util::sha256_hex(source0),
                "edited image terminal does not name slot-0 and its exact SHA-256");
        }
        if (leg.find("slot_id")->as_str() == "slot-2") {
          const Value* anchor = leg.find("anchor_slot_index");
          const Value* sha = leg.find("anchor_sha256");
          check(anchor != nullptr && anchor->as_int() == 1 && sha != nullptr &&
                    sha->as_str() == util::sha256_hex(source1),
                "I2V terminal does not name slot-1 and its exact SHA-256");
        }
      }
    }
    check(count_kind(log, UnifiedEvent::Kind::kArtifact) == 3,
          "three-slot reference chain did not emit exactly three artifacts");
    engine.release(id);
  }
  {
    // A failed intermediate image must fail its already-pending dependent too;
    // otherwise the public request can never join after the VLM completes.
    Harness h("explicitimagecascadefailure");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(3));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[image: root]]"
                        "[[image: edit | source=slot-0]]"
                        "[[video: move | source=slot-1]]", false));
    auto root = h.next_submit(h.image_recv);
    check(h.no_submit(h.image_recv) && h.no_submit(h.video_recv),
          "dependent chain dispatched before its sources settled");
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    if (root) {
      h.complete(root->request_id, "image", false, Value(),
                 std::string("root source failed"));
    }
    check(engine.wait(id, 5.0), "failed source left a transitive dependent pending");
    check(h.no_submit(h.image_recv) && h.no_submit(h.video_recv),
          "failed source chain generated a dependent artifact");
    const auto terminal = log.terminal_for(id);
    check(terminal && log.terminal_count(id) == 1,
          "failed source chain did not produce exactly one public terminal");
    size_t failed = 0;
    if (terminal && terminal->legs.is_array()) {
      for (const Value& leg : terminal->legs.as_array()) {
        if (leg.find("state")->as_str() == "failed") ++failed;
      }
    }
    check(failed == 3 && count_kind(log, UnifiedEvent::Kind::kLegError) == 3,
          "failed source chain did not finalize every dependent as failed");
    engine.release(id);
  }
}

void test_explicit_anchor_failure_cancel_and_validation() {
  {
    Harness h("explicitanchorfirstself");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(2));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[video: first dynamic | source=slot-0]]", false));
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    check(engine.wait(id, 5.0), "first-video self-reference did not settle");
    check(h.no_submit(h.video_recv) && h.no_submit(h.image_recv),
          "first-video self-reference was rewritten into generation");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              log.terminal_count(id) == 1 &&
              count_kind(log, UnifiedEvent::Kind::kArtifact) == 0 &&
              count_kind(log, UnifiedEvent::Kind::kLegError) == 1,
          "first-video self-reference did not fail closed with one public terminal");
    bool exact_failure = false;
    if (terminal && terminal->legs.is_array() &&
        terminal->legs.as_array().size() == 1) {
      const Value& leg = terminal->legs.as_array().front();
      exact_failure = leg.find("slot_id")->as_str() == "slot-0" &&
                      leg.find("state")->as_str() == "failed" &&
                      leg.find("error")->as_str() ==
                          "source slot slot-0 is not an earlier slot in this request";
    }
    check(exact_failure, "first-video self-reference lost its exact slot/error identity");
    engine.release(id);
  }
  {
    Harness h("explicitanchorfailure");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(2));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[image: source]][[video: move | source=slot-0]]", false));
    auto image = h.next_submit(h.image_recv);
    check(h.no_submit(h.video_recv), "failed explicit source dispatched early");
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    if (image) {
      h.complete(image->request_id, "image", false, Value(),
                 std::string("image exploded"));
    }
    check(engine.wait(id, 5.0), "explicit dependency failure did not settle");
    check(h.no_submit(h.video_recv),
          "explicit source failure silently fell back to T2V");
    const auto terminal = log.terminal_for(id);
    bool dependent_failed = false;
    if (terminal && terminal->legs.is_array()) {
      for (const Value& leg : terminal->legs.as_array()) {
        if (leg.find("slot_id")->as_str() == "slot-1") {
          dependent_failed = leg.find("state")->as_str() == "failed" &&
                             leg.find("anchor_explicit")->as_bool();
        }
      }
    }
    check(dependent_failed &&
              count_kind(log, UnifiedEvent::Kind::kLegError) == 2,
          "explicit dependency failure did not fail the dependent slot");
    engine.release(id);
  }
  {
    Harness h("explicitanchorlatefailure");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(2));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm", text_chunk("[[image: source]]", false));
    auto image = h.next_submit(h.image_recv);
    if (image) {
      h.complete(image->request_id, "image", false, Value(),
                 std::string("source already failed"));
    }
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[video: late | source=slot-0]]", false));
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    check(engine.wait(id, 5.0), "late explicit reference did not settle");
    check(h.no_submit(h.video_recv),
          "late explicit reference reused a failed source or fell back to T2V");
    const auto terminal = log.terminal_for(id);
    bool dependent_failed = false;
    if (terminal && terminal->legs.is_array()) {
      for (const Value& leg : terminal->legs.as_array()) {
        if (leg.find("slot_id")->as_str() == "slot-1") {
          dependent_failed = leg.find("state")->as_str() == "failed";
        }
      }
    }
    check(dependent_failed, "late reference did not retain a failed dependent leg");
    engine.release(id);
  }
  {
    Harness h("explicitanchorcancel");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(2));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[image: source]][[video: move | source=slot-0]]", false));
    auto image = h.next_submit(h.image_recv);
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    check(engine.cancel(id), "could not cancel pending explicit dependency");
    if (image) {
      check(h.wait_abort_on(h.image_recv, image->request_id),
            "cancel did not abort explicit source image");
      // A source can finish successfully before observing the abort. Its late
      // bytes and any later VLM/directive events must not revive the request.
      h.stream(vlm->request_id, "vlm",
               text_chunk("late [[image: stale slot]]", false));
      h.complete(image->request_id, "image", true,
                 artifact_result("image", "image/png", "LATE-STALE-PNG"));
    }
    check(engine.wait(id, 5.0), "cancelled explicit dependency did not settle");
    const size_t settled_events = log.snapshot().size();
    h.stream(vlm->request_id, "vlm",
             text_chunk("after terminal [[video: stale]]", false));
    if (image) {
      h.complete(image->request_id, "image", true,
                 artifact_result("image", "image/png", "AFTER-TERMINAL-PNG"));
    }
    check(log.snapshot().size() == settled_events,
          "post-terminal late events were not suppressed");
    check(h.no_submit(h.video_recv) && h.no_submit(h.image_recv),
          "cancelled explicit dependency dispatched late work");
    check(count_kind(log, UnifiedEvent::Kind::kArtifact) == 0,
          "late source completion leaked stale image bytes");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
              log.terminal_count(id) == 1,
          "late completion replaced or duplicated the cancellation terminal");
    bool source_aborted = false;
    bool dependent_aborted = false;
    for (const LegSnapshot& leg : engine.legs_snapshot(id)) {
      if (leg.slot_id == "slot-0") source_aborted = leg.state == LegState::kAborted;
      if (leg.slot_id == "slot-1") {
        dependent_aborted = leg.state == LegState::kAborted;
      }
    }
    check(source_aborted, "late source completion did not retain aborted source leg");
    check(dependent_aborted, "cancellation did not abort explicit dependent slot");
    engine.release(id);
  }
  {
    Harness h("explicitanchorinvalid");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request());
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(
        vlm->request_id, "vlm",
        text_chunk(
            "[[video: independent]]"
            "[[video: wrong modality | source=slot-0]]"
            "[[video: missing | source=slot-9]]"
            "[[video: forward | source=slot-4]]"
            "[[video: duplicate | source=slot-0 | source=slot-0]]"
            "[[image: wrong target | source=slot-0]]",
            false));
    auto independent = h.next_submit(h.video_recv);
    check(independent.has_value() && independent->request_id == id + ":slot:0",
          "ordinary i2v_chain auto-tag behavior changed");
    check(h.no_submit(h.video_recv) && h.no_submit(h.image_recv),
          "invalid/wrong/duplicate/forward explicit reference dispatched");
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    if (independent) {
      h.complete(independent->request_id, "video", true,
                 artifact_result("video", "video/mp4", "T2V"));
    }
    check(engine.wait(id, 5.0), "invalid explicit references did not settle");
    const auto terminal = log.terminal_for(id);
    size_t failed = 0;
    if (terminal && terminal->legs.is_array()) {
      check(terminal->legs.as_array().size() == 6,
            "invalid explicit references lost monotonic slot ids");
      for (const Value& leg : terminal->legs.as_array()) {
        if (leg.find("state")->as_str() == "failed") ++failed;
      }
    }
    check(failed == 5 && count_kind(log, UnifiedEvent::Kind::kLegError) == 5,
          "invalid explicit references did not fail all dependent slots");
    engine.release(id);
  }
  {
    Harness h("explicitanchormalformedself");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(2));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[video: malformed | source=slot-]]"
                        "[[video: self | source=slot-1]]", false));
    h.complete(vlm->request_id, "vlm", true, text_result("clean"));
    check(engine.wait(id, 5.0), "malformed/self references did not settle");
    check(h.no_submit(h.video_recv) && h.no_submit(h.image_recv),
          "malformed/self explicit reference dispatched generation");
    check(count_kind(log, UnifiedEvent::Kind::kArtifact) == 0 &&
              count_kind(log, UnifiedEvent::Kind::kLegError) == 2,
          "malformed/self controls produced bytes or missed slot errors");
    const auto terminal = log.terminal_for(id);
    size_t failed = 0;
    if (terminal && terminal->legs.is_array()) {
      check(terminal->legs.as_array().size() == 2,
            "malformed/self controls lost slot identity");
      for (const Value& leg : terminal->legs.as_array()) {
        if (leg.find("state")->as_str() == "failed") ++failed;
      }
    }
    check(failed == 2, "malformed/self controls did not fail closed");
    engine.release(id);
  }
  {
    Harness h("explicitanchoremptyprompt");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    const std::string id = engine.submit(explicit_anchor_request(1));
    auto vlm = h.next_submit(h.vlm_recv);
    if (!vlm) return;
    h.stream(vlm->request_id, "vlm",
             text_chunk("Clean prose. [[video: | source=slot-0]]", false));
    h.complete(vlm->request_id, "vlm", true, text_result("Clean prose."));
    check(engine.wait(id, 5.0), "empty explicit reference did not settle");
    check(h.no_submit(h.image_recv) && h.no_submit(h.video_recv),
          "empty explicit reference triggered an unrelated fallback visual");
    const auto terminal = log.terminal_for(id);
    check(count_kind(log, UnifiedEvent::Kind::kVisualSlot) == 1 &&
              count_kind(log, UnifiedEvent::Kind::kLegError) == 1 && terminal &&
              terminal->legs.is_array() && terminal->legs.as_array().size() == 1 &&
              terminal->legs.as_array().front().find("state")->as_str() == "failed",
          "empty explicit reference did not produce one failed slot");
    engine.release(id);
  }
  {
    // Session A's completed slot-0 is not a source for session B.
    Harness h("explicitanchorsession");
    UnifiedEngine engine(h.coordinator, h.mixed_config());
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
    engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };
    StagePayload first = narrative_request(true, false, false, "auto", 1);
    first.request.inputs["session_id"] = Value("session-A");
    const std::string first_id = engine.submit(std::move(first));
    auto first_vlm = h.next_submit(h.vlm_recv);
    if (!first_vlm) return;
    h.stream(first_vlm->request_id, "vlm", text_chunk("[[image: A]]", false));
    auto image = h.next_submit(h.image_recv);
    if (image) {
      h.complete(image->request_id, "image", true,
                 artifact_result("image", "image/png", "SESSION-A"));
    }
    h.complete(first_vlm->request_id, "vlm", true, text_result("A"));
    check(engine.wait(first_id, 5.0), "session A did not complete");
    engine.release(first_id);

    StagePayload second = explicit_anchor_request(1);
    second.request.inputs["session_id"] = Value("session-B");
    const std::string second_id = engine.submit(std::move(second));
    auto second_vlm = h.next_submit(h.vlm_recv);
    if (!second_vlm) return;
    h.stream(second_vlm->request_id, "vlm",
             text_chunk("[[video: cross session | source=slot-0]]", false));
    h.complete(second_vlm->request_id, "vlm", true, text_result("B"));
    check(engine.wait(second_id, 5.0), "cross-session reference did not settle");
    check(h.no_submit(h.video_recv),
          "cross-session explicit reference reused stale bytes");
    engine.release(second_id);
  }
}

void test_explicit_anchor_http_wire() {
  Harness h("explicitanchorhttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "explicit-anchor HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const json body = {
      {"model", "qwen3-unified"},
      {"messages", json::array({json{{"role", "user"}, {"content", "go"}}})},
      {"stream", true},
      {"modalities", json::array({"text", "image", "video"})},
      {"video", {{"size", "704x480"}, {"seed", 42}, {"steps", 7},
                  {"num_frames", 13}, {"guidance_scale", 4.5}}},
      {"narrative", {{"enabled", true}, {"max_slots", 2},
                      {"continuity_mode", "i2v_chain"}}}};
  auto future = std::async(std::launch::async, [&] {
    test_http::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    return test_http::perform(request);
  });
  auto vlm = h.next_submit(h.vlm_recv);
  if (vlm) {
    h.stream(vlm->request_id, "vlm",
             text_chunk("A. [[image: red boat]]", false));
    auto image = h.next_submit(h.image_recv);
    if (image) {
      h.complete(image->request_id, "image", true,
                 artifact_result("image", "image/png", "BOATPNG"));
    }
    h.stream(vlm->request_id, "vlm",
             text_chunk("[[video: camera orbits | source=slot-0]]", false));
    auto video = h.next_submit(h.video_recv);
    check_explicit_anchor_submit(video, vlm->request_id + ":slot:1", "BOATPNG", 0);
    if (video) {
      h.complete(video->request_id, "video", true,
                 artifact_result("video", "video/mp4", "BOATMP4"));
    }
    h.complete(vlm->request_id, "vlm", true, text_result("A."));
  }
  const auto response = future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "explicit-anchor HTTP SSE failed");
  bool visual_ok = false;
  bool artifact_ok = false;
  bool terminal_ok = false;
  for (const std::string& payload : sse_data_payloads(response.body)) {
    const json frame = json::parse(payload);
    if (!frame.contains("unified")) continue;
    const json& unified = frame["unified"];
    if (unified["kind"] == "visual_slot" &&
        unified.value("slot_id", "") == "slot-1") {
      visual_ok = unified.value("continuity_mode", "") == "i2v_chain" &&
                  unified.value("anchor_slot_index", -1) == 0 &&
                  unified.value("anchor_explicit", false) &&
                  !unified.contains("anchor_sha256") &&
                  unified["prompt"].get<std::string>().find("source=") ==
                      std::string::npos;
    }
    if (unified["kind"] == "artifact" &&
        unified["artifact"].value("media_type", "") == "video") {
      const json& artifact = unified["artifact"];
      artifact_ok = artifact.value("anchor_slot_index", -1) == 0 &&
                    artifact.value("anchor_explicit", false) &&
                    artifact.value("anchor_sha256", "") ==
                        util::sha256_hex("BOATPNG");
    }
    if (unified["kind"] == "complete" && unified.contains("legs")) {
      for (const json& leg : unified["legs"]) {
        if (leg.value("slot_id", "") == "slot-1") {
          terminal_ok = leg.value("anchor_slot_index", -1) == 0 &&
                        leg.value("anchor_explicit", false) &&
                        leg.value("anchor_sha256", "") ==
                            util::sha256_hex("BOATPNG");
        }
      }
    }
  }
  check(visual_ok && artifact_ok && terminal_ok,
        "explicit-anchor HTTP visual/artifact/terminal wire metadata is incomplete");
}

Value native_stage_telemetry(const std::string& request_id,
                             const std::string& lifecycle = "completed") {
  Value::Map out;
  out["schema_version"] = Value(int64_t{1});
  out["coverage"] = Value("COVERED");
  out["scope"] = Value("request-native-cuda-runtime");
  out["request_id"] = Value(request_id);
  out["stage"] = Value("vlm-native");
  out["backend"] = Value("qwen38");
  out["device_index"] = Value(int64_t{0});
  out["lifecycle_status"] = Value(lifecycle);
  out["execution"] = Value(Value::Map{{"mode", Value("serial")},
                                      {"batch_capacity", Value(int64_t{1})}});
  out["epoch"] = Value(Value::Map{
      {"id", Value(int64_t{7})}, {"reset", Value(true)},
      {"semantics", Value("new_allocation_ownership_epoch")}});
  out["allocated_bytes_peak"] = Value(int64_t{8192});
  out["reserved_bytes_peak"] = Value(int64_t{8192});
  out["allocated_bytes_final"] = Value(int64_t{4096});
  out["reserved_bytes_final"] = Value(int64_t{4096});
  out["sample_coverage"] = Value(Value::Map{
      {"allocation_events", Value(int64_t{2})},
      {"free_events", Value(int64_t{1})},
      {"terminal_snapshots", Value(int64_t{1})}});
  out["attribution"] = Value("native fixture");
  return Value(std::move(out));
}

Value asr_native_stage_telemetry(const std::string& request_id,
                                 const std::string& lifecycle = "completed",
                                 int64_t epoch = 7) {
  Value raw = native_stage_telemetry(request_id, lifecycle);
  raw["stage"] = Value("asr");
  raw["backend"] = Value("asr");
  raw["epoch"]["id"] = Value(epoch);
  for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                          "allocated_bytes_final", "reserved_bytes_final"}) {
    raw[key] = Value(int64_t{0});
  }
  raw["sample_coverage"] = Value(Value::Map{
      {"allocation_events", Value(int64_t{0})},
      {"free_events", Value(int64_t{0})},
      {"terminal_snapshots", Value(int64_t{1})}});
  return raw;
}

void test_public_http_request_native_allocator_telemetry() {
  Harness h("requesttelemetryhttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "request telemetry HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url = "http://127.0.0.1:" +
                          std::to_string(api.bound_port()) +
                          "/v1/chat/completions";
  auto post = [&](json body) {
    return std::async(std::launch::async, [url, body = std::move(body)] {
      test_http::HttpRequest request;
      request.url = url;
      request.body = body.dump();
      request.headers.emplace("Content-Type", "application/json");
      request.timeout = client::Milliseconds{15000};
      return test_http::perform(request);
    });
  };
  const json messages = json::array(
      {json{{"role", "user"}, {"content", "telemetry please"}}});

  // Invalid opt-ins fail before a stage request is created.
  httplib::Client raw_client("127.0.0.1", api.bound_port());
  for (const json& telemetry : {json{{"native_allocator", "yes"}},
                                json{{"unknown", true}}, json::array()}) {
    json bad = {{"model", "qwen3-unified"}, {"messages", messages},
                {"stream", false}, {"telemetry", telemetry}};
    auto response = raw_client.Post("/v1/chat/completions", bad.dump(),
                                    "application/json");
    check(response && response->status == 400,
          "malformed telemetry opt-in was not rejected");
  }

  json stream_body = {{"model", "qwen3-unified"}, {"messages", messages},
                      {"stream", true},
                      {"telemetry", {{"native_allocator", true}}}};
  auto stream_future = post(stream_body);
  auto stream_submit = h.next_submit(h.vlm_recv);
  check(stream_submit.has_value(), "telemetry SSE request did not reach VLM");
  StagePayload stream_payload;
  if (stream_submit) {
    stream_payload = StagePayload::from_value(stream_submit->data);
    check(obs::native_allocator_telemetry_requested(stream_payload.request.params),
          "HTTP telemetry opt-in did not reach native stage params");
    Value result = Value(text_result("covered answer"));
    result[obs::kRequestTelemetryResultKey] =
        native_stage_telemetry(stream_submit->request_id);
    h.complete(stream_submit->request_id, "vlm", true, std::move(result));
  }
  const auto stream_response = stream_future.get();
  check(stream_response.status == 200, "telemetry SSE request failed");
  check(stream_response.body.find("\"telemetry\"") != std::string::npos &&
            stream_response.body.find("\"coverage\":\"COVERED\"") !=
                std::string::npos &&
            stream_response.body.find("\"allocated_bytes_peak\":8192") !=
                std::string::npos &&
            stream_response.body.find("\"native_stage\":\"vlm-native\"") !=
                std::string::npos,
        "covered native allocator envelope missing from terminal SSE");
  check(stream_response.body.find("umm_gpu_memory") == std::string::npos,
        "whole-device NVML samples leaked into request allocator telemetry");

  json missing_body = {{"model", "qwen3-unified"}, {"messages", messages},
                       {"stream", false},
                       {"telemetry", {{"native_allocator", true}}}};
  auto missing_future = post(missing_body);
  auto missing_submit = h.next_submit(h.vlm_recv);
  check(missing_submit.has_value(), "missing telemetry request did not reach VLM");
  if (missing_submit) {
    h.complete(missing_submit->request_id, "vlm", true,
               Value(text_result("unsupported answer")));
  }
  const auto missing_response = missing_future.get();
  const json missing = json::parse(missing_response.body);
  check(missing["choices"][0]["message"]["content"] == "unsupported answer" &&
            missing["telemetry"]["coverage"] == "NOT_COVERED" &&
            missing["telemetry"]["stages"][0]["reason"] ==
                "stage_terminal_snapshot_missing" &&
            missing["telemetry"]["stages"][0]["allocated_bytes_peak"].is_null(),
        "unsupported stage did not produce explicit null-byte NOT_COVERED");

  json malformed_body = missing_body;
  auto malformed_future = post(malformed_body);
  auto malformed_submit = h.next_submit(h.vlm_recv);
  check(malformed_submit.has_value(), "malformed telemetry request did not reach VLM");
  if (malformed_submit) {
    Value raw = native_stage_telemetry(malformed_submit->request_id);
    raw["allocated_bytes_peak"] = Value(int64_t{-1});
    raw["model_path"] = Value("/secret/internal/model");
    Value result = Value(text_result("malformed answer"));
    result[obs::kRequestTelemetryResultKey] = std::move(raw);
    h.complete(malformed_submit->request_id, "vlm", true, std::move(result));
  }
  const auto malformed_response = malformed_future.get();
  const json malformed = json::parse(malformed_response.body);
  check(malformed["telemetry"]["coverage"] == "NOT_COVERED" &&
            malformed["telemetry"]["stages"][0]["reason"] ==
                "malformed_native_telemetry" &&
            malformed_response.body.find("/secret/internal/model") ==
                std::string::npos,
        "malformed metrics were not redacted and converted to NOT_COVERED");

  auto failed_future = std::async(std::launch::async, [&] {
    json body = missing_body;
    httplib::Client client("127.0.0.1", api.bound_port());
    return client.Post("/v1/chat/completions", body.dump(), "application/json");
  });
  auto failed_submit = h.next_submit(h.vlm_recv);
  check(failed_submit.has_value(), "failed telemetry request did not reach VLM");
  if (failed_submit) {
    Value result = Value(text_result("unused failed answer"));
    result[obs::kRequestTelemetryResultKey] =
        native_stage_telemetry(failed_submit->request_id, "failed");
    h.complete(failed_submit->request_id, "vlm", false, std::move(result),
               std::string("injected native failure"));
  }
  auto failed_response = failed_future.get();
  check(failed_response && failed_response->status == 500,
        "failed native request did not retain HTTP failure");
  if (failed_response) {
    const json failed = json::parse(failed_response->body);
    check(failed["telemetry"]["coverage"] == "COVERED" &&
              failed["telemetry"]["lifecycle_status"] == "failed" &&
              failed["telemetry"]["stages"][0]["lifecycle_status"] == "failed" &&
              failed["telemetry"]["stages"][0]["allocated_bytes_peak"] == 8192,
          "failed native request lost its covered terminal peak");
  }

  auto cancel_future = post(stream_body);
  auto cancel_submit = h.next_submit(h.vlm_recv);
  check(cancel_submit.has_value(), "cancel telemetry request did not reach VLM");
  if (cancel_submit) {
    check(engine.cancel(cancel_submit->request_id),
          "telemetry cancellation was not accepted");
  }
  const auto cancel_response = cancel_future.get();
  check(cancel_response.body.find("\"kind\":\"cancelled\"") != std::string::npos &&
            cancel_response.body.find("\"lifecycle_status\":\"cancelled\"") !=
                std::string::npos &&
            cancel_response.body.find("stage_terminal_snapshot_missing") !=
                std::string::npos,
        "immediate public cancellation did not retain truthful status/coverage");

  json default_body = {{"model", "qwen3-unified"}, {"messages", messages},
                       {"stream", false}};
  auto default_future = post(default_body);
  auto default_submit = h.next_submit(h.vlm_recv);
  check(default_submit.has_value(), "default telemetry regression request missing");
  if (default_submit) {
    StagePayload payload = StagePayload::from_value(default_submit->data);
    check(!obs::native_allocator_telemetry_requested(payload.request.params),
          "default request unexpectedly enabled native telemetry");
    h.complete(default_submit->request_id, "vlm", true,
               Value(text_result("legacy answer")));
  }
  const auto default_response = default_future.get();
  const json legacy = json::parse(default_response.body);
  check(!legacy.contains("telemetry") &&
            legacy["choices"][0]["message"]["content"] == "legacy answer",
        "default public JSON gained telemetry or changed its content");

  json mixed_body = missing_body;
  mixed_body["modalities"] = json::array({"text", "image"});
  mixed_body["image"] = {{"output_format", "png"}, {"size", "256x256"}};
  auto mixed_future = post(mixed_body);
  auto mixed_vlm = h.next_submit(h.vlm_recv);
  check(mixed_vlm.has_value(), "mixed telemetry request did not reach VLM");
  if (mixed_vlm) {
    Value result = Value(text_result("mixed answer"));
    result[obs::kRequestTelemetryResultKey] =
        native_stage_telemetry(mixed_vlm->request_id);
    h.complete(mixed_vlm->request_id, "vlm", true, std::move(result));
  }
  auto mixed_image = h.next_submit(h.image_recv);
  check(mixed_image.has_value(), "mixed telemetry request did not reach image");
  if (mixed_image) {
    Value result = artifact_result("image", "image/png", "PNG");
    Value raw = native_stage_telemetry(mixed_image->request_id);
    raw["stage"] = Value("image");
    raw["native_stage"] = Value("image");
    raw["backend"] = Value("image2");
    result[obs::kRequestTelemetryResultKey] = std::move(raw);
    h.complete(mixed_image->request_id, "image", true, std::move(result));
  }
  const auto mixed_response = mixed_future.get();
  const json mixed = json::parse(mixed_response.body);
  check(mixed_response.status == 200 && mixed.contains("telemetry") &&
            mixed["artifacts"].size() == 1 &&
            !mixed["artifacts"][0].contains(obs::kRequestTelemetryResultKey) &&
            mixed_response.body.find(obs::kRequestTelemetryResultKey) ==
                std::string::npos,
        "internal stage telemetry leaked through a nonstream artifact");

  // TTS off/on requests preserve the complete public JSON/SSE shape and exact
  // audio bytes. The fake native terminal intentionally models a warm TTS
  // request: no new direct allocations, covered zero bytes, and zero events.
  struct AudioTelemetryRun {
    test_http::HttpResponse response;
    std::string media_bytes;
    json normalized;
  };
  std::string pcm_bytes(4096, '\0');
  for (size_t i = 0; i < pcm_bytes.size(); ++i) {
    pcm_bytes[i] = static_cast<char>((i * 37 + 11) & 0xff);
  }
  const std::string wav_bytes = audio::wav_from_pcm16(pcm_bytes, 24000);
  auto run_tts_audio = [&](bool enabled, bool streamed) {
    json body = {
        {"model", "qwen3-unified"}, {"messages", messages},
        {"stream", streamed},
        {"modalities", json::array({"text", "audio"})},
        {"audio", {{"voice", "vivian"}, {"format", "wav"},
                   {"stream", streamed}}}};
    if (enabled) body["telemetry"] = {{"native_allocator", true}};
    auto future = post(body);
    auto vlm_submit = h.next_submit(h.vlm_recv);
    check(vlm_submit.has_value(), "TTS telemetry request did not reach VLM");
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    check(obs::native_allocator_telemetry_requested(vlm_payload.request.params) ==
              enabled,
          "TTS request changed the VLM telemetry opt-in");
    Value vlm_result = text_result("PCM parity.");
    if (enabled) {
      vlm_result[obs::kRequestTelemetryResultKey] =
          native_stage_telemetry(vlm_submit->request_id);
    }
    h.complete(vlm_submit->request_id, "vlm", true, std::move(vlm_result));

    auto tts_submit = h.next_submit(h.tts_recv);
    check(tts_submit.has_value(), "TTS telemetry request did not reach TTS");
    StagePayload tts_payload = StagePayload::from_value(tts_submit->data);
    check(obs::native_allocator_telemetry_requested(tts_payload.request.params) ==
              enabled,
          "TTS output leg did not preserve its telemetry opt-in");
    if (streamed) {
      Value::Map chunk;
      chunk["audio"] = Value::bin(pcm_bytes);
      chunk["index"] = Value(int64_t{0});
      chunk["first_frame"] = Value(int64_t{0});
      chunk["frames"] = Value(int64_t{1});
      chunk["sample_rate"] = Value(int64_t{24000});
      chunk["channels"] = Value(int64_t{1});
      chunk["format"] = Value("pcm16");
      h.stream(tts_submit->request_id, "tts", Value(std::move(chunk)));
    }
    Value::Map extra;
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    extra["streamed"] = Value(streamed);
    Value tts_result = artifact_result("audio", "audio/wav", wav_bytes,
                                       std::move(extra));
    if (enabled) {
      Value raw = native_stage_telemetry(tts_submit->request_id);
      raw["stage"] = Value("tts");
      raw["backend"] = Value("tts");
      for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                              "allocated_bytes_final", "reserved_bytes_final"}) {
        raw[key] = Value(int64_t{0});
      }
      raw["sample_coverage"] = Value(Value::Map{
          {"allocation_events", Value(int64_t{0})},
          {"free_events", Value(int64_t{0})},
          {"terminal_snapshots", Value(int64_t{1})}});
      tts_result[obs::kRequestTelemetryResultKey] = std::move(raw);
    }
    h.complete(tts_submit->request_id, "tts", true, std::move(tts_result));
    const auto response = future.get();
    check(response.status == 200, "TTS off/on public request failed");

    AudioTelemetryRun out{response, {}, {}};
    json public_body;
    std::string public_id;
    if (streamed) {
      public_body = json::array();
      for (const std::string& payload : sse_data_payloads(response.body)) {
        if (payload == "[DONE]") continue;
        json frame = json::parse(payload);
        if (public_id.empty() && frame.contains("id")) {
          public_id = frame["id"].get<std::string>();
        }
        if (frame.contains("unified") &&
            frame["unified"].value("kind", "") == "artifact") {
          const json& artifact = frame["unified"]["artifact"];
          if (artifact.value("mime_type", "") == "audio/pcm") {
            out.media_bytes += util::base64_decode(
                artifact["data"].get<std::string>(), pcm_bytes.size() + 16);
          }
        }
        public_body.push_back(std::move(frame));
      }
      check(out.media_bytes == pcm_bytes,
            "streamed public PCM changed under telemetry");
    } else {
      public_body = json::parse(response.body);
      public_id = public_body["id"].get<std::string>();
      check(public_body["artifacts"].size() == 1 &&
                !public_body["artifacts"][0].contains(
                    obs::kRequestTelemetryResultKey),
            "nonstream TTS telemetry leaked into its artifact");
      out.media_bytes = util::base64_decode(
          public_body["artifacts"][0]["data"].get<std::string>(),
          wav_bytes.size() + 16);
      check(out.media_bytes == wav_bytes, "nonstream TTS WAV changed");
    }
    check((response.body.find("\"telemetry\"") != std::string::npos) == enabled,
          "TTS telemetry field presence differs from its opt-in");
    if (enabled) {
      check(response.body.find("\"native_stage\":\"tts\"") != std::string::npos &&
                response.body.find("\"allocated_bytes_peak\":0") !=
                    std::string::npos,
            "covered warm TTS terminal telemetry is missing");
    }

    std::function<json(const json&)> normalize = [&](const json& value) -> json {
      if (value.is_object()) {
        json object = json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
          if (it.key() == "created" || it.key() == "session_id" ||
              it.key() == "telemetry") {
            continue;
          }
          object[it.key()] = normalize(it.value());
        }
        return object;
      }
      if (value.is_array()) {
        json array = json::array();
        for (const json& item : value) array.push_back(normalize(item));
        return array;
      }
      if (value.is_string()) {
        std::string text = value.get<std::string>();
        for (const std::string& id : {public_id, vlm_submit->request_id}) {
          if (id.empty()) continue;
          size_t at = 0;
          while ((at = text.find(id, at)) != std::string::npos) {
            text.replace(at, id.size(), "<request>");
            at += 9;
          }
        }
        return text;
      }
      return value;
    };
    out.normalized = normalize(public_body);
    return out;
  };

  const auto audio_off = run_tts_audio(false, false);
  const auto audio_on = run_tts_audio(true, false);
  const auto audio_warm = run_tts_audio(true, false);
  check(audio_off.normalized == audio_on.normalized &&
            audio_on.normalized == audio_warm.normalized &&
            audio_off.media_bytes == audio_on.media_bytes &&
            audio_on.media_bytes == audio_warm.media_bytes,
        "TTS JSON/repeated-warm protocol or audio parity failed");
  const auto pcm_off = run_tts_audio(false, true);
  const auto pcm_on = run_tts_audio(true, true);
  check(pcm_off.normalized == pcm_on.normalized &&
            pcm_off.media_bytes == pcm_on.media_bytes &&
            util::sha256_hex(pcm_off.media_bytes) ==
                util::sha256_hex(pcm_on.media_bytes),
        "TTS SSE/PCM off-on protocol parity failed");

  api.stop();
  server_thread.join();
}

void test_public_http_asr_request_native_allocator_telemetry() {
  Harness h("asrrequesttelemetryhttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "ASR request telemetry HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url = "http://127.0.0.1:" +
                          std::to_string(api.bound_port()) +
                          "/v1/chat/completions";
  auto post = [&](json body) {
    return std::async(std::launch::async, [url, body = std::move(body)] {
      test_http::HttpRequest request;
      request.url = url;
      request.body = body.dump();
      request.headers.emplace("Content-Type", "application/json");
      request.timeout = client::Milliseconds{15000};
      return test_http::perform(request);
    });
  };

  std::string pcm(3200, '\0');
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = static_cast<char>((i * 29 + 7) & 0xff);
  }
  const std::string wav = audio::wav_from_pcm16(pcm, 16000);
  const json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", util::base64_encode(wav)}, {"format", "wav"}}}};
  const json messages = json::array(
      {json{{"role", "user"}, {"content", json::array({audio_part})}}});
  auto body_for = [&](bool enabled, bool streamed) {
    json body = {{"model", "qwen3-unified"}, {"messages", messages},
                 {"stream", streamed}, {"max_tokens", 16}, {"temperature", 0}};
    if (enabled) body["telemetry"] = {{"native_allocator", true}};
    return body;
  };

  struct AsrTelemetryRun {
    test_http::HttpResponse response;
    json normalized;
    std::string transcript;
    std::string answer;
  };
  int64_t next_epoch = 20;
  auto run_asr = [&](bool enabled, bool streamed) {
    auto future = post(body_for(enabled, streamed));
    auto asr_submit = h.next_submit(h.asr_recv);
    check(asr_submit.has_value(), "ASR telemetry request did not reach ASR");
    StagePayload asr_payload = StagePayload::from_value(asr_submit->data);
    check(asr_payload.request.inputs.find("audio")->as_bin() == pcm &&
              obs::native_allocator_telemetry_requested(asr_payload.request.params) ==
                  enabled,
          "ASR telemetry changed PCM or its stage opt-in");
    Value asr_result = text_result("allocator transcript");
    if (enabled) {
      asr_result[obs::kRequestTelemetryResultKey] = asr_native_stage_telemetry(
          asr_submit->request_id, "completed", next_epoch++);
    }
    h.complete(asr_submit->request_id, "asr", true, std::move(asr_result));

    auto vlm_submit = h.next_submit(h.vlm_recv);
    check(vlm_submit.has_value(), "ASR telemetry request did not reach VLM");
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& vlm_messages = *vlm_payload.request.inputs.find("messages");
    check(vlm_messages.as_array().at(0).find("content")->as_str() ==
              "allocator transcript" &&
              obs::native_allocator_telemetry_requested(vlm_payload.request.params) ==
                  enabled,
          "ASR transcript or VLM telemetry opt-in changed");
    if (streamed) {
      h.stream(vlm_submit->request_id, "vlm", text_chunk("telemetry answer", false));
    }
    Value vlm_result = text_result("telemetry answer");
    vlm_result["finish_reason"] = Value("stop");
    if (enabled) {
      vlm_result[obs::kRequestTelemetryResultKey] =
          native_stage_telemetry(vlm_submit->request_id);
    }
    h.complete(vlm_submit->request_id, "vlm", true, std::move(vlm_result));

    AsrTelemetryRun out{future.get(), json(), {}, {}};
    check(out.response.status == 200, "ASR off/on public request failed");
    json public_body;
    std::string public_id;
    if (streamed) {
      public_body = json::array();
      for (const std::string& payload : sse_data_payloads(out.response.body)) {
        if (payload == "[DONE]") continue;
        json frame = json::parse(payload);
        if (public_id.empty() && frame.contains("id")) {
          public_id = frame["id"].get<std::string>();
        }
        if (frame.contains("unified")) {
          const json& unified = frame["unified"];
          if (unified.value("kind", "") == "asr_final") {
            out.transcript = unified.value("text", "");
          }
        }
        if (frame.contains("choices") && !frame["choices"].empty() &&
            frame["choices"][0].contains("delta")) {
          out.answer += frame["choices"][0]["delta"].value("content", "");
        }
        public_body.push_back(std::move(frame));
      }
    } else {
      public_body = json::parse(out.response.body);
      public_id = public_body["id"].get<std::string>();
      out.transcript = vlm_messages.as_array().at(0).find("content")->as_str();
      out.answer = public_body["choices"][0]["message"]["content"];
    }
    check(out.transcript == "allocator transcript" &&
              out.answer == "telemetry answer",
          "ASR off/on transcript or answer changed");
    check((out.response.body.find("\"telemetry\"") != std::string::npos) == enabled,
          "ASR telemetry field presence differs from its opt-in");
    if (enabled) {
      check(out.response.body.find("\"native_stage\":\"asr\"") != std::string::npos &&
                out.response.body.find("\"allocated_bytes_peak\":0") !=
                    std::string::npos,
            "covered warm ASR zero-event telemetry is missing");
    }

    std::function<json(const json&)> normalize = [&](const json& value) -> json {
      if (value.is_object()) {
        json object = json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
          if (it.key() == "created" || it.key() == "session_id" ||
              it.key() == "telemetry") {
            continue;
          }
          object[it.key()] = normalize(it.value());
        }
        return object;
      }
      if (value.is_array()) {
        json array = json::array();
        for (const json& item : value) array.push_back(normalize(item));
        return array;
      }
      if (value.is_string()) {
        std::string text = value.get<std::string>();
        for (const std::string& id :
             {public_id, asr_submit->request_id, vlm_submit->request_id}) {
          if (id.empty()) continue;
          size_t at = 0;
          while ((at = text.find(id, at)) != std::string::npos) {
            text.replace(at, id.size(), "<request>");
            at += 9;
          }
        }
        return text;
      }
      return value;
    };
    out.normalized = normalize(public_body);
    return out;
  };

  const auto json_off = run_asr(false, false);
  const auto json_on = run_asr(true, false);
  const auto json_warm = run_asr(true, false);
  check(json_off.normalized == json_on.normalized &&
            json_on.normalized == json_warm.normalized,
        "ASR JSON/repeated-warm protocol parity failed");
  const auto sse_off = run_asr(false, true);
  const auto sse_on = run_asr(true, true);
  check(sse_off.normalized == sse_on.normalized,
        "ASR SSE/transcript off-on protocol parity failed");

  // A covered native ASR failure retains its observed terminal record and never
  // starts VLM. The zero values mean no new allocations, not missing coverage.
  auto failed_future = std::async(std::launch::async, [&] {
    json body = body_for(true, false);
    httplib::Client client("127.0.0.1", api.bound_port());
    return client.Post("/v1/chat/completions", body.dump(), "application/json");
  });
  auto failed_submit = h.next_submit(h.asr_recv);
  check(failed_submit.has_value(), "failed ASR telemetry request missing");
  Value failed_result = text_result("unused failed transcript");
  failed_result[obs::kRequestTelemetryResultKey] = asr_native_stage_telemetry(
      failed_submit->request_id, "failed", next_epoch++);
  h.complete(failed_submit->request_id, "asr", false, std::move(failed_result),
             std::string("injected invalid native audio"));
  const auto failed_response = failed_future.get();
  check(failed_response && failed_response->status == 500 &&
            failed_response->body.find("\"native_stage\":\"asr\"") !=
                std::string::npos &&
            failed_response->body.find("\"lifecycle_status\":\"failed\"") !=
                std::string::npos &&
            failed_response->body.find("\"allocated_bytes_peak\":0") !=
                std::string::npos,
        "failed ASR request lost its covered terminal telemetry");

  // Malformed ASR telemetry is allow-list sanitized at the public projection;
  // native paths and negative byte counts never escape into the response.
  auto malformed_future = post(body_for(true, false));
  auto malformed_submit = h.next_submit(h.asr_recv);
  check(malformed_submit.has_value(), "malformed ASR telemetry request missing");
  Value malformed = asr_native_stage_telemetry(malformed_submit->request_id);
  malformed["allocated_bytes_peak"] = Value(int64_t{-1});
  malformed["model_path"] = Value("/secret/internal/asr-model");
  Value malformed_result = text_result("sanitized transcript");
  malformed_result[obs::kRequestTelemetryResultKey] = std::move(malformed);
  h.complete(malformed_submit->request_id, "asr", true, std::move(malformed_result));
  auto malformed_vlm = h.next_submit(h.vlm_recv);
  check(malformed_vlm.has_value(), "sanitization request did not reach VLM");
  Value sanitized_vlm_result = text_result("sanitized answer");
  sanitized_vlm_result[obs::kRequestTelemetryResultKey] =
      native_stage_telemetry(malformed_vlm->request_id);
  h.complete(malformed_vlm->request_id, "vlm", true,
             std::move(sanitized_vlm_result));
  const auto malformed_response = malformed_future.get();
  check(malformed_response.status == 200 &&
            malformed_response.body.find("malformed_native_telemetry") !=
                std::string::npos &&
            malformed_response.body.find("/secret/internal/asr-model") ==
                std::string::npos,
        "malformed ASR telemetry was not sanitized to NOT_COVERED");

  // Immediate public cancellation is truthful about the missing native terminal
  // instead of inventing a late covered ASR allocation snapshot.
  auto cancel_future = post(body_for(true, true));
  auto cancel_submit = h.next_submit(h.asr_recv);
  check(cancel_submit.has_value(), "cancel ASR telemetry request missing");
  check(engine.cancel(cancel_submit->request_id),
        "ASR telemetry cancellation was not accepted");
  const auto cancel_response = cancel_future.get();
  check(cancel_response.body.find("\"kind\":\"cancelled\"") != std::string::npos &&
            cancel_response.body.find("\"lifecycle_status\":\"cancelled\"") !=
                std::string::npos &&
            cancel_response.body.find("stage_terminal_snapshot_missing") !=
                std::string::npos,
        "ASR immediate cancellation did not retain truthful status/coverage");

  api.stop();
  server_thread.join();
}

void test_openai_client_teacache_and_narrative() {
  Harness h("clientparity");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "client parity server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  client::OpenAIConfig cfg;
  cfg.base_url = "http://127.0.0.1:" + std::to_string(api.bound_port()) + "/v1";
  cfg.model = "qwen3-unified";
  cfg.timeout = client::Milliseconds{8000};
  client::OpenAIClient cli(cfg);

  auto expect_config_error = [&](const client::ChatRequest& req,
                                 const char* label) {
    bool caught = false;
    try {
      (void)cli.chat(req);
    } catch (const client::Error& err) {
      caught = err.code() == client::ErrorCode::kConfig;
    }
    check(caught, label);
  };

  // 1. Fail-closed client-side validation for video.teacache_thresh and narrative.
  for (double bad_thresh :
       {-0.01, 0.251, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    client::ChatRequest req;
    req.messages.push_back({"user", "bad thresh"});
    req.modalities = {"text", "video"};
    client::VideoOutput vo;
    vo.teacache_thresh = bad_thresh;
    req.video = vo;
    expect_config_error(req, "OpenAIClient accepted invalid video.teacache_thresh");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "bad visual_mode"});
    req.stream = true;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    n.visual_mode = "audio";
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted invalid narrative.visual_mode");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "bad continuity"});
    req.stream = true;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    n.continuity_mode = "chain";
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted invalid narrative.continuity_mode");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "bad max_slots"});
    req.stream = true;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    n.max_slots = 7;
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted out-of-range narrative.max_slots");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "oversized style_anchor"});
    req.stream = true;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    n.style_anchor = std::string(257, 'x');
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted >256-byte narrative.style_anchor");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "nonstream narrative"});
    req.stream = false;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted narrative without stream=true");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "no visual modality"});
    req.stream = true;
    req.modalities = {"text"};
    client::NarrativeOptions n;
    n.enabled = true;
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted narrative without visual modality");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "mismatched mode"});
    req.stream = true;
    req.modalities = {"text", "image"};
    client::NarrativeOptions n;
    n.enabled = true;
    n.visual_mode = "video";
    req.narrative = n;
    expect_config_error(req, "OpenAIClient accepted narrative.visual_mode=video without video modality");
  }

  // 2. VideoOutput without teacache_thresh omits it; with teacache_thresh forwards it.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "default video"});
    req.modalities = {"text", "video"};
    req.video = client::VideoOutput{};
    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    if (auto vlm = h.next_submit(h.vlm_recv)) {
      h.complete(vlm->request_id, "vlm", true, text_result("ok"));
    }
    if (auto vid = h.next_submit(h.video_recv)) {
      StagePayload p = StagePayload::from_value(vid->data);
      check(p.request.inputs.find("teacache_thresh") == nullptr,
            "OpenAIClient injected teacache_thresh when unset");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4DEF"));
    }
    check(fut.get().artifacts.size() == 1,
          "OpenAIClient default video roundtrip failed");
  }
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "fast video"});
    req.modalities = {"text", "video"};
    client::VideoOutput vo;
    vo.teacache_thresh = 0.05;
    req.video = vo;
    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    if (auto vlm = h.next_submit(h.vlm_recv)) {
      h.complete(vlm->request_id, "vlm", true, text_result("fast"));
    }
    if (auto vid = h.next_submit(h.video_recv)) {
      StagePayload p = StagePayload::from_value(vid->data);
      const Value* thresh = p.request.inputs.find("teacache_thresh");
      check(thresh != nullptr && std::abs(thresh->as_double() - 0.05) < 1e-9,
            "OpenAIClient did not forward video.teacache_thresh=0.05");
      h.complete(vid->request_id, "video", true,
                 artifact_result("video", "video/mp4", "MP4FAST"));
    }
    check(fut.get().artifacts.size() == 1,
          "OpenAIClient fast video roundtrip failed");
  }

  // 3. Streaming narrative with continuity_mode, style_anchor, allow_text_only,
  //    and video.teacache_thresh roundtrips end-to-end through UnifiedEngine.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "tell a story"});
    req.stream = true;
    req.modalities = {"text", "image", "video"};
    client::VideoOutput vo;
    vo.teacache_thresh = 0.05;
    req.video = vo;
    client::NarrativeOptions n;
    n.enabled = true;
    n.visual_mode = "auto";
    n.allow_text_only = false;
    n.continuity_mode = "i2v_chain";
    n.style_anchor = "  35mm film grain  ";
    req.narrative = n;

    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "OpenAIClient narrative request did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("Scene one. [[image: warm lantern]] Scene two. [[video: drifting boat]]",
                          false));
      auto img = h.next_submit(h.image_recv);
      check(img.has_value(), "OpenAIClient narrative slot 0 image missing");
      if (img) {
        StagePayload p = StagePayload::from_value(img->data);
        check(p.request.inputs.find("prompt")->as_str().find("35mm film grain") !=
                  std::string::npos,
              "OpenAIClient narrative style_anchor did not reach slot 0");
        h.complete(img->request_id, "image", true,
                   artifact_result("image", "image/png", "PNG_KEYFRAME"));
      }
      auto vid = h.next_submit(h.video_recv);
      check(vid.has_value(), "OpenAIClient narrative slot 1 video missing");
      if (vid) {
        StagePayload p = StagePayload::from_value(vid->data);
        const Value* thresh = p.request.inputs.find("teacache_thresh");
        check(thresh != nullptr && std::abs(thresh->as_double() - 0.05) < 1e-9,
              "OpenAIClient narrative video slot missing teacache_thresh=0.05");
        check(p.request.inputs.find("image") != nullptr &&
                  p.request.inputs.find("image")->as_str().rfind(
                      "data:image/png;base64,", 0) == 0,
              "OpenAIClient narrative i2v_chain did not attach slot 0 image");
        h.complete(vid->request_id, "video", true,
                   artifact_result("video", "video/mp4", "MP4_CHAIN"));
      }
      h.complete(vlm->request_id, "vlm", true, text_result("Scene one. Scene two."));
    }
    const client::ChatResponse resp = fut.get();
    check(resp.text == "Scene one.  Scene two. " &&
              resp.text.find("[[") == std::string::npos,
          ("OpenAIClient narrative clean prose mismatch: '" + resp.text + "'")
              .c_str());
    check(resp.artifacts.size() == 2,
          "OpenAIClient narrative expected 2 artifacts");
  }

  api.stop();
  server_thread.join();
  wait_until([&]() -> bool { return engine.in_flight() == 0; },
             "client parity requests stayed in flight");
}

Value native_behavior_result(const std::string& text, int64_t count = 3) {
  Value result = Value(text_result(text));
  result["generated_token_count"] = Value(count);
  result["finish_reason"] = Value("stop");
  result[obs::kBehaviorLogprobsResultKey] = obs::behavior_logprobs_value(
      {11, 12, 13}, {-0.125, -0.25, -0.5}, {1, 0, 1}, std::string(64, 'a'));
  return result;
}

std::string behavior_response_header(const test_http::HttpResponse& response,
                                     const std::string& name) {
  for (const auto& [key, value] : response.headers) {
    std::string lower = key;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (lower == name) return value;
  }
  return {};
}

void test_public_http_behavior_logprobs() {
  Harness h("behaviorlogprobshttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "behavior HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string url = "http://127.0.0.1:" +
                          std::to_string(api.bound_port()) +
                          "/v1/chat/completions";
  auto post = [&](json body, std::map<std::string, std::string> headers = {}) {
    return std::async(std::launch::async,
                      [url, body = std::move(body), headers = std::move(headers)] {
      test_http::HttpRequest request;
      request.url = url;
      request.body = body.dump();
      request.headers = headers;
      request.headers.emplace("Content-Type", "application/json");
      request.timeout = client::Milliseconds{15000};
      return test_http::perform(request);
    });
  };
  const json messages = json::array(
      {json{{"role", "user"}, {"content", "show evidence"}}});
  const json base = {{"model", "qwen3-unified"}, {"messages", messages},
                     {"stream", true}};
  const std::map<std::string, std::string> attestation = {
      {"x-umminfer-online-rollout-contract", "1"},
      {"x-umminfer-rollout-generation", "fresh"},
      {"x-umminfer-rollout-batch-id", "batch-42"},
      {"x-umminfer-rollout-group-id", "group-42"},
      {"x-umminfer-rollout-id", "rollout-42"},
      {"x-umminfer-rollout-index", "0"},
      {"x-umminfer-rollout-attempt", "0"},
      {"x-umminfer-request-sha256", std::string(64, '1')},
      {"x-umminfer-behavior-policy-sha256", std::string(64, '2')},
      {"x-umminfer-behavior-policy-revision", "policy-v1"},
      {"x-umminfer-behavior-model-id", "qwen3-unified"},
      {"x-umminfer-behavior-model-revision", "UNKNOWN"},
      {"x-umminfer-behavior-model-sha256", std::string(64, '3')},
      {"x-umminfer-behavior-model-config-sha256", std::string(64, '4')},
      {"x-umminfer-sampling-sha256", std::string(64, '5')},
  };

  httplib::Client raw_client("127.0.0.1", api.bound_port());
  for (const json& behavior : {json{{"logprobs", "yes"}},
                               json{{"unknown", true}}, json::array()}) {
    json bad = base;
    bad["behavior"] = behavior;
    httplib::Headers headers;
    for (const auto& [key, value] : attestation) headers.emplace(key, value);
    auto response = raw_client.Post("/v1/chat/completions", headers, bad.dump(),
                                    "application/json");
    check(response && response->status == 400 &&
              response->get_header_value("x-umminfer-behavior-evidence") ==
                  "not_exposed" &&
              response->get_header_value("x-umminfer-online-rollout-contract") == "1",
          "malformed behavior opt-in failed closed with attestation");
  }
  json nonstream = base;
  nonstream["stream"] = false;
  nonstream["behavior"] = json{{"logprobs", true}};
  auto nonstream_response = raw_client.Post(
      "/v1/chat/completions", nonstream.dump(), "application/json");
  check(nonstream_response && nonstream_response->status == 400,
        "behavior evidence requires the final SSE contract");

  json covered_body = base;
  covered_body["behavior"] = json{{"logprobs", true}};
  auto covered_future = post(covered_body, attestation);
  auto covered_submit = h.next_submit(h.vlm_recv);
  check(covered_submit.has_value(), "covered behavior request did not reach VLM");
  if (covered_submit) {
    StagePayload payload = StagePayload::from_value(covered_submit->data);
    check(obs::behavior_logprobs_requested(payload.request.params) &&
              !payload.request.params.contains(obs::kBehaviorNarrativeParam),
          "text behavior opt-in reached only the native VLM parameters");
    h.stream(covered_submit->request_id, "vlm", text_chunk("covered", false));
    h.complete(covered_submit->request_id, "vlm", true,
               native_behavior_result("covered"));
  }
  const auto covered = covered_future.get();
  check(covered.status == 200 &&
            behavior_response_header(covered, "x-umminfer-behavior-evidence") ==
                "exposed" &&
            behavior_response_header(covered, "x-umminfer-rollout-id") ==
                "rollout-42" &&
            behavior_response_header(covered, "x-umminfer-sampling-sha256") ==
                std::string(64, '5'),
        "covered response preserved exact online attestation headers");
  const auto covered_frames = sse_data_payloads(covered.body);
  size_t behavior_blocks = 0;
  for (size_t i = 0; i < covered_frames.size(); ++i) {
    const json frame = json::parse(covered_frames[i]);
    if (frame.contains("behavior")) {
      ++behavior_blocks;
      check(i + 1 == covered_frames.size() &&
                frame["behavior"].size() == 5 &&
                frame["behavior"]["format"] == obs::kBehaviorLogprobsFormat &&
                frame["behavior"]["token_ids"] == json::array({11, 12, 13}) &&
                frame["behavior"]["mask"] == json::array({1, 0, 1}),
            "behavior evidence was not the exact final five-field block");
    }
  }
  check(behavior_blocks == 1 && covered.body.ends_with("data: [DONE]\n\n"),
        "covered SSE contains one final behavior block before DONE");

  auto default_future = post(base);
  auto default_submit = h.next_submit(h.vlm_recv);
  if (default_submit) {
    StagePayload payload = StagePayload::from_value(default_submit->data);
    check(!obs::behavior_logprobs_requested(payload.request.params),
          "default request unexpectedly enabled behavior capture");
    h.complete(default_submit->request_id, "vlm", true,
               native_behavior_result("legacy"));
  }
  const auto legacy = default_future.get();
  check(legacy.status == 200 && legacy.body.find("\"behavior\"") == std::string::npos &&
            behavior_response_header(legacy, "x-umminfer-behavior-evidence").empty(),
        "default protocol projected unsolicited native behavior fields");

  std::vector<Value> invalid;
  Value missing = Value(text_result("missing"));
  missing["generated_token_count"] = Value(int64_t{3});
  missing["finish_reason"] = Value("stop");
  invalid.push_back(std::move(missing));
  Value truncated = native_behavior_result("truncated");
  truncated["finish_reason"] = Value("length");
  invalid.push_back(std::move(truncated));
  Value positive = native_behavior_result("positive");
  {
    Value::Map evidence = positive["behavior"].as_map();
    evidence["logprobs"] = Value(Value::Array{Value(0.1), Value(-0.2), Value(-0.3)});
    positive["behavior"] = Value(std::move(evidence));
  }
  invalid.push_back(std::move(positive));
  Value misaligned = native_behavior_result("misaligned");
  {
    Value::Map evidence = misaligned["behavior"].as_map();
    evidence["mask"] = Value(Value::Array{Value(int64_t{1})});
    misaligned["behavior"] = Value(std::move(evidence));
  }
  invalid.push_back(std::move(misaligned));
  Value zero_mask = native_behavior_result("zero-mask");
  {
    Value::Map evidence = zero_mask["behavior"].as_map();
    evidence["mask"] = Value(Value::Array{Value(int64_t{0}), Value(int64_t{0}),
                                          Value(int64_t{0})});
    zero_mask["behavior"] = Value(std::move(evidence));
  }
  invalid.push_back(std::move(zero_mask));
  Value bad_hash = native_behavior_result("bad-hash");
  {
    Value::Map evidence = bad_hash["behavior"].as_map();
    evidence["tokenizer_sha256"] = Value("bad");
    bad_hash["behavior"] = Value(std::move(evidence));
  }
  invalid.push_back(std::move(bad_hash));
  Value bad_count = native_behavior_result("bad-count", 2);
  invalid.push_back(std::move(bad_count));
  for (Value& result : invalid) {
    auto future = post(covered_body);
    auto submit = h.next_submit(h.vlm_recv);
    check(submit.has_value(), "invalid behavior case did not reach VLM");
    if (submit) h.complete(submit->request_id, "vlm", true, std::move(result));
    const auto response = future.get();
    check(response.status == 200 &&
              behavior_response_header(response, "x-umminfer-behavior-evidence") ==
                  "not_exposed" &&
              response.body.find("\"behavior\"") == std::string::npos,
          "invalid/truncated evidence did not fail closed to not_exposed");
  }

  auto cancel_future = post(covered_body);
  auto cancel_submit = h.next_submit(h.vlm_recv);
  if (cancel_submit) {
    check(engine.cancel(cancel_submit->request_id),
          "behavior cancellation was not accepted");
  }
  const auto cancelled = cancel_future.get();
  check(cancelled.body.find("\"kind\":\"cancelled\"") != std::string::npos &&
            cancelled.body.find("\"behavior\"") == std::string::npos &&
            behavior_response_header(cancelled, "x-umminfer-behavior-evidence") ==
                "not_exposed",
        "cancelled behavior stream retained partial or misleading evidence");

  json narrative_body = covered_body;
  narrative_body["modalities"] = json::array({"text", "image"});
  narrative_body["image"] = json{{"output_format", "png"}, {"size", "256x256"}};
  narrative_body["narrative"] = json{{"enabled", true}, {"visual_mode", "image"},
                                     {"max_slots", 1}};
  auto narrative_future = post(narrative_body);
  auto narrative_vlm = h.next_submit(h.vlm_recv);
  if (narrative_vlm) {
    StagePayload payload = StagePayload::from_value(narrative_vlm->data);
    check(payload.request.params.contains(obs::kBehaviorNarrativeParam),
          "narrative behavior alignment flag did not reach the VLM");
    h.stream(narrative_vlm->request_id, "vlm",
             text_chunk("A. [[image: red boat]] Tail", false));
  }
  auto narrative_image = h.next_submit(h.image_recv);
  if (narrative_image) {
    StagePayload payload = StagePayload::from_value(narrative_image->data);
    check(!obs::behavior_logprobs_requested(payload.request.params),
          "behavior capture leaked into the image output leg");
    h.complete(narrative_image->request_id, "image", true,
               artifact_result("image", "image/png", "PNG"));
  }
  if (narrative_vlm) {
    Value result = native_behavior_result("A. [[image: red boat]] Tail", 4);
    Value::Map evidence = result["behavior"].as_map();
    evidence["token_ids"] = Value(Value::Array{
        Value(int64_t{21}), Value(int64_t{22}), Value(int64_t{23}), Value(int64_t{24})});
    evidence["logprobs"] = Value(Value::Array{
        Value(-0.1), Value(-0.2), Value(-0.3), Value(-0.4)});
    evidence["mask"] = Value(Value::Array{
        Value(int64_t{1}), Value(int64_t{0}), Value(int64_t{0}), Value(int64_t{1})});
    result["behavior"] = Value(std::move(evidence));
    h.complete(narrative_vlm->request_id, "vlm", true, std::move(result));
  }
  const auto narrative = narrative_future.get();
  check(narrative.status == 200 &&
            behavior_response_header(narrative, "x-umminfer-behavior-evidence") ==
                "exposed" &&
            narrative.body.find("\"mask\":[1,0,0,1]") != std::string::npos &&
            narrative.body.find("\"kind\":\"artifact\"") != std::string::npos,
        "supported interleaved image narrative lost its aligned behavior mask");

  api.stop();
  server_thread.join();
}

void test_openai_client_behavior_logprobs() {
  Harness h("clientbehaviorlogprobs");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "client behavior logprobs server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  client::OpenAIConfig cfg;
  cfg.base_url = "http://127.0.0.1:" + std::to_string(api.bound_port()) + "/v1";
  cfg.model = "qwen3-unified";
  cfg.timeout = client::Milliseconds{8000};
  client::OpenAIClient cli(cfg);

  // 1. Default OFF: omits "behavior" from request, receives no header or block.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "default off"});
    req.stream = true;
    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "default-off behavior request did not reach VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(!obs::behavior_logprobs_requested(payload.request.params),
            "OpenAIClient default-off request enabled behavior_logprobs");
      h.stream(vlm->request_id, "vlm", text_chunk("plain", false));
      h.complete(vlm->request_id, "vlm", true, native_behavior_result("plain"));
    }
    const client::ChatResponse resp = fut.get();
    check(resp.text == "plain", "default-off client text mismatch");
    check(resp.behavior_evidence_header.empty(),
          "default-off response unexpectedly populated behavior_evidence_header");
    check(!resp.behavior_evidence.has_value() && resp.behavior_evidence_json.empty(),
          "default-off response unexpectedly populated behavior_evidence");
  }

  // 2. Opt-in with stream=false passes through and surfaces server HTTP 400.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "nonstream behavior"});
    req.stream = false;
    req.behavior_logprobs = true;
    bool caught_400 = false;
    try {
      (void)cli.chat(req);
    } catch (const client::Error& err) {
      caught_400 = err.code() == client::ErrorCode::kBadRequest &&
                   err.status() == 400 &&
                   std::string(err.what()).find("stream=true") != std::string::npos;
    }
    check(caught_400,
          "OpenAIClient behavior_logprobs with stream=false did not surface HTTP 400");
  }

  // 3. Opt-in with stream=true and valid native evidence -> header=exposed + exact 5-field block.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "exposed behavior"});
    req.stream = true;
    req.behavior_logprobs = true;
    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "exposed behavior request did not reach VLM");
    if (vlm) {
      StagePayload payload = StagePayload::from_value(vlm->data);
      check(obs::behavior_logprobs_requested(payload.request.params),
            "OpenAIClient did not serialize behavior.logprobs=true");
      h.stream(vlm->request_id, "vlm", text_chunk("covered", false));
      h.complete(vlm->request_id, "vlm", true, native_behavior_result("covered"));
    }
    const client::ChatResponse resp = fut.get();
    check(resp.text == "covered", "exposed behavior client text mismatch");
    check(resp.behavior_evidence_header == "exposed",
          "OpenAIClient did not capture x-umminfer-behavior-evidence=exposed");
    check(resp.behavior_evidence.has_value() && !resp.behavior_evidence_json.empty(),
          "OpenAIClient did not capture exposed behavior payload");
    if (resp.behavior_evidence) {
      const json& ev = *resp.behavior_evidence;
      check(ev.size() == 5 &&
                ev["format"] == obs::kBehaviorLogprobsFormat &&
                ev["tokenizer_sha256"] == std::string(64, 'a') &&
                ev["token_ids"] == json::array({11, 12, 13}) &&
                ev["logprobs"] == json::array({-0.125, -0.25, -0.5}) &&
                ev["mask"] == json::array({1, 0, 1}),
            "OpenAIClient behavior_evidence fields did not match server contract");
      check(json::parse(resp.behavior_evidence_json) == ev,
            "OpenAIClient behavior_evidence_json did not round-trip to behavior_evidence");
    }
  }

  // 4. Opt-in with stream=true and truncated/missing evidence -> header=not_exposed, zero fabrication.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "truncated behavior"});
    req.stream = true;
    req.behavior_logprobs = true;
    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "not_exposed behavior request did not reach VLM");
    if (vlm) {
      Value truncated = native_behavior_result("cut");
      truncated["finish_reason"] = Value("length");
      h.stream(vlm->request_id, "vlm", text_chunk("cut", false));
      h.complete(vlm->request_id, "vlm", true, std::move(truncated));
    }
    const client::ChatResponse resp = fut.get();
    check(resp.text == "cut" && resp.finish_reason == "length",
          "not_exposed behavior client text/finish mismatch");
    check(resp.behavior_evidence_header == "not_exposed",
          "OpenAIClient did not capture x-umminfer-behavior-evidence=not_exposed");
    check(!resp.behavior_evidence.has_value() && resp.behavior_evidence_json.empty(),
          "OpenAIClient fabricated behavior evidence on not_exposed");
  }

  // 5. Opt-in with interleaved image narrative captures exposed masked behavior + artifact.
  {
    client::ChatRequest req;
    req.messages.push_back({"user", "narrative behavior"});
    req.stream = true;
    req.behavior_logprobs = true;
    req.modalities = {"text", "image"};
    client::ImageOutput img_out;
    img_out.size = "256x256";
    img_out.output_format = "png";
    req.image = img_out;
    client::NarrativeOptions n;
    n.enabled = true;
    n.visual_mode = "image";
    n.max_slots = 1;
    req.narrative = n;

    auto fut = std::async(std::launch::async, [&] { return cli.chat(req); });
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "narrative behavior request did not reach VLM");
    if (vlm) {
      h.stream(vlm->request_id, "vlm",
               text_chunk("A. [[image: red boat]] Tail", false));
    }
    auto img = h.next_submit(h.image_recv);
    check(img.has_value(), "narrative behavior image slot missing");
    if (img) {
      h.complete(img->request_id, "image", true,
                 artifact_result("image", "image/png", "PNG_BEH"));
    }
    if (vlm) {
      Value result = native_behavior_result("A. [[image: red boat]] Tail", 4);
      Value::Map evidence = result["behavior"].as_map();
      evidence["token_ids"] = Value(Value::Array{
          Value(int64_t{21}), Value(int64_t{22}), Value(int64_t{23}), Value(int64_t{24})});
      evidence["logprobs"] = Value(Value::Array{
          Value(-0.1), Value(-0.2), Value(-0.3), Value(-0.4)});
      evidence["mask"] = Value(Value::Array{
          Value(int64_t{1}), Value(int64_t{0}), Value(int64_t{0}), Value(int64_t{1})});
      result["behavior"] = Value(std::move(evidence));
      h.complete(vlm->request_id, "vlm", true, std::move(result));
    }
    const client::ChatResponse resp = fut.get();
    check(resp.behavior_evidence_header == "exposed" &&
              resp.behavior_evidence.has_value() &&
              (*resp.behavior_evidence)["mask"] == json::array({1, 0, 0, 1}) &&
              resp.artifacts.size() == 1 &&
              resp.text.find("[[") == std::string::npos,
          "OpenAIClient narrative behavior roundtrip mismatch");
  }

  api.stop();
  server_thread.join();
  wait_until([&]() -> bool { return engine.in_flight() == 0; },
             "client behavior logprobs requests stayed in flight");
}

void test_online_attestation_python_canonical_fixtures() {
  std::ifstream input("tests/online_attestation_canonical_fixtures.json",
                      std::ios::binary);
  check(input.good(), "Python canonical attestation fixtures are missing");
  const json fixtures = json::parse(input);
  check(fixtures.at("schema") ==
            "umminfer-online-attestation-canonical-fixtures-v1",
        "unexpected canonical fixture schema");
  check(fixtures.at("cases").size() == 3,
        "canonical fixture coverage was reduced");
  for (const auto& fixture : fixtures.at("cases")) {
    const json body = json::parse(fixture.at("input").get<std::string>());
    const std::string canonical = openai_detail::online_canonical_json(body);
    check(canonical == fixture.at("canonical").get<std::string>(),
          "C++ canonical JSON differs from the independent Python fixture");
    check(util::sha256_hex(canonical) == fixture.at("sha256").get<std::string>(),
          "C++ canonical request SHA-256 differs from Python");
    check(canonical.find(std::filesystem::temp_directory_path().string()) ==
              std::string::npos,
          "canonical request bytes leaked a host temporary path");
  }
  json nonfinite = json::object();
  nonfinite["temperature"] = std::numeric_limits<double>::infinity();
  bool rejected = false;
  try {
    (void)openai_detail::online_canonical_json(nonfinite);
  } catch (const std::exception&) {
    rejected = true;
  }
  check(rejected, "canonical JSON accepted a non-finite float");
}

struct OnlineCppFixture {
  std::filesystem::path root;
  json plan;
  std::vector<openai_detail::OnlineScheduleEntry> schedule;
  std::string config_path;

  ~OnlineCppFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
};

OnlineCppFixture make_online_cpp_fixture() {
  namespace fs = std::filesystem;
  OnlineCppFixture fixture;
  fixture.root = fs::temp_directory_path() /
                 ("umminfer-online-attestation-cpp-" + std::to_string(getpid()));
  fs::remove_all(fixture.root);
  fs::create_directories(fixture.root / "model");
  const std::string config_bytes = "{\"model_type\":\"fake-contract\"}\n";
  const std::string weight_bytes("fake native weights\0\xff\n", 22);
  {
    std::ofstream(fixture.root / "model/config.json", std::ios::binary)
        << config_bytes;
    std::ofstream out(fixture.root / "model/weights.safetensors", std::ios::binary);
    out.write(weight_bytes.data(), weight_bytes.size());
  }
  json weight_files = {
      {"weights.safetensors",
       {{"sha256", util::sha256_hex(weight_bytes)},
        {"bytes", static_cast<int64_t>(weight_bytes.size())}}}};
  const std::string weight_set_sha =
      util::sha256_hex(openai_detail::online_canonical_json(weight_files));
  const std::string model_config_sha = util::sha256_hex(config_bytes);
  fixture.plan = {
      {"schema", "umminfer-online-grpo-plan-v1"},
      {"batch_id", "cpp-attestation-batch"},
      {"behavior_policy",
       {{"policy_id", "cpp-policy"},
        {"revision", "cpp-policy-revision"},
        {"model_id", "qwen3-unified"},
        {"model_revision", "cpp-native-revision"},
        {"model_sha256", weight_set_sha},
        {"model_config_sha256", model_config_sha}}},
      {"sampling", {{"max_tokens", 16}, {"temperature", 0.25}, {"top_p", 0.9}}},
      {"request_template", json::object()},
      {"reward", {{"spec", "fixture:reward"}}},
      {"split_seed", "cpp-split"},
      {"eval_ratio", 0.0},
      {"transport",
       {{"timeout_s", 2.0}, {"max_attempts", 3},
        {"retry_statuses", {503}}, {"retry_transport", false},
        {"retry_timeouts", false}}},
      {"groups",
       {{{"prompt_id", "cpp-group"},
         {"messages", {{{"role", "user"}, {"content", "held VLM request"}}}},
         {"expected_visual_modalities", json::array()},
         {"seeds", {-42, 43}}}}},
  };
  fixture.schedule = openai_detail::online_derive_schedule(fixture.plan);
  const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  const json identity = {
      {"schema", "umminfer-native-stage-identity-v1"},
      {"stage_name", "vlm"}, {"state", "loaded"},
      {"loaded_at_epoch_ms", now},
      {"model_id", "qwen3-unified"},
      {"model_revision", "cpp-native-revision"},
      {"model_root", "model"},
      {"config", {{"path", "config.json"}, {"sha256", model_config_sha},
                  {"bytes", static_cast<int64_t>(config_bytes.size())}}},
      {"weight_files", weight_files},
      {"model_sha256", weight_set_sha},
  };
  json server = {
      {"schema", "umminfer-online-rollout-server-v1"},
      {"plan_path", "plan.json"},
      {"native_identity_path", "identity.json"},
      {"native_stage_name", "vlm"},
      {"max_identity_age_ms", 300000},
  };
  const auto write_document = [&](const char* name, const json& value) {
    const std::string bytes = openai_detail::online_canonical_json(value) + "\n";
    std::ofstream out(fixture.root / name, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    out.close();
    return util::sha256_hex(bytes);
  };
  server["plan_sha256"] = write_document("plan.json", fixture.plan);
  server["native_identity_sha256"] = write_document("identity.json", identity);
  (void)write_document("server.json", server);
  fixture.config_path = (fixture.root / "server.json").string();
  return fixture;
}

OnlineCppFixture make_online_cpp_process_fixture() {
  auto fixture = make_online_cpp_fixture();
  namespace fs = std::filesystem;
#if defined(__APPLE__)
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, ::getpid()};
  struct kinfo_proc proc_info {};
  size_t info_size = sizeof(proc_info);
  check(::sysctl(mib, 4, &proc_info, &info_size, nullptr, 0) == 0 &&
            info_size >= sizeof(proc_info),
        "could not read own process start time");
  const int64_t start_ticks =
      static_cast<int64_t>(proc_info.kp_proc.p_starttime.tv_sec) * 1000000LL +
      static_cast<int64_t>(proc_info.kp_proc.p_starttime.tv_usec);
  char exe_path[PROC_PIDPATHINFO_MAXSIZE];
  const int exe_size = ::proc_pidpath(::getpid(), exe_path, sizeof(exe_path));
  check(exe_size > 0, "could not resolve own executable");
  const std::string executable(exe_path, static_cast<size_t>(exe_size));
#else
  std::ifstream stat_input("/proc/self/stat");
  std::string stat;
  std::getline(stat_input, stat);
  const size_t close = stat.rfind(')');
  check(close != std::string::npos, "could not parse own process stat");
  std::istringstream fields(stat.substr(close + 1));
  std::vector<std::string> values;
  for (std::string value; fields >> value;) values.push_back(value);
  check(values.size() > 19, "own process stat lacks starttime");
  const int64_t start_ticks = std::stoll(values[19]);
  char exe_path[4096];
  const ssize_t exe_size = ::readlink("/proc/self/exe", exe_path,
                                      sizeof(exe_path) - 1);
  check(exe_size > 0, "could not resolve own executable");
  const std::string executable(exe_path, static_cast<size_t>(exe_size));
#endif
  std::ifstream exe_input(executable, std::ios::binary);
  const std::string exe_bytes((std::istreambuf_iterator<char>(exe_input)),
                              std::istreambuf_iterator<char>());
  json identity;
  {
    std::ifstream input(fixture.root / "identity.json", std::ios::binary);
    input >> identity;
  }
  identity["schema"] = "umminfer-native-stage-identity-v2";
  identity["process"] = {
      {"pid", static_cast<int64_t>(getpid())},
      {"start_time_ticks", start_ticks},
      {"executable_path", executable},
      {"executable_sha256", util::sha256_hex(exe_bytes)},
      {"executable_bytes", static_cast<int64_t>(exe_bytes.size())},
  };
  identity["stage_config"] = {
      {"kind", "qwen38"}, {"gpu", 7}, {"batch_size", 1},
      {"max_new_tokens", 512}, {"max_seq", 8192},
  };
  json server;
  {
    std::ifstream input(fixture.root / "server.json", std::ios::binary);
    input >> server;
  }
  server["schema"] = "umminfer-online-rollout-server-v2";
  const auto write_document = [&](const fs::path& path, const json& value) {
    const std::string bytes = openai_detail::online_canonical_json(value) + "\n";
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    out.close();
    return util::sha256_hex(bytes);
  };
  server["native_identity_sha256"] = write_document(
      fixture.root / "identity.json", identity);
  (void)write_document(fixture.root / "server.json", server);
  return fixture;
}

std::string online_test_value(const std::string& key,
                              const openai_detail::OnlineScheduleEntry& entry,
                              int attempt) {
  if (key == "contract") return "1";
  if (key == "generation") return "fresh";
  if (key == "batch_id") return entry.batch_id;
  if (key == "group_id") return entry.group_id;
  if (key == "rollout_id") return entry.rollout_id;
  if (key == "rollout_index") return std::to_string(entry.rollout_index);
  if (key == "attempt") return std::to_string(attempt);
  if (key == "request_sha256") return entry.request_sha256;
  if (key == "behavior_policy_sha256") return entry.behavior_policy_sha256;
  if (key == "behavior_policy_revision") return entry.behavior_policy_revision;
  if (key == "model_id") return entry.model_id;
  if (key == "model_revision") return entry.model_revision;
  if (key == "model_sha256") return entry.model_sha256;
  if (key == "model_config_sha256") return entry.model_config_sha256;
  if (key == "sampling_sha256") return entry.sampling_sha256;
  if (key == "behavior_evidence") return "not_exposed";
  throw std::runtime_error("unknown online test header");
}

test_http::HttpRequest online_test_request(
    int port, const openai_detail::OnlineScheduleEntry& entry, int attempt,
    bool marked = true) {
  test_http::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(port) +
                "/v1/chat/completions";
  request.body = openai_detail::online_canonical_json(entry.body);
  request.headers.emplace("Content-Type", "application/json");
  request.allow_http_errors = true;
  if (marked) {
    for (const auto& [key, header] : openai_detail::online_response_headers()) {
      if (key != "behavior_evidence") {
        request.headers.emplace(header, online_test_value(key, entry, attempt));
      }
    }
  }
  return request;
}

void check_online_test_headers(
    const test_http::HttpResponse& response,
    const openai_detail::OnlineScheduleEntry& entry, int attempt) {
  for (const auto& [key, header] : openai_detail::online_response_headers()) {
    const auto found = response.headers.find(header);
    check(found != response.headers.end() &&
              found->second == online_test_value(key, entry, attempt),
          ("online response attestation mismatch for " + key).c_str());
  }
}

void test_public_http_online_attestation_retry_cancel() {
  auto fixture = make_online_cpp_fixture();
  Harness h("onlineattestation");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  config.online_rollout_config = fixture.config_path;
  OpenAIApi api(h.coordinator, config);
  UnifiedConfig engine_config;
  engine_config.max_in_flight = 1;
  UnifiedEngine engine(h.coordinator, engine_config);
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "online attestation server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const auto& first_entry = fixture.schedule[0];
  const auto& retry_entry = fixture.schedule[1];
  auto first = std::async(std::launch::async, [&] {
    return test_http::perform(online_test_request(api.bound_port(), first_entry, 0));
  });
  auto first_submit = h.next_submit(h.vlm_recv);
  check(first_submit.has_value(), "first attested request did not reach the fake VLM");

  const auto overloaded = test_http::perform(
      online_test_request(api.bound_port(), retry_entry, 0));
  check(overloaded.status == 503, "full unified admission did not answer 503");
  check_online_test_headers(overloaded, retry_entry, 0);
  const auto duplicate = test_http::perform(
      online_test_request(api.bound_port(), retry_entry, 0));
  check(duplicate.status == 409, "duplicate rollout attempt was not rejected");
  check_online_test_headers(duplicate, retry_entry, 0);

  if (first_submit) {
    auto result = text_result("first answer");
    result["finish_reason"] = Value("stop");
    h.complete(first_submit->request_id, "vlm", true, std::move(result));
  }
  const auto first_response = first.get();
  check(first_response.status == 200, "first attested stream did not complete");
  check_online_test_headers(first_response, first_entry, 0);

  auto cancelled = std::async(std::launch::async, [&] {
    return test_http::perform(online_test_request(api.bound_port(), retry_entry, 1));
  });
  auto cancel_submit = h.next_submit(h.vlm_recv);
  check(cancel_submit.has_value(), "retry attempt did not reach the fake VLM");
  if (cancel_submit) {
    test_http::HttpRequest abort;
    abort.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) + "/abort";
    abort.body = json{{"request_id", cancel_submit->request_id}}.dump();
    abort.allow_http_errors = true;
    const auto abort_response = test_http::perform(abort);
    check(abort_response.status == 200, "public abort did not accept the retry id");
    h.complete(cancel_submit->request_id, "vlm", false, Value{}, "request aborted");
  }
  const auto cancelled_response = cancelled.get();
  check(cancelled_response.status == 200, "cancelled stream changed HTTP status");
  check_online_test_headers(cancelled_response, retry_entry, 1);
  check(cancelled_response.body.find("\"kind\":\"cancelled\"") != std::string::npos,
        "cancelled attested stream lacks its cancelled terminal");

  auto recovered = std::async(std::launch::async, [&] {
    return test_http::perform(online_test_request(api.bound_port(), retry_entry, 2));
  });
  auto recovered_submit = h.next_submit(h.vlm_recv);
  check(recovered_submit.has_value(), "second retry did not reach the fake VLM");
  if (recovered_submit) {
    auto result = text_result("retry answer");
    result["finish_reason"] = Value("stop");
    h.complete(recovered_submit->request_id, "vlm", true, std::move(result));
  }
  const auto recovered_response = recovered.get();
  check(recovered_response.status == 200, "successful retry did not complete");
  check_online_test_headers(recovered_response, retry_entry, 2);
  check(recovered_response.body.find("retry answer") != std::string::npos,
        "successful retry lost its generated text");

  auto ordinary = std::async(std::launch::async, [&] {
    return test_http::perform(
        online_test_request(api.bound_port(), first_entry, 0, false));
  });
  auto ordinary_submit = h.next_submit(h.vlm_recv);
  check(ordinary_submit.has_value(), "ordinary request did not reach the fake VLM");
  if (ordinary_submit) {
    auto result = text_result("ordinary answer");
    result["finish_reason"] = Value("stop");
    h.complete(ordinary_submit->request_id, "vlm", true, std::move(result));
  }
  const auto ordinary_response = ordinary.get();
  check(ordinary_response.status == 200, "ordinary request failed on an opted-in server");
  for (const auto& [key, header] : openai_detail::online_response_headers()) {
    (void)key;
    check(ordinary_response.headers.find(header) == ordinary_response.headers.end(),
          "ordinary default response gained an attestation header");
  }

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "online attestation requests stayed in flight");
}

void test_public_http_process_bound_stage_identity() {
  auto fixture = make_online_cpp_process_fixture();
  Harness h("onlineprocessidentity");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  config.online_rollout_config = fixture.config_path;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "process-bound attestation server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const auto& entry = fixture.schedule[0];
  json server;
  {
    std::ifstream input(fixture.root / "server.json", std::ios::binary);
    input >> server;
  }
  const std::string identity_sha = server["native_identity_sha256"];
  const auto no_proof = [](const test_http::HttpResponse& response) {
    for (const auto& [key, header] : openai_detail::online_response_headers()) {
      (void)key;
      check(response.headers.find(header) == response.headers.end(),
            "missing/mismatched live identity produced client-echoed proof");
    }
  };

  const auto missing = test_http::perform(
      online_test_request(api.bound_port(), entry, 0));
  check(missing.status == 503, "missing live stage identity was accepted");
  no_proof(missing);
  h.event_push.send(Message{StageReadyMessage{"vlm", std::string(64, '0')}});
  wait_until([&] {
    return h.coordinator.stage_native_identity("vlm") ==
           std::optional<std::string>(std::string(64, '0'));
  }, "incorrect test stage identity was not registered");
  const auto mismatched = test_http::perform(
      online_test_request(api.bound_port(), entry, 0));
  check(mismatched.status == 503, "mismatched live stage identity was accepted");
  no_proof(mismatched);

  h.event_push.send(Message{StageReadyMessage{"vlm", identity_sha}});
  wait_until([&] {
    return h.coordinator.stage_native_identity("vlm") ==
           std::optional<std::string>(identity_sha);
  }, "matching test stage identity was not registered");
  auto accepted = std::async(std::launch::async, [&] {
    return test_http::perform(online_test_request(api.bound_port(), entry, 0));
  });
  auto submit = h.next_submit(h.vlm_recv);
  check(submit.has_value(),
        "matching native/gateway/live identity did not admit the scheduled request");
  if (submit) {
    auto result = text_result("process bound answer");
    result["finish_reason"] = Value("stop");
    h.complete(submit->request_id, "vlm", true, std::move(result));
  }
  const auto response = accepted.get();
  check(response.status == 200, "matching process-bound request did not complete");
  check_online_test_headers(response, entry, 0);

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "process-bound identity requests stayed in flight");
}

}  // namespace

int main() {
  // Hard stop for a wedged run. Measured wall time for the whole suite: 38 s
  // idle and 41 s under 13 CPU-bound processes; 300 s leaves 7x headroom for a
  // parallel `zig build test` run. Each individual wait is bounded by
  // kStateWait, so this only catches a run that stops making progress.
  alarm(300);
  RUN(test_streaming_asr_to_vlm());
  RUN(test_asr_partials_opt_in());
  RUN(test_asr_partials_default_off());
  RUN(test_asr_partials_cancel());
  RUN(test_asr_partials_error());
  RUN(test_nonstream_asr_synthesizes_final());
  RUN(test_asr_plain_string_result());
  RUN(test_asr_malformed_result_fails());
  RUN(test_streaming_asr_malformed_final_fails());
  RUN(test_unknown_id_wait_fails());
  RUN(test_cancel_propagates_to_both_stages());
  RUN(test_asr_failure());
  RUN(test_vlm_failure());
  RUN(test_vlm_result_text_contract());
  RUN(test_queue_cap_backpressure());
  RUN(test_queue_cap_concurrent_submit());
  RUN(test_duplicate_final_and_out_of_order());
  RUN(test_joint_image_audio());
  RUN(test_joint_image_remote_url());
  RUN(test_media_contract_errors());
  RUN(test_vlm_only_string_content());
  RUN(test_vlm_only_image_text());
  RUN(test_vlm_only_session_history());
  RUN(test_session_reuse());
  RUN(test_session_reuse_concurrent());
  RUN(test_session_history_order());
  RUN(test_session_history_default_off());
  RUN(test_session_history_isolation());
  RUN(test_session_history_cancel_error_excluded());
  RUN(test_session_history_turn_eviction());
  RUN(test_session_history_char_eviction());
  RUN(test_session_history_session_eviction());
  RUN(test_session_history_concurrent());
  RUN(test_session_history_restart_drops());
  RUN(test_cancel_before_pumped_completion());
  RUN(test_cancel_after_asr_completed_swallows_vlm());
  RUN(test_cancel_handoff_race_stress());
  RUN(test_previous_sink_reentry_does_not_deadlock());
  RUN(test_cancel_cannot_overtake_partial_dispatch());
  RUN(test_forwarded_sink_cancel_order(false));
  RUN(test_forwarded_sink_cancel_order(true));
  RUN(test_construction_after_pump_start_rejected());
  RUN(test_destructor_synchronizes_inflight_callback());
  RUN(test_retire_wait_fallback_order());
  RUN(test_retired_cross_thread_order());
  RUN(test_retired_same_thread_reentry());
  RUN(test_active_reentry_exception_drains());
  RUN(test_retired_reentry_exception_drains());
  RUN(test_destructor_suppresses_previous_sink_exception());
  RUN(test_public_http_unified_sse());
  RUN(test_public_http_overload_503());
  RUN(test_public_http_cancel_during_vlm());
  RUN(test_public_http_asr_model_failure());
  RUN(test_public_http_joint_input());
  RUN(test_public_http_vlm_only());
  RUN(test_public_http_vlm_only_stream());
  RUN(test_public_http_top_p_seed_passthrough());
  RUN(test_public_http_nonunified_top_p_seed_passthrough());
  RUN(test_public_http_unified_errors());
  RUN(test_public_http_session_history());
  RUN(test_public_http_session_id_requires_unified());
  RUN(test_mixed_fanout_join_and_prompt_rules());
  RUN(test_mixed_image_prompt_falls_back_to_answer());
  RUN(test_mixed_partial_leg_failure_keeps_artifacts());
  RUN(test_mixed_cancel_aborts_every_submitted_leg());
  RUN(test_streamed_tts_submit_failure_becomes_leg_error());
  RUN(test_mixed_cancel_before_fanout_blocks_legs());
  RUN(test_mixed_join_release_then_late_complete());
  RUN(test_mixed_artifact_limits_become_leg_errors());
  RUN(test_mixed_chunk_seq_reservation());
  RUN(test_mixed_previous_stream_tolerates_bin_chunk());
  RUN(test_public_http_mixed_artifact_stream());
  RUN(test_public_http_mixed_partial_failure());
  RUN(test_public_http_nonstream_mixed_all_modalities());
  RUN(test_public_http_nonstream_mixed_partial_failure());
  RUN(test_public_http_nonstream_mixed_rejections());
  RUN(test_public_http_nonstream_text_only_compatible());
  RUN(test_openai_client_nonstream_mixed_roundtrip());
  RUN(test_realtime_spoken_audio_format_and_leg_error());
  RUN(test_realtime_barge_in_history_before_and_after_vlm());
  RUN(test_realtime_playback_completion_and_default_exclusion());
  RUN(test_realtime_deterministic_coordinator_history_stress());
  RUN(test_public_http_finish_reason_length_and_router_reset());
  RUN(test_public_http_audio_speech());
  RUN(test_public_http_audio_speech_without_tts());
  RUN(test_cancel_for_session_rejects_foreign_session());
  RUN(test_streamed_tts_long_stream_drains_past_aggregate_cap());
  RUN(test_streamed_tts_active_bound_drains_all_sentences());
  RUN(test_streamed_tts_sentence_fanout());
  RUN(test_streamed_tts_markdown_spans_and_cursors());
  RUN(test_streamed_tts_chunk_limit_aborts());
  RUN(test_simple_scheduler_batch_and_abort_tombstone());
  RUN(test_mixed_video_lifecycle_and_prompt_rules());
  RUN(test_mixed_video_failure_and_artifact_limit());
  RUN(test_mixed_video_cancel_and_session_ownership());
  RUN(test_public_http_mixed_video_stream_and_abort());
  RUN(test_openai_client_video_and_wire_limit_boundary());
  RUN(test_non_unified_null_video_field_compatible());
  RUN(test_direct_engine_video_input_modality_rejected_even_with_video_stage());
  RUN(test_mixed_artifact_aggregate_60mib_boundary());
  RUN(test_narrative_parser_unit());
  RUN(test_narrative_streaming_opening_director_prompt());
  RUN(test_narrative_streaming_opening_slots_and_tts());
  RUN(test_narrative_interleaved_slots_and_streaming_tts());
  RUN(test_narrative_max_slots_and_zero_marker_fallback());
  RUN(test_narrative_cancel_aborts_slots_and_tts());
  RUN(test_narrative_unterminated_directive_never_reaches_tts());
  RUN(test_narrative_eof_and_terminal_result());
  RUN(test_narrative_vlm_failure_aborts_slots());
  RUN(test_public_http_narrative_validation());
  RUN(test_public_http_visual_render());
  RUN(test_public_http_narrative_sse());
  RUN(test_public_http_video_url_input());
  RUN(test_unified_multiturn_engine_and_schema());
  RUN(test_public_http_multiturn_multimodal());
  RUN(test_control_message_parse_contract());
  RUN(test_energy_vad_speech_transitions());
  RUN(test_echo_robust_vad_echo_and_double_talk());
  RUN(test_echo_robust_vad_delay_bounds_and_session_isolation());
  RUN(test_stage_submit_fallback_and_failure_completion());
  RUN(test_stage_dual_channel_abort_emits_one_terminal());
  RUN(test_stage_heartbeat_reregisters_after_coordinator_restart());
  RUN(test_upstream_media_proxy_routes());
  RUN(test_unified_metrics_cancel_race_and_release_gauge());
  RUN(test_public_http_wave7_media_passthrough_and_tts_chunk_meta());
  RUN(test_public_http_video_teacache_thresh());
  RUN(test_narrative_continuity_http_and_engine());
  RUN(test_explicit_anchor_completion());
  RUN(test_explicit_image_reference_chain());
  RUN(test_explicit_anchor_failure_cancel_and_validation());
  RUN(test_explicit_anchor_http_wire());
  RUN(test_public_http_request_native_allocator_telemetry());
  RUN(test_public_http_asr_request_native_allocator_telemetry());
  RUN(test_openai_client_teacache_and_narrative());
  RUN(test_public_http_behavior_logprobs());
  RUN(test_openai_client_behavior_logprobs());
  RUN(test_online_attestation_python_canonical_fixtures());
  RUN(test_public_http_online_attestation_retry_cancel());
  RUN(test_public_http_process_bound_stage_identity());
  RUN(test_public_http_behavior_logprobs());
  return g_failures == 0 ? 0 : 1;
}
