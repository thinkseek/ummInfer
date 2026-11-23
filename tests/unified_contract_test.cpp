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
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
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
#include "umminfer/obs/metrics.hpp"
#include "serve/energy_vad.hpp"
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
  if (out.status < 200 || out.status >= 300) {
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
    if (!closed_.exchange(true) && sock_ != CURL_SOCKET_BAD) {
      ::shutdown(static_cast<int>(sock_), SHUT_RDWR);
    }
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

void test_realtime_spoken_audio_format_and_leg_error() {
  Harness h("rtspoken");
  ApiConfig rt_config;
  rt_config.host = "127.0.0.1";
  rt_config.port = 0;
  rt_config.model = "qwen3-unified";
  rt_config.entry_stage = "asr";
  rt_config.request_timeout_s = 10.0;
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  RealtimeApi realtime(h.coordinator, rt_config);
  realtime.attach_unified(engine);
  engine.on_event = [&](const UnifiedEvent& event) {
    realtime.forward_event(event);
  };
  realtime.install_stream_sink();
  check(realtime.bind(), "realtime server did not bind");
  h.start_pump();
  std::thread rt_thread([&] { realtime.listen(); });

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

  wait_until([&] { return engine.in_flight() == 0; },
             "realtime spoken requests stayed in flight");
  cancel.store(true, std::memory_order_relaxed);
  ws.close();
  if (ws_reader.joinable()) ws_reader.join();
  realtime.stop();
  rt_thread.join();
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

  // 1) stream=false with video output must be rejected with HTTP 400.
  {
    test_http::HttpRequest req;
    req.url = base_url + "/v1/chat/completions";
    req.body = nlohmann::json{
                   {"model", "qwen3-unified"},
                   {"messages", nlohmann::json::array(
                                    {nlohmann::json{{"role", "user"},
                                                    {"content", "make video"}}})},
                   {"stream", false},
                   {"modalities", nlohmann::json::array({"text", "video"})}}
                   .dump();
    req.headers.emplace("Content-Type", "application/json");
    bool got_400 = false;
    try {
      test_http::perform(req);
    } catch (const client::Error& err) {
      got_400 = (err.status() == 400);
    }
    check(got_400, "non-streaming video request was not rejected with 400");
  }

  // 2) Streaming 5 MiB fake MP4 video artifact -> 3 base64 chunks + sha256.
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
    bad.modalities = {"text", "video"};
    bad.stream = false;
    bad.video = client::VideoOutput{};
    bool rejected = false;
    try {
      client.chat(bad);
    } catch (const client::Error& err) {
      rejected = (err.code() == client::ErrorCode::kConfig);
    }
    check(rejected, "OpenAIClient accepted non-streaming video request");
  }
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
    check(done[0].request_id == "stage-fallback" && done[0].success,
          "the fallback submit did not complete successfully");
    const Value* probe = done[0].result.find("probe");
    check(probe != nullptr && probe->as_str() == "inputs-fallback",
          "a submit without payload.data did not compute on request.inputs");
    check(work.size() == 1 && work[0].find("probe") != nullptr,
          "the scheduler received the wrong fallback input");
    check(done[1].request_id == "stage-mismatch" && !done[1].success,
          "a payload request_id mismatch did not fail its request");
    check(done[1].error && done[1].error->find("differs") != std::string::npos,
          "the failure completion lost its reason");
  }
  stage.stop();
  stage_thread.join();
  submits.close();
  completions.close();
  aborts.close();
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
  RUN(test_realtime_spoken_audio_format_and_leg_error());
  RUN(test_public_http_finish_reason_length_and_router_reset());
  RUN(test_public_http_audio_speech());
  RUN(test_public_http_audio_speech_without_tts());
  RUN(test_cancel_for_session_rejects_foreign_session());
  RUN(test_streamed_tts_sentence_fanout());
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
  RUN(test_control_message_parse_contract());
  RUN(test_energy_vad_speech_transitions());
  RUN(test_stage_submit_fallback_and_failure_completion());
  RUN(test_upstream_media_proxy_routes());
  RUN(test_unified_metrics_cancel_race_and_release_gauge());
  return g_failures == 0 ? 0 : 1;
}
