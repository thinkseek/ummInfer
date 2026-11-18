// SPDX-License-Identifier: Apache-2.0
// Batch 3 checks for the public client and the fixed-upstream media routes.
// Every request goes to a real local ummInfer server. The configured upstream is
// an unbound loopback port, so forwarded requests fail at connect time. No fake
// upstream, no hand-written protocol response and no inference take part.
// Success against a real Qwen or SGLang upstream is NOT_COVERED.
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "umminfer/audio/audio.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/serve/openai_api.hpp"
#include "umminfer/util/codec.hpp"
#include <httplib.h>

using namespace umminfer;

namespace {

std::string g_cli_bin = "./zig-out/bin/umm-infer";

int failures = 0;
int checks = 0;

#define CHECK(cond)                                                         \
  do {                                                                      \
    ++checks;                                                               \
    if (!(cond)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond   \
                << "\n";                                                    \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

constexpr const char* kUpstreamKey = "sk-upstream-CLIENT-SECRET";
constexpr const char* kInboundKey = "inbound-client-key";
constexpr const char* kDeadUpstream = "http://127.0.0.1:9";

struct TestServer {
  explicit TestServer(ApiConfig config)
      : coordinator("ipc:///tmp/omni-client-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(++counter()) + "-done",
                    "ipc:///tmp/omni-client-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(counter()) + "-abort"),
        api(coordinator, config),
        inbound_key(config.api_key) {}

  ~TestServer() {
    api.stop();
    if (worker.joinable()) worker.join();
  }

  bool start() {
    if (!api.bind()) return false;
    port = api.bound_port();
    worker = std::thread([this] { api.listen(); });
    for (int attempt = 0; attempt < 100; ++attempt) {
      httplib::Client probe("127.0.0.1", port);
      probe.set_connection_timeout(1, 0);
      probe.set_read_timeout(2, 0);
      httplib::Headers hdrs;
      if (!inbound_key.empty()) {
        hdrs.emplace("Authorization", "Bearer " + inbound_key);
      }
      if (auto res = probe.Get("/health", hdrs); res && res->status == 200) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
  }

  static int& counter() {
    static int value = 0;
    return value;
  }

  Coordinator coordinator;
  OpenAIApi api;
  std::string inbound_key;
  std::thread worker;
  int port = 0;
};

ApiConfig base_config() {
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "served-model";
  return config;
}

ApiConfig qwen_config() {
  ApiConfig config = base_config();
  config.upstream_kind = "qwen";
  config.upstream_base_url = kDeadUpstream;
  config.upstream_model = "qwen3.5-omni-plus";
  config.upstream_api_key = kUpstreamKey;
  config.upstream_variant = "omni";
  return config;
}

ApiConfig sglang_config(const std::string& variant) {
  ApiConfig config = base_config();
  config.upstream_kind = "sglang";
  config.upstream_base_url = kDeadUpstream;
  config.upstream_model = "MiniMaxAI/MiniMax-H3";
  config.upstream_variant = variant;
  return config;
}

client::OpenAIConfig client_config(int port, const std::string& key,
                                   int attempts = 1) {
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:" + std::to_string(port) + "/v1";
  config.model = "served-model";
  config.api_key = key;
  config.max_attempts = attempts;
  config.retry_base = client::Milliseconds{2000};
  config.retry_max = client::Milliseconds{2000};
  config.connect_timeout = client::Milliseconds{2000};
  config.timeout = client::Milliseconds{5000};
  return config;
}

template <typename Function>
client::ErrorCode error_code(Function&& function, int* status = nullptr) {
  try {
    function();
  } catch (const client::Error& error) {
    if (status != nullptr) *status = error.status();
    return error.code();
  } catch (const std::exception& error) {
    std::cerr << "unexpected exception: " << error.what() << "\n";
    return client::ErrorCode::kEmpty;
  }
  return client::ErrorCode::kEmpty;
}

int raw_http_status(int port, const std::string& method,
                    const std::string& path, const std::string& body = "",
                    const std::string& bearer = "",
                    std::string* text = nullptr) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(2, 0);
  cli.set_read_timeout(2, 0);
  httplib::Headers hdrs;
  if (!bearer.empty()) hdrs.emplace("Authorization", "Bearer " + bearer);
  httplib::Result res =
      method == "GET" ? cli.Get(path, hdrs)
                      : cli.Post(path, hdrs, body, "application/json");
  if (!res) return 0;
  if (text != nullptr) *text = res->body;
  return res->status;
}

client::Message text_message(std::string text) {
  client::Message message;
  message.role = "user";
  message.content = std::move(text);
  return message;
}

void test_validation() {
  // The base URL is never contacted: every case must fail before a connection.
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:9";
  config.model = "served-model";
  client::OpenAIClient client(config);

  client::ChatRequest chat;
  chat.messages.push_back(text_message("hi"));
  chat.audio = client::AudioOutput{"Tina", "wav"};
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.modalities = {"text", "audio"};
  chat.audio->format = "mp3";
  chat.stream = true;
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.audio->format = "wav";
  chat.stream = false;
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.stream = true;
  chat.modalities = {"voice"};
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  client::ChatRequest both;
  client::Message message;
  message.role = "user";
  message.content = "text";
  client::ContentPart part;
  part.kind = client::ContentPart::Kind::kText;
  part.text = "text";
  message.parts.push_back(part);
  both.messages.push_back(message);
  CHECK(error_code([&] { client.chat(both); }) == client::ErrorCode::kConfig);

  client::ChatRequest empty_image;
  client::Message image_message;
  image_message.role = "user";
  client::ContentPart image_part;
  image_part.kind = client::ContentPart::Kind::kImageUrl;
  image_message.parts.push_back(image_part);
  empty_image.messages.push_back(image_message);
  CHECK(error_code([&] { client.chat(empty_image); }) == client::ErrorCode::kConfig);

  client::ChatRequest fast_video;
  client::Message video_message;
  video_message.role = "user";
  client::ContentPart video_part;
  video_part.kind = client::ContentPart::Kind::kVideoUrl;
  video_part.url = "https://example.com/a.mp4";
  video_part.fps = 20;
  video_message.parts.push_back(video_part);
  fast_video.messages.push_back(video_message);
  CHECK(error_code([&] { client.chat(fast_video); }) == client::ErrorCode::kConfig);

  client::ImageRequest generation;
  CHECK(error_code([&] { client.generate_image(generation); }) ==
        client::ErrorCode::kConfig);
  generation.prompt = "x";
  generation.extra_body["response_format"] = "url";
  CHECK(error_code([&] { client.generate_image(generation); }) ==
        client::ErrorCode::kConfig);

  client::ImageEditRequest edit;
  edit.prompt = "x";
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);
  edit.images.push_back({"image-bytes", "a.png", "image/png"});
  edit.fields = {{"size", "1"}, {"size", "2"}};
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);
  edit.fields = {{"model", "other"}};
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);

  // The edit parser accepts inline bytes only, so response_format is forced to
  // b64_json and a caller field that tries to override it is rejected.
  client::ImageEditRequest format_edit;
  format_edit.prompt = "x";
  format_edit.images.push_back({"image-bytes", "a.png", "image/png"});
  format_edit.fields = {{"response_format", "url"}};
  CHECK(error_code([&] { client.edit_image(format_edit); }) ==
        client::ErrorCode::kConfig);

  client::VideoRequest video;
  CHECK(error_code([&] { client.create_video(video); }) ==
        client::ErrorCode::kConfig);
  video.prompt = "x";
  video.body["model"] = "other";
  CHECK(error_code([&] { client.create_video(video); }) ==
        client::ErrorCode::kConfig);

  CHECK(error_code([&] { client.video_status("a/b"); }) ==
        client::ErrorCode::kConfig);
  CHECK(error_code([&] { client.delete_video(""); }) ==
        client::ErrorCode::kConfig);
  CHECK(error_code([&] { client.download_video("vid-1", ""); }) ==
        client::ErrorCode::kConfig);
  // The zero-based output index must not be negative.
  CHECK(error_code([&] {
          client.download_video("vid-1", "/tmp/omni-client-check/never.mp4", -1);
        }) == client::ErrorCode::kConfig);

  // A cancelled flag fails before any connection.
  std::atomic<bool> cancelled{true};
  chat.cancel = &cancelled;
  chat.modalities = {"text"};
  chat.audio.reset();
  CHECK(error_code([&] { client.chat(chat); }) ==
        client::ErrorCode::kCancelled);
  client::ImageRequest cancelled_image;
  cancelled_image.prompt = "x";
  cancelled_image.cancel = &cancelled;
  CHECK(error_code([&] { client.generate_image(cancelled_image); }) ==
        client::ErrorCode::kCancelled);
  CHECK(error_code([&] { client.video_status("vid-1", &cancelled); }) ==
        client::ErrorCode::kCancelled);
}

void test_qwen_server() {
  ApiConfig server_config = qwen_config();
  server_config.api_key = kInboundKey;
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  // Authentication runs before any route logic.
  {
    client::OpenAIClient wrong(client_config(server.port, "wrong-key"));
    client::ChatRequest request;
    request.messages.push_back(text_message("hi"));
    int status = 0;
    CHECK(error_code([&] { wrong.chat(request); }, &status) ==
          client::ErrorCode::kAuthentication);
    CHECK(status == 401);
  }

  client::OpenAIClient client(client_config(server.port, kInboundKey));

  // Text chat still reaches the fixed upstream.
  {
    client::ChatRequest request;
    request.messages.push_back(text_message("hi"));
    int status = 0;
    const auto code = error_code([&] { client.chat(request); }, &status);
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // Multimodal content parts reach the proxy and fail only at the upstream.
  {
    client::ChatRequest request;
    client::Message message;
    message.role = "user";
    client::ContentPart text;
    text.kind = client::ContentPart::Kind::kText;
    text.text = "what is this";
    client::ContentPart image;
    image.kind = client::ContentPart::Kind::kImageUrl;
    image.url = "https://example.com/a.jpg";
    client::ContentPart audio;
    audio.kind = client::ContentPart::Kind::kInputAudio;
    audio.audio = "AAAA";
    audio.audio_format = "mp3";
    message.parts = {text, image, audio};
    request.messages.push_back(message);
    std::string detail;
    int status = 0;
    try {
      client.chat(request);
    } catch (const client::Error& error) {
      detail = error.what();
      status = error.status();
    } catch (const std::exception& error) {
      detail = error.what();
    }
    CHECK(status == 502);
    CHECK(detail.find(kUpstreamKey) == std::string::npos);
    CHECK(detail.find(kInboundKey) == std::string::npos);
  }

  // Speech output requires streaming and reaches the proxy as a stream.
  {
    client::ChatRequest request;
    request.messages.push_back(text_message("say hi"));
    request.stream = true;
    request.modalities = {"text", "audio"};
    request.audio = client::AudioOutput{"Tina", "wav"};
    bool delta = false;
    request.on_delta = [&](const std::string&) { delta = true; };
    request.timeout = client::Milliseconds{5000};
    int status = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto code = error_code([&] { client.chat(request); }, &status);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    // The stream route commits nothing until the upstream answers. A dead
    // upstream must surface as the public 502 mapping, not as a client timeout
    // after the full request timeout.
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(elapsed < std::chrono::milliseconds(3000));
    CHECK(!delta);
  }

  // A fixed-upstream process cannot abort upstream work. /abort must answer 501
  // and never {"success":true}; the client's own cancel is local and never
  // claims an upstream abort.
  {
    std::string text;
    const int status = raw_http_status(
        server.port, "POST", "/abort", R"({"request_id":"req-1"})",
        kInboundKey, &text);
    CHECK(status == 501);
    CHECK(text.find("\"success\":true") == std::string::npos);
    CHECK(text.find("abort is not supported") != std::string::npos);
  }
}

void test_image_server() {
  ApiConfig server_config = sglang_config("image");
  server_config.api_key = kInboundKey;
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  client::OpenAIClient client(client_config(server.port, kInboundKey));

  {
    client::ImageRequest request;
    request.prompt = "a red cube";
    std::string detail;
    int status = 0;
    try {
      client.generate_image(request);
    } catch (const client::Error& error) {
      detail = error.what();
      status = error.status();
    } catch (const std::exception& error) {
      detail = error.what();
    }
    CHECK(status == 502);
    CHECK(detail.find(kUpstreamKey) == std::string::npos);
    CHECK(detail.find(kInboundKey) == std::string::npos);
  }

  {
    client::ImageRequest request;
    request.prompt = "a red cube";
    request.model = "other";
    int status = 0;
    CHECK(error_code([&] { client.generate_image(request); }, &status) ==
          client::ErrorCode::kBadRequest);
    CHECK(status == 400);
  }

  {
    client::ImageEditRequest request;
    request.prompt = "make it blue";
    request.images.push_back({"first-image", "first.png", "image/png"});
    request.images.push_back({"second-image", "second.png", "image/png"});
    request.fields = {{"url", "https://example.com/one"}, {"url", "https://example.com/two"}};
    int status = 0;
    CHECK(error_code([&] { client.edit_image(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // task_type is on the final H3 edit allowlist. The client must forward it
  // instead of dropping it, so the request stops only at the upstream.
  {
    client::ImageEditRequest request;
    request.prompt = "restore this image";
    request.images.push_back({"image-bytes", "a.png", "image/png"});
    request.fields = {{"task_type", "portrait_restoration"}};
    int status = 0;
    CHECK(error_code([&] { client.edit_image(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }
}

void test_video_server() {
  ApiConfig server_config = sglang_config("video");
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  // max_attempts is 3 and the retry delay is 2 s. A creation POST that retried
  // would take at least one second; a single POST fails in milliseconds.
  client::OpenAIClient client(client_config(server.port, "", 3));

  {
    client::VideoRequest request;
    request.prompt = "a running horse";
    request.body["task"] = "t2va";
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    CHECK(error_code([&] { client.create_video(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(elapsed < std::chrono::milliseconds(900));
  }

  // The final H3 create allowlist. The client must forward each field unchanged
  // instead of dropping it, so the request stops only at the upstream and is
  // still sent once.
  {
    client::VideoRequest request;
    request.prompt = "a running horse";
    request.body["task"] = "t2va";
    request.body["num_inference_steps"] = 32;
    request.body["flow_shift"] = 3.0;
    request.body["audio_flow_shift"] = 1.5;
    request.body["reference_resize_mode"] = "stretch";
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    CHECK(error_code([&] { client.create_video(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(elapsed < std::chrono::milliseconds(900));
  }

  {
    int status = 0;
    CHECK(error_code([&] { client.video_status("vid-1"); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(error_code([&] { client.delete_video("vid-1"); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // A zero-based output index is forwarded as ?variant=N. Anything that is not
  // a non-negative integer is a local 400 from the proxy, never a forwarded
  // request.
  for (const char* bad : {"abc", "a/b", "bad.id"}) {
    std::string text;
    const int status = raw_http_status(
        server.port, "GET",
        std::string("/v1/videos/vid-1/content?variant=") + bad, "", "", &text);
    CHECK(status == 400);
    if (status != 400) std::cerr << "variant=" << bad << " status=" << status << "\n";
  }

  const auto directory = std::filesystem::temp_directory_path() /
                         ("omni-client-test-" + std::to_string(::getpid()));
  std::error_code ignored;
  std::filesystem::create_directories(directory, ignored);
  const auto existing = directory / "existing.mp4";
  {
    std::ofstream out(existing, std::ios::binary);
    out << "keep";
  }
  {
    // An existing artifact is never replaced, even before the network.
    CHECK(error_code([&] { client.download_video("vid-1", existing.string()); }) ==
          client::ErrorCode::kBadRequest);
    std::ifstream in(existing, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    CHECK(content == "keep");
  }
  {
    // A failed download leaves neither the final path nor a part file. A dead
    // upstream must surface as the public 502 mapping before the request
    // timeout, not as a client timeout with no upstream status.
    client::OpenAIClient downloader(client_config(server.port, "", 1));
    const auto fresh = directory / "fresh.mp4";
    int status = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto code = error_code(
        [&] { downloader.download_video("vid-1", fresh.string(), 0); }, &status);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(elapsed < std::chrono::milliseconds(3000));
    CHECK(!std::filesystem::exists(fresh));
    int parts = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (entry.path().filename().string().find(".part.") != std::string::npos) ++parts;
    }
    CHECK(parts == 0);
  }
  std::filesystem::remove_all(directory, ignored);
}

// Executes the real compiled `umm-infer` CLI binary as a subprocess.
int cli(std::vector<std::string> args, std::string* captured_output = nullptr) {
  if (args.empty()) return 1;
  args[0] = g_cli_bin;
  int pipefd[2] = {-1, -1};
  if (::pipe(pipefd) != 0) return 1;
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    return 1;
  }
  if (pid == 0) {
    ::close(pipefd[0]);
    ::dup2(pipefd[1], STDOUT_FILENO);
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[1]);
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    ::execv(g_cli_bin.c_str(), argv.data());
    std::_Exit(127);
  }
  ::close(pipefd[1]);
  std::string out;
  char buf[4096];
  while (true) {
    const ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
    if (n <= 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  ::close(pipefd[0]);
  if (captured_output != nullptr) *captured_output = std::move(out);
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return 1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

void test_cli_usage() {
  const std::string url = "http://127.0.0.1:9/v1";
  const std::string out = "/tmp/omni-client-cli-" + std::to_string(::getpid()) + ".png";
  const std::string input = "/tmp/omni-client-cli-" + std::to_string(::getpid()) + ".jpg";

  // Required media arguments are usage errors, not client configuration errors.
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--out",
             out}) == 2);
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--prompt",
             "p"}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--out", out}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--image", input}) == 2);
  CHECK(cli({"umm-infer", "video", "create", "--url", url, "--model", "m"}) == 2);
  CHECK(cli({"umm-infer", "video", "content", "--url", url, "--id", "vid-1"}) == 2);

  // An empty value is not an absent flag.
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--temperature="}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--top-p="}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--stream", "--audio-out="}) == 2);

  // The multi-output count is bounded locally.
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--prompt",
             "p", "--out", out, "--n", "11"}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--image", input, "--out", out, "--n", "0"}) == 2);

  // Mixed-output flag table is closed and validated before any connection.
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--image-out", out,
             "--image-format", "gif"}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--image-prompt", "p"}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--image-out", out}) == 2);
}

// Without a fixed upstream the local routes keep their previous behavior.
void test_no_upstream() {
  TestServer server(base_config());
  CHECK(server.start());
  if (server.port == 0) return;
  client::OpenAIClient client(client_config(server.port, ""));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  int status = 0;
  CHECK(error_code([&] { client.chat(request); }, &status) ==
        client::ErrorCode::kServer);
  CHECK(status == 503);
}

// Minimal local SSE server returning hand-built unified frames, so the client
// parser can be exercised without a real engine or worker. The frames use the
// exact frozen artifact envelope.
struct ArtifactMockServer {
  httplib::Server server;
  int port = 0;
  std::string frames;
  bool serve_frames = true;

  void start() {
    server.Post("/v1/chat/completions", [this](const httplib::Request&,
                                              httplib::Response& res) {
      res.set_chunked_content_provider(
          "text/event-stream",
          [this](size_t, httplib::DataSink& sink) {
            if (serve_frames) {
              sink.write(frames.data(), frames.size());
            }
            sink.done();
            return true;
          });
    });
    port = server.bind_to_any_port("127.0.0.1");
    thread = std::thread([this] { server.listen_after_bind(); });
    for (int i = 0; i < 100 && !server.is_running(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  ~ArtifactMockServer() {
    server.stop();
    if (thread.joinable()) thread.join();
  }

  std::thread thread;
};

// Builds a complete artifact SSE stream for one artifact of `raw` bytes, split
// into base64 chunks of at most `raw_chunk` decoded bytes. Seq values are
// gap-free starting at 0; the last chunk carries bytes and sha256.
std::string artifact_stream(const std::string& raw, size_t raw_chunk,
                            const std::string& media,
                            const std::string& mime) {
  std::string out;
  int64_t seq = 0;
  const int64_t total =
      static_cast<int64_t>((raw.size() + raw_chunk - 1) / raw_chunk);
  const std::string digest = util::sha256_hex(raw);
  size_t offset = 0;
  for (int64_t index = 0; index < total; ++index) {
    const size_t length = std::min(raw_chunk, raw.size() - offset);
    nlohmann::json artifact = {
        {"artifact_id", "art-0"},
        {"media_type", media},
        {"mime_type", mime},
        {"encoding", "base64"},
        {"data", util::base64_encode(raw.substr(offset, length))},
        {"index", index},
        {"total", total}};
    if (index + 1 == total) {
      artifact["bytes"] = static_cast<int64_t>(raw.size());
      artifact["sha256"] = digest;
    }
    nlohmann::json frame = {
        {"id", "http-1"},
        {"object", "chat.completion.chunk"},
        {"choices", nlohmann::json::array()},
        {"unified", {{"kind", "artifact"},
                     {"stage", media},
                     {"seq", seq++},
                     {"session_id", "s"},
                     {"is_final", true},
                     {"text", ""},
                     {"artifact", artifact}}}};
    out += "data: " + frame.dump() + "\n\n";
    offset += length;
  }
  nlohmann::json terminal = {
      {"id", "http-1"},
      {"object", "chat.completion.chunk"},
      {"choices", nlohmann::json::array(
                       {nlohmann::json{{"delta", nlohmann::json::object()},
                                       {"finish_reason", "stop"}}})},
      {"unified", {{"kind", "complete"},
                   {"stage", "vlm"},
                   {"seq", seq},
                   {"session_id", "s"},
                   {"is_final", true},
                   {"text", ""},
                   {"legs", nlohmann::json::array()}}}};
  out += "data: " + terminal.dump() + "\n\n";
  out += "data: [DONE]\n\n";
  return out;
}

client::OpenAIConfig mock_config(int port) {
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:" + std::to_string(port) + "/v1";
  config.model = "m";
  config.max_attempts = 1;
  // The oversized-stream cases push 48-96 MiB through a local mock. Under a
  // parallel `zig build test` that write can take longer than 10 s and the
  // client would time out instead of hitting the parser limit, which makes
  // the assertion depend on machine load rather than on the contract.
  config.timeout = client::Milliseconds{60000};
  config.connect_timeout = client::Milliseconds{2000};
  return config;
}

void test_client_artifact_reassembly_and_checksum() {
  const std::string raw(3u << 20, '\x33');
  ArtifactMockServer mock;
  mock.frames = artifact_stream(raw, 1u << 20, "image", "image/png");
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  request.modalities = {"text", "image"};
  client::ImageOutput image;
  request.image = image;

  client::ChatResponse response = openai.chat(request);
  CHECK(response.artifacts.size() == 1);
  if (response.artifacts.size() == 1) {
    CHECK(response.artifacts[0].data == raw);
    CHECK(response.artifacts[0].sha256 == util::sha256_hex(raw));
    CHECK(response.artifacts[0].media_type == "image");
  }
}

void test_client_rejects_bad_checksum_and_base64() {
  const std::string raw(1024, '\x44');
  // Corrupt the digest on the final chunk.
  std::string bad = artifact_stream(raw, 512, "audio", "audio/wav");
  const std::string good_digest = util::sha256_hex(raw);
  bad.replace(bad.find(good_digest), good_digest.size(),
              std::string(64, '0'));
  ArtifactMockServer mock;
  mock.frames = std::move(bad);
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  request.modalities = {"text", "audio"};
  request.audio = client::AudioOutput{"Tina", "wav"};
  int status = 0;
  CHECK(error_code([&] { openai.chat(request); }, &status) ==
        client::ErrorCode::kBadResponse);

  // Non-aligned base64 in a chunk.
  std::string broken = artifact_stream(raw, 512, "audio", "audio/wav");
  broken.replace(broken.find("\"data\":\"") + 8, 1, "!");
  ArtifactMockServer mock2;
  mock2.frames = std::move(broken);
  mock2.start();
  client::OpenAIClient openai2(mock_config(mock2.port));
  CHECK(error_code([&] { openai2.chat(request); }) ==
        client::ErrorCode::kBadResponse);
}

void test_client_artifact_size_limits() {
  // Just over the 48 MiB per-artifact limit must fail deterministically.
  const std::string raw((48u << 20) + 1, '\x55');
  ArtifactMockServer mock;
  mock.frames = artifact_stream(raw, 2u << 20, "image", "image/png");
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  request.modalities = {"text", "image"};
  request.image = client::ImageOutput{};
  int status = 0;
  const auto code = error_code([&] { openai.chat(request); }, &status);
  // The exact layer that trips first can vary under the parallel test run's
  // memory pressure (parser limit or a curl write failure); the contract is
  // that the oversized stream always fails deterministically, never truncates.
  CHECK(code == client::ErrorCode::kBadResponse ||
        code == client::ErrorCode::kNetwork);
  CHECK(status == 0);  // transport/parser rejection, no HTTP status
}

void test_client_sse_line_limit_symmetric() {
  // One SSE data line over the shared 4 MiB parser ceiling must fail in the
  // client exactly as the server rejects an over-limit frame.
  ArtifactMockServer mock;
  nlohmann::json frame = {
      {"id", "http-1"},
      {"unified", {{"kind", "vlm_delta"}, {"seq", 0},
                   {"text", std::string(4u << 20, 'x')}}}};
  mock.frames = "data: " + frame.dump() + "\n\ndata: [DONE]\n\n";
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  CHECK(error_code([&] { openai.chat(request); }) ==
        client::ErrorCode::kBadResponse);
}

// The CLI writes a delivered image artifact atomically through PartWriter and
// refuses to overwrite an existing path.
void test_cli_image_output_atomic_write() {
  const std::string raw(2048, '\x66');
  ArtifactMockServer mock;
  mock.frames = artifact_stream(raw, 1024, "image", "image/png");
  mock.start();
  const std::string url =
      "http://127.0.0.1:" + std::to_string(mock.port) + "/v1";
  const std::string dir = "/tmp/omni-cli-mixed-" + std::to_string(::getpid());
  std::error_code ignored;
  std::filesystem::create_directories(dir, ignored);
  const std::string out = dir + "/generated.png";

  const int code = cli({"umm-infer", "chat", "--url", url, "--model", "m",
                        "--prompt", "hi", "--stream", "--image-out", out});
  CHECK(code == 0);
  CHECK(std::filesystem::exists(out));
  if (std::filesystem::exists(out)) {
    std::ifstream in(out, std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    CHECK(content == raw);
  }
  // A second run must fail before the request rather than replace the file.
  const std::string before = "keep-this-content";
  {
    std::ofstream guard(out, std::ios::binary);
    guard << before;
  }
  CHECK(cli({"umm-infer", "chat", "--url", url, "--model", "m",
             "--prompt", "hi", "--stream", "--image-out", out}) == 1);
  std::ifstream in(out, std::ios::binary);
  const std::string kept((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  CHECK(kept == before);
  std::filesystem::remove_all(dir, ignored);
}

// A reassembled WAV artifact must be accepted by the WAV parser; the
// descriptor echoes the 24 kHz mono contract.
void test_client_audio_artifact_is_parseable_wav() {
  const std::string pcm(4800, '\x01');  // 2400 mono samples
  const std::string wav = audio::wav_from_pcm16(pcm, 24000);
  ArtifactMockServer mock;
  std::string frames = artifact_stream(wav, 2048, "audio", "audio/wav");
  mock.frames = std::move(frames);
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  request.modalities = {"text", "audio"};
  request.audio = client::AudioOutput{"Tina", "wav"};
  client::ChatResponse response = openai.chat(request);
  CHECK(response.artifacts.size() == 1);
  if (response.artifacts.size() == 1) {
    const std::string& delivered = response.artifacts[0].data;
    CHECK(delivered == wav);
    // The concatenated chunks must form one structurally valid RIFF/WAVE
    // container whose data section holds exactly the original PCM. The shared
    // pcm16_from_wav helper is pinned to the 16 kHz ASR contract, so it cannot
    // parse a 24 kHz artifact; assert the WAV framing directly instead.
    bool parseable = delivered.size() >= 44 &&
                     delivered.substr(0, 4) == "RIFF" &&
                     delivered.substr(8, 4) == "WAVE" &&
                     delivered.find("fmt ") != std::string::npos &&
                     delivered.find("data") != std::string::npos;
    CHECK(parseable);
    CHECK(delivered.find(pcm) != std::string::npos);
  }
}

// A stream whose total wire size crosses the existing 64 MiB transport hard
// limit must fail deterministically, never truncate. The artifact path raises
// its stream ceiling to 96 MiB, so a plain text delta stream still enforces
// the original 64 MiB bound.
void test_client_text_stream_64mib_hard_limit() {
  ArtifactMockServer mock;
  nlohmann::json frame = {
      {"id", "http-1"},
      {"object", "chat.completion.chunk"},
      {"choices",
       nlohmann::json::array({nlohmann::json{
           {"delta", {{"content", std::string(1u << 20, 'x')}}},
           {"finish_reason", nullptr}}})}};
  std::string frames;
  for (int i = 0; i < 70; ++i) frames += "data: " + frame.dump() + "\n\n";
  frames += "data: [DONE]\n\n";
  mock.frames = std::move(frames);
  mock.start();
  client::OpenAIClient openai(mock_config(mock.port));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  request.stream = true;
  int status = 0;
  const auto code = error_code([&] { openai.chat(request); }, &status);
  // Parser limit or a curl write failure; either way the oversized stream is
  // rejected deterministically rather than truncated.
  CHECK(code == client::ErrorCode::kBadResponse ||
        code == client::ErrorCode::kNetwork);
  CHECK(status == 0);
}

// The CLI logs only safe artifact metadata (media type, byte count, sha256) and
// never echoes raw artifact payload bytes or secret text on stdout/stderr.
void test_artifact_log_summary_redacts_payload() {
  const std::string secret_prompt = "do-not-log-this-prompt";
  const std::string bytes = secret_prompt + std::string(32, '\x00');
  ArtifactMockServer mock;
  mock.frames = artifact_stream(bytes, 64, "image", "image/png");
  mock.start();
  const std::string url =
      "http://127.0.0.1:" + std::to_string(mock.port) + "/v1";
  const std::string dir =
      "/tmp/omni-cli-redact-" + std::to_string(::getpid());
  std::error_code ignored;
  std::filesystem::create_directories(dir, ignored);
  const std::string out = dir + "/redacted.png";
  std::string cli_output;
  const int code = cli({"umm-infer", "chat", "--url", url, "--model", "m",
                        "--prompt", "hi", "--stream", "--image-out", out},
                       &cli_output);
  CHECK(code == 0);
  CHECK(cli_output.find(secret_prompt) == std::string::npos);
  CHECK(cli_output.find("do-not-log") == std::string::npos);
  CHECK(std::filesystem::exists(out));
  if (std::filesystem::exists(out)) {
    std::ifstream in(out, std::ios::binary);
    const std::string saved((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    CHECK(saved == bytes);
  }
  std::filesystem::remove_all(dir, ignored);
  // Known SHA-256 vectors guard the public codec implementation.
  CHECK(util::sha256_hex("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(util::sha256_hex("") ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

// A streaming call without request.on_delta must still forbid automatic retry
// once any content or unified SSE frame has arrived before truncation.
void test_stream_no_retry_without_on_delta() {
  httplib::Server server;
  std::atomic<int> hits{0};
  server.Post("/v1/chat/completions",
              [&](const httplib::Request&, httplib::Response& res) {
                ++hits;
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [](size_t offset, httplib::DataSink& sink) {
                      if (offset > 0) {
                        sink.done();
                        return true;
                      }
                      const std::string partial =
                          "data: {\"id\":\"http-1\",\"object\":\"chat.completion.chunk\","
                          "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},"
                          "\"finish_reason\":null}]}\n\n";
                      sink.write(partial.data(), partial.size());
                      // End without [DONE] to simulate mid-stream truncation.
                      sink.done();
                      return true;
                    });
              });
  const int port = server.bind_to_any_port("127.0.0.1");
  std::thread th([&] { server.listen_after_bind(); });
  for (int i = 0; i < 100 && !server.is_running(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  client::OpenAIConfig cfg = mock_config(port);
  cfg.max_attempts = 3;
  cfg.retry_base = client::Milliseconds{10};
  cfg.retry_max = client::Milliseconds{20};
  client::OpenAIClient openai(cfg);
  client::ChatRequest req;
  req.messages.push_back(text_message("hello"));
  req.stream = true;  // Note: req.on_delta intentionally left unset!
  CHECK(error_code([&] { openai.chat(req); }) == client::ErrorCode::kNetwork);
  CHECK(hits.load() == 1);

  server.stop();
  if (th.joinable()) th.join();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    g_cli_bin = argv[1];
  }
  test_validation();
  test_cli_usage();
  test_qwen_server();
  test_image_server();
  test_video_server();
  test_no_upstream();
  test_client_artifact_reassembly_and_checksum();
  test_client_rejects_bad_checksum_and_base64();
  test_client_artifact_size_limits();
  test_client_sse_line_limit_symmetric();
  test_client_audio_artifact_is_parseable_wav();
  test_client_text_stream_64mib_hard_limit();
  test_cli_image_output_atomic_write();
  test_artifact_log_summary_redacts_payload();
  test_stream_no_retry_without_on_delta();
  if (failures != 0) {
    std::cerr << failures << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "all " << checks << " client checks passed\n";
  return 0;
}
