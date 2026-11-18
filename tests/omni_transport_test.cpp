// SPDX-License-Identifier: Apache-2.0
// Blackbox transport & codec API checks: RFC 4648 base64, WAV/PCM16 framing,
// streamed multipart uploads, Content-Type enforcement, request validation, and
// atomic part-file publishing via public client/audio/codec headers only.
#include <httplib.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include <nlohmann/json.hpp>

#include "umminfer/audio/audio.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/util/codec.hpp"

using namespace umminfer;

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond    \
                << "\n";                                                     \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

template <typename Fn>
void expect_error(Fn fn, const char* label) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  std::cerr << "FAIL expected an error: " << label << "\n";
  ++failures;
}

uint16_t read_u16(std::string_view bytes, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset])) |
         static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1]) << 8);
}

uint32_t read_u32(std::string_view bytes, size_t offset) {
  return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3])) << 24);
}

std::string slurp(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

struct TempDir {
  std::string path;
  TempDir() {
    char pattern[] = "/tmp/umminfer-omni-XXXXXX";
    char* made = ::mkdtemp(pattern);
    if (made == nullptr) throw std::runtime_error("mkdtemp failed");
    path = made;
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

struct LocalHttpServer {
  httplib::Server srv;
  std::thread th;
  int port = 0;

  void start() {
    port = srv.bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("bind_to_any_port failed");
    th = std::thread([this] { srv.listen_after_bind(); });
    while (!srv.is_running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  ~LocalHttpServer() {
    srv.stop();
    if (th.joinable()) th.join();
  }
};

void test_base64() {
  CHECK(util::base64_decode("", 0).empty());
  CHECK(util::base64_decode("TQ==", 1) == "M");
  CHECK(util::base64_decode("TWE=", 2) == "Ma");
  CHECK(util::base64_decode("TWFu", 3) == "Man");
  CHECK(util::base64_decode("TWFuTQ==", 4) == "ManM");

  const std::string bytes("\x00\x01\xfe\xffhello", 11);
  CHECK(util::base64_decode(util::base64_encode(bytes), 11) == bytes);

  const char* invalid[] = {
      "TQ=",     // length is not a multiple of 4
      "T=Q=",    // padding before the end
      "====",    // padding only
      "A===",    // padding in the second position
      "AA=A",    // padding inside the group
      "AB==",    // non-zero trailing bits
      "AAB=",    // non-zero trailing bits
      "TQ===",   // bad length
      "T WFu",   // whitespace
      "TWFu\n",  // newline
      "TWF*",    // non-alphabet character
  };
  for (const char* text : invalid) {
    expect_error([&] { util::base64_decode(text, 1024); }, text);
  }
  expect_error([] { util::base64_decode("TWFu", 2); }, "output limit");
  expect_error([] { util::base64_decode("", 16, /*require_nonempty=*/true); },
               "empty rejected when require_nonempty=true");
}

void test_wav() {
  const std::string pcm("\x01\x02\x03\x04", 4);
  const std::string wav = audio::wav_from_pcm16(pcm);
  CHECK(wav.size() == 48);
  CHECK(wav.substr(0, 4) == "RIFF");
  CHECK(wav.substr(8, 4) == "WAVE");
  CHECK(read_u32(wav, 4) == 40);
  CHECK(read_u16(wav, 20) == 1);   // PCM
  CHECK(read_u16(wav, 22) == 1);   // mono
  CHECK(read_u32(wav, 24) == 16000);
  CHECK(read_u32(wav, 28) == 32000);
  CHECK(read_u16(wav, 34) == 16);
  CHECK(read_u32(wav, 40) == 4);
  CHECK(audio::pcm16_from_wav(wav) == pcm);

  const std::string wav24 = audio::wav_from_pcm16(pcm, 24000);
  CHECK(read_u32(wav24, 24) == 24000);
  CHECK(read_u32(wav24, 28) == 48000);
  CHECK(read_u32(wav24, 40) == 4);
  const auto view24 = audio::parse_wav_pcm16(wav24);
  CHECK(view24.sample_rate == 24000 && view24.channels == 1 &&
        view24.pcm == pcm);

  const std::string empty = audio::wav_from_pcm16("", 8000);
  CHECK(empty.size() == 44);
  CHECK(read_u32(empty, 40) == 0);

  expect_error([&] { audio::wav_from_pcm16("\x01", 16000); }, "odd PCM");
  expect_error([&] { audio::wav_from_pcm16(pcm, 0); }, "zero rate");
  expect_error([&] { audio::wav_from_pcm16(pcm, 0x80000000u); },
               "rate overflow");
}

void test_multipart_upload_and_content_type_and_atomic_download() {
  TempDir dir;
  LocalHttpServer http;
  std::string uploaded_wav;
  std::string uploaded_model;
  std::atomic<int> mode{0};

  http.srv.Post("/v1/audio/transcriptions",
                [&](const httplib::Request& req, httplib::Response& res) {
                  if (req.form.has_file("file")) {
                    uploaded_wav = req.form.get_file("file").content;
                  }
                  if (req.form.has_field("model")) {
                    uploaded_model = req.form.get_field("model");
                  }
                  const nlohmann::json event = {
                      {"type", "transcript.text.done"},
                      {"text", "ok"},
                      {"generation_metadata",
                       {{"generated_token_count", 2},
                        {"finish_reason", "stop"}}}};
                  res.set_content(
                      "data: " + event.dump() + "\n\ndata: [DONE]\n\n",
                      "text/event-stream");
                });

  http.srv.Get(R"(/v1/videos/([^/]+)/content)",
               [&](const httplib::Request&, httplib::Response& res) {
                 const int m = mode.load();
                 if (m == 0) {
                   res.set_content("MP4-PAYLOAD-BYTES",
                                   "Video/MP4; charset=binary");
                 } else if (m == 1) {
                   res.set_content("<html>wrong type</html>", "text/html");
                 } else {
                   res.status = 500;
                   res.set_content(R"({"error":{"message":"boom"}})",
                                   "application/json");
                 }
               });

  http.start();

  // 1. Streamed multipart WAV upload via public ASRClient.
  {
    client::ASRConfig acfg;
    acfg.base_url = "http://127.0.0.1:" + std::to_string(http.port);
    acfg.model = "qwen3-asr";
    client::ASRClient asr(acfg);
    const std::string pcm(640, '\x05');
    client::TranscribeRequest treq;
    treq.pcm = pcm;
    const auto tresp = asr.transcribe(treq);
    CHECK(tresp.text == "ok");
    CHECK(tresp.finish_reason == "stop");
    CHECK(uploaded_model == "qwen3-asr");
    CHECK(audio::pcm16_from_wav(uploaded_wav) == pcm);
  }

  // 2. Atomic download_video publishing, case-insensitive Content-Type params,
  // Content-Type mismatch rejection, and refusal to overwrite existing files.
  {
    client::OpenAIConfig ocfg;
    ocfg.base_url = "http://127.0.0.1:" + std::to_string(http.port) + "/v1";
    ocfg.model = "vid-model";
    ocfg.max_attempts = 1;
    client::OpenAIClient cli(ocfg);

    const std::string target = dir.path + "/video.mp4";
    mode.store(0);
    cli.download_video("job-1", target);
    CHECK(std::filesystem::exists(target));
    CHECK(slurp(target) == "MP4-PAYLOAD-BYTES");

    // Existing file must never be overwritten, and no leftover .part remains.
    expect_error([&] { cli.download_video("job-1", target); },
                 "refuse overwrite existing output");
    CHECK(slurp(target) == "MP4-PAYLOAD-BYTES");

    // Wrong Content-Type (text/html instead of video/mp4 or octet-stream) fails
    // and leaves neither target nor temporary .part behind.
    const std::string bad_ct = dir.path + "/bad_ct.mp4";
    mode.store(1);
    expect_error([&] { cli.download_video("job-1", bad_ct); },
                 "reject unexpected text/html Content-Type");
    CHECK(!std::filesystem::exists(bad_ct));

    // Directory target fails cleanly.
    expect_error([&] { cli.download_video("job-1", dir.path + "/"); },
                 "reject directory output path");

    // Verify no hidden .part files remain in TempDir.
    int dot_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir.path)) {
      const std::string name = entry.path().filename().string();
      if (!name.empty() && name[0] == '.') ++dot_files;
    }
    CHECK(dot_files == 0);
  }
}

void test_public_client_validation() {
  expect_error([] {
    client::OpenAIConfig cfg;
    cfg.base_url = "";
    cfg.model = "m";
    client::OpenAIClient c(cfg);
  }, "empty base_url");

  client::OpenAIConfig cfg;
  cfg.base_url = "http://127.0.0.1:9/v1";
  cfg.model = "m";
  cfg.max_attempts = 1;
  client::OpenAIClient cli(cfg);

  expect_error([&] {
    client::ImageEditRequest req;
    req.prompt = "edit";
    cli.edit_image(req);
  }, "edit_image without image part");

  expect_error([&] {
    client::ASRConfig acfg;
    acfg.base_url = "http://127.0.0.1:9/v1";
    acfg.model = "m";
    client::ASRClient asr(acfg);
    client::TranscribeRequest treq;
    treq.pcm = "\x01";  // odd byte count
    asr.transcribe(treq);
  }, "odd PCM16 length rejected before network");
}

}  // namespace

int main() {
  test_base64();
  test_wav();
  test_multipart_upload_and_content_type_and_atomic_download();
  test_public_client_validation();
  if (failures != 0) {
    std::cerr << failures << " transport blackbox check(s) failed\n";
    return 1;
  }
  std::cout << "omni_transport_test: all blackbox checks passed\n";
  return 0;
}
