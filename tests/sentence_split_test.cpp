// SPDX-License-Identifier: Apache-2.0
// Blackbox API test for streamed TTS sentence splitting over
// POST /v1/chat/completions (SSE + audio.stream=true). Exercises CJK
// terminators, decimal numbers ("3.14"), closing quotes, repeated terminators,
// 48-codepoint weak-break windows, UTF-8 integrity, and the 32-leg fan-out cap
// strictly through OpenAIApi + OpenAIClient without calling internal helpers.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "umminfer/audio/audio.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

struct StreamTtsHarness {
  std::string ipc_dir;
  std::unique_ptr<umminfer::Coordinator> coordinator;
  std::unique_ptr<umminfer::Stage> vlm_stage;
  std::unique_ptr<umminfer::Stage> tts_stage;
  std::unique_ptr<umminfer::UnifiedEngine> unified;
  std::unique_ptr<umminfer::OpenAIApi> api;
  std::thread vlm_thread;
  std::thread tts_thread;
  std::thread coord_thread;
  std::thread api_thread;

  std::mutex mu;
  std::vector<std::string> vlm_chunks;
  std::vector<std::string> tts_texts;

  StreamTtsHarness() {
    static std::atomic<int> seq{0};
    ipc_dir = "/tmp/umm-ssplit-bb-" + std::to_string(::getpid()) + "-" +
              std::to_string(++seq);
    std::filesystem::create_directories(ipc_dir);
    const std::string done_ep = "ipc://" + ipc_dir + "/done";
    const std::string abort_ep = "ipc://" + ipc_dir + "/abort";
    const std::string vlm_ep = "ipc://" + ipc_dir + "/vlm";
    const std::string tts_ep = "ipc://" + ipc_dir + "/tts";

    coordinator = std::make_unique<umminfer::Coordinator>(done_ep, abort_ep);
    coordinator->register_stage("vlm", vlm_ep, /*require_ready=*/true);
    coordinator->register_stage("tts", tts_ep, /*require_ready=*/true);

    umminfer::ApiConfig acfg;
    acfg.host = "127.0.0.1";
    acfg.port = 0;
    acfg.model = "served-model";
    acfg.entry_stage = "vlm";
    acfg.request_timeout_s = 10.0;
    api = std::make_unique<umminfer::OpenAIApi>(*coordinator, acfg);

    umminfer::UnifiedConfig ucfg;
    ucfg.vlm_stage = "vlm";
    ucfg.tts_stage = "tts";
    unified = std::make_unique<umminfer::UnifiedEngine>(*coordinator, ucfg);
    api->set_unified_engine(*unified, "vlm");
    coordinator->start();
    coord_thread = std::thread([this] { coordinator->run(); });

    auto make_cfg = [&](const std::string& name, const std::string& ep) {
      umminfer::Stage::Config scfg;
      scfg.name = name;
      scfg.recv_endpoint = ep;
      scfg.coordinator_endpoint = done_ep;
      scfg.abort_endpoint = abort_ep;
      scfg.announce_ready = true;
      scfg.max_concurrency = 1;
      return scfg;
    };

    vlm_stage = std::make_unique<umminfer::Stage>(
        make_cfg("vlm", vlm_ep),
        [this](const umminfer::Value&, const umminfer::AbortCheck&) {
          std::vector<std::string> chunks;
          {
            std::lock_guard<std::mutex> lock(mu);
            chunks = vlm_chunks;
          }
          std::string full;
          for (const std::string& piece : chunks) {
            full += piece;
            umminfer::Value::Map delta;
            delta["text"] = umminfer::Value(piece);
            umminfer::emit_stage_stream(umminfer::Value(std::move(delta)));
          }
          umminfer::Value::Map out;
          out["content"] = umminfer::Value(full);
          out["finish_reason"] = umminfer::Value("stop");
          return umminfer::Value(std::move(out));
        });

    tts_stage = std::make_unique<umminfer::Stage>(
        make_cfg("tts", tts_ep),
        [this](const umminfer::Value& work, const umminfer::AbortCheck&) {
          std::string text;
          if (const auto* t = work.find("text"); t != nullptr && t->is_str()) {
            text = t->as_str();
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            tts_texts.push_back(text);
          }
          const std::string pcm(480, '\x02');
          umminfer::Value::Map chunk;
          chunk["audio"] = umminfer::Value::bin(pcm);
          chunk["index"] = umminfer::Value(int64_t{0});
          umminfer::emit_stage_stream(umminfer::Value(std::move(chunk)));

          umminfer::Value::Map out;
          out["media_type"] = umminfer::Value("audio");
          out["mime_type"] = umminfer::Value("audio/wav");
          out["format"] = umminfer::Value("wav");
          out["sample_rate"] = umminfer::Value(int64_t{24000});
          out["channels"] = umminfer::Value(int64_t{1});
          out["streamed"] = umminfer::Value(true);
          out["data"] =
              umminfer::Value::bin(umminfer::audio::wav_from_pcm16(pcm, 24000));
          return umminfer::Value(std::move(out));
        });

    vlm_stage->start();
    tts_stage->start();
    vlm_thread = std::thread([this] { vlm_stage->run(); });
    tts_thread = std::thread([this] { tts_stage->run(); });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (coordinator->stage_ready("vlm") &&
          coordinator->stage_ready("tts")) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!api->bind()) throw std::runtime_error("OpenAIApi bind failed");
    api_thread = std::thread([this] { api->listen(); });
  }

  ~StreamTtsHarness() {
    if (api) api->stop();
    if (api_thread.joinable()) api_thread.join();
    if (vlm_stage) vlm_stage->stop();
    if (tts_stage) tts_stage->stop();
    if (vlm_thread.joinable()) vlm_thread.join();
    if (tts_thread.joinable()) tts_thread.join();
    if (coordinator) coordinator->stop();
    if (coord_thread.joinable()) coord_thread.join();
    std::error_code ec;
    std::filesystem::remove_all(ipc_dir, ec);
  }

  struct SseResult {
    std::string text;
    size_t audio_artifacts = 0;
  };

  std::vector<std::string> run_streamed_tts(
      std::vector<std::string> chunks, SseResult* out_resp = nullptr) {
    {
      std::lock_guard<std::mutex> lock(mu);
      vlm_chunks = std::move(chunks);
      tts_texts.clear();
    }
    httplib::Client cli("127.0.0.1", api->bound_port());
    cli.set_read_timeout(10, 0);
    const nlohmann::json body = {
        {"model", "served-model"},
        {"stream", true},
        {"modalities", {"text", "audio"}},
        {"audio", {{"voice", "vivian"}, {"format", "wav"}, {"stream", true}}},
        {"messages", {{{"role", "user"}, {"content", "speak"}}}}};
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    check(res && res->status == 200, "POST /v1/chat/completions returned 200");
    if (res && out_resp != nullptr) {
      SseResult parsed;
      size_t pos = 0;
      while (pos < res->body.size()) {
        const size_t end = res->body.find('\n', pos);
        const std::string line =
            end == std::string::npos ? res->body.substr(pos)
                                     : res->body.substr(pos, end - pos);
        pos = end == std::string::npos ? res->body.size() : end + 1;
        if (line.rfind("data: ", 0) != 0) continue;
        const std::string payload = line.substr(6);
        if (payload == "[DONE]") break;
        const auto ev = nlohmann::json::parse(payload, nullptr, false);
        if (ev.is_discarded()) continue;
        if (ev.contains("choices") && ev["choices"].is_array() &&
            !ev["choices"].empty()) {
          const auto& delta = ev["choices"][0].value("delta", nlohmann::json::object());
          if (delta.contains("content") && delta["content"].is_string()) {
            parsed.text += delta["content"].get<std::string>();
          }
        }
        if (ev.contains("unified") && ev["unified"].is_object() &&
            ev["unified"].value("kind", "") == "artifact") {
          ++parsed.audio_artifacts;
        }
      }
      *out_resp = std::move(parsed);
    }
    std::lock_guard<std::mutex> lock(mu);
    return tts_texts;
  }
};

bool valid_utf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 0;
    if (c < 0x80) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    else return false;
    if (i + len > s.size()) return false;
    for (size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

}  // namespace

int main() {
  StreamTtsHarness h;

  // 1. CJK terminator splits mid-stream into separate TTS legs and reassembles audio.
  {
    StreamTtsHarness::SseResult resp;
    const auto legs = h.run_streamed_tts({"你好。世界"}, &resp);
    check(legs.size() == 2 && legs[0] == "你好。" && legs[1] == "世界",
          "CJK terminator split into exact TTS legs");
    check(resp.text == "你好。世界", "full streamed text preserved");
    check(resp.audio_artifacts == 2, "streamed PCM16 chunks delivered per leg");
  }

  // 2. Decimal period ("3.14") is not split; ASCII period before whitespace is.
  {
    const auto legs =
        h.run_streamed_tts({"version 3.14 is out. ", "Next clause"});
    check(legs.size() == 2 && legs[0] == "version 3.14 is out." &&
              legs[1] == "Next clause",
          "decimal 3.14 kept intact while ASCII sentence period split");
  }

  // 3. Closing quotes and repeated terminators stay attached to their sentences.
  {
    const auto legs =
        h.run_streamed_tts({"他说：“好。”真的吗？！然后走了。"});
    check(legs.size() == 3 && legs[0] == "他说：“好。”" &&
              legs[1] == "真的吗？！" && legs[2] == "然后走了。",
          "closing quotes and repeated terminators stayed with their sentence");
  }

  // 4. Weak break (comma + space) wins over hard 48-char edge on unpunctuated run.
  {
    const std::string clause =
        std::string(30, 'a') + ", " + std::string(30, 'b');
    const auto legs = h.run_streamed_tts({clause});
    check(legs.size() == 2 &&
              (legs[0] == std::string(30, 'a') + "," ||
               legs[0] == std::string(30, 'a') + ", ") &&
              legs[1] == std::string(30, 'b'),
          "48-codepoint window cut at weak comma+space boundary");
  }

  // 5. Long unpunctuated CJK run cuts at 48 codepoints (144 UTF-8 bytes) cleanly.
  {
    std::string multi;
    for (int i = 0; i < 25; ++i) multi += "中文";  // 50 codepoints
    const auto legs = h.run_streamed_tts({multi});
    check(legs.size() == 2 && legs[0].size() == 48 * 3 &&
              legs[1].size() == 2 * 3 && valid_utf8(legs[0]) &&
              valid_utf8(legs[1]),
          "48-codepoint CJK cut preserved valid UTF-8 boundaries");
  }

  // 6. Pathological rapid punctuation is bounded by the 32-leg cap without dropping text.
  {
    std::vector<std::string> many;
    std::string expected;
    for (int i = 0; i < 45; ++i) {
      std::string s = "Word " + std::to_string(i) + " ends. ";
      expected += s;
      many.push_back(std::move(s));
    }
    StreamTtsHarness::SseResult resp;
    const auto legs = h.run_streamed_tts(many, &resp);
    check(legs.size() == 32, "streamed TTS fan-out capped at 32 legs");
    check(resp.text == expected, "all 45 sentences preserved in chat text");
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "%d sentence_split blackbox check(s) failed\n",
                 g_failures);
    return 1;
  }
  std::puts("sentence_split blackbox API tests: PASS");
  return 0;
}
