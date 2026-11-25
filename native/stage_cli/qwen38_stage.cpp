// SPDX-License-Identifier: Apache-2.0
// qwen38 kind: Qwen3.8-27B (text + one optional image) behind the Stage
// protocol. Prompt rendering and image preprocessing follow native/qwen38_cli.
//
// Deltas: every generated token is detokenized and the new suffix is sent as a
// stream chunk {"text": ...}. The HTTP layer forwards them as SSE deltas and
// still receives the aggregate result at completion.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Same subset the accepted image_preprocess.cpp compiles in, so the decoder
// declarations match its implementation.
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_THREAD_LOCALS
#include "qwen38/stb_image.h"

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/tokenizer.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxNewTokens = 256;
constexpr int kMinPixels = 65536;
constexpr int kMaxPixels = 16777216;

// Adapts the shared, host-tested sampling parse to the engine's struct. The
// engine's type is local to the CUDA tree, so the fields are copied explicitly
// rather than depending on layout compatibility.
qwen38::SamplingParams parse_sampling(const Value::Map& params) {
  const SamplingParams parsed = parse_sampling_params(params);
  qwen38::SamplingParams sampling;
  sampling.temperature = parsed.temperature;
  sampling.top_p = parsed.top_p;
  sampling.seed = parsed.seed;
  sampling.has_seed = parsed.has_seed;
  return sampling;
}

qwen38::PreprocessedImage decode_image(const std::string& url) {
  const std::string bytes = load_image_bytes(url);
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes.data()),
      static_cast<int>(bytes.size()), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw ProtocolError(std::string("image decode failed: ") + stbi_failure_reason());
  }
  std::vector<uint8_t> rgb(pixels, pixels + static_cast<size_t>(width) * height * 3);
  stbi_image_free(pixels);
  return qwen38::preprocess_rgb(rgb.data(), height, width, kMinPixels, kMaxPixels);
}

struct DecodedVideo {
  qwen38::PreprocessedImage preprocessed;
  std::vector<double> timestamps;
};

struct TempFileGuard {
  std::filesystem::path path;
  ~TempFileGuard() {
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
};

// Runs a POSIX child process directly (no shell) and captures stdout.
std::string run_capture(const std::vector<std::string>& args,
                        size_t max_bytes = 64u << 20) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    throw ProtocolError("pipe() failed while invoking media decoder");
  }
  const pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    throw ProtocolError("fork() failed while invoking media decoder");
  }
  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);
    const int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  close(pipefd[1]);
  std::string out;
  char buf[65536];
  ssize_t n = 0;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
    if (out.size() + static_cast<size_t>(n) > max_bytes) {
      close(pipefd[0]);
      int status = 0;
      waitpid(pid, &status, 0);
      throw ProtocolError("video decoder stdout exceeded safety limit");
    }
    out.append(buf, static_cast<size_t>(n));
  }
  close(pipefd[0]);
  int status = 0;
  if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw ProtocolError("video decoder (" + args[0] + ") failed");
  }
  return out;
}

DecodedVideo decode_video(const std::string& url) {
  const std::string bytes = load_video_bytes(url);
  static std::atomic<uint64_t> seq{0};
  TempFileGuard guard;
  guard.path = std::filesystem::temp_directory_path() /
               ("umm-qwen38-vid-" + std::to_string(getpid()) + "-" +
                std::to_string(++seq) + ".mp4");
  {
    std::ofstream f(guard.path, std::ios::binary | std::ios::trunc);
    if (!f) throw ProtocolError("failed to create temp file for video decode");
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  // Probe source width, height, and duration (seconds).
  const std::string probe = run_capture(
      {"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=width,height,duration:format=duration", "-of",
       "default=noprint_wrappers=1", guard.path.string()},
      64u << 10);
  int src_w = 0, src_h = 0;
  double duration = 0.0;
  std::istringstream iss(probe);
  std::string line;
  while (std::getline(iss, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string val = line.substr(eq + 1);
    try {
      if (key == "width" && src_w == 0) src_w = std::stoi(val);
      else if (key == "height" && src_h == 0) src_h = std::stoi(val);
      else if (key == "duration" && duration <= 0.0 && val != "N/A") {
        duration = std::stod(val);
      }
    } catch (...) {
    }
  }
  if (src_w <= 0 || src_h <= 0) {
    throw ProtocolError("video probe failed: no valid video stream found");
  }
  if (!(duration > 0.0) || !std::isfinite(duration)) duration = 2.0;

  // Target <= 384px on the long edge (divisible by 2) for fast decode;
  // preprocess_video_frames then applies exact Qwen smart_resize (divisible by 32).
  constexpr int kMaxSide = 384;
  int dst_w = src_w, dst_h = src_h;
  if (std::max(dst_w, dst_h) > kMaxSide) {
    const double s = static_cast<double>(kMaxSide) / std::max(dst_w, dst_h);
    dst_w = std::max(32, static_cast<int>(std::round(dst_w * s / 2.0)) * 2);
    dst_h = std::max(32, static_cast<int>(std::round(dst_h * s / 2.0)) * 2);
  }
  constexpr int kTargetFrames = 8;  // 4 temporal patches (temporal_patch_size=2)
  const double sample_fps =
      std::max(0.25, static_cast<double>(kTargetFrames) / std::max(0.5, duration));
  char vf[160];
  std::snprintf(vf, sizeof(vf), "fps=%.4f,scale=%d:%d:flags=bicubic",
                sample_fps, dst_w, dst_h);
  const std::string raw = run_capture(
      {"ffmpeg", "-v", "error", "-i", guard.path.string(), "-vf", vf,
       "-frames:v", std::to_string(kTargetFrames), "-f", "rawvideo",
       "-pix_fmt", "rgb24", "pipe:1"},
      static_cast<size_t>(kTargetFrames + 2) * static_cast<size_t>(dst_w) *
          static_cast<size_t>(dst_h) * 3u);

  const size_t frame_bytes =
      static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3u;
  const size_t n_frames = frame_bytes > 0 ? raw.size() / frame_bytes : 0;
  if (n_frames == 0) {
    throw ProtocolError("video decode produced zero RGB frames");
  }
  std::vector<std::vector<uint8_t>> frames;
  frames.reserve(n_frames);
  for (size_t i = 0; i < n_frames; ++i) {
    const auto* ptr = reinterpret_cast<const uint8_t*>(raw.data() + i * frame_bytes);
    frames.emplace_back(ptr, ptr + frame_bytes);
  }

  DecodedVideo out;
  out.preprocessed =
      qwen38::preprocess_video_frames(frames, dst_h, dst_w, 32768, 131072, 3584);
  const int slices = out.preprocessed.grid_t;
  out.timestamps.reserve(static_cast<size_t>(slices));
  for (int s = 0; s < slices; ++s) {
    const double t_sec =
        slices > 1 ? (duration * static_cast<double>(s) / static_cast<double>(slices))
                   : 0.0;
    out.timestamps.push_back(t_sec);
  }
  return out;
}

}  // namespace

const char* compiled_kind() { return "qwen38"; }

const char* stage_usage_flags() {
  return "[--max-new-tokens N] [--max-seq N] [--batch-size N] ";
}

bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options) {
  if (key == "max-new-tokens") {
    options.max_new_tokens = parse_stage_int(value, "--max-new-tokens");
    return true;
  }
  if (key == "max-seq") {
    options.max_seq = parse_stage_int(value, "--max-seq");
    return true;
  }
  if (key == "batch-size") {
    options.batch_size = parse_stage_int(value, "--batch-size");
    return true;
  }
  return false;
}

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_new_tokens =
      options.max_new_tokens > 0 ? options.max_new_tokens : kDefaultMaxNewTokens;
  auto tokenizer = std::make_shared<qwen38::Qwen2BpeTokenizer>(
      options.model_dir + "/vocab.json", options.model_dir + "/merges.txt",
      options.model_dir + "/tokenizer_config.json");
  // batch_size > 1 enables continuous batching; the engine owns one batch
  // thread and is safe to call from several scheduler workers.
  const int batch_size = options.batch_size > 1 ? options.batch_size : 1;
  const int max_seq = options.max_seq > 0 ? options.max_seq : 8192;
  auto engine = std::make_shared<qwen38::Engine>(
      options.model_dir, options.gpu, max_seq, batch_size);

  const int gpu = options.gpu;
  return [tokenizer, engine, max_new_tokens, max_seq, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const OmniWork input = parse_qwen38_work(work);
    const int req_max_new_tokens =
        resolve_max_new_tokens(bound_request_params(), max_new_tokens, max_seq);
    const qwen38::SamplingParams sampling =
        parse_sampling(bound_request_params());

    std::unique_ptr<qwen38::PreprocessedImage> image;
    std::vector<double> video_timestamps;
    bool is_video = false;
    int image_tokens = 0;
    if (!input.image_url.empty()) {
      image = std::make_unique<qwen38::PreprocessedImage>(decode_image(input.image_url));
      if (image->grid_t != 1) {
        throw ProtocolError("qwen38 image path expects grid_t == 1");
      }
      image_tokens = image->rows() / 4;
    } else if (!input.video_url.empty()) {
      DecodedVideo decoded = decode_video(input.video_url);
      video_timestamps = std::move(decoded.timestamps);
      image = std::make_unique<qwen38::PreprocessedImage>(
          std::move(decoded.preprocessed));
      is_video = true;
      image_tokens = (image->grid_h * image->grid_w) / 4;
    }
    std::vector<qwen38::ChatTurn> turns;
    turns.reserve(input.messages.size() + 1);
    for (size_t i = 0; i < input.messages.size(); ++i) {
      qwen38::ChatTurn turn;
      turn.role = input.messages[i].role;
      turn.text = input.messages[i].text;
      const bool attach_here =
          is_video
              ? (input.messages[i].has_video || i + 1 == input.messages.size())
              : (input.messages[i].has_image || i + 1 == input.messages.size());
      if (image != nullptr && attach_here) {
        if (is_video) {
          turn.video_slices = image->grid_t;
          turn.video_timestamps = video_timestamps;
        } else {
          turn.has_image = true;
        }
      }
      turns.push_back(std::move(turn));
    }
    if (turns.empty()) {
      qwen38::ChatTurn turn;
      turn.role = "user";
      turn.text = input.text;
      if (image != nullptr) {
        if (is_video) {
          turn.video_slices = image->grid_t;
          turn.video_timestamps = video_timestamps;
        } else {
          turn.has_image = true;
        }
      }
      turns.push_back(std::move(turn));
    }
    // The prompt splits into the completed turns and the open assistant header.
    // The header is not cacheable: the next turn replaces it with the reply, so
    // a snapshot taken after it never matches.
    const qwen38::RenderedChatPrompt parts =
        qwen38::render_chat_prompt_parts(turns, /*enable_thinking=*/false);
    std::vector<int> ids = tokenizer->encode(parts.full());
    // Snapshot at the turn boundary. Only a prefix that tokenizes to a leading
    // slice of the full prompt is usable; anything else keeps the whole-prompt
    // snapshot rather than risk a wrong prefix.
    int cache_prefix_len = -1;
    if (image == nullptr) {
      const std::vector<int> prefix_ids = tokenizer->encode(parts.prefix);
      if (prefix_ids.size() < ids.size() &&
          std::equal(prefix_ids.begin(), prefix_ids.end(), ids.begin())) {
        cache_prefix_len = static_cast<int>(prefix_ids.size());
      }
    }
    if (image != nullptr) {
      if (is_video) {
        ids = qwen38::expand_video_pads(ids, image_tokens, image->grid_t);
      } else {
        ids = qwen38::expand_image_pads(ids, image_tokens);
      }
    }

    // Byte-level BPE decode is additive per token: buffer only incomplete
    // trailing UTF-8 bytes so streaming stays O(1) per token instead of O(T^2).
    std::string pending_utf8;
    // Copy the sink and id bound to this compute call. In batched mode the
    // callback fires on the engine's batch thread, where the scheduler's
    // thread-locals are not bound.
    const StreamFn stream = bound_stream_fn();
    const std::string req_id = bound_request_id();
    auto emit_token = [&tokenizer, &pending_utf8, stream, req_id](int id) {
      pending_utf8 += tokenizer->decode({id});
      const size_t safe = utf8_complete_prefix(pending_utf8);
      if (safe == 0) return;
      Value::Map chunk;
      chunk["text"] = Value(pending_utf8.substr(0, safe));
      Value value(std::move(chunk));
      if (stream) {
        stream(req_id, std::move(value));
      } else {
        emit_stage_stream(std::move(value));
      }
      pending_utf8.erase(0, safe);
    };

    qwen38::GenResult result;
    try {
      if (engine->batch_capacity() > 1) {
        qwen38::BatchRequest request;
        request.prompt_ids = std::move(ids);
        request.image = image.get();
        request.max_new_tokens = req_max_new_tokens;
        request.sampling = sampling;
        request.abort = [&tracker] { return tracker(); };
        request.on_token = emit_token;
        request.cache_prefix_len = cache_prefix_len;
        result = engine->generate_batched(request);
      } else {
        result = engine->generate(ids, image.get(), req_max_new_tokens, "",
                                  [&tracker] { return tracker(); }, emit_token,
                                  sampling, cache_prefix_len);
      }
    } catch (const qwen38::GenerationAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "qwen38 request aborted request_id=" << req_id
                << " boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    const std::string text = tokenizer->decode(result.ids);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    const double decode_tok_s = result.decode_ms > 0.0
                                    ? result.decode_steps * 1000.0 / result.decode_ms
                                    : 0.0;
    const char* finish_reason = result.stopped_by_eos ? "stop" : "length";
    std::cerr << "qwen38 request request_id=" << req_id
              << " prompt_tokens=" << result.prompt_tokens
              << " prefill_tokens=" << result.prefill_tokens
              << " prefix_reused=" << result.reused_tokens
              << " image_tokens=" << result.image_tokens
              << " generated_tokens=" << result.ids.size()
              << " finish_reason=" << finish_reason
              << " chars=" << text.size()
              << " prefill_ms=" << std::fixed << std::setprecision(1)
              << result.prefill_ms
              << " decode_ms=" << std::fixed << std::setprecision(1)
              << result.decode_ms
              << " decode_tok_per_s=" << std::fixed << std::setprecision(1)
              << decode_tok_s
              << " elapsed_ms=" << elapsed.count() << "\n";
    return omni_result(text, static_cast<int>(result.ids.size()),
                       result.prefill_ms, result.decode_ms, finish_reason);
  };
}

}  // namespace umminfer::stage
