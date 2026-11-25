// SPDX-License-Identifier: Apache-2.0
#include "video_input.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace umminfer::qwen38 {
namespace {

constexpr int kVideoMinPixels = 32768;
constexpr int kVideoMaxFramePixels = 131072;
constexpr int kVideoMaxTotalPatches = 3584;
constexpr size_t kDecoderMetadataLimit = 1u << 20;

struct TempFileGuard {
  std::filesystem::path path;
  ~TempFileGuard() {
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
};

std::filesystem::path temp_media_path(const char* suffix) {
  static std::atomic<uint64_t> seq{0};
  return std::filesystem::temp_directory_path() /
         ("umm-qwen38-vid-" + std::to_string(getpid()) + "-" +
          std::to_string(++seq) + suffix);
}

// Runs a POSIX child process directly (no shell) and captures bounded stdout.
// Temporal mode redirects stderr to a bounded-after-exit showinfo metadata file
// so two pipes cannot deadlock while raw RGB is drained.
std::string run_capture(const std::vector<std::string>& args, size_t max_bytes,
                        const std::filesystem::path& stderr_path = {}) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    throw std::runtime_error("pipe() failed while invoking media decoder");
  }
  const pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    throw std::runtime_error("fork() failed while invoking media decoder");
  }
  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);
    int devnull = -1;
    if (stderr_path.empty()) {
      devnull = open("/dev/null", O_WRONLY);
    } else {
      devnull = open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    }
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
      throw std::runtime_error("video decoder stdout exceeded safety limit");
    }
    out.append(buf, static_cast<size_t>(n));
  }
  close(pipefd[0]);
  int status = 0;
  if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw std::runtime_error("video decoder (" + args[0] + ") failed");
  }
  return out;
}

struct VideoProbe {
  int width = 0;
  int height = 0;
  double duration = 0.0;
  int64_t frame_count = 0;  // zero when the container does not declare it
  double average_fps = 0.0;
};

VideoProbe probe_video(const std::filesystem::path& path) {
  const std::string probe = run_capture(
      {"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=width,height,duration,nb_frames,avg_frame_rate:format=duration", "-of",
       "default=noprint_wrappers=1", path.string()},
      64u << 10);
  VideoProbe out;
  std::istringstream iss(probe);
  std::string line;
  while (std::getline(iss, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string val = line.substr(eq + 1);
    try {
      if (key == "width" && out.width == 0) out.width = std::stoi(val);
      else if (key == "height" && out.height == 0) out.height = std::stoi(val);
      else if (key == "duration" && out.duration <= 0.0 && val != "N/A") {
        out.duration = std::stod(val);
      } else if (key == "nb_frames" && out.frame_count == 0 && val != "N/A") {
        out.frame_count = std::stoll(val);
      } else if (key == "avg_frame_rate" && out.average_fps <= 0.0 && val != "N/A") {
        const size_t slash = val.find('/');
        if (slash == std::string::npos) {
          out.average_fps = std::stod(val);
        } else {
          const double numerator = std::stod(val.substr(0, slash));
          const double denominator = std::stod(val.substr(slash + 1));
          if (denominator > 0.0) out.average_fps = numerator / denominator;
        }
      }
    } catch (...) {
    }
  }
  if (out.width <= 0 || out.height <= 0) {
    throw std::runtime_error("video probe failed: no valid video stream found");
  }
  if (!(out.duration > 0.0) || !std::isfinite(out.duration)) out.duration = 2.0;
  if (out.frame_count < 0) out.frame_count = 0;
  if (!(out.average_fps > 0.0) || !std::isfinite(out.average_fps)) {
    out.average_fps = 0.0;
  }
  return out;
}

void target_dimensions(const VideoProbe& probe, int* dst_h, int* dst_w) {
  // Compute the exact Qwen smart_resize target (divisible by 32) once before
  // ffmpeg so ffmpeg performs the single bicubic scale and preprocessing skips
  // a second resize pass.
  smart_resize(probe.height, probe.width, 32, kVideoMinPixels,
               std::min(kVideoMaxFramePixels, (kVideoMaxTotalPatches / 4) * 256),
               dst_h, dst_w);
}

std::string read_metadata(const std::filesystem::path& path) {
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec || size > kDecoderMetadataLimit) {
    throw std::runtime_error("video decoder metadata exceeded safety limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to read video decoder metadata");
  std::string out;
  out.resize(static_cast<size_t>(size));
  if (size != 0) input.read(out.data(), static_cast<std::streamsize>(size));
  if (!input && size != 0) throw std::runtime_error("truncated video decoder metadata");
  return out;
}

std::vector<double> parse_showinfo_timestamps(const std::string& metadata) {
  std::vector<double> out;
  std::istringstream lines(metadata);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find(" n:") == std::string::npos) continue;
    const size_t marker = line.find("pts_time:");
    if (marker == std::string::npos) continue;
    const char* begin = line.c_str() + marker + 9;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) continue;
    out.push_back(value);
  }
  return out;
}

std::vector<std::vector<uint8_t>> split_rgb_frames(const std::string& raw,
                                                   int dst_h, int dst_w) {
  const size_t frame_bytes = static_cast<size_t>(dst_w) * dst_h * 3u;
  if (frame_bytes == 0) {
    throw std::runtime_error("video decode produced a partial RGB frame");
  }
  const size_t count = raw.size() / frame_bytes;
  if (count == 0) throw std::runtime_error("video decode produced zero RGB frames");
  std::vector<std::vector<uint8_t>> frames;
  frames.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto* ptr = reinterpret_cast<const uint8_t*>(raw.data() + i * frame_bytes);
    frames.emplace_back(ptr, ptr + frame_bytes);
  }
  return frames;
}

DecodedVideo decode_uniform(const std::filesystem::path& path,
                            const VideoProbe& probe,
                            const VideoFrameSelection& options) {
  const int target_frames = options.max_frames;
  int dst_h = 0, dst_w = 0;
  target_dimensions(probe, &dst_h, &dst_w);
  const double sample_fps =
      std::max(0.25, static_cast<double>(target_frames) /
                         std::max(0.5, probe.duration));
  char vf[160];
  std::snprintf(vf, sizeof(vf), "fps=%.4f,scale=%d:%d:flags=bicubic", sample_fps,
                dst_w, dst_h);
  // This is intentionally the historical command and output cap. No showinfo,
  // re-timestamping, candidate analysis, or reordering occurs in uniform mode.
  const std::string raw = run_capture(
      {"ffmpeg", "-v", "error", "-i", path.string(), "-an", "-sn", "-dn", "-vf",
       vf, "-frames:v", std::to_string(target_frames), "-f", "rawvideo",
       "-pix_fmt", "rgb24", "pipe:1"},
      static_cast<size_t>(target_frames + 2) * static_cast<size_t>(dst_w) *
          static_cast<size_t>(dst_h) * 3u);
  std::vector<std::vector<uint8_t>> frames =
      split_rgb_frames(raw, dst_h, dst_w);

  DecodedVideo out;
  out.preprocessed = preprocess_video_frames(
      frames, dst_h, dst_w, kVideoMinPixels, kVideoMaxFramePixels,
      kVideoMaxTotalPatches);
  out.decoded_frame_count = frames.size();
  out.selected_frame_indices.reserve(frames.size());
  out.selected_frame_timestamps.reserve(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    out.selected_frame_indices.push_back(i);
    out.selected_frame_timestamps.push_back(
        probe.duration * static_cast<double>(i) / static_cast<double>(frames.size()));
  }
  const int slices = out.preprocessed.grid_t;
  out.timestamps.reserve(static_cast<size_t>(slices));
  for (int s = 0; s < slices; ++s) {
    const double t_sec = slices > 1
                             ? (probe.duration * static_cast<double>(s) /
                                static_cast<double>(slices))
                             : 0.0;
    out.timestamps.push_back(t_sec);
  }
  return out;
}

DecodedVideo decode_temporal(const std::filesystem::path& path,
                             const VideoProbe& probe,
                             const VideoFrameSelection& options) {
  int dst_h = 0, dst_w = 0;
  target_dimensions(probe, &dst_h, &dst_w);
  int64_t source_frames = probe.frame_count;
  if (source_frames == 0 && probe.average_fps > 0.0) {
    const double estimate = probe.duration * probe.average_fps;
    if (std::isfinite(estimate) && estimate > 0.0 && estimate < 9.0e18) {
      source_frames = static_cast<int64_t>(std::llround(estimate));
    }
  }
  const bool preserve_source_pts = source_frames > 0;
  std::string vf;
  if (preserve_source_pts) {
    // select+n refers to decoded source order and does not rewrite PTS. A
    // declared/estimated stride covers the whole clip while -frames:v enforces
    // the same hard output ceiling as the unknown-metadata fallback.
    const int64_t limit = options.max_candidate_frames;
    const int64_t stride = std::max<int64_t>(1, source_frames / limit +
                                                   (source_frames % limit != 0));
    vf = "select='not(mod(n," + std::to_string(stride) +
         "))',scale=" + std::to_string(dst_w) + ":" + std::to_string(dst_h) +
         ":flags=bicubic,showinfo";
  } else {
    const double candidate_fps =
        std::max(0.25, static_cast<double>(options.max_candidate_frames) /
                           std::max(0.5, probe.duration));
    char fallback_vf[192];
    std::snprintf(fallback_vf, sizeof(fallback_vf),
                  "fps=%.4f,scale=%d:%d:flags=bicubic,showinfo", candidate_fps,
                  dst_w, dst_h);
    vf = fallback_vf;
  }
  TempFileGuard metadata_guard;
  metadata_guard.path = temp_media_path(".log");
  // The bound is computed from the already smart-resized dimensions and the
  // hard candidate ceiling before run_capture allocates its stdout string.
  const size_t output_cap =
      static_cast<size_t>(options.max_candidate_frames + 2) *
      static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3u;
  std::vector<std::string> args{"ffmpeg", "-v", "info", "-i", path.string(),
                                "-an", "-sn", "-dn", "-vf", vf};
  if (preserve_source_pts) args.insert(args.end(), {"-vsync", "0"});
  args.insert(args.end(), {"-frames:v", std::to_string(options.max_candidate_frames),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"});
  const std::string raw = run_capture(args, output_cap, metadata_guard.path);
  std::vector<std::vector<uint8_t>> frames =
      split_rgb_frames(raw, dst_h, dst_w);
  const std::vector<double> source_timestamps =
      parse_showinfo_timestamps(read_metadata(metadata_guard.path));

  std::vector<VideoCandidate> candidates;
  candidates.reserve(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    double timestamp = probe.duration * static_cast<double>(i) /
                       static_cast<double>(frames.size());
    if (i < source_timestamps.size() && std::isfinite(source_timestamps[i])) {
      timestamp = source_timestamps[i];
    }
    // A short decode or missing showinfo rows is a decode gap, not permission
    // to allocate more frames or synthesize extra content. Keep valid frames
    // and use deterministic nominal timestamps only for missing/non-finite PTS.
    candidates.push_back(VideoCandidate{std::move(frames[i]), timestamp});
  }
  return preprocess_temporal_candidates(candidates, dst_h, dst_w, options);
}

}  // namespace

DecodedVideo preprocess_temporal_candidates(
    const std::vector<VideoCandidate>& candidates, int src_h, int src_w,
    const VideoFrameSelection& options) {
  validate_video_frame_selection(options);
  if (options.policy != VideoFramePolicy::kTemporal) {
    throw std::invalid_argument("preprocess_temporal_candidates requires temporal policy");
  }
  if (candidates.empty()) {
    throw std::invalid_argument("temporal selection requires at least one candidate");
  }
  if (candidates.size() > static_cast<size_t>(options.max_candidate_frames)) {
    throw std::invalid_argument("temporal candidates exceed the configured frame budget");
  }
  const std::vector<TemporalFrameScore> scores =
      analyze_temporal_frames(candidates, src_h, src_w);
  const std::vector<size_t> selected = select_temporal_keyframes(scores, options);
  if (selected.empty() || selected.size() > static_cast<size_t>(options.max_frames)) {
    throw std::runtime_error("temporal selection violated the frame budget");
  }

  std::vector<std::vector<uint8_t>> frames;
  frames.reserve(selected.size());
  DecodedVideo out;
  out.decoded_frame_count = candidates.size();
  out.selected_frame_indices = selected;
  out.selected_frame_timestamps.reserve(selected.size());
  for (size_t index : selected) {
    frames.push_back(candidates[index].rgb);
    out.selected_frame_timestamps.push_back(candidates[index].timestamp_seconds);
  }
  out.preprocessed = preprocess_video_frames(
      frames, src_h, src_w, kVideoMinPixels, kVideoMaxFramePixels,
      kVideoMaxTotalPatches);
  out.timestamps.reserve(static_cast<size_t>(out.preprocessed.grid_t));
  for (int slice = 0; slice < out.preprocessed.grid_t; ++slice) {
    const size_t pair_first = std::min(static_cast<size_t>(slice * 2), frames.size() - 1);
    out.timestamps.push_back(candidates[selected[pair_first]].timestamp_seconds);
  }
  return out;
}

DecodedVideo decode_video_bytes(std::string_view bytes,
                                const VideoFrameSelection& options) {
  validate_video_frame_selection(options);
  if (bytes.empty()) throw std::invalid_argument("video input is empty");
  TempFileGuard guard;
  guard.path = temp_media_path(".mp4");
  {
    std::ofstream f(guard.path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("failed to create temp file for video decode");
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!f) throw std::runtime_error("failed to write temp file for video decode");
  }
  const VideoProbe probe = probe_video(guard.path);
  if (options.policy == VideoFramePolicy::kUniform) {
    return decode_uniform(guard.path, probe, options);
  }
  return decode_temporal(guard.path, probe, options);
}

}  // namespace umminfer::qwen38
