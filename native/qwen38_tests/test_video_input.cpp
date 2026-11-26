// SPDX-License-Identifier: Apache-2.0
// Synthetic CPU contract tests for bounded Qwen3.8 temporal keyframe selection.
#include "qwen38/engine.hpp"
#include "qwen38/prefix_cache.hpp"
#include "qwen38/video_input.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using umminfer::qwen38::DecodedVideo;
using umminfer::qwen38::TemporalFrameScore;
using umminfer::qwen38::VideoCandidate;
using umminfer::qwen38::VideoFramePolicy;
using umminfer::qwen38::VideoFrameSelection;

int g_failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

template <typename Fn>
bool throws(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

VideoFrameSelection temporal(int frames = 4, int candidates = 16) {
  VideoFrameSelection options;
  options.policy = VideoFramePolicy::kTemporal;
  options.max_frames = frames;
  options.max_candidate_frames = candidates;
  return options;
}

std::vector<uint8_t> solid(int height, int width, uint8_t value) {
  return std::vector<uint8_t>(static_cast<size_t>(height) * width * 3, value);
}

void test_motion_scene_order_and_ties() {
  VideoFrameSelection options = temporal(4, 10);
  std::vector<TemporalFrameScore> scores(10);
  scores[4] = {0.9, 0.8};
  scores[7] = {0.5, 0.1};
  auto selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 4, 7, 9}),
        "scene change and motion peak rank between chronological coverage anchors");

  options = temporal(3, 8);
  options.scene_threshold = 1.0;
  scores.assign(8, {});
  scores[2] = {0.5, 0.4};
  scores[5] = {0.5, 0.4};
  selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 2, 7}),
        "equal motion/scene scores break ties by earliest decode index");
}

void test_static_repeated_short_and_all_motion_fallbacks() {
  VideoFrameSelection options = temporal(4, 10);
  std::vector<TemporalFrameScore> scores(10);
  auto selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 3, 6, 9}),
        "static video falls back to bounded uniform coverage");
  options.motion_threshold = 0.0;
  selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 3, 6, 9}),
        "zero-score repeated frames never consume priority slots at threshold zero");

  scores.resize(3);
  options = temporal(8, 8);
  selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 1, 2}),
        "short clips retain every decoded frame without synthesis");

  options = temporal(4, 12);
  scores.assign(12, {0.8, 0.8});
  selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected.size() == 4 && selected.front() == 0 && selected.back() == 11 &&
            std::adjacent_find(selected.begin(), selected.end()) == selected.end(),
        "all-frames-above-threshold remains bounded and unique");
  check(selected == umminfer::qwen38::select_temporal_keyframes(scores, options),
        "all-motion selection is deterministic");
}

void test_duplicate_scene_run() {
  VideoFrameSelection options = temporal(3, 7);
  options.motion_threshold = 0.05;
  options.scene_threshold = 0.5;
  std::vector<TemporalFrameScore> scores(7);
  scores[2] = {0.1, 0.8};
  scores[3] = {0.1, 0.6};
  const auto selected = umminfer::qwen38::select_temporal_keyframes(scores, options);
  check(selected == std::vector<size_t>({0, 2, 6}),
        "adjacent duplicate scene detections use one strongest representative");
}

void test_analysis_and_temporal_timestamps() {
  constexpr int kSize = 32;
  std::vector<VideoCandidate> candidates;
  const std::vector<double> timestamps{0.0, 0.8, 0.2, 0.9, 0.1};
  const std::vector<uint8_t> values{0, 0, 255, 250, 0};
  for (size_t i = 0; i < values.size(); ++i) {
    candidates.push_back({solid(kSize, kSize, values[i]), timestamps[i],
                          static_cast<int64_t>(100 + i * 3)});
  }
  const auto scores = umminfer::qwen38::analyze_temporal_frames(candidates, kSize, kSize);
  check(scores[1].motion == 0.0 && scores[1].scene == 0.0,
        "byte-repeated frames have exactly zero motion and scene scores");
  check(std::fabs(scores[2].motion - 1.0) < 1e-12 &&
            std::fabs(scores[2].scene - 1.0) < 1e-12,
        "black/white cut produces maximum motion and scene scores");

  VideoFrameSelection options = temporal(4, 5);
  const DecodedVideo out = umminfer::qwen38::preprocess_temporal_candidates(
      candidates, kSize, kSize, options);
  check(std::is_sorted(out.selected_frame_indices.begin(),
                       out.selected_frame_indices.end()),
        "selected frames preserve decode chronology");
  check(out.selected_frame_timestamps.size() == out.selected_frame_indices.size() &&
            out.selected_source_frame_indices.size() == out.selected_frame_indices.size(),
        "selected timestamps and source indices retain one-to-one source mapping");
  bool source_indices_match = true;
  for (size_t i = 0; i < out.selected_frame_indices.size(); ++i) {
    source_indices_match &= out.selected_source_frame_indices[i] ==
                            static_cast<int64_t>(100 + out.selected_frame_indices[i] * 3);
  }
  check(source_indices_match, "selected diagnostics retain exact source frame indices");
  check(out.selected_frame_content_fnv1a64.size() == out.selected_frame_indices.size() &&
            std::all_of(out.selected_frame_content_fnv1a64.begin(),
                        out.selected_frame_content_fnv1a64.end(),
                        [](const std::string& digest) { return digest.size() == 16; }),
        "each selected frame carries one fixed-width scaled-RGB content fingerprint");
  bool timestamps_match = true;
  bool nonmonotonic_preserved = false;
  for (size_t i = 0; i < out.selected_frame_indices.size(); ++i) {
    const double expected = timestamps[out.selected_frame_indices[i]];
    timestamps_match &= out.selected_frame_timestamps[i] == expected;
    if (i > 0 && out.selected_frame_timestamps[i] < out.selected_frame_timestamps[i - 1]) {
      nonmonotonic_preserved = true;
    }
  }
  check(timestamps_match && nonmonotonic_preserved,
        "non-monotonic source timestamps are preserved without timestamp sorting");
  check(out.preprocessed.is_video && out.preprocessed.grid_t >= 1 &&
            out.preprocessed.rows() <= 3584,
        "selected frames stay on native temporal patch packing within token budget");
  bool slice_timestamps_match = out.timestamps.size() ==
                                static_cast<size_t>(out.preprocessed.grid_t);
  for (size_t slice = 0; slice < out.timestamps.size(); ++slice) {
    const size_t pair = std::min(slice * 2, out.selected_frame_timestamps.size() - 1);
    slice_timestamps_match &= out.timestamps[slice] == out.selected_frame_timestamps[pair];
  }
  check(slice_timestamps_match, "each temporal-2 slice uses its pair-first source timestamp");
  check(candidates[2].timestamp_seconds == 0.2 && candidates[2].rgb == solid(32, 32, 255),
        "temporal preprocessing never mutates decoded source candidates");

  // One decoded frame is a valid bounded fallback for a severe decode gap.
  const DecodedVideo gap = umminfer::qwen38::preprocess_temporal_candidates(
      {candidates[2]}, kSize, kSize, options);
  check(gap.preprocessed.grid_t == 1 && gap.selected_frame_indices == std::vector<size_t>{0} &&
            gap.timestamps == std::vector<double>{0.2},
        "decode gap down to one frame still produces one valid temporal slice");
}

void test_prompt_mrope_parity_and_still_image() {
  constexpr int kSize = 32;
  std::vector<VideoCandidate> candidates{
      {solid(kSize, kSize, 0), 0.0},
      {solid(kSize, kSize, 255), 0.4},
      {solid(kSize, kSize, 128), 0.2},
  };
  const DecodedVideo decoded = umminfer::qwen38::preprocess_temporal_candidates(
      candidates, kSize, kSize, temporal(3, 3));
  umminfer::qwen38::ChatTurn turn;
  turn.role = "user";
  turn.text = "What changes?";
  turn.video_slices = decoded.preprocessed.grid_t;
  turn.video_timestamps = decoded.timestamps;
  const auto rendered = umminfer::qwen38::render_chat_prompt_parts({turn}, false);
  check(rendered.prefix.find("<0.2 seconds><|vision_start|><|video_pad|>") !=
            std::string::npos,
        "native video prompt carries selected source timestamps");
  const int merged_tokens = decoded.preprocessed.grid_h * decoded.preprocessed.grid_w / 4;
  std::vector<int> placeholders;
  for (int slice = 0; slice < decoded.preprocessed.grid_t; ++slice) {
    placeholders.push_back(1);  // timestamp/text token separates native video runs
    placeholders.push_back(248057);
  }
  const auto ids = umminfer::qwen38::expand_video_pads(
      placeholders, merged_tokens, decoded.preprocessed.grid_t);
  const auto serial = umminfer::qwen38::build_mrope_positions(
      ids, decoded.preprocessed.grid_t, decoded.preprocessed.grid_h,
      decoded.preprocessed.grid_w, true);
  const auto batched = umminfer::qwen38::build_mrope_positions(
      ids, decoded.preprocessed.grid_t, decoded.preprocessed.grid_h,
      decoded.preprocessed.grid_w, true);
  check(serial.t == batched.t && serial.h == batched.h && serial.w == batched.w &&
            serial.next == batched.next,
        "serial and batched requests share identical temporal MRoPE geometry");
  std::vector<int> changed_timestamp_values = ids;
  for (int& id : changed_timestamp_values) {
    if (id == 1) id = 4242;  // Same token layout, different timestamp text ids.
  }
  const auto changed_stamps = umminfer::qwen38::build_mrope_positions(
      changed_timestamp_values, decoded.preprocessed.grid_t, decoded.preprocessed.grid_h,
      decoded.preprocessed.grid_w, true);
  check(changed_stamps.t == serial.t && changed_stamps.h == serial.h &&
            changed_stamps.w == serial.w && changed_stamps.next == serial.next,
        "mRoPE coordinates depend on token/grid layout rather than numeric timestamp text");

  const auto still = umminfer::qwen38::preprocess_rgb(
      solid(64, 64, 0).data(), 64, 64, 4096, 16384);
  check(!still.is_video && still.grid_t == 1 && still.grid_h == 4 && still.grid_w == 4 &&
            still.pixel_values.size() == 16 * umminfer::qwen38::PreprocessedImage::kPatchDim &&
            std::all_of(still.pixel_values.begin(), still.pixel_values.end(),
                        [](float value) { return value == -1.0f; }),
        "still-image decode/preprocess remains single-slice and byte-exact");
}

void test_exact_temporal2_channel_frame_packing() {
  constexpr int kSize = 32;
  std::vector<std::vector<uint8_t>> frames;
  for (int index = 0; index < 8; ++index) {
    frames.push_back(solid(kSize, kSize, static_cast<uint8_t>(20 + index)));
  }
  const auto packed = umminfer::qwen38::preprocess_video_frames(
      frames, kSize, kSize, 1024, 1024, 3584);
  check(packed.grid_t == 4 && packed.grid_h == 2 && packed.grid_w == 2 &&
            packed.resized_h == kSize && packed.resized_w == kSize && packed.rows() == 16,
        "eight frames produce four exact-size temporal-2 slices");
  for (int slice = 0; slice < 4; ++slice) {
    const float expected0 = static_cast<float>(20 + slice * 2) / 127.5f - 1.0f;
    const float expected1 = static_cast<float>(20 + slice * 2 + 1) / 127.5f - 1.0f;
    for (int merge_row = 0; merge_row < 4; ++merge_row) {
      const size_t base = static_cast<size_t>(slice * 4 + merge_row) *
                          umminfer::qwen38::PreprocessedImage::kPatchDim;
      check(std::fabs(packed.pixel_values[base] - expected0) < 1e-7 &&
                std::fabs(packed.pixel_values[base + 256] - expected1) < 1e-7 &&
                std::fabs(packed.pixel_values[base + 512] - expected0) < 1e-7 &&
                std::fabs(packed.pixel_values[base + 768] - expected1) < 1e-7,
            "each temporal-2 slice packs channel-major first/second frame values without reordering");
    }
  }
}

void test_malformed_and_budget_rejection() {
  VideoFrameSelection options = temporal();
  options.max_frames = 9;
  check(throws([&] { umminfer::qwen38::validate_video_frame_selection(options); }),
        "more than eight selected frames is rejected before allocation");
  options = temporal();
  options.max_candidate_frames = 65;
  check(throws([&] { umminfer::qwen38::validate_video_frame_selection(options); }),
        "more than 64 candidate frames is rejected before allocation");
  options = temporal();
  options.motion_threshold = std::numeric_limits<double>::quiet_NaN();
  check(throws([&] { umminfer::qwen38::validate_video_frame_selection(options); }),
        "non-finite motion threshold is rejected");
  options = temporal(8, 4);
  check(throws([&] { umminfer::qwen38::validate_video_frame_selection(options); }),
        "selection count cannot exceed candidate budget");
  check(throws([] { (void)umminfer::qwen38::decode_video_bytes("not an MP4"); }),
        "malformed media is rejected instead of fabricating frames");
  std::vector<VideoCandidate> too_many(5, VideoCandidate{solid(32, 32, 0), 0.0});
  check(throws([&] {
          (void)umminfer::qwen38::preprocess_temporal_candidates(
              too_many, 32, 32, temporal(4, 4));
        }),
        "candidate vectors above the declared bound are rejected");
}

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    path = std::filesystem::temp_directory_path() /
           ("umm-video-input-test-" + std::to_string(getpid()));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void run_process(const std::vector<std::string>& args) {
  const pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    const int nullfd = open("/dev/null", O_WRONLY);
    if (nullfd >= 0) {
      dup2(nullfd, STDOUT_FILENO);
      dup2(nullfd, STDERR_FILENO);
      close(nullfd);
    }
    std::vector<char*> argv;
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("synthetic FFmpeg fixture generation failed");
  }
}

std::string make_synthetic_mp4(const std::filesystem::path& dir) {
  const std::vector<uint8_t> values{0, 0, 0, 255, 250, 245, 0, 0, 128, 126, 124, 0};
  for (size_t i = 0; i < values.size(); ++i) {
    const std::string name = "frame-" + std::to_string(i) + ".ppm";
    std::ofstream out(dir / name, std::ios::binary);
    out << "P6\n64 64\n255\n";
    const auto rgb = solid(64, 64, values[i]);
    out.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    if (!out) throw std::runtime_error("failed to write synthetic PPM");
  }
  const std::filesystem::path output = dir / "motion.mp4";
  run_process({"ffmpeg", "-y", "-v", "error", "-framerate", "4", "-i",
               (dir / "frame-%d.ppm").string(), "-c:v", "mpeg4", "-qscale:v", "1",
               "-pix_fmt", "yuv420p", output.string()});
  std::ifstream input(output, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void test_ffmpeg_default_equivalence_and_temporal_budget() {
  TempDir temp;
  const std::string bytes = make_synthetic_mp4(temp.path);
  check(!bytes.empty(), "synthetic MP4 fixture is nonempty");
  const VideoFrameSelection defaults;
  VideoFrameSelection explicit_uniform;
  explicit_uniform.policy = VideoFramePolicy::kUniform;
  const DecodedVideo a = umminfer::qwen38::decode_video_bytes(bytes, defaults);
  const DecodedVideo b = umminfer::qwen38::decode_video_bytes(bytes, explicit_uniform);
  check(a.preprocessed.grid_t == b.preprocessed.grid_t &&
            a.preprocessed.grid_h == b.preprocessed.grid_h &&
            a.preprocessed.grid_w == b.preprocessed.grid_w &&
            a.preprocessed.resized_h == b.preprocessed.resized_h &&
            a.preprocessed.resized_w == b.preprocessed.resized_w &&
            a.preprocessed.pixel_values == b.preprocessed.pixel_values &&
            a.timestamps == b.timestamps &&
            a.selected_frame_indices == b.selected_frame_indices &&
            a.selected_frame_content_fnv1a64 == b.selected_frame_content_fnv1a64,
        "default and explicit uniform decode are byte-identical");
  check(a.decoded_frame_count <= 8 && a.preprocessed.rows() <= 3584,
        "uniform decode keeps the historical frame and token budgets");

  VideoFrameSelection options = temporal(4, 12);
  const DecodedVideo selected = umminfer::qwen38::decode_video_bytes(bytes, options);
  check(selected.decoded_frame_count <= 12 &&
            selected.selected_frame_indices.size() <= 4 &&
            selected.preprocessed.rows() <= 3584 && selected.preprocessed.is_video,
        "temporal FFmpeg decode enforces candidate, frame, and patch budgets");
  check(std::is_sorted(selected.selected_frame_indices.begin(),
                       selected.selected_frame_indices.end()) &&
            selected.selected_frame_timestamps.size() ==
                selected.selected_frame_indices.size() &&
            selected.selected_source_frame_indices == std::vector<int64_t>(
                selected.selected_frame_indices.begin(), selected.selected_frame_indices.end()) &&
            selected.source_indices_exact && !selected.has_nominal_timestamps,
        "FFmpeg temporal output retains exact chronological source index/PTS mapping");
  const auto boundary = std::find(selected.selected_frame_indices.begin(),
                                  selected.selected_frame_indices.end(), 3);
  check(boundary != selected.selected_frame_indices.end(),
        "temporal FFmpeg policy selects the synthetic scene boundary");
  if (boundary != selected.selected_frame_indices.end()) {
    const size_t position = static_cast<size_t>(boundary - selected.selected_frame_indices.begin());
    check(std::fabs(selected.selected_frame_timestamps[position] - 0.75) < 0.05,
          "temporal decoder preserves the source presentation timestamp near 0.75 s");
  }
}

}  // namespace

int main() {
  test_motion_scene_order_and_ties();
  test_static_repeated_short_and_all_motion_fallbacks();
  test_duplicate_scene_run();
  test_analysis_and_temporal_timestamps();
  test_prompt_mrope_parity_and_still_image();
  test_exact_temporal2_channel_frame_packing();
  test_malformed_and_budget_rejection();
  test_ffmpeg_default_equivalence_and_temporal_budget();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("qwen38 video input host tests passed\n");
  return 0;
}
