// SPDX-License-Identifier: Apache-2.0
// Host checks for the stage-specific CLI hooks: the shared integer parser and
// the video stage's flag set. Links video_stage.cpp, so the checks run against
// the same parse_stage_flag the executable uses.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "stage_kind.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

template <typename Fn>
bool throws_runtime(Fn fn) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

void test_parse_stage_int() {
  using umminfer::stage::parse_stage_int;
  check(parse_stage_int("0", "--fps") == 0, "zero accepted");
  check(parse_stage_int("16", "--fps") == 16, "16 accepted");
  check(throws_runtime([] { (void)parse_stage_int("-1", "--fps"); }),
        "negative rejected");
  check(throws_runtime([] { (void)parse_stage_int("16x", "--fps"); }),
        "trailing text rejected");
  check(throws_runtime([] { (void)parse_stage_int("", "--fps"); }),
        "empty rejected");
  check(throws_runtime([] { (void)parse_stage_int("abc", "--fps"); }),
        "non-numeric rejected");
}

void test_video_flags_accepted() {
  using umminfer::stage::parse_stage_flag;
  umminfer::stage::StageOptions options;
  check(parse_stage_flag("video-backend", "contract-stub", options),
        "--video-backend accepted");
  check(options.video_backend == "contract-stub", "--video-backend stored");
  check(parse_stage_flag("size", "704x480", options), "--size accepted");
  check(options.image_size == "704x480", "--size stored");
  check(parse_stage_flag("fps", "16", options), "--fps accepted");
  check(options.video_fps == 16, "--fps stored");
  check(parse_stage_flag("max-seconds", "2", options), "--max-seconds accepted");
  check(options.video_max_seconds == 2, "--max-seconds stored");
  check(parse_stage_flag("max-frames", "29", options), "--max-frames accepted");
  check(options.max_frames == 29, "--max-frames stored");
  check(parse_stage_flag("steps", "4", options), "--steps accepted");
  check(options.steps == 4, "--steps stored");
  check(parse_stage_flag("step-delay-ms", "250", options),
        "--step-delay-ms accepted");
  check(options.video_step_delay_ms == 250, "--step-delay-ms stored");
  check(parse_stage_flag("ffmpeg-bin", "/usr/bin/ffmpeg", options),
        "--ffmpeg-bin accepted");
  check(options.ffmpeg_bin == "/usr/bin/ffmpeg", "--ffmpeg-bin stored");
}

void test_video_flags_invalid_value() {
  using umminfer::stage::parse_stage_flag;
  umminfer::stage::StageOptions options;
  check(throws_runtime([&] { (void)parse_stage_flag("fps", "-4", options); }),
        "--fps rejects a negative value");
  check(throws_runtime([&] { (void)parse_stage_flag("steps", "many", options); }),
        "--steps rejects a non-numeric value");
}

void test_video_rejects_other_stage_flags() {
  using umminfer::stage::parse_stage_flag;
  umminfer::stage::StageOptions options;
  check(!parse_stage_flag("image-backend", "2512", options),
        "--image-backend is image-only");
  check(!parse_stage_flag("batch-size", "4", options),
        "--batch-size is qwen38-only");
  check(!parse_stage_flag("max-new-tokens", "64", options),
        "--max-new-tokens is text-only");
  check(!parse_stage_flag("max-seq", "2048", options),
        "--max-seq is text-only");
  check(!parse_stage_flag("gpu", "0", options), "transport flags are not stage flags");
}

void test_usage_flags() {
  using umminfer::stage::stage_usage_flags;
  const std::string usage = stage_usage_flags();
  check(usage.find("--video-backend") != std::string::npos,
        "usage lists --video-backend");
  check(usage.find("--ffmpeg-bin") != std::string::npos,
        "usage lists --ffmpeg-bin");
  check(usage.find("--image-backend") == std::string::npos,
        "usage omits --image-backend");
  check(usage.empty() || usage.back() == ' ', "usage fragment ends with a space");
}

}  // namespace

int main() {
  test_parse_stage_int();
  test_video_flags_accepted();
  test_video_flags_invalid_value();
  test_video_rejects_other_stage_flags();
  test_usage_flags();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all stage flag checks passed\n");
  return 0;
}
