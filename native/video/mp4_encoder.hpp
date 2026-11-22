// SPDX-License-Identifier: Apache-2.0
// Native C++ H.264 MP4 encoder over POSIX fork/execvp(ffmpeg) with per-frame
// cancellation, ISO BMFF ftyp validation, and RAII child/temp-file cleanup.
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace umminfer::video {

class RequestAborted : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

using AbortCheck = std::function<bool()>;

// Quick structural check for an ISO BMFF ftyp box.
bool is_valid_mp4_header(std::string_view bytes);

// Encodes RGB24 frames (each width * height * 3 bytes) into a faststart H.264
// MP4 byte string using ffmpeg. Checks `abort()` before each frame and while
// waiting for finalization; on abort terminates and reaps ffmpeg and throws
// RequestAborted. Never invokes a shell.
std::string encode_rgb24_frames_mp4(
    const std::vector<std::vector<uint8_t>>& frames, int width, int height,
    int fps, const AbortCheck& abort = {},
    const std::string& ffmpeg_bin = "ffmpeg");

}  // namespace umminfer::video
