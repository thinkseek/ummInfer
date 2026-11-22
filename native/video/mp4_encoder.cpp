// SPDX-License-Identifier: Apache-2.0
#include "mp4_encoder.hpp"

#include "config.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace umminfer::video {
namespace {

struct TempFileGuard {
  std::string path;
  ~TempFileGuard() {
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
};

struct FdGuard {
  int fd = -1;
  ~FdGuard() { close_now(); }
  void close_now() {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
};

struct ChildGuard {
  pid_t pid = -1;
  bool reaped = false;
  ~ChildGuard() { kill_and_reap(); }
  void kill_and_reap() {
    if (pid <= 0 || reaped) return;
    ::kill(pid, SIGTERM);
    for (int i = 0; i < 30; ++i) {
      int status = 0;
      const pid_t r = ::waitpid(pid, &status, WNOHANG);
      if (r == pid || r < 0) {
        reaped = true;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ::kill(pid, SIGKILL);
    int status = 0;
    ::waitpid(pid, &status, 0);
    reaped = true;
  }
};

bool write_all(int fd, const uint8_t* data, size_t size) {
  size_t written = 0;
  while (written < size) {
    const ssize_t n = ::write(fd, data + written, size - written);
    if (n > 0) {
      written += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

std::string read_fd_all(int fd) {
  std::string out;
  char buf[1024];
  while (true) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    break;
  }
  return out;
}

}  // namespace

bool is_valid_mp4_header(std::string_view bytes) {
  return bytes.size() >= 12 && bytes.substr(4, 4) == "ftyp";
}

std::string encode_rgb24_frames_mp4(
    const std::vector<std::vector<uint8_t>>& frames, int width, int height,
    int fps, const AbortCheck& abort, const std::string& ffmpeg_bin) {
  if (frames.empty()) {
    throw std::invalid_argument("video encoder received zero frames");
  }
  if (width <= 0 || height <= 0 || fps <= 0) {
    throw std::invalid_argument("video encoder dimensions and fps must be positive");
  }
  const size_t expected_bytes =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
  for (const auto& frame : frames) {
    if (frame.size() != expected_bytes) {
      throw std::invalid_argument("RGB24 frame byte size does not match WxHx3");
    }
  }
  if (abort && abort()) {
    throw RequestAborted("aborted before MP4 encoding");
  }

  // Create unique temporary MP4 path in system temp dir.
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  std::string tmpl = (tmp_dir / "umminfer-video-XXXXXX").string();
  std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
  tmpl_buf.push_back('\0');
  const int tmp_fd = ::mkstemp(tmpl_buf.data());
  if (tmp_fd < 0) {
    throw std::runtime_error("mkstemp failed for MP4 encoding: " +
                             std::string(std::strerror(errno)));
  }
  ::close(tmp_fd);
  const std::string base_path(tmpl_buf.data());
  const std::string out_path = base_path + ".mp4";
  std::error_code ec;
  std::filesystem::remove(base_path, ec);
  TempFileGuard out_guard{out_path};

  int stdin_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  if (::pipe(stdin_pipe) != 0) {
    throw std::runtime_error("pipe(stdin) failed for ffmpeg");
  }
  if (::pipe(stderr_pipe) != 0) {
    ::close(stdin_pipe[0]);
    ::close(stdin_pipe[1]);
    throw std::runtime_error("pipe(stderr) failed for ffmpeg");
  }

  // Ignore SIGPIPE in case ffmpeg exits early while we write.
  std::signal(SIGPIPE, SIG_IGN);

  const std::string size_arg = std::to_string(width) + "x" + std::to_string(height);
  const std::string fps_arg = std::to_string(fps);
  std::vector<std::string> args = {
      ffmpeg_bin,  "-y",       "-loglevel", "error",      "-f",      "rawvideo",
      "-pix_fmt",  "rgb24",    "-s",        size_arg,     "-r",      fps_arg,
      "-i",        "-",        "-c:v",      "libx264",    "-preset", "veryfast",
      "-crf",      "20",       "-pix_fmt",  "yuv420p",    "-movflags", "+faststart",
      "-f",        "mp4",      out_path,
  };

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(stdin_pipe[0]);
    ::close(stdin_pipe[1]);
    ::close(stderr_pipe[0]);
    ::close(stderr_pipe[1]);
    throw std::runtime_error("fork failed for ffmpeg");
  }

  if (pid == 0) {
    // Child process: wire stdin and stderr, redirect stdout to /dev/null.
    ::dup2(stdin_pipe[0], STDIN_FILENO);
    ::dup2(stderr_pipe[1], STDERR_FILENO);
    const int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDOUT_FILENO);
      ::close(devnull);
    }
    ::close(stdin_pipe[0]);
    ::close(stdin_pipe[1]);
    ::close(stderr_pipe[0]);
    ::close(stderr_pipe[1]);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& s : args) argv.push_back(s.data());
    argv.push_back(nullptr);
    ::execvp(ffmpeg_bin.c_str(), argv.data());
    const char* msg = "execvp(ffmpeg) failed\n";
    (void)::write(STDERR_FILENO, msg, std::strlen(msg));
    ::_exit(127);
  }

  ::close(stdin_pipe[0]);
  ::close(stderr_pipe[1]);
  FdGuard in_fd{stdin_pipe[1]};
  FdGuard err_fd{stderr_pipe[0]};
  ChildGuard child{pid, false};

  for (const auto& frame : frames) {
    if (abort && abort()) {
      in_fd.close_now();
      child.kill_and_reap();
      throw RequestAborted("aborted during MP4 encoding");
    }
    if (!write_all(in_fd.fd, frame.data(), frame.size())) {
      break;
    }
  }
  in_fd.close_now();

  int status = 0;
  while (true) {
    if (abort && abort()) {
      child.kill_and_reap();
      throw RequestAborted("aborted while finalizing MP4");
    }
    const pid_t r = ::waitpid(child.pid, &status, WNOHANG);
    if (r == child.pid) {
      child.reaped = true;
      break;
    }
    if (r < 0) {
      if (errno == EINTR) continue;
      child.reaped = true;
      throw std::runtime_error("waitpid(ffmpeg) failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  const std::string err_text = read_fd_all(err_fd.fd);
  err_fd.close_now();

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    throw std::runtime_error("ffmpeg failed (rc=" + std::to_string(code) +
                             "): " + err_text);
  }

  std::ifstream in(out_path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    throw std::runtime_error("failed to read encoded MP4 output");
  }
  const std::streamsize size = in.tellg();
  if (size <= 0) {
    throw std::runtime_error("ffmpeg produced empty MP4 output");
  }
  if (static_cast<size_t>(size) > kMaxVideoArtifactBytes) {
    throw std::runtime_error("encoded MP4 exceeds 48 MiB UnifiedEngine limit");
  }
  in.seekg(0, std::ios::beg);
  std::string mp4_bytes(static_cast<size_t>(size), '\0');
  if (!in.read(mp4_bytes.data(), size)) {
    throw std::runtime_error("failed reading encoded MP4 bytes");
  }
  if (!is_valid_mp4_header(mp4_bytes)) {
    throw std::runtime_error("ffmpeg produced invalid MP4 output (missing ftyp box)");
  }
  return mp4_bytes;
}

}  // namespace umminfer::video
