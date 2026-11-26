// SPDX-License-Identifier: Apache-2.0
#include "native_identity.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libproc.h>
#include <sys/sysctl.h>
#endif

#include <nlohmann/json.hpp>

#include "umminfer/util/codec.hpp"

namespace umminfer::stage {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

[[noreturn]] void fail(const std::string& message) {
  throw ProtocolError("native stage identity: " + message);
}

class Fd {
 public:
  explicit Fd(int value) : value_(value) {}
  ~Fd() {
    if (value_ >= 0) ::close(value_);
  }
  int get() const { return value_; }

 private:
  int value_;
};

Fd open_regular(const fs::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
  if (fd < 0) fail("required file cannot be opened without following symlinks");
  struct stat info {};
  if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    fail("required path is not a regular file");
  }
  return Fd(fd);
}

Json hash_file(const fs::path& path) {
  Fd file = open_regular(path);
  util::Sha256 digest;
  std::array<char, 1u << 16> buffer{};
  int64_t bytes = 0;
  while (true) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    if (count < 0) fail("file read failed while hashing");
    if (count == 0) break;
    if (bytes > std::numeric_limits<int64_t>::max() - count) {
      fail("file size overflows int64");
    }
    bytes += count;
    digest.update(std::string_view(buffer.data(), static_cast<size_t>(count)));
  }
  return Json{{"sha256", digest.finish_hex()}, {"bytes", bytes}};
}

std::string read_small_file(const fs::path& path) {
  Fd file = open_regular(path);
  std::string out;
  std::array<char, 4096> buffer{};
  while (true) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    if (count < 0) fail("process identity read failed");
    if (count == 0) break;
    out.append(buffer.data(), static_cast<size_t>(count));
    if (out.size() > (1u << 20)) fail("process identity document is too large");
  }
  return out;
}

std::string executable_path(int pid) {
#if defined(__APPLE__)
  std::array<char, PROC_PIDPATHINFO_MAXSIZE> buffer{};
  const int size = ::proc_pidpath(pid, buffer.data(), buffer.size());
  if (size <= 0) fail("native process executable is unavailable");
  return std::string(buffer.data(), static_cast<size_t>(size));
#else
  std::array<char, 4096> buffer{};
  const std::string proc = "/proc/" + std::to_string(pid) + "/exe";
  const ssize_t size = ::readlink(proc.c_str(), buffer.data(), buffer.size() - 1);
  if (size <= 0) fail("native process executable is unavailable");
  return std::string(buffer.data(), static_cast<size_t>(size));
#endif
}

void write_new_file(const fs::path& path, const std::string& bytes) {
  if (!path.is_absolute()) fail("identity output path must be absolute");
  if (!fs::is_directory(path.parent_path())) fail("identity output directory is absent");
  const fs::path temporary = path.parent_path() /
      (path.filename().string() + ".tmp." + std::to_string(::getpid()));
  const int raw = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,
                         S_IRUSR | S_IWUSR);
  if (raw < 0) fail("identity output or its temporary file already exists");
  Fd file(raw);
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = ::write(file.get(), bytes.data() + offset, bytes.size() - offset);
    if (count <= 0) fail("identity document write failed");
    offset += static_cast<size_t>(count);
  }
  if (::fsync(file.get()) != 0) fail("identity document fsync failed");
  if (::link(temporary.c_str(), path.c_str()) != 0) {
    const std::string reason = std::strerror(errno);
    ::unlink(temporary.c_str());
    fail("identity output must be a new file: " + reason);
  }
  ::unlink(temporary.c_str());
}

}  // namespace

int64_t native_process_start_time_ticks(int pid) {
  if (pid <= 0) fail("process id must be positive");
#if defined(__APPLE__)
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  struct kinfo_proc info {};
  size_t size = sizeof(info);
  if (::sysctl(mib, 4, &info, &size, nullptr, 0) != 0 || size < sizeof(info)) {
    fail("process starttime is unavailable");
  }
  const int64_t sec = static_cast<int64_t>(info.kp_proc.p_starttime.tv_sec);
  const int64_t usec = static_cast<int64_t>(info.kp_proc.p_starttime.tv_usec);
  if (sec < 0 || usec < 0) fail("process starttime is invalid");
  return sec * 1000000LL + usec;
#else
  const std::string stat = read_small_file(
      "/proc/" + std::to_string(pid) + "/stat");
  const size_t close = stat.rfind(')');
  if (close == std::string::npos) fail("process stat is malformed");
  std::vector<std::string> fields;
  size_t offset = close + 1;
  while (offset < stat.size()) {
    while (offset < stat.size() && stat[offset] == ' ') ++offset;
    if (offset >= stat.size()) break;
    const size_t end = stat.find(' ', offset);
    fields.push_back(stat.substr(offset, end - offset));
    offset = end == std::string::npos ? stat.size() : end + 1;
  }
  // fields[0] is field 3 (state); starttime is field 22.
  if (fields.size() <= 19) fail("process stat lacks starttime");
  const std::string& value = fields[19];
  if (value.empty() ||
      !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
      })) {
    fail("process starttime is not numeric");
  }
  try {
    size_t used = 0;
    const unsigned long long parsed = std::stoull(value, &used);
    if (used != value.size() || parsed > static_cast<unsigned long long>(
            std::numeric_limits<int64_t>::max())) {
      fail("process starttime is outside int64");
    }
    return static_cast<int64_t>(parsed);
  } catch (const std::exception&) {
    fail("process starttime is invalid");
  }
#endif
}

WrittenNativeIdentity write_native_qwen_identity(const StageOptions& options,
                                                 const std::string& stage_name) {
  if (options.identity_out.empty()) fail("identity output path is empty");
  if (options.model_id.empty() || options.model_revision.empty()) {
    fail("real identity requires --model-id and --model-revision");
  }
  const fs::path model_root = fs::absolute(options.model_dir).lexically_normal();
  if (!fs::is_directory(model_root)) fail("model root is not a directory");
  const Json config = hash_file(model_root / "config.json");
  Json weights = Json::object();
  for (const auto& entry : fs::directory_iterator(model_root)) {
    const fs::path path = entry.path();
    if (path.extension() != ".safetensors") continue;
    // Enumeration is evidence coverage, not permission to follow a symlink.
    weights[path.filename().string()] = hash_file(path);
  }
  if (weights.empty()) fail("no available .safetensors weight files were found");

  const int pid = static_cast<int>(::getpid());
  const int64_t process_start = native_process_start_time_ticks(pid);
  const std::string executable = executable_path(pid);
  const Json executable_identity = hash_file(executable);
  Json identity = {
      {"schema", "umminfer-native-stage-identity-v2"},
      {"stage_name", stage_name},
      {"state", "loaded"},
      {"loaded_at_epoch_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count()},
      {"model_id", options.model_id},
      {"model_revision", options.model_revision},
      {"model_root", model_root.string()},
      {"config", {{"path", "config.json"}, {"sha256", config["sha256"]},
                  {"bytes", config["bytes"]}}},
      {"weight_files", weights},
      {"model_sha256", util::sha256_hex(weights.dump())},
      {"process", {{"pid", pid}, {"start_time_ticks", process_start},
                   {"executable_path", executable},
                   {"executable_sha256", executable_identity["sha256"]},
                   {"executable_bytes", executable_identity["bytes"]}}},
      {"stage_config", {{"kind", "qwen38"}, {"gpu", options.gpu},
                        {"batch_size", options.batch_size > 1 ? options.batch_size : 1},
                        {"max_new_tokens", options.max_new_tokens > 0
                                               ? options.max_new_tokens : 256},
                        {"max_seq", options.max_seq > 0 ? options.max_seq : 8192}}},
  };
  const std::string bytes = identity.dump() + "\n";
  const fs::path output(options.identity_out);
  write_new_file(output, bytes);
  return {util::sha256_hex(bytes), process_start};
}

}  // namespace umminfer::stage
