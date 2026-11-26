// SPDX-License-Identifier: Apache-2.0
#include "native_identity.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "umminfer/util/codec.hpp"

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() /
      ("umminfer-native-identity-test-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "model");
  const std::string config = "{\"model_type\":\"qwen3_5\"}\n";
  const std::string weight = std::string("fixture\0weights\n", 16);
  std::ofstream(root / "model/config.json", std::ios::binary) << config;
  std::ofstream(root / "model/model.safetensors", std::ios::binary).write(
      weight.data(), weight.size());
  const fs::path output = root / "identity.json";

  umminfer::stage::StageOptions options;
  options.model_dir = (root / "model").string();
  options.identity_out = output.string();
  options.model_id = "qwen3-native";
  options.model_revision = "UNKNOWN";
  options.gpu = 7;
  options.batch_size = 1;
  const auto written = umminfer::stage::write_native_qwen_identity(options, "vlm");
  std::ifstream input(output, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  const auto identity = nlohmann::json::parse(bytes);
  check(identity["schema"] == "umminfer-native-stage-identity-v2",
        "identity schema mismatch");
  check(identity["state"] == "loaded" && identity["stage_name"] == "vlm",
        "identity did not retain its loaded stage");
  check(identity["process"]["pid"] == ::getpid(), "identity pid mismatch");
  check(identity["process"]["start_time_ticks"] ==
            umminfer::stage::native_process_start_time_ticks(::getpid()),
        "identity process start mismatch");
  check(identity["config"]["sha256"] == umminfer::util::sha256_hex(config),
        "identity config hash mismatch");
  check(identity["weight_files"]["model.safetensors"]["sha256"] ==
            umminfer::util::sha256_hex(weight),
        "identity weight hash mismatch");
  check(identity["model_sha256"] ==
            umminfer::util::sha256_hex(identity["weight_files"].dump()),
        "identity weight-set hash mismatch");
  check(written.document_sha256 == umminfer::util::sha256_hex(bytes),
        "stage_ready digest does not bind exact identity bytes");
  check(identity["stage_config"]["gpu"] == 7 &&
            identity["stage_config"]["batch_size"] == 1,
        "identity stage config mismatch");

  bool duplicate_rejected = false;
  try {
    (void)umminfer::stage::write_native_qwen_identity(options, "vlm");
  } catch (const std::exception&) {
    duplicate_rejected = true;
  }
  check(duplicate_rejected, "identity overwrote an existing evidence file");
  fs::remove_all(root);
  std::puts("native_identity_test: PASS");
  return 0;
}
