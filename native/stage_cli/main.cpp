// SPDX-License-Identifier: Apache-2.0
// umm-native-stage-<kind>: one native model behind the existing C++ Stage and
// Coordinator protocol. One process, one model, one GPU.
//
//   umm-native-stage-asr  --kind asr  --model DIR --gpu 2 \
//     --recv-endpoint ipc:///tmp/x/asr --coordinator-endpoint ipc:///tmp/x/completion \
//     --abort-endpoint ipc:///tmp/x/abort [--next name=endpoint] [--peer name=endpoint]
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

#include "stage_kind.hpp"
#include "umminfer/pipeline/pipeline.hpp"

namespace {

using Flags = std::map<std::string, std::string>;
using Repeats = std::multimap<std::string, std::string>;

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

void usage() {
  std::cerr << "usage: umm-native-stage-<asr|omni|qwen38|tts|image> --kind KIND --model DIR "
               "--recv-endpoint EP --coordinator-endpoint EP --abort-endpoint EP "
               "[--name NAME] [--gpu N] [--max-new-tokens N] [--max-frames N] "
               "[--max-seq N] [--steps N] [--batch-size N] "
               "[--next name=EP] [--peer name=EP]\n";
}

// ZMQ binds an ipc path but never creates its parent directory. Restrict newly
// created directories to the owner (0700) so unprivileged local users cannot
// connect to internal ZMQ sockets.
void ensure_ipc_parent(const std::string& endpoint) {
  const std::string prefix = "ipc://";
  if (endpoint.rfind(prefix, 0) != 0) return;
  const size_t slash = endpoint.rfind('/');
  if (slash == std::string::npos || slash <= prefix.size()) return;
  const std::filesystem::path dir(
      endpoint.substr(prefix.size(), slash - prefix.size()));
  std::error_code ec;
  const bool existed = std::filesystem::exists(dir, ec);
  std::filesystem::create_directories(dir, ec);
  if (!existed && !ec) {
    std::filesystem::permissions(
        dir, std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace, ec);
  }
}

int parse_int(const std::string& text, const std::string& option) {
  size_t used = 0;
  int value = 0;
  try {
    value = std::stoi(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  if (used != text.size() || value < 0) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  Flags flags;
  Repeats repeats;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    }
    if (arg.rfind("--", 0) != 0 || index + 1 >= argc) {
      std::cerr << "unexpected argument: " << arg << "\n";
      usage();
      return 2;
    }
    const std::string key = arg.substr(2);
    const std::string value = argv[++index];
    if (key == "next" || key == "peer") {
      repeats.emplace(key, value);
    } else {
      flags[key] = value;
    }
  }

  const std::string kind = flags.count("kind") ? flags["kind"] : "";
  if (kind != umminfer::stage::compiled_kind()) {
    std::cerr << "this executable implements --kind " << umminfer::stage::compiled_kind()
              << ", got '" << kind << "'\n";
    return 2;
  }
  const std::string model = flags.count("model") ? flags["model"] : "";
  const std::string recv = flags.count("recv-endpoint") ? flags["recv-endpoint"] : "";
  const std::string coordinator =
      flags.count("coordinator-endpoint") ? flags["coordinator-endpoint"] : "";
  const std::string abort = flags.count("abort-endpoint") ? flags["abort-endpoint"] : "";
  if (model.empty() || recv.empty() || coordinator.empty() || abort.empty()) {
    usage();
    return 2;
  }

  try {
    umminfer::stage::StageOptions options;
    options.model_dir = model;
    options.gpu = flags.count("gpu") ? parse_int(flags["gpu"], "--gpu") : 0;
    options.max_new_tokens = flags.count("max-new-tokens")
                                 ? parse_int(flags["max-new-tokens"], "--max-new-tokens")
                                 : 0;
    options.max_frames =
        flags.count("max-frames") ? parse_int(flags["max-frames"], "--max-frames") : 0;
    options.max_seq =
        flags.count("max-seq") ? parse_int(flags["max-seq"], "--max-seq") : 0;
    options.steps = flags.count("steps") ? parse_int(flags["steps"], "--steps") : 0;
    options.batch_size =
        flags.count("batch-size") ? parse_int(flags["batch-size"], "--batch-size") : 0;

    umminfer::Stage::Config config;
    config.name = flags.count("name") ? flags["name"] : kind;
    config.recv_endpoint = recv;
    config.coordinator_endpoint = coordinator;
    config.abort_endpoint = abort;
    config.gpu_id = options.gpu;
    config.announce_ready = true;
    if (options.batch_size > 1) {
      // The engine runs one batch thread; concurrent compute calls only queue
      // work into it. This is the thread-safe contract, not serial compute.
      config.max_concurrency = options.batch_size;
      config.engine_contract = umminfer::EngineContract::kThreadSafe;
    }
    config.peer_endpoints[config.name] = recv;
    for (const auto& [key, value] : repeats) {
      const size_t equals = value.find('=');
      if (equals == std::string::npos || equals == 0 || equals + 1 == value.size()) {
        throw std::runtime_error("--" + key + " requires name=endpoint");
      }
      const std::string target = value.substr(0, equals);
      const std::string endpoint = value.substr(equals + 1);
      ensure_ipc_parent(endpoint);
      if (key == "next") config.next_stages[target] = endpoint;
      config.peer_endpoints[target] = endpoint;
    }
    ensure_ipc_parent(recv);
    ensure_ipc_parent(coordinator);
    ensure_ipc_parent(abort);

    const auto load_started = std::chrono::steady_clock::now();
    umminfer::CancellableComputeFn compute = umminfer::stage::make_compute(options);
    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - load_started);
    std::cout << "native stage model loaded kind=" << kind << " model=" << model
              << " gpu=" << options.gpu << " load_ms=" << load_ms.count() << std::endl;

    umminfer::Stage stage(config, std::move(compute));
    stage.start();
    std::cout << "native stage " << config.name << " ready kind=" << kind << std::endl;

    std::atomic<bool> done{false};
    std::atomic<bool> failed{false};
    std::thread runner([&stage, &done, &failed] {
      try {
        stage.run();
      } catch (const std::exception& error) {
        // An uncaught exception on a thread calls std::terminate.
        std::cerr << "stage run failed: " << error.what() << "\n";
        failed = true;
      }
      done = true;
    });
    while (g_running && !done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    stage.stop();
    runner.join();
    if (failed) return 1;
  } catch (const std::exception& error) {
    std::cerr << "native stage failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
