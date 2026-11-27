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
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "stage_kind.hpp"
#include "native_identity.hpp"
#include "request_telemetry_stage.hpp"
#include "umminfer/pipeline/pipeline.hpp"

namespace {

using Flags = std::map<std::string, std::string>;
using Repeats = std::multimap<std::string, std::string>;

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

void usage() {
  std::cerr << "usage: umm-native-stage-<asr|omni|qwen38|tts|image|video> --kind KIND --model DIR "
               "--recv-endpoint EP --coordinator-endpoint EP --abort-endpoint EP "
               "[--name NAME] [--gpu N] "
            << umminfer::stage::stage_usage_flags()
            << "[--next name=EP] [--peer name=EP]\n";
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

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // Transport flags: the Stage protocol wiring. Every other flag belongs to
  // the stage and is parsed by its *_stage.cpp.
  static const std::set<std::string> kTransportFlags = {
      "kind", "model", "name", "gpu", "recv-endpoint", "coordinator-endpoint",
      "abort-endpoint", "identity-out", "model-id", "model-revision"};
  Flags flags;
  Repeats repeats;
  umminfer::stage::StageOptions options;
  try {
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
      } else if (kTransportFlags.count(key)) {
        flags[key] = value;
      } else if (!umminfer::stage::parse_stage_flag(key, value, options)) {
        std::cerr << "unknown option: --" << key << "\n";
        usage();
        return 2;
      }
    }

    const std::string kind = flags.count("kind") ? flags["kind"] : "";
    if (kind != umminfer::stage::compiled_kind()) {
      std::cerr << "this executable implements --kind "
                << umminfer::stage::compiled_kind() << ", got '" << kind << "'\n";
      return 2;
    }
    const std::string model = flags.count("model") ? flags["model"] : "";
    const std::string recv = flags.count("recv-endpoint") ? flags["recv-endpoint"] : "";
    const std::string coordinator =
        flags.count("coordinator-endpoint") ? flags["coordinator-endpoint"] : "";
    const std::string abort =
        flags.count("abort-endpoint") ? flags["abort-endpoint"] : "";
    if (model.empty() || recv.empty() || coordinator.empty() || abort.empty()) {
      usage();
      return 2;
    }

    options.model_dir = model;
    options.identity_out = flags.count("identity-out") ? flags["identity-out"] : "";
    options.model_id = flags.count("model-id") ? flags["model-id"] : "";
    options.model_revision = flags.count("model-revision") ? flags["model-revision"] : "";
    if (kind != "qwen38" &&
        (!options.identity_out.empty() || !options.model_id.empty() ||
         !options.model_revision.empty())) {
      throw std::runtime_error("native identity emission is supported only by qwen38");
    }
    const bool identity_requested = !options.identity_out.empty() ||
                                    !options.model_id.empty() ||
                                    !options.model_revision.empty();
    if (identity_requested && (options.identity_out.empty() || options.model_id.empty() ||
                               options.model_revision.empty())) {
      throw std::runtime_error(
          "--identity-out, --model-id, and --model-revision are required together");
    }
    options.gpu = flags.count("gpu")
                      ? umminfer::stage::parse_stage_int(flags["gpu"], "--gpu")
                      : 0;

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
    if (kind == "tts") {
      // Unified streamed first legs are :tts:0 and one-shot legs are :tts.
      // Later :tts:N chunks stay FIFO with ordinary work and share fair drain.
      config.request_priority = [](const umminfer::IncomingMessage& msg) {
        auto ends = [&msg](std::string_view suffix) {
          return msg.request_id.size() >= suffix.size() &&
                 std::string_view(msg.request_id).substr(
                     msg.request_id.size() - suffix.size()) == suffix;
        };
        return msg.type == "new_request" && (ends(":tts:0") || ends(":tts"));
      };
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
#ifdef UMM_NATIVE_REQUEST_ALLOCATOR
    constexpr bool kInstrumentedAllocator = true;
#else
    constexpr bool kInstrumentedAllocator = false;
#endif
    compute = umminfer::stage::with_native_request_telemetry(
        std::move(compute), options, config.name, kInstrumentedAllocator);
    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - load_started);
    std::cout << "native stage model loaded kind=" << kind << " model=" << model
              << " gpu=" << options.gpu << " load_ms=" << load_ms.count() << std::endl;

    if (!options.identity_out.empty()) {
      // make_compute returned only after the tokenizer, config, every weight
      // shard, and CUDA engine were loaded. Evidence cannot be pre-fabricated by
      // a client: this process hashes the files and its own executable now.
      config.native_identity_sha256 =
          umminfer::stage::write_native_qwen_identity(options, config.name)
              .document_sha256;
    }
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
