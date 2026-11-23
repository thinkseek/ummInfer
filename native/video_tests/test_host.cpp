// SPDX-License-Identifier: Apache-2.0
// Host unit tests for the native C++/CUDA video vertical slice:
//   - Wan checkpoint metadata & revision detection
//   - 32-px spatial alignment & 4k+1 temporal frame alignment
//   - Flow-matching shifted sigma schedule & Euler step math
//   - POSIX ffmpeg H.264 MP4 encoding, ftyp header check & mid-encode abort
//   - VideoEngine contract-stub execution, step abort, and default CUDA safety gate
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "video/config.hpp"
#include "video/engine.hpp"
#include "video/mp4_encoder.hpp"
#include "video/safetensors_mmap.hpp"
#include "video/scheduler.hpp"
#include "video/wan_vae.hpp"

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const std::string& label) {
  if (!cond) {
    std::cerr << "FAIL: " << label << "\n";
    ++g_failed;
    throw std::runtime_error("assertion failed: " + label);
  }
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const std::string& tag) {
    path = std::filesystem::temp_directory_path() /
           ("umminfer-video-test-" + tag);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << text;
}

void append_u16(std::ofstream& out, uint16_t value) {
  const char bytes[] = {static_cast<char>(value & 0xff),
                        static_cast<char>(value >> 8)};
  out.write(bytes, 2);
}

void append_u32(std::ofstream& out, uint32_t value) {
  const char bytes[] = {static_cast<char>(value & 0xff),
                        static_cast<char>((value >> 8) & 0xff),
                        static_cast<char>((value >> 16) & 0xff),
                        static_cast<char>(value >> 24)};
  out.write(bytes, 4);
}

void test_safetensors_mmap() {
  using namespace umminfer::video;
  TempDir tmp("safetensors");
  const auto path = tmp.path / "synthetic.safetensors";
  std::string header =
      R"({"__metadata__":{"test":"native"},"f32":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"bf16":{"dtype":"BF16","shape":[3],"data_offsets":[8,14]},"f16":{"dtype":"F16","shape":[2],"data_offsets":[14,18]}})";
  const size_t padded_header = (header.size() + 7) / 8 * 8;
  header.resize(padded_header, ' ');
  {
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_size = header.size();
    for (unsigned i = 0; i < 8; ++i) {
      out.put(static_cast<char>((header_size >> (i * 8)) & 0xff));
    }
    out.write(header.data(), header.size());
    float f32[] = {1.25f, -2.5f};
    for (const float value : f32) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      append_u32(out, bits);
    }
    append_u16(out, 0x3f80);  // BF16 1.0
    append_u16(out, 0xc000);  // BF16 -2.0
    append_u16(out, 0x3f40);  // BF16 0.75
    append_u16(out, 0x3c00);  // F16 1.0
    append_u16(out, 0x0001);  // Smallest positive F16 subnormal.
  }

  SafetensorsMmap file(path);
  check(file.contains("f32") && !file.contains("missing"),
        "tensor lookup by name");
  const auto& f32_view = file.tensor("f32", {2});
  check(f32_view.numel() == 2 && f32_view.size_bytes == 8,
        "F32 shape and byte size");
  const auto f32 = f32_view.to_f32();
  check(f32[0] == 1.25f && f32[1] == -2.5f, "F32 lookup values");
  const auto bf16 = file.tensor("bf16", {3}).to_f32();
  check(bf16[0] == 1.0f && bf16[1] == -2.0f && bf16[2] == 0.75f,
        "BF16 conversion");
  const auto f16 = file.tensor("f16", {2}).to_f32();
  check(f16[0] == 1.0f && std::fabs(f16[1] - 5.9604644775390625e-8f) < 1e-12f,
        "F16 conversion including subnormal");

  bool wrong_shape = false;
  try {
    (void)file.tensor("f32", {1, 2});
  } catch (const std::invalid_argument&) {
    wrong_shape = true;
  }
  check(wrong_shape, "shape mismatch rejected");

  bool missing = false;
  try {
    (void)file.tensor("missing");
  } catch (const std::out_of_range&) {
    missing = true;
  }
  check(missing, "missing tensor rejected");

  const auto truncated = tmp.path / "truncated.safetensors";
  std::filesystem::copy_file(path, truncated);
  std::filesystem::resize_file(truncated, std::filesystem::file_size(path) - 1);
  bool invalid = false;
  try {
    SafetensorsMmap bad(truncated);
  } catch (const std::runtime_error&) {
    invalid = true;
  }
  check(invalid, "out-of-file tensor data rejected");
}

void test_size_and_frame_alignment() {
  using namespace umminfer::video;
  const auto [w1, h1] = parse_video_size("480x320", 480, 320, 32);
  check(w1 == 480 && h1 == 320, "valid aligned size");

  const auto [w2, h2] = parse_video_size("", 704, 480, 32);
  check(w2 == 704 && h2 == 480, "empty size falls back to defaults");

  bool threw = false;
  try {
    (void)parse_video_size("500x480", 480, 320, 32);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "unaligned 500x480 rejected");

  threw = false;
  try {
    (void)parse_video_size("64x64", 480, 320, 32);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "below min_dim rejected");

  check(align_video_frames(1, 16, 81, 4) == 13, "1s@16fps aligns to 13 (4*3+1)");
  check(align_video_frames(2, 16, 81, 4) == 29, "2s@16fps aligns to 29 (4*7+1)");
  check(align_video_frames(10, 16, 49, 4) == 49, "clamps to max_frames 49");
}

void test_wan_config_loading() {
  using namespace umminfer::video;
  TempDir tmp("wan-cfg");
  const auto snap =
      tmp.path / "snapshots" / "0123456789abcdef0123456789abcdef01234567";
  write_text(snap / "model_index.json", R"({"_class_name":"WanPipeline"})");
  write_text(snap / "vae" / "config.json",
             R"({"z_dim":16,"scale_factor_spatial":16,"scale_factor_temporal":4,"latents_mean":[0.1,-0.2],"latents_std":[1.5,0.5]})");
  write_text(snap / "transformer" / "config.json",
             R"({"patch_size":[1,2,2],"in_channels":48,"out_channels":48,"num_layers":30})");
  write_text(snap / "scheduler" / "scheduler_config.json",
             R"({"flow_shift":5.0,"num_train_timesteps":1000})");

  const WanModelConfig cfg = load_wan_model_config(snap.string());
  check(cfg.model_class == "WanPipeline", "model_class parsed");
  check(cfg.revision == "0123456789abcdef0123456789abcdef01234567",
        "HF snapshot revision detected");
  check(cfg.spatial_align == 32, "spatial_align == 32");
  check(cfg.temporal_stride == 4, "temporal_stride == 4");
  check(cfg.latent_channels == 16, "latent_channels == 16");
  check(cfg.in_channels == 48 && cfg.num_layers == 30, "transformer dims parsed");
  check(std::fabs(cfg.flow_shift - 5.0) < 1e-6, "flow_shift == 5.0");
  check(cfg.latents_mean.size() == 2 && std::fabs(cfg.latents_mean[0] - 0.1f) < 1e-5f,
        "latents_mean parsed");
}

void test_wan_vae_config_loading() {
  using namespace umminfer::video;
  TempDir tmp("wan-vae-cfg");
  const auto path = tmp.path / "config.json";
  write_text(path,
             R"({"z_dim":1,"base_dim":2,"decoder_base_dim":null,"num_res_blocks":1,"out_channels":3,"patch_size":null,"is_residual":true,"dim_mult":[1],"temperal_downsample":[],"attn_scales":[],"latents_mean":[0.0],"latents_std":[1.0]})");
  const auto config = load_wan_vae_config(tmp.path);
  check(config.dim == 2 && config.decoder_base_dim == 2,
        "VAE decoder width falls back to base_dim");
  check(config.patch_size == 1 && config.is_residual,
        "null VAE patch size uses identity unpatchify");

  write_text(path,
             R"({"z_dim":1,"base_dim":2,"num_res_blocks":1,"out_channels":12,"patch_size":65536,"is_residual":true,"dim_mult":[1],"temperal_downsample":[],"attn_scales":[],"latents_mean":[0.0],"latents_std":[1.0]})");
  bool rejected = false;
  try {
    (void)load_wan_vae_config(tmp.path);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "overflowing VAE patch size rejected without a signal");
}

void test_flow_match_schedule_and_euler() {
  using namespace umminfer::video;
  FlowMatchConfig cfg;
  cfg.steps = 10;
  cfg.flow_shift = 5.0;
  cfg.num_train_timesteps = 1000;
  const FlowMatchSchedule s = build_flow_match_schedule(cfg);
  check(s.sigmas.size() == 11, "sigmas has steps + 1 entries");
  check(std::fabs(s.sigmas.front() - 1.0f) < 1e-5f, "sigma[0] == 1.0");
  check(std::fabs(s.sigmas.back() - 0.0f) < 1e-6f, "sigma[steps] == 0.0");
  for (size_t i = 0; i < 10; ++i) {
    check(s.sigmas[i] > s.sigmas[i + 1], "sigmas strictly decreasing");
    check(s.dt[i] < 0.0f, "dt negative");
    check(std::fabs((s.sigmas[i + 1] - s.sigmas[i]) - s.dt[i]) < 1e-6f,
          "dt matches sigma diff");
  }
  float lat[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float vel[4] = {0.5f, -1.0f, 2.0f, 0.0f};
  euler_step_host(lat, vel, -0.5f, 4);
  check(std::fabs(lat[0] - 0.75f) < 1e-6f && std::fabs(lat[1] - 2.5f) < 1e-6f &&
            std::fabs(lat[2] - 2.0f) < 1e-6f && std::fabs(lat[3] - 4.0f) < 1e-6f,
        "euler_step_host exact");
}

void test_mp4_encode_and_abort() {
  using namespace umminfer::video;
  constexpr int kW = 128;
  constexpr int kH = 128;
  std::vector<std::vector<uint8_t>> frames(5, std::vector<uint8_t>(kW * kH * 3, 64));
  for (size_t f = 0; f < frames.size(); ++f) {
    for (size_t i = 0; i < frames[f].size(); i += 3) {
      frames[f][i] = static_cast<uint8_t>((f * 40) % 255);
    }
  }
  const std::string mp4 = encode_rgb24_frames_mp4(frames, kW, kH, 8);
  check(is_valid_mp4_header(mp4), "encoded MP4 has valid ftyp header");
  check(mp4.size() > 256, "encoded MP4 is non-trivial");

  int polls = 0;
  bool aborted = false;
  try {
    (void)encode_rgb24_frames_mp4(frames, kW, kH, 8, [&polls] {
      return ++polls >= 2;
    });
  } catch (const RequestAborted&) {
    aborted = true;
  }
  check(aborted, "mid-encode abort throws RequestAborted");
}

void test_engine_contract_stub_and_cuda_gate() {
  using namespace umminfer::video;
  TempDir tmp("engine");
  write_text(tmp.path / "model_index.json", R"({"_class_name":"WanPipeline"})");

  // Default kCuda mode must fail loudly without CUDA or real Wan weights.
  bool cuda_refused = false;
  try {
    VideoEngine engine(0, tmp.path.string(), VideoBackendMode::kCuda);
  } catch (const std::runtime_error&) {
    cuda_refused = true;
  }
  check(cuda_refused, "default kCuda mode refuses silent fallback");

  VideoEngine stub(0, tmp.path.string(), VideoBackendMode::kContractStub);
  GenerateParams params;
  params.prompt = "A paper boat floating on a calm pond";
  params.width = 160;
  params.height = 128;
  params.seconds = 1;
  params.fps = 8;
  params.num_frames = 5;
  params.steps = 4;
  params.seed = 42;

  const GenerateResult res = stub.generate(params);
  check(is_valid_mp4_header(res.mp4_bytes), "engine produces valid MP4");
  check(res.width == 160 && res.height == 128, "result dimensions match");
  check(res.frames == 5 && res.fps == 8, "result frames and fps match");
  check(res.backend == "native-cpp-contract-stub", "explicit stub backend tag");

  // Step-level abort check.
  std::atomic<int> calls{0};
  bool step_aborted = false;
  try {
    (void)stub.generate(params, [&calls] { return ++calls >= 3; });
  } catch (const RequestAborted&) {
    step_aborted = true;
  }
  check(step_aborted, "engine aborts cleanly at denoising step boundary");
}

}  // namespace

int main() {
  const std::pair<const char*, void (*)()> tests[] = {
      {"size_and_frame_alignment", test_size_and_frame_alignment},
      {"wan_config_loading", test_wan_config_loading},
      {"wan_vae_config_loading", test_wan_vae_config_loading},
      {"safetensors_mmap", test_safetensors_mmap},
      {"flow_match_schedule_and_euler", test_flow_match_schedule_and_euler},
      {"mp4_encode_and_abort", test_mp4_encode_and_abort},
      {"engine_contract_stub_and_cuda_gate", test_engine_contract_stub_and_cuda_gate},
  };
  for (const auto& [name, fn] : tests) {
    try {
      fn();
      std::cout << "PASS: " << name << "\n";
      ++g_passed;
    } catch (const std::exception& err) {
      std::cerr << "ERROR in " << name << ": " << err.what() << "\n";
      ++g_failed;
    }
  }
  std::cout << "video_host_tests: " << g_passed << " passed, " << g_failed
            << " failed\n";
  return g_failed == 0 ? 0 : 1;
}
