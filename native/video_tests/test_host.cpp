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

#include <nlohmann/json.hpp>

#include "video/config.hpp"
#include "video/engine.hpp"
#include "video/mp4_encoder.hpp"
#include "video/safetensors_mmap.hpp"
#include "video/scheduler.hpp"
#include "video/umt5_encoder.hpp"
#include "video/umt5_tokenizer.hpp"
#include "video/wan_vae.hpp"
#include "video/wan_dit.hpp"

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

void write_f32_shard(const std::filesystem::path& path,
                     const std::string& name,
                     const std::vector<float>& values) {
  nlohmann::json header = {
      {name,
       {{"dtype", "F32"},
        {"shape", {values.size()}},
        {"data_offsets", {0, values.size() * sizeof(float)}}}}};
  std::string text = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t size = text.size();
  for (unsigned i = 0; i < 8; ++i) out.put((size >> (i * 8)) & 0xff);
  out.write(text.data(), text.size());
  out.write(reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(float));
}

void test_sharded_safetensors_mmap() {
  using namespace umminfer::video;
  TempDir tmp("sharded-safetensors");
  write_f32_shard(tmp.path / "a.safetensors", "alpha", {1.0f, 2.0f});
  write_f32_shard(tmp.path / "b.safetensors", "beta", {3.0f});
  const nlohmann::json index = {
      {"metadata", {{"total_size", 12}}},
      {"weight_map", {{"alpha", "a.safetensors"}, {"beta", "b.safetensors"}}}};
  write_text(tmp.path / "model.safetensors.index.json", index.dump());

  ShardedSafetensorsMmap shards(tmp.path / "model.safetensors.index.json");
  check(shards.contains("alpha") && shards.contains("beta") &&
            !shards.contains("missing"),
        "multi-shard tensor lookup");
  check(shards.tensor("alpha", {2}).to_f32() == std::vector<float>({1.0f, 2.0f}),
        "tensor loaded from first shard");
  check(shards.tensor("beta", {1}).to_f32() == std::vector<float>({3.0f}),
        "tensor loaded from second shard");

  ShardedSafetensorsMmap single(tmp.path / "a.safetensors");
  check(single.contains("alpha") && !single.contains("beta") &&
            single.tensor("alpha", {2}).numel() == 2,
        "single-file sharded reader compatibility");
}

void test_umt5_tokenizer() {
  using namespace umminfer::video;
  TempDir tmp("umt5-tokenizer");
  nlohmann::json vocab = nlohmann::json::array({
      {"<pad>", 0.0}, {"</s>", 0.0}, {"<unk>", -10.0},
      {"▁hello", -0.1}, {"hello", -0.1}, {"▁", -0.5},
      {"▁world", -0.1}, {"▁影", -0.2}, {"像", -0.3},
      {"▁影像", -5.0}, {"▁cinematic", -0.2}, {"!", -0.1},
      {"<0xF0>", 0.0}, {"<0x9F>", 0.0},
      {"<0x98>", 0.0}, {"<0x80>", 0.0},
      {"<extra_id_0>", 0.0}, {"▁a", -0.1}, {"▁▁", 0.0},
      {"▁&", -0.1},
  });
  const auto write_tokenizer = [&](const std::filesystem::path& path,
                                   bool byte_fallback) {
    const nlohmann::json root = {
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Metaspace"},
          {"replacement", "▁"},
          {"prepend_scheme", "always"},
          {"split", true}}},
        {"added_tokens",
         {{{"id", 16}, {"content", "<extra_id_0>"}, {"special", true}}}},
        {"model",
         {{"type", "Unigram"},
          {"unk_id", 2},
          {"byte_fallback", byte_fallback},
          {"vocab", vocab}}}};
    write_text(path, root.dump());
  };
  write_tokenizer(tmp.path / "tokenizer.json", true);
  Umt5Tokenizer tokenizer(tmp.path / "tokenizer.json");

  check(tokenizer.encode("  hello   world! ").ids ==
            std::vector<int32_t>({3, 6, 11, 1}),
        "ASCII whitespace cleanup and Unigram tokenization");
  check(tokenizer.encode("影像").ids == std::vector<int32_t>({7, 8, 1}),
        "CJK Viterbi path uses the best score sum");
  check(tokenizer.encode("cinematic!").ids ==
            std::vector<int32_t>({10, 11, 1}),
        "ASCII punctuation tokenization");
  check(tokenizer.encode("&amp;amp;").ids == std::vector<int32_t>({19, 1}),
        "Wan prompt cleaning applies two HTML unescapes");
  check(tokenizer.encode("<extra_id_0>").ids ==
            std::vector<int32_t>({16, 1}) &&
            tokenizer.encode("hello<extra_id_0>world").ids ==
                std::vector<int32_t>({3, 16, 6, 1}),
        "added tokens bypass Metaspace and split adjacent text");
  check(tokenizer.encode("▁hello").ids == std::vector<int32_t>({3, 1}) &&
            tokenizer.encode("a▁▁").ids ==
                std::vector<int32_t>({17, 5, 5, 1}),
        "Metaspace prefix and split boundaries");
  check(tokenizer.encode("😀").ids ==
            std::vector<int32_t>({5, 12, 13, 14, 15, 1}),
        "unknown Unicode uses byte fallback tokens");
  check(tokenizer.encode("").ids == std::vector<int32_t>({1}),
        "empty prompt produces EOS only");

  const auto padded = tokenizer.encode("hello", 5, true);
  check(padded.ids == std::vector<int32_t>({3, 1, 0, 0, 0}) &&
            padded.attention_mask == std::vector<int32_t>({1, 1, 0, 0, 0}),
        "EOS and padding with attention mask");
  check(tokenizer.encode("hello world", 2).ids ==
            std::vector<int32_t>({3, 1}),
        "truncation preserves final EOS");

  write_tokenizer(tmp.path / "tokenizer-unk.json", false);
  Umt5Tokenizer unk_tokenizer(tmp.path / "tokenizer-unk.json");
  check(unk_tokenizer.encode("影像😀").ids ==
            std::vector<int32_t>({7, 8, 2, 1}),
        "unknown Unicode uses <unk> without byte fallback");
  check(unk_tokenizer.encode("😀😀").ids ==
            std::vector<int32_t>({5, 2, 1}),
        "adjacent unknown codepoints fuse into one <unk>");
}

void test_umt5_relative_position_buckets() {
  using umminfer::video::umt5_relative_position_bucket;
  const std::pair<int, int> cases[] = {
      {0, 0},   {1, 17},  {-1, 1},   {7, 23},  {-7, 7},
      {8, 24},  {-8, 8},  {16, 26},  {-16, 10},
      {128, 31}, {-128, 15}, {512, 31}, {-512, 15},
  };
  for (const auto& [distance, expected] : cases) {
    check(umt5_relative_position_bucket(distance) == expected,
          "UMT5 relative bucket at distance " + std::to_string(distance));
  }
  for (int distance = -512; distance <= 512; ++distance) {
    const int bucket = umt5_relative_position_bucket(distance);
    check(bucket >= 0 && bucket < 32, "UMT5 bucket remains in range");
  }
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

void test_wan_dit_host_helpers() {
  using namespace umminfer::video;
  TempDir tmp("wan-dit-host");
  write_text(tmp.path / "transformer" / "config.json",
             R"({"num_layers":30,"num_attention_heads":24,"attention_head_dim":128,"ffn_dim":14336,"in_channels":48,"out_channels":48,"patch_size":[1,2,2],"text_dim":4096,"freq_dim":256,"rope_max_seq_len":1024,"cross_attn_norm":true,"qk_norm":"rms_norm_across_heads","eps":0.000001,"image_dim":null,"added_kv_proj_dim":null})");
  const WanDitConfig config = load_wan_dit_config(tmp.path.string());
  check(config.inner_dim == 3072 && config.num_layers == 30 &&
            config.ffn_dim == 14336 && config.patch_h == 2,
        "Wan DiT config dimensions");

  const auto zero = wan_sinusoidal_timestep_embedding(0.0f, 256);
  check(zero.size() == 256, "timestep embedding size");
  for (int i = 0; i < 128; ++i) {
    check(zero[i] == 1.0f && zero[i + 128] == 0.0f,
          "timestep cosine/sine layout at zero");
  }
  const auto embedding = wan_sinusoidal_timestep_embedding(500.0f, 256);
  const double frequency = std::exp(-std::log(10000.0) * 7 / 128);
  check(std::fabs(embedding[7] - std::cos(500.0 * frequency)) < 2e-5 &&
            std::fabs(embedding[135] - std::sin(500.0 * frequency)) < 2e-5,
        "timestep frequencies match Diffusers");

  const auto rope = wan_build_rope_3d(2, 4, 6, 128);
  check(rope.frames == 2 && rope.height == 2 && rope.width == 3 &&
            rope.cos.size() == 12 * 128 && rope.sin.size() == rope.cos.size(),
        "3D RoPE post-patch shape");
  for (int d = 0; d < 128; d += 2) {
    check(rope.cos[d] == 1.0f && rope.sin[d] == 0.0f &&
              rope.cos[d] == rope.cos[d + 1] && rope.sin[d] == rope.sin[d + 1],
          "RoPE origin and repeat-interleave pairs");
  }
  const size_t last_token = 11 * 128;
  check(std::fabs(rope.cos[last_token] - std::cos(1.0)) < 1e-6f &&
            std::fabs(rope.sin[last_token] - std::sin(1.0)) < 1e-6f &&
            std::fabs(rope.cos[last_token + 44] - std::cos(1.0)) < 1e-6f &&
            std::fabs(rope.sin[last_token + 86] - std::sin(2.0)) < 1e-6f,
        "RoPE temporal, height, and width axes");
  const double rope_frequency = std::pow(10000.0, -2.0 / 44.0);
  check(std::fabs(rope.cos[last_token + 2] - std::cos(rope_frequency)) < 1e-6f,
        "RoPE FP64 frequency calculation");

  std::vector<float> latent(2 * 2 * 4 * 6);
  for (size_t i = 0; i < latent.size(); ++i) latent[i] = static_cast<float>(i) * 0.125f - 3.0f;
  const auto tokens = wan_patchify_host(latent.data(), 2, 2, 4, 6);
  const auto restored = wan_unpatchify_host(tokens.data(), 2, 2, 4, 6);
  check(tokens.size() == 2 * 2 * 3 * 2 * 4 && restored == latent,
        "patchify/unpatchify exact round trip and component order");

  std::vector<float> projected(16);
  for (size_t i = 0; i < projected.size(); ++i) projected[i] = (i / 8) * 10 + (i % 8);
  const auto explicit_output = wan_unpatchify_host(projected.data(), 2, 1, 2, 4);
  const std::vector<float> expected = {
      0, 2, 10, 12, 4, 6, 14, 16,
      1, 3, 11, 13, 5, 7, 15, 17,
  };
  check(explicit_output == expected, "proj_out patch, row, column, channel order");
}

void test_wan_dit_perf_helpers() {
  using namespace umminfer::video;
  check(kWanDitQueryTile == 512, "Wan DiT query tile raised to 512");

  // Verify attention query tile partitioning across <512, ==512, and >512 token counts.
  const int token_cases[] = {1, 63, 64, 264, 511, 512, 513, 1024, 2640};
  for (int tokens : token_cases) {
    int covered = 0;
    int tiles = 0;
    for (int begin = 0; begin < tokens; begin += kWanDitQueryTile) {
      const int len = wan_attention_tile_len(tokens, begin);
      check(len > 0 && len <= kWanDitQueryTile && begin + len <= tokens,
            "tile bounds valid for tokens=" + std::to_string(tokens));
      covered += len;
      ++tiles;
    }
    check(covered == tokens,
          "tile partition covers all tokens for tokens=" + std::to_string(tokens));
    const int expected_tiles = (tokens + kWanDitQueryTile - 1) / kWanDitQueryTile;
    check(tiles == expected_tiles,
          "tile count matches ceil(tokens/512) for tokens=" + std::to_string(tokens));
  }
  check(wan_attention_tile_len(2640, 2560) == 80,
        "704x480 29f final query tile has 80 tokens");

  bool bad_tile = false;
  try {
    (void)wan_attention_tile_len(512, 512);
  } catch (const std::invalid_argument&) {
    bad_tile = true;
  }
  check(bad_tile, "out-of-range q_start rejected");

  // Verify text-embedding content hashing: deterministic, shape-sensitive, bit-sensitive.
  std::vector<float> small = {1.0f, -2.5f, 0.25f, 4.0f, -0.5f, 3.0f, 0.125f, 7.5f};
  const uint64_t h_2x4 = wan_hash_text_embeds(small.data(), 2, 4);
  const uint64_t h_2x4_repeat = wan_hash_text_embeds(small.data(), 2, 4);
  const uint64_t h_4x2 = wan_hash_text_embeds(small.data(), 4, 2);
  check(h_2x4 == h_2x4_repeat, "text embed hash is deterministic");
  check(h_2x4 != h_4x2, "text embed hash distinguishes shape with identical data");

  // Exercise unaligned tail sizes (1 float = 4B, 3 floats = 12B, 9 floats = 36B).
  for (int elems : {1, 3, 9}) {
    std::vector<float> buf(static_cast<size_t>(elems), 1.25f);
    const uint64_t h0 = wan_hash_text_embeds(buf.data(), 1, elems);
    buf.back() = 1.250001f;
    const uint64_t h1 = wan_hash_text_embeds(buf.data(), 1, elems);
    check(h0 != h1, "tail float mutation changes hash for elems=" + std::to_string(elems));
  }

  // Exercise realistic [512, 4096] prompt tensor (~8 MiB).
  std::vector<float> full(512u * 4096u, 0.0f);
  for (size_t i = 0; i < full.size(); ++i) {
    full[i] = static_cast<float>((i * 2654435761u) & 0xFFFFu) * 1e-4f;
  }
  const uint64_t h_full = wan_hash_text_embeds(full.data(), 512, 4096);
  const WanTextEmbedCacheKey key_a{h_full, 512, 4096};
  const WanTextEmbedCacheKey key_a_copy{wan_hash_text_embeds(full.data(), 512, 4096), 512, 4096};
  check(key_a == key_a_copy, "WanTextEmbedCacheKey matches identical prompt");

  for (size_t idx : {size_t{0}, full.size() / 2, full.size() - 1}) {
    const float saved = full[idx];
    full[idx] = saved + 1e-5f;
    const uint64_t mutated = wan_hash_text_embeds(full.data(), 512, 4096);
    check(mutated != h_full,
          "full [512,4096] hash detects mutation at index " + std::to_string(idx));
    full[idx] = saved;
  }

  bool null_rejected = false;
  try {
    (void)wan_hash_text_embeds(nullptr, 512, 4096);
  } catch (const std::invalid_argument&) {
    null_rejected = true;
  }
  check(null_rejected, "null text embedding pointer rejected");
}

void test_wan_dit_teacache_controller() {
  using namespace umminfer::video;

  // 1. Threshold validation and basic relative L1 + rescale helpers.
  wan_validate_teacache_thresh(0.0f);
  wan_validate_teacache_thresh(0.15f);
  wan_validate_teacache_thresh(1.0f);
  for (float bad : {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity()}) {
    bool threw = false;
    try {
      wan_validate_teacache_thresh(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "invalid TeaCache threshold rejected");
  }

  const std::vector<float> e_base = {1.0f, -2.0f, 3.0f, -4.0f};  // L1 = 10.0
  const std::vector<float> e_same = e_base;
  const std::vector<float> e_step = {1.2f, -2.2f, 3.2f, -4.2f};  // diff L1 = 0.8 => rel = 0.08
  check(std::fabs(wan_teacache_relative_l1(e_same.data(), e_base.data(), e_base.size())) < 1e-7f,
        "identical modulation has 0 relative L1");
  check(std::fabs(wan_teacache_relative_l1(e_step.data(), e_base.data(), e_base.size()) - 0.08f) < 1e-6f,
        "relative L1 matches ||cur-prev||_1 / ||prev||_1");
  check(wan_teacache_rescale(-0.5f) == 0.0f, "rescale clamps negative input to 0");
  check(std::fabs(wan_teacache_rescale(0.12f) - 0.12f) < 1e-6f,
        "rescale is monotonic and preserves non-negative delta");

  // 2. Disabled controller (thresh == 0.0f) never skips and never requests saving.
  {
    TeaCacheController ctrl(0.0f);
    check(!ctrl.enabled(), "thresh == 0 disables TeaCache");
    for (int step = 0; step < 6; ++step) {
      const TeaCacheDecision d =
          ctrl.evaluate(0, step, 6, 8, 30, 44, false, e_base.data(), e_base.size());
      check(!d.use_cached_residual && !d.save_residual,
            "disabled TeaCache never skips or saves at step " + std::to_string(step));
    }
    check(ctrl.hits() == 0 && ctrl.total_calls() == 6,
          "disabled TeaCache reports 0/6 hits");
  }

  // 3. First step, last step, and <= 2 total steps never skip.
  {
    TeaCacheController ctrl(0.99f);
    const auto s0 = ctrl.evaluate(0, 0, 2, 8, 30, 44, false, e_base.data(), e_base.size());
    const auto s1 = ctrl.evaluate(0, 1, 2, 8, 30, 44, false, e_base.data(), e_base.size());
    check(!s0.use_cached_residual && !s1.use_cached_residual && ctrl.hits() == 0,
          "total_steps <= 2 never skips");
    const auto unsched = ctrl.evaluate(0, -1, 15, 8, 30, 44, false, e_base.data(), e_base.size());
    check(!unsched.use_cached_residual && !unsched.save_residual,
          "negative step_index never skips");
  }

  // 4. Accumulation, cache hit, threshold crossing recompute + reset, and final step.
  {
    TeaCacheController ctrl(0.15f);
    // Construct a sequence where each step shifts every element by +0.15 (L1 = 0.6),
    // starting from L1 = 10.0 so step-to-step rel_l1 is ~0.06 -> hits on steps 1 & 2,
    // crosses 0.15 on step 3 (recomputes & resets), hits on step 4, full on final step 5.
    std::vector<float> e = {2.5f, 2.5f, 2.5f, 2.5f};  // L1 = 10.0
    const auto d0 = ctrl.evaluate(0, 0, 6, 8, 30, 44, false, e.data(), e.size());
    check(!d0.use_cached_residual && d0.save_residual, "step 0 computes and saves");

    // Step 1: identical e -> rel_l1 = 0 < 0.15 -> hit.
    const auto d1 = ctrl.evaluate(0, 1, 6, 8, 30, 44, false, e.data(), e.size());
    check(d1.use_cached_residual && !d1.save_residual, "step 1 identical e0 hits");

    // Step 2: rel_l1 = 0.08 -> accum = 0.08 < 0.15 -> hit.
    for (float& v : e) v += 0.2f;  // delta L1 = 0.8, prev L1 = 10.0 -> 0.08
    const auto d2 = ctrl.evaluate(0, 2, 6, 8, 30, 44, false, e.data(), e.size());
    check(d2.use_cached_residual && !d2.save_residual &&
              std::fabs(d2.accumulated_rel_l1 - 0.08f) < 1e-5f,
          "step 2 below threshold hits and accumulates");

    // Step 3: rel_l1 = 0.864 / 10.8 = 0.08 -> accum = 0.16 >= 0.15 -> miss & reset!
    for (float& v : e) v += 0.216f;
    const auto d3 = ctrl.evaluate(0, 3, 6, 8, 30, 44, false, e.data(), e.size());
    check(!d3.use_cached_residual && d3.save_residual &&
              d3.accumulated_rel_l1 == 0.0f,
          "step 3 crossing threshold recomputes and resets accumulator");

    // Step 4: small delta (0.04 < 0.15) after reset -> hit again!
    for (float& v : e) v += 0.1f;
    const auto d4 = ctrl.evaluate(0, 4, 6, 8, 30, 44, false, e.data(), e.size());
    check(d4.use_cached_residual && !d4.save_residual,
          "step 4 hits after accumulator reset");

    // Step 5 (final step): identical e0 must STILL compute full blocks and not save.
    const auto d5 = ctrl.evaluate(0, 5, 6, 8, 30, 44, false, e.data(), e.size());
    check(!d5.use_cached_residual && !d5.save_residual,
          "final step always computes full blocks");
    check(ctrl.hits() == 3 && ctrl.total_calls() == 6,
          "hit/call telemetry matches 3/6");
  }

  // 5. Independent CFG branches (cond = 0, uncond = 1) never interfere.
  {
    TeaCacheController ctrl(0.15f);
    std::vector<float> e_cond = {2.5f, 2.5f, 2.5f, 2.5f};
    std::vector<float> e_uncond = {2.5f, 2.5f, 2.5f, 2.5f};
    check(!ctrl.evaluate(0, 0, 5, 8, 30, 44, false, e_cond.data(), e_cond.size()).use_cached_residual,
          "cond step 0 computes");
    // Branch 1 has never run step 0; step 1 on branch 1 must miss (not reuse branch 0!).
    const auto uncond_cold =
        ctrl.evaluate(1, 1, 5, 8, 30, 44, false, e_uncond.data(), e_uncond.size());
    check(!uncond_cold.use_cached_residual && uncond_cold.save_residual,
          "uncond branch does not borrow cond branch residual");

    // Mutate cond heavily so cond misses on step 1, while uncond has identical e0 on step 2 -> uncond hits!
    std::vector<float> e_cond_big = {10.0f, 10.0f, 10.0f, 10.0f};
    check(!ctrl.evaluate(0, 1, 5, 8, 30, 44, false, e_cond_big.data(), e_cond_big.size()).use_cached_residual,
          "cond step 1 large jump misses");
    check(ctrl.evaluate(1, 2, 5, 8, 30, 44, false, e_uncond.data(), e_uncond.size()).use_cached_residual,
          "uncond step 2 still hits independently of cond miss");
  }

  // 6. Shape change, I2V mode change, invalidate_branch(), and reset() invalidate cache.
  {
    TeaCacheController ctrl(0.50f);
    (void)ctrl.evaluate(0, 0, 6, 8, 30, 44, false, e_base.data(), e_base.size());
    check(ctrl.evaluate(0, 1, 6, 8, 30, 44, false, e_base.data(), e_base.size()).use_cached_residual,
          "matching shape hits before mutation");
    // Geometry change -> miss!
    const auto shape_miss =
        ctrl.evaluate(0, 2, 6, 4, 30, 44, false, e_base.data(), e_base.size());
    check(!shape_miss.use_cached_residual && shape_miss.save_residual,
          "frame count change invalidates cached residual");
    // First-frame I2V flag flip -> miss!
    const auto mode_miss =
        ctrl.evaluate(0, 3, 6, 4, 30, 44, true, e_base.data(), e_base.size());
    check(!mode_miss.use_cached_residual && mode_miss.save_residual,
          "first_frame_zero_timestep change invalidates cached residual");
    // Explicit invalidate_branch -> miss!
    ctrl.invalidate_branch(0);
    check(!ctrl.evaluate(0, 4, 6, 4, 30, 44, true, e_base.data(), e_base.size()).use_cached_residual,
          "invalidate_branch forces recompute");
    // reset() preserves threshold, clears telemetry and cached state.
    ctrl.reset();
    check(ctrl.threshold() == 0.50f && ctrl.hits() == 0 && ctrl.total_calls() == 0,
          "reset() clears counters and keeps threshold");
    check(!ctrl.evaluate(0, 1, 6, 4, 30, 44, true, e_base.data(), e_base.size()).use_cached_residual,
          "post-reset intermediate step cannot hit stale cache");
  }

  // 7. Invalid branch index rejected.
  {
    TeaCacheController ctrl(0.15f);
    bool bad_branch = false;
    try {
      (void)ctrl.evaluate(2, 0, 6, 8, 30, 44, false, e_base.data(), e_base.size());
    } catch (const std::invalid_argument&) {
      bad_branch = true;
    }
    check(bad_branch, "out-of-range TeaCache branch rejected");
  }
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
  check(res.umt5_ms == 0.0 && res.dit_ms == 0.0 && res.vae_ms == 0.0,
        "contract stub reports zeroed phase timings");

  // Step-level abort check.
  std::atomic<int> calls{0};
  bool step_aborted = false;
  try {
    (void)stub.generate(params, [&calls] { return ++calls >= 3; });
  } catch (const RequestAborted&) {
    step_aborted = true;
  }
  check(step_aborted, "engine aborts cleanly at denoising step boundary");

  // The contract stub must accept image bytes without decoding them.
  GenerateParams image_params = params;
  image_params.init_image_bytes = "not-a-real-image";
  const GenerateResult image_res = stub.generate(image_params);
  check(is_valid_mp4_header(image_res.mp4_bytes),
        "contract stub accepts init image bytes");

  // VideoEngineConfig + teacache_thresh validation & plumbing.
  VideoEngineConfig cfg;
  cfg.gpu = 0;
  cfg.model_dir = tmp.path.string();
  cfg.mode = VideoBackendMode::kContractStub;
  cfg.teacache_thresh = 0.15f;
  VideoEngine configured(cfg);
  check(std::fabs(configured.teacache_thresh() - 0.15f) < 1e-6f,
        "VideoEngineConfig stores teacache_thresh");
  const GenerateResult tc_res = configured.generate(params);
  check(std::fabs(tc_res.teacache_thresh - 0.15f) < 1e-6f &&
            tc_res.teacache_hits == 0 && tc_res.teacache_total_calls == 0,
        "GenerateResult reports effective teacache_thresh and stub telemetry");

  GenerateParams override_params = params;
  override_params.teacache_thresh = 0.0f;
  check(configured.generate(override_params).teacache_thresh == 0.0f,
        "per-request teacache_thresh override applies");

  bool bad_tc_rejected = false;
  try {
    GenerateParams bad_params = params;
    bad_params.teacache_thresh = 1.5f;
    (void)configured.generate(bad_params);
  } catch (const std::invalid_argument&) {
    bad_tc_rejected = true;
  }
  check(bad_tc_rejected, "out-of-range per-request teacache_thresh rejected");
}

// Reference values produced by Diffusers 0.43 UniPCMultistepScheduler loaded
// from Wan2.2-TI2V-5B-Diffusers/scheduler/scheduler_config.json (solver_order=2,
// solver_type="bh2", predict_x0=true, prediction_type="flow_prediction",
// use_flow_sigmas=true, flow_shift=5.0, lower_order_final=true,
// final_sigmas_type="zero", disable_corrector=[]).
void test_unipc_bh2_matches_diffusers() {
  using namespace umminfer::video;
  FlowMatchConfig cfg;
  cfg.flow_shift = 5.0;
  cfg.num_train_timesteps = 1000;

  cfg.steps = 20;
  const FlowMatchSchedule s20 = build_flow_match_schedule(cfg);
  const float ref_sigmas20[21] = {
      0.9999989867210388f, 0.9895941615104675f, 0.978284478187561f,
      0.9659478068351746f, 0.9524376392364502f, 0.9375780820846558f,
      0.9211564660072327f, 0.9029127359390259f, 0.8825258612632751f,
      0.8595945835113525f, 0.8336109519004822f, 0.8039219379425049f,
      0.7696741223335266f, 0.7297303080558777f, 0.6825404167175293f,
      0.6259360909461975f, 0.55678790807724f,   0.4704066216945648f,
      0.3594328761100769f, 0.21162153780460358f, 0.0f};
  const float ref_timesteps20[20] = {
      999, 989, 978, 965, 952, 937, 921, 902, 882, 859,
      833, 803, 769, 729, 682, 625, 556, 470, 359, 211};
  check(s20.sigmas.size() == 21 && s20.timesteps.size() == 20,
        "20-step schedule sizes");
  for (size_t i = 0; i < 21; ++i) {
    check(std::fabs(s20.sigmas[i] - ref_sigmas20[i]) < 1e-6f,
          "sigma[" + std::to_string(i) + "] matches Diffusers");
  }
  for (size_t i = 0; i < 20; ++i) {
    check(s20.timesteps[i] == ref_timesteps20[i],
          "timestep[" + std::to_string(i) + "] matches Diffusers int64 cast");
  }

  cfg.steps = 6;
  const FlowMatchSchedule s6 = build_flow_match_schedule(cfg);
  const float ref_sigmas6[7] = {
      0.9999989867210388f, 0.9615828394889832f, 0.9092148542404175f,
      0.8336109519004822f, 0.7148972749710083f, 0.5014970302581787f, 0.0f};
  for (size_t i = 0; i < 7; ++i) {
    check(std::fabs(s6.sigmas[i] - ref_sigmas6[i]) < 1e-6f,
          "6-step sigma[" + std::to_string(i) + "] matches Diffusers");
  }

  // Fixed model-output sequence (all values are exact binary fractions), so the
  // only difference between this loop and the Diffusers loop is solver math.
  const float ref_steps[6][4] = {
      {0.7403959035873413f, 1.2307919263839722f, 1.721187949180603f,
       2.2115838527679443f},
      {0.7633533477783203f, 1.231645107269287f, 1.6999366283416748f,
       2.1682281494140625f},
      {0.8466852903366089f, 1.2705178260803223f, 1.6943503618240356f,
       2.11818265914917f},
      {1.0312597751617432f, 1.3718507289886475f, 1.7124415636062622f,
       2.053032159805298f},
      {1.4824192523956299f, 1.6435328722000122f, 1.8046461343765259f,
       1.9657593965530396f},
      {2.6107876300811768f, 2.3330912590026855f, 2.0553946495056152f,
       1.777698040008545f},
  };
  const int expected_order[6] = {1, 2, 2, 2, 2, 1};

  UniPCFlowMatchSolver solver;
  solver.reset(s6);
  std::vector<float> sample = {0.75f, 1.25f, 1.75f, 2.25f};
  for (int k = 0; k < 6; ++k) {
    const float model_output[4] = {
        static_cast<float>((0 + 1) * 0.25 - k * 0.5 + 0 * k * 0.125),
        static_cast<float>((1 + 1) * 0.25 - k * 0.5 + 1 * k * 0.125),
        static_cast<float>((2 + 1) * 0.25 - k * 0.5 + 2 * k * 0.125),
        static_cast<float>((3 + 1) * 0.25 - k * 0.5 + 3 * k * 0.125)};
    solver.step(k, sample.data(), model_output, sample.size());
    check(solver.last_order() == expected_order[k],
          "UniPC order at step " + std::to_string(k) + " (lower_order_final)");
    for (size_t j = 0; j < sample.size(); ++j) {
      check(std::fabs(sample[j] - ref_steps[k][j]) < 1e-5f,
            "UniPC step " + std::to_string(k) + " element " +
                std::to_string(j) + " matches Diffusers within 1e-5");
    }
  }

  // reset() must fully clear the multistep history.
  solver.reset(s6);
  std::vector<float> replay = {0.75f, 1.25f, 1.75f, 2.25f};
  const float first_model_output[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  solver.step(0, replay.data(), first_model_output, replay.size());
  for (size_t j = 0; j < replay.size(); ++j) {
    check(std::fabs(replay[j] - ref_steps[0][j]) < 1e-5f,
          "UniPC reset() replays step 0 identically");
  }

  bool rejected = false;
  try {
    solver.step(6, replay.data(), first_model_output, replay.size());
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  check(rejected, "UniPC rejects out-of-range step index");
}

void test_wan_vae_causal_chunk_schedule() {
  using namespace umminfer::video;
  WanVaeConfig cfg;
  cfg.temperal_downsample = {false, true, true};
  const int64_t tf = wan_vae_temporal_factor(cfg);
  check(tf == 4, "Wan2.2 temporal factor is 4");
  check(wan_vae_decoded_frame_count(1, tf) == 1, "latent_t=1 -> 1 frame");
  check(wan_vae_decoded_frame_count(4, tf) == 13, "latent_t=4 -> 13 frames");
  check(wan_vae_decoded_frame_count(8, tf) == 29, "latent_t=8 -> 29 frames");
  check(wan_vae_decoded_frame_count(20, tf) == 77, "latent_t=20 -> 77 frames");
  check(wan_vae_decoded_frame_count(0, tf) == 0, "latent_t=0 -> 0 frames");

  int64_t total = 0;
  for (int64_t t = 0; t < 20; ++t) {
    check(wan_vae_chunk_frame_offset(t, tf) == total,
          "chunk frame offset is contiguous at latent step " + std::to_string(t));
    total += wan_vae_chunk_frame_count(t, tf);
  }
  check(total == 77, "20 latent chunks sum cleanly to 77 frames");
}

void test_wan_vae_pool_policy_and_pointwise() {
  using namespace umminfer::video;

  // 1. Pointwise 1x1x1 convolution fast-path predicate.
  check(wan_conv3d_is_pointwise(1, 1, 1), "1x1x1 is pointwise");
  check(!wan_conv3d_is_pointwise(3, 3, 3), "3x3x3 is not pointwise");
  check(!wan_conv3d_is_pointwise(3, 1, 1), "3x1x1 temporal conv is not pointwise");
  check(!wan_conv3d_is_pointwise(1, 3, 3), "1x3x3 spatial conv is not pointwise");
  check(!wan_conv3d_is_pointwise(1, 1, 3), "1x1x3 is not pointwise");
  check(!wan_conv3d_is_pointwise(1, 3, 1), "1x3x1 is not pointwise");
  check(!wan_conv3d_is_pointwise(0, 1, 1), "degenerate kd=0 is not pointwise");

  // 2. Default policy upper bound (1.25x) and overflow safety.
  WanVaePoolPolicy policy;
  check(policy.max_cached_bytes == (1536ULL << 20),
        "default VAE pool cache cap is 1.5 GiB");
  check(policy.max_reusable_bytes(0) == 0, "0-byte request has 0 upper bound");
  check(policy.max_reusable_bytes(1000) == 1250,
        "1000 bytes -> exact 1250 byte 1.25x upper bound");
  check(policy.max_reusable_bytes(std::numeric_limits<size_t>::max()) ==
            std::numeric_limits<size_t>::max(),
        "overflow-safe upper bound saturates at size_t::max");

  // 3. Best-fit selection within [req, 1.25 * req].
  {
    const std::vector<size_t> idle = {999, 1251, 1250, 1100, 1000, 1100};
    const auto exact = policy.select_best_fit(idle, 1000);
    check(exact.has_value() && *exact == 4,
          "best-fit prefers exact 1000B match over 1100B/1250B");

    const auto tight = policy.select_best_fit(idle, 1001);
    check(tight.has_value() && *tight == 3,
          "best-fit picks smallest qualifying block (1100B) and earliest tie");

    const auto boundary = policy.select_best_fit({999, 1251, 1250}, 1000);
    check(boundary.has_value() && *boundary == 2,
          "exact 1.25x boundary (1250B for 1000B) is accepted");

    const auto oversized = policy.select_best_fit({999, 1251, 2000}, 1000);
    check(!oversized.has_value(),
          "blocks < req or > 1.25x req are rejected");

    check(!policy.select_best_fit(idle, 0).has_value(),
          "zero-byte request never selects a cached block");
  }

  // 4. Bounded cache byte cap eviction (oldest-first prefix).
  {
    WanVaePoolPolicy capped;
    capped.max_cached_bytes = 1000;

    // Fits comfortably under cap -> cache with 0 evictions.
    const auto fit = capped.plan_release({300, 200}, 500, 400);
    check(fit.cache_incoming && fit.evict_prefix_count == 0,
          "release fitting within cap evicts 0 blocks");

    // 500 + 600 = 1100 > 1000 -> evict oldest 300 -> 200 + 600 = 800 <= 1000.
    const auto evict_one = capped.plan_release({300, 200}, 500, 600);
    check(evict_one.cache_incoming && evict_one.evict_prefix_count == 1,
          "evicting oldest block brings total back within cap");

    // 900 + 800 = 1700 > 1000 -> evict 300, then 400 -> 200 + 800 = 1000 <= 1000.
    const auto evict_two = capped.plan_release({300, 400, 200}, 900, 800);
    check(evict_two.cache_incoming && evict_two.evict_prefix_count == 2,
          "evicts exact prefix count needed to fit incoming block");

    // Incoming block alone exceeds cap -> do not cache, evict 0 existing blocks.
    const auto too_big = capped.plan_release({300, 200}, 500, 1001);
    check(!too_big.cache_incoming && too_big.evict_prefix_count == 0,
          "oversized single block bypasses cache without thrashing idle pool");

    // Zero-byte block is never cached.
    const auto zero = capped.plan_release({300}, 300, 0);
    check(!zero.cache_incoming && zero.evict_prefix_count == 0,
          "zero-byte release is ignored");
  }
}

void test_wan_vae_gemm_prec_policy() {
  using namespace umminfer::video;

  // 1. Default and empty/null values resolve to kBf16.
  check(parse_wan_vae_gemm_prec(nullptr, nullptr) == WanVaeGemmPrec::kBf16,
        "null envs default to kBf16");
  check(parse_wan_vae_gemm_prec("", nullptr) == WanVaeGemmPrec::kBf16,
        "empty gemm_prec defaults to kBf16");
  check(parse_wan_vae_gemm_prec("", "") == WanVaeGemmPrec::kBf16,
        "empty envs default to kBf16");
  check(parse_wan_vae_gemm_prec("", "0") == WanVaeGemmPrec::kBf16,
        "pedantic=0 with empty gemm_prec defaults to kBf16");

  // 2. Explicit precision names (lowercase, uppercase, mixed case, whitespace).
  check(parse_wan_vae_gemm_prec("tf32", nullptr) == WanVaeGemmPrec::kTf32,
        "tf32 parses to kTf32");
  check(parse_wan_vae_gemm_prec("bf16", nullptr) == WanVaeGemmPrec::kBf16,
        "bf16 parses to kBf16");
  check(parse_wan_vae_gemm_prec("fp16", nullptr) == WanVaeGemmPrec::kFp16,
        "fp16 parses to kFp16");
  check(parse_wan_vae_gemm_prec("fp32", nullptr) == WanVaeGemmPrec::kFp32,
        "fp32 parses to kFp32");

  check(parse_wan_vae_gemm_prec("BF16", nullptr) == WanVaeGemmPrec::kBf16,
        "BF16 is case-insensitive");
  check(parse_wan_vae_gemm_prec("Bf16", "0") == WanVaeGemmPrec::kBf16,
        "Bf16 with pedantic=0 parses to kBf16");
  check(parse_wan_vae_gemm_prec("  fp16\t\n", "") == WanVaeGemmPrec::kFp16,
        "trimmed fp16 parses to kFp16");
  check(parse_wan_vae_gemm_prec("FP32", nullptr) == WanVaeGemmPrec::kFp32,
        "FP32 parses to kFp32");
  check(parse_wan_vae_gemm_prec("TF32", nullptr) == WanVaeGemmPrec::kTf32,
        "TF32 parses to kTf32");

  // 3. Unknown/invalid strings fall back safely to kBf16.
  for (const char* bad :
       {"fp8", "int8", "bf32", "float16", "bfloat16", "garbage", "   "}) {
    check(parse_wan_vae_gemm_prec(bad, nullptr) == WanVaeGemmPrec::kBf16,
          std::string("unknown gemm_prec '") + bad + "' falls back to kBf16");
  }

  // 4. Legacy UMM_VAE_PEDANTIC_FP32=1 takes precedence over UMM_VAE_GEMM_PREC.
  check(parse_wan_vae_gemm_prec(nullptr, "1") == WanVaeGemmPrec::kFp32,
        "pedantic=1 with null gemm_prec forces kFp32");
  check(parse_wan_vae_gemm_prec("tf32", "1") == WanVaeGemmPrec::kFp32,
        "pedantic=1 overrides tf32");
  check(parse_wan_vae_gemm_prec("bf16", "1") == WanVaeGemmPrec::kFp32,
        "pedantic=1 overrides bf16");
  check(parse_wan_vae_gemm_prec("fp16", "1") == WanVaeGemmPrec::kFp32,
        "pedantic=1 overrides fp16");
  check(parse_wan_vae_gemm_prec("bf16", "true") == WanVaeGemmPrec::kFp32,
        "pedantic=true overrides bf16");
  check(parse_wan_vae_gemm_prec("fp16", "0") == WanVaeGemmPrec::kFp16,
        "pedantic=0 does not override fp16");

  // 5. Canonical precision names and conv kernel specialization predicates.
  check(std::strcmp(wan_vae_gemm_prec_name(WanVaeGemmPrec::kTf32), "tf32") == 0,
        "kTf32 name is tf32");
  check(std::strcmp(wan_vae_gemm_prec_name(WanVaeGemmPrec::kBf16), "bf16") == 0,
        "kBf16 name is bf16");
  check(std::strcmp(wan_vae_gemm_prec_name(WanVaeGemmPrec::kFp16), "fp16") == 0,
        "kFp16 name is fp16");
  check(std::strcmp(wan_vae_gemm_prec_name(WanVaeGemmPrec::kFp32), "fp32") == 0,
        "kFp32 name is fp32");

  check(wan_conv3d_is_spatial_3x3(1, 3, 3), "1x3x3 is spatial 3x3");
  check(!wan_conv3d_is_spatial_3x3(3, 3, 3), "3x3x3 is not spatial 3x3");
  check(!wan_conv3d_is_spatial_3x3(1, 1, 1), "1x1x1 is not spatial 3x3");
  check(wan_conv3d_is_causal_3x3x3(3, 3, 3), "3x3x3 is causal 3x3x3");
  check(!wan_conv3d_is_causal_3x3x3(1, 3, 3), "1x3x3 is not causal 3x3x3");
  check(!wan_conv3d_is_causal_3x3x3(3, 1, 1), "3x1x1 is not causal 3x3x3");
}

void test_i2v_center_crop_preserves_aspect_ratio() {
  using namespace umminfer::video;
  // 1. Square 64x64 image with red top/bottom 16-px bands and pure green middle
  // 32-px band -> resized to 64x32 (2:1). Center-crop must discard the red
  // bands and keep pure green (R=-1, G=+1, B=-1) with zero vertical squash.
  std::vector<uint8_t> sq(64 * 64 * 3, 0);
  for (int y = 0; y < 64; ++y) {
    const bool middle = (y >= 16 && y < 48);
    for (int x = 0; x < 64; ++x) {
      const size_t idx = (static_cast<size_t>(y) * 64 + x) * 3;
      sq[idx + 0] = middle ? 0 : 255;
      sq[idx + 1] = middle ? 255 : 0;
      sq[idx + 2] = 0;
    }
  }
  const std::vector<float> wide =
      resize_center_crop_rgb_to_nchw(sq.data(), 64, 64, 64, 32);
  const size_t wide_plane = 64 * 32;
  check(wide.size() == 3 * wide_plane, "wide center-crop output size");
  for (size_t i = 0; i < wide_plane; ++i) {
    check(wide[0 * wide_plane + i] < -0.99f &&
              wide[1 * wide_plane + i] > 0.99f &&
              wide[2 * wide_plane + i] < -0.99f,
          "square->landscape center-crop discards top/bottom bands cleanly");
  }

  // 2. Landscape 64x32 image with blue left/right 16-px margins and white
  // center 32x32 -> resized to 32x32 (1:1). Must discard left/right blue bars.
  std::vector<uint8_t> land(64 * 32 * 3, 0);
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 64; ++x) {
      const bool center = (x >= 16 && x < 48);
      const size_t idx = (static_cast<size_t>(y) * 64 + x) * 3;
      land[idx + 0] = center ? 255 : 0;
      land[idx + 1] = center ? 255 : 0;
      land[idx + 2] = 255;
    }
  }
  const std::vector<float> cropped_sq =
      resize_center_crop_rgb_to_nchw(land.data(), 64, 32, 32, 32);
  const size_t sq_plane = 32 * 32;
  for (size_t i = 0; i < sq_plane; ++i) {
    check(cropped_sq[0 * sq_plane + i] > 0.99f &&
              cropped_sq[1 * sq_plane + i] > 0.99f &&
              cropped_sq[2 * sq_plane + i] > 0.99f,
          "landscape->square center-crop discards side margins cleanly");
  }
}

}  // namespace

int main() {
  const std::pair<const char*, void (*)()> tests[] = {
      {"size_and_frame_alignment", test_size_and_frame_alignment},
      {"wan_config_loading", test_wan_config_loading},
      {"wan_vae_config_loading", test_wan_vae_config_loading},
      {"wan_vae_causal_chunk_schedule", test_wan_vae_causal_chunk_schedule},
      {"wan_vae_pool_policy_and_pointwise", test_wan_vae_pool_policy_and_pointwise},
      {"wan_vae_gemm_prec_policy", test_wan_vae_gemm_prec_policy},
      {"safetensors_mmap", test_safetensors_mmap},
      {"sharded_safetensors_mmap", test_sharded_safetensors_mmap},
      {"umt5_tokenizer", test_umt5_tokenizer},
      {"umt5_relative_position_buckets", test_umt5_relative_position_buckets},
      {"wan_dit_host_helpers", test_wan_dit_host_helpers},
      {"wan_dit_perf_helpers", test_wan_dit_perf_helpers},
      {"wan_dit_teacache_controller", test_wan_dit_teacache_controller},
      {"flow_match_schedule_and_euler", test_flow_match_schedule_and_euler},
      {"unipc_bh2_matches_diffusers", test_unipc_bh2_matches_diffusers},
      {"i2v_center_crop_preserves_aspect_ratio",
       test_i2v_center_crop_preserves_aspect_ratio},
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
