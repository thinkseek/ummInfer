// SPDX-License-Identifier: Apache-2.0
// Minimal assert-based host tests for the GPU-free pieces of the native
// Qwen-Image engine: scheduler math, PNG encoding, tensor conversion.
// One binary, nonzero exit on failure, no framework.
#include "image/png.hpp"
#include "image/scheduler.hpp"
#include "image/safetensors_shard.hpp"
#include "image/packing.hpp"
#include "image/tokenizer.hpp"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int g_failures = 0;
void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

uint32_t read_u32(const std::string& b, size_t off) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(b[off])) << 24) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 1])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 2])) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(b[off + 3]));
}

void test_schedule_50_steps_1024px() {
  // 1024x1024 -> latent 128x128 -> packed 64x64 = 4096 tokens.
  umminfer::image::SchedulerConfig cfg;
  auto s = umminfer::image::build_schedule(cfg, 4096, 50);
  check(s.timesteps.size() == 50 && s.sigmas.size() == 51, "schedule sizes");
  check(std::fabs(s.sigmas[0] - 1.0f) < 1e-6f, "first sigma is 1.0");
  check(s.sigmas[50] == 0.0f, "terminal appended sigma is 0");
  check(std::fabs(s.sigmas[49] - 0.02f) < 1e-5f, "last real sigma = shift_terminal .02");
  // mu = 4096 * (0.4/7936) + (0.5 - 256 * slope) = 0.693548...
  check(std::fabs(s.timesteps[0] - 1000.0f) < 1e-4f, "first timestep 1000");
  // Reference values computed independently with numpy double precision.
  check(std::fabs(s.sigmas[1] - 0.9897009f) < 1e-5f, "sigma[1] matches numpy");
  check(std::fabs(s.sigmas[48] - 0.05847578f) < 1e-5f, "sigma[48] matches numpy");
  // Monotone decreasing.
  for (int i = 1; i < 50; ++i) check(s.sigmas[i] < s.sigmas[i - 1], "sigmas monotone");
  // Smaller image (512px -> 1024 tokens) gets smaller mu -> smaller sigma[1].
  auto s512 = umminfer::image::build_schedule(cfg, 1024, 50);
  check(s512.sigmas[1] < s.sigmas[1], "dynamic shift depends on image seq len");
}

void test_schedule_errors() {
  umminfer::image::SchedulerConfig cfg;
  bool threw = false;
  try { umminfer::image::build_schedule(cfg, 4096, 0); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "zero steps rejected");
  threw = false;
  try { umminfer::image::build_schedule(cfg, 0, 50); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "zero seq len rejected");
}

void test_euler_step() {
  std::vector<float> sample = {1.0f, -2.0f, 0.5f};
  const std::vector<float> vel = {0.5f, 4.0f, -1.0f};
  // dt = 0.2 - 0.8 = -0.6
  umminfer::image::euler_step_f32(sample.data(), vel.data(), 0.8, 0.2, 3);
  check(std::fabs(sample[0] - 0.7f) < 1e-6f, "euler sample[0]");
  check(std::fabs(sample[1] + 4.4f) < 1e-6f, "euler sample[1]");
  check(std::fabs(sample[2] - 1.1f) < 1e-6f, "euler sample[2]");
}

void test_png_structure_and_roundtrip() {
  umminfer::image::RgbImage img(4, 3);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 4; ++x) {
      const size_t p = (y * 4 + x) * 3;
      img.rgb[p] = static_cast<uint8_t>(x * 60);
      img.rgb[p + 1] = static_cast<uint8_t>(y * 80);
      img.rgb[p + 2] = 128;
    }
  const std::string png = umminfer::image::encode_png(img);
  static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  check(png.size() > 40 && std::memcmp(png.data(), sig, 8) == 0, "png signature");

  // IHDR fields.
  check(read_u32(png, 8) == 13, "ihdr length");
  check(png.compare(12, 4, "IHDR") == 0, "ihdr type");
  check(read_u32(png, 16) == 4 && read_u32(png, 20) == 3, "ihdr dimensions");
  check(static_cast<uint8_t>(png[24]) == 8 && static_cast<uint8_t>(png[25]) == 2,
        "ihdr 8-bit RGB");

  // Walk every chunk: verify declared CRC and collect IDAT bytes.
  std::vector<uint8_t> idat;
  size_t pos = 8;
  bool saw_iend = false;
  while (pos < png.size()) {
    const uint32_t len = read_u32(png, pos);
    const std::string type = png.substr(pos + 4, 4);
    uLong expect = crc32(0L, Z_NULL, 0);
    expect = crc32(expect, reinterpret_cast<const Bytef*>(png.data() + pos + 4), 4 + len);
    check(read_u32(png, pos + 8 + len) == static_cast<uint32_t>(expect), "chunk crc valid");
    if (type == "IDAT")
      idat.insert(idat.end(), png.begin() + pos + 8, png.begin() + pos + 8 + len);
    if (type == "IEND") {
      saw_iend = true;
      check(len == 0, "iend empty");
    }
    pos += 12 + len;
  }
  check(saw_iend, "iend present");

  // Inflate IDAT and confirm raw scanlines round-trip the source pixels.
  const size_t row_bytes = 4 * 3;
  std::vector<uint8_t> raw((row_bytes + 1) * 3);
  uLongf raw_len = raw.size();
  int rc = uncompress(raw.data(), &raw_len, idat.data(), idat.size());
  check(rc == Z_OK && raw_len == raw.size(), "idat inflates to full raw size");
  for (int y = 0; y < 3; ++y) {
    check(raw[y * (row_bytes + 1)] == 0, "filter byte none");
    check(std::memcmp(raw.data() + y * (row_bytes + 1) + 1,
                      img.rgb.data() + y * row_bytes, row_bytes) == 0,
          "scanline pixels round-trip");
  }
}

void test_png_errors() {
  bool threw = false;
  try { umminfer::image::encode_png(umminfer::image::RgbImage(0, 4)); }
  catch (const std::invalid_argument&) { threw = true; }
  check(threw, "zero width rejected");
  umminfer::image::RgbImage bad(2, 2);
  bad.rgb.resize(4);  // mismatched buffer
  threw = false;
  try { umminfer::image::encode_png(bad); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "buffer size mismatch rejected");
}

void test_tensor_conversion() {
  // CHW, H=2, W=1. Channel 0: {-1, 1}, channel 1: {0, -2}, channel 2: {0.5, 2}.
  const float chw[6] = {-1.0f, 1.0f,   // r: rows 0,1
                         0.0f, -2.0f,  // g
                         0.5f, 2.0f};  // b
  auto img = umminfer::image::tensor_n1p1_to_rgb(chw, 2, 1);
  check(img.width == 1 && img.height == 2, "converted dims");
  check(img.rgb[0] == 0, "-1 -> 0");
  check(std::abs(static_cast<int>(img.rgb[1]) - 128) <= 1, "0 -> ~128");
  check(std::abs(static_cast<int>(img.rgb[2]) - 191) <= 1, "0.5 -> ~191");
  check(img.rgb[3] == 255, "1 -> 255");
  check(img.rgb[4] == 0 && img.rgb[5] == 255, "out-of-range clamps");
}

void test_packing_roundtrip() {
  // 2 channels, 4x4 latent -> 2x2 tokens of 8 features; assign unique values.
  const int C = 2, H = 4, W = 4;
  std::vector<float> chw(C * H * W);
  for (size_t i = 0; i < chw.size(); ++i) chw[i] = static_cast<float>(i) - 10.0f;
  auto packed = umminfer::image::pack_latents(chw.data(), C, H, W);
  check(packed.tokens == 4 && packed.features == 8, "packed shape");
  // Token (0,0), channel 1, patch offset (dy=1,dx=0) -> src [1,1,0] = 16+4-10=10.
  const int tok00 = 0, feat = 1 * 4 + 1 * 2 + 0;
  check(std::fabs(packed.data[tok00 * 8 + feat] - 10.0f) < 1e-6f, "pack index math");
  auto back = umminfer::image::unpack_latents(packed, C, H, W);
  check(back == chw, "unpack restores every element");
  bool threw = false;
  try { umminfer::image::pack_latents(chw.data(), C, 3, W); }
  catch (const std::invalid_argument&) { threw = true; }
  check(threw, "odd latent dimension rejected");
}

// Write a minimal two-shard safetensors directory by hand.
void write_st_file(const std::string& path, const std::string& name,
                   const std::vector<size_t>& shape, const std::vector<uint16_t>& data) {
  nlohmann::json hdr;
  size_t bytes = data.size() * 2;
  hdr[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {0, bytes}}};
  std::string hj = hdr.dump();
  std::ofstream f(path, std::ios::binary);
  uint64_t n = hj.size();
  char lenb[8];
  for (int i = 0; i < 8; ++i) lenb[i] = static_cast<char>((n >> (8 * i)) & 0xff);
  f.write(lenb, 8);
  f.write(hj.data(), static_cast<std::streamsize>(hj.size()));
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(bytes));
}

void test_sharded_safetensors() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "umm_image_st_fixture";
  fs::remove_all(dir);
  fs::create_directories(dir);
  // Two shards, different tensors; BF16 values stored as raw 16-bit patterns.
  std::vector<uint16_t> a(6, 0x3f80);  // bit pattern, interpreted opaquely
  std::vector<uint16_t> b(4, 0x4000);
  write_st_file((dir / "m-00001-of-00002.safetensors").string(), "block.0.w", {2, 3}, a);
  write_st_file((dir / "m-00002-of-00002.safetensors").string(), "block.1.w", {4}, b);
  umminfer::image::ShardedSafeTensors st(dir.string());
  check(st.size() == 2 && st.shards().size() == 2, "two shards parsed");
  const auto& t0 = st.info("block.0.w");
  check(t0.shape == std::vector<size_t>({2, 3}) && t0.shard == 0, "tensor metadata");
  check(st.info("block.1.w").shard == 1, "second tensor lands in shard 1");
  const auto* p = static_cast<const uint16_t*>(st.host_ptr(t0));
  check(p[0] == 0x3f80 && p[5] == 0x3f80, "mmap host pointer reads tensor bytes");
  bool threw = false;
  try { (void)st.info("absent.tensor"); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "missing tensor throws");
  fs::remove_all(dir);
}

void test_tokenizer_bpe_and_specials() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "umm_image_tok_fixture";
  fs::remove_all(dir);
  fs::create_directories(dir);
  // Vocab: every GPT-2 byte-mapped symbol as its own id (so any text encodes),
  // plus one merged symbol and the special tokens.
  nlohmann::json vocab = nlohmann::json::object();
  int next_id = 0;
  for (int b = 0; b < 256; ++b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    uint32_t cp;
    if (printable) cp = static_cast<uint32_t>(b);
    else {
      int n = 0;
      for (int x = 0; x < b; ++x) {
        bool px = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255);
        if (!px) ++n;
      }
      cp = 256u + n;
    }
    std::string sym;
    if (cp < 0x80) sym.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
      sym.push_back(static_cast<char>(0xc0 | (cp >> 6)));
      sym.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
      sym.push_back(static_cast<char>(0xe0 | (cp >> 12)));
      sym.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      sym.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    if (!vocab.contains(sym)) vocab[sym] = next_id++;
  }
  // One merge: "t" + "h" -> "th".
  auto sym_of = [&](char c) { return std::string(1, c); };
  vocab[sym_of('t') + sym_of('h')] = next_id;
  const int merged_id = next_id;
  std::ofstream mf(dir / "merges.txt", std::ios::binary);
  mf << "#version: 0.2\n" << sym_of('t') << " " << sym_of('h') << "\n";
  mf.close();
  nlohmann::json added = {{"<|im_start|>", 151644}, {"<|im_end|>", 151645}};
  std::ofstream af(dir / "added_tokens.json", std::ios::binary);
  af << added.dump();
  af.close();
  for (auto it = added.begin(); it != added.end(); ++it) vocab[it.key()] = it.value();
  std::ofstream vf(dir / "vocab.json", std::ios::binary);
  vf << vocab.dump();
  vf.close();

  umminfer::image::Qwen2BpeTokenizer tok(dir.string());
  auto ids = tok.encode_prompt("th");
  // The merged token must appear; the two special frame tokens are present.
  bool has_merge = false, has_start = false, has_end = false;
  int end_count = 0;
  for (int id : ids) {
    if (id == merged_id) has_merge = true;
    if (id == 151644) has_start = true;
    if (id == 151645) { has_end = true; ++end_count; }
  }
  check(has_merge, "BPE merge applied to th");
  check(has_start && has_end && end_count >= 2, "template special tokens split out");
  // Chinese text must encode without throwing and stay deterministic.
  auto zh1 = tok.encode_prompt("一只猫");
  auto zh2 = tok.encode_prompt("一只猫");
  check(zh1 == zh2 && !zh1.empty(), "Chinese prompt deterministic");
  fs::remove_all(dir);
}
}  // namespace

int main() {
  test_schedule_50_steps_1024px();
  test_schedule_errors();
  test_euler_step();
  test_png_structure_and_roundtrip();
  test_png_errors();
  test_tensor_conversion();
  test_packing_roundtrip();
  test_sharded_safetensors();
  test_tokenizer_bpe_and_specials();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all image host checks passed\n");
  return 0;
}
