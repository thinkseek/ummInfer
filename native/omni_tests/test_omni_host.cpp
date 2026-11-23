// SPDX-License-Identifier: Apache-2.0
// Assert-based host tests for the native omni slice. No GPU, no framework.
// Golden data under golden/ is downloaded from the Linux SM90 GPU reference and
// is not committed; tests needing it skip cleanly when the directory is absent.
#include "omni/image_preprocess.hpp"
#include "omni/mrope.hpp"
#include "omni/tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {
int g_failures = 0;
void check(bool cond, const std::string& what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Minimal .npy v1/v2 reader for the little-endian float32 arrays the golden
// dumper writes. Returns false on anything unexpected.
bool load_npy_f32(const std::string& path, std::vector<float>* out,
                  std::vector<long>* shape) {
  const std::string bytes = read_file(path);
  if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
    return false;
  const unsigned char major = static_cast<unsigned char>(bytes[6]);
  size_t header_len = 0;
  size_t pos = 8;
  if (major == 1) {
    header_len = static_cast<unsigned char>(bytes[8]) |
                 (static_cast<unsigned char>(bytes[9]) << 8);
    pos = 10;
  } else if (major == 2) {
    header_len = 0;
    for (int i = 0; i < 4; ++i)
      header_len |= static_cast<size_t>(static_cast<unsigned char>(bytes[8 + i]))
                    << (8 * i);
    pos = 12;
  } else {
    return false;
  }
  const std::string header = bytes.substr(pos, header_len);
  if (header.find("'descr': '<f4'") == std::string::npos &&
      header.find("\"descr\": \"<f4\"") == std::string::npos)
    return false;
  const size_t data_pos = pos + header_len;
  const size_t n = (bytes.size() - data_pos) / sizeof(float);
  out->resize(n);
  std::memcpy(out->data(), bytes.data() + data_pos, n * sizeof(float));
  const size_t lp = header.find("'shape': (");
  if (lp != std::string::npos) {
    size_t q = lp + 10;
    std::string tup;
    while (q < header.size() && header[q] != ')') tup += header[q++];
    std::stringstream ss(tup);
    long v;
    while (ss >> v) {
      shape->push_back(v);
      if (ss.peek() == ',') ss.ignore();
    }
  }
  return true;
}

void test_smart_resize() {
  using umminfer::omni::smart_resize;
  int h = 0, w = 0;
  // Reference image H=380 W=570, factor 32 -> 384x576.
  smart_resize(380, 570, 32, 3136, 12845056, &h, &w);
  check(h == 384 && w == 576, "smart_resize 380x570 -> 384x576");
  check(h % 32 == 0 && w % 32 == 0, "smart_resize factor divisibility");
  // Small image gets scaled up to the min-pixel bound.
  smart_resize(28, 28, 32, 3136, 12845056, &h, &w);
  check(h * w >= 3136 && h % 32 == 0 && w % 32 == 0,
        "smart_resize upscales below min_pixels");
  // Wide image stays inside the max bound.
  smart_resize(4000, 300, 32, 3136, 12845056, &h, &w);
  check(static_cast<long long>(h) * w <= 12845056 + 32LL * 32,
        "smart_resize respects max_pixels");
  bool threw = false;
  try {
    smart_resize(10000, 10, 32, 3136, 12845056, &h, &w);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "smart_resize rejects aspect ratio > 200");
}

void test_preprocess_rgb_normalization() {
  using umminfer::omni::preprocess_rgb;
  // 32x32 frame: already a multiple of 32, so resize is identity.
  std::vector<uint8_t> white(32 * 32 * 3, 255);
  const auto img = preprocess_rgb(white.data(), 32, 32, 1024, 12845056);
  check(img.grid_t == 1 && img.grid_h == 2 && img.grid_w == 2,
        "preprocess_rgb white 32x32 grid [1,2,2]");
  check(img.resized_h == 32 && img.resized_w == 32,
        "preprocess_rgb keeps the aligned 32x32 source");
  check(img.rows() == 4 && img.pixel_values.size() == 4u * 1536u,
        "preprocess_rgb white rows and element count");
  bool all_one = true;
  for (float v : img.pixel_values)
    if (!std::isfinite(v) || std::fabs(v - 1.0f) > 1e-6f) all_one = false;
  check(all_one, "white 255 normalizes to +1.0 (x/127.5 - 1)");

  std::vector<uint8_t> black(32 * 32 * 3, 0);
  const auto dark = preprocess_rgb(black.data(), 32, 32, 1024, 12845056);
  bool all_minus_one = true;
  for (float v : dark.pixel_values)
    if (!std::isfinite(v) || std::fabs(v + 1.0f) > 1e-6f) all_minus_one = false;
  check(all_minus_one, "black 0 normalizes to -1.0");

  // Mid gray 128 is the neutral point: 128/127.5 - 1 = 0.00392.
  std::vector<uint8_t> gray(32 * 32 * 3, 128);
  const auto mid = preprocess_rgb(gray.data(), 32, 32, 1024, 12845056);
  check(std::fabs(mid.pixel_values[0] - (128.0f / 127.5f - 1.0f)) < 1e-6f,
        "gray 128 normalizes near zero");

  // Channel order stays RGB: a pure red frame lights the C0 slab of every
  // row and leaves C1/C2 at -1. Row layout is (c, temporal, py, px).
  std::vector<uint8_t> red(32 * 32 * 3, 0);
  for (size_t i = 0; i < red.size(); i += 3) red[i] = 255;
  const auto rgb = preprocess_rgb(red.data(), 32, 32, 1024, 12845056);
  bool channel_ok = true;
  for (size_t r = 0; r < 4; ++r) {
    const float* row = rgb.pixel_values.data() + r * 1536;
    for (int i = 0; i < 512; ++i)
      if (std::fabs(row[i] - 1.0f) > 1e-6f) channel_ok = false;
    for (int i = 512; i < 1536; ++i)
      if (std::fabs(row[i] + 1.0f) > 1e-6f) channel_ok = false;
  }
  check(channel_ok, "preprocess_rgb keeps RGB channel order");
}

void test_build_positions() {
  using umminfer::omni::build_positions;
  // Text-only prompt: t = h = w = index.
  const std::vector<int> text = {10, 11, 12, 13};
  const auto flat = build_positions(text, 151676, 151669, 151670, 0, 0, false);
  check(flat.t == std::vector<int>({0, 1, 2, 3}), "mrope t text-only");
  check(flat.h == flat.t && flat.w == flat.t, "mrope h/w text-only");
  check(flat.next_pos == 4, "mrope next_pos text-only");

  // Image prompt: 2 prefix tokens, four pads, two suffix tokens; a 4x4
  // patch grid merges to 2x2 = 4 decoder pads.
  const std::vector<int> ids = {1, 2, 7, 7, 7, 7, 3, 4};
  const auto pos = build_positions(ids, 7, 151669, 151670, 4, 4, true);
  check(pos.t == std::vector<int>({0, 1, 2, 2, 2, 2, 4, 5}),
        "mrope t image region shares the first grid slot");
  check(pos.h == std::vector<int>({0, 1, 2, 2, 3, 3, 4, 5}), "mrope h grid rows");
  check(pos.w == std::vector<int>({0, 1, 2, 3, 2, 3, 4, 5}), "mrope w grid cols");
  check(pos.next_pos == 6, "mrope next_pos leaves room for the full grid");

  bool threw = false;
  try {
    (void)build_positions({1, 7, 7, 7, 2}, 7, 151669, 151670, 4, 4, true);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "mrope rejects a pad count that does not match the grid");
}

void test_preprocess_errors() {
  bool threw = false;
  try {
    umminfer::omni::preprocess_image("/nonexistent/path/image.png");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "preprocess rejects missing image");
  threw = false;
  try {
    umminfer::omni::preprocess_image(__FILE__);  // text, not an image
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "preprocess rejects non-image bytes");
}

void test_image_golden(const std::filesystem::path& golden,
                       const std::string& image_path) {
  if (!std::filesystem::exists(golden / "pixel_values_golden.npy") ||
      image_path.empty() || !std::filesystem::exists(image_path)) {
    std::printf("skip: image golden comparison (data or image missing)\n");
    return;
  }
  // Stage 1: JPEG decode must match the PIL reference byte-for-byte.
  int dh0 = 0, dw0 = 0;
  auto decoded = umminfer::omni::decode_image_rgb(image_path, &dh0, &dw0);
  const std::string ref_dec = read_file((golden / "decoded_rgb_hwc_u8.bin").string());
  check(ref_dec.size() == decoded.size(), "decoded dump size matches");
  if (ref_dec.size() == decoded.size()) {
    size_t dec_bad = 0, dec_max = 0;
    for (size_t i = 0; i < decoded.size(); ++i) {
      const int d = std::abs(static_cast<int>(decoded[i]) -
                             static_cast<unsigned char>(ref_dec[i]));
      dec_max = std::max(dec_max, static_cast<size_t>(d));
      if (d > 1) ++dec_bad;
    }
    // stb_image and libjpeg-turbo use different IDCT paths; per-pixel deltas
    // of a few LSB on ~1-2% of pixels are expected and mostly average out in
    // the bicubic resize (resize fed these bytes is within 2 LSB almost
    // everywhere). Ponytail ceiling: replace stb with libjpeg-turbo if a
    // stricter decode match is ever required.
    check(dec_max <= 4 && 100.0 * dec_bad / decoded.size() < 3.0,
          "JPEG decode close to libjpeg (max<=4, <3% beyond 1 LSB)");
    std::printf("decode %zux%zu max_diff=%zu pct_gt1=%.3f\n",
                static_cast<size_t>(dw0), static_cast<size_t>(dh0), dec_max,
                100.0 * dec_bad / decoded.size());
  }
  // Stage 2: bicubic resize vs torchvision, fed the SAME HF-decoded bytes so
  // the stb/libjpeg IDCT difference is excluded from this check.
  const std::string ref_rsz = read_file((golden / "resized_chw_u8.bin").string());
  constexpr int kRszH = 384, kRszW = 576;
  if (ref_rsz.size() == static_cast<size_t>(3 * kRszH * kRszW) &&
      ref_dec.size() == decoded.size()) {
    auto resized = umminfer::omni::resize_image_rgb(
        reinterpret_cast<const uint8_t*>(ref_dec.data()), dh0, dw0, kRszH, kRszW);
    size_t rsz_bad = 0, rsz_max = 0;
    double rsz_sum = 0.0;
    for (int y = 0; y < kRszH; ++y)
      for (int x = 0; x < kRszW; ++x)
        for (int c = 0; c < 3; ++c) {
          const int ref = static_cast<unsigned char>(
              ref_rsz[(c * kRszH + y) * kRszW + x]);
          const int d = std::abs(static_cast<int>(resized[(y * kRszW + x) * 3 + c]) - ref);
          rsz_max = std::max(rsz_max, static_cast<size_t>(d));
          rsz_sum += d;
          if (d > 2) ++rsz_bad;
        }
    check(rsz_bad * 1000 <= 3 * kRszH * kRszW,  // <=0.1% of pixels beyond 2 LSB
          "bicubic resize within 2 LSB of torchvision");
    std::printf("resize max_diff=%zu mean_diff=%.4f bad_gt2=%zu\n", rsz_max,
                rsz_sum / (3 * kRszH * kRszW), rsz_bad);
  }

  auto img = umminfer::omni::preprocess_image(image_path);
  check(img.grid_t == 1 && img.grid_h == 24 && img.grid_w == 36,
        "image grid_thw = [1,24,36]");
  check(img.rows() == 864, "pre-merge patch rows = 864");
  check(img.pixel_values.size() == 864u * 1536u, "pixel_values element count");

  std::vector<float> ref;
  std::vector<long> shape;
  check(load_npy_f32((golden / "pixel_values_golden.npy").string(), &ref, &shape),
        "golden npy parsed");
  check(shape.size() == 2 && shape[0] == 864 && shape[1] == 1536,
        "golden npy shape [864,1536]");
  if (ref.size() != img.pixel_values.size()) {
    check(false, "golden/native size mismatch");
    return;
  }
  double max_abs = 0.0, sum = 0.0;
  size_t first32_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(ref[i] - img.pixel_values[i]);
    max_abs = std::max(max_abs, d);
    sum += img.pixel_values[i];
    if (i < 32 && d > 0.02) ++first32_bad;
  }
  check(first32_bad == 0, "first32 pixel_values match within 0.02");
  check(max_abs < 0.05, "whole pixel buffer max abs diff < 0.05 (~1.3/255)");
  const double mean = sum / static_cast<double>(img.pixel_values.size());
  const auto stats = nlohmann::json::parse(read_file((golden / "image_golden.json").string()));
  check(std::fabs(mean - stats["mean"].get<double>()) < 0.01,
        "pixel mean matches golden");
  std::printf("image max_abs_diff=%.5f mean=%.5f\n", max_abs, mean);
}

// Tiny synthetic BPE fixture: byte symbols plus two merges.
void test_tokenizer_synthetic(const std::filesystem::path& tmp) {
  const std::string vp = (tmp / "mini_vocab.json").string();
  const std::string mp = (tmp / "mini_merges.txt").string();
  const std::string cp = (tmp / "mini_config.json").string();
  {
    std::ofstream f(vp);
    f << R"({"a":0,"b":1,"c":2,"ab":3,"Ġ":4,"Ġa":5})";
  }
  {
    std::ofstream f(mp);
    f << "a b\nĠ a\n";
  }
  {
    std::ofstream f(cp);
    f << R"({"added_tokens_decoder":{
      "151643":{"content":"<|endoftext|>","special":true},
      "151644":{"content":"<|im_start|>","special":true},
      "151645":{"content":"<|im_end|>","special":true}}})";
  }
  umminfer::omni::Qwen2BpeTokenizer tok(vp, mp, cp);
  check(tok.encode("ab") == std::vector<int>({3}), "synthetic merge a+b -> ab");
  check(tok.encode("c") == std::vector<int>({2}), "synthetic single symbol");
  auto fr = tok.encode("<|im_start|>ab<|im_end|>");
  check(fr == std::vector<int>({151644, 3, 151645}),
        "synthetic specials protected inline");
  check(tok.encode("").empty(), "empty encode is empty");
  check(tok.is_eos(151645) && tok.is_eos(151643), "eos ids");
}

void test_tokenizer_golden(const std::filesystem::path& golden) {
  const std::string vp = (golden / "vocab.json").string();
  if (!std::filesystem::exists(vp)) {
    std::printf("skip: tokenizer golden comparison (vocab missing)\n");
    return;
  }
  umminfer::omni::Qwen2BpeTokenizer tok(
      vp, (golden / "merges.txt").string(),
      (golden / "tokenizer_config.json").string());
  const auto cases =
      nlohmann::json::parse(read_file((golden / "tokenizer_golden.json").string()));
  int matched = 0;
  for (const auto& c : cases) {
    const auto got = tok.encode(c["text"].get<std::string>());
    const auto want = c["ids"].get<std::vector<int>>();
    if (got != want) {
      std::printf("FAIL: tokenizer golden case %zu\n",
                  static_cast<size_t>(&c - &cases[0]));
      std::printf("  text: %s\n", c["text"].get<std::string>().substr(0, 80).c_str());
      ++g_failures;
    } else {
      ++matched;
    }
  }
  std::printf("tokenizer golden cases matched: %d/%zu\n", matched, cases.size());
  // Decode round-trip on a plain English case.
  const auto ids = cases[0]["ids"].get<std::vector<int>>();
  const std::string back = tok.decode(ids);
  check(back == cases[0]["text"].get<std::string>(), "decode round-trip English");
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path here =
      argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
  const std::filesystem::path golden = here / "golden";
  const char* env_img = std::getenv("UMM_OMNI_TEST_IMAGE");
  const std::string image_path = env_img ? env_img : "";

  test_smart_resize();
  test_preprocess_rgb_normalization();
  test_build_positions();
  test_preprocess_errors();
  test_image_golden(golden, image_path);
  test_tokenizer_synthetic(here);
  test_tokenizer_golden(golden);

  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all omni host checks passed\n");
  return 0;
}
