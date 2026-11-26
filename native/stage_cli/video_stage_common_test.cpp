// SPDX-License-Identifier: Apache-2.0
// Host contract test for stage_common video wire parsing and artifact shaping.
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "stage_common.hpp"
#include "umminfer/util/codec.hpp"

namespace {

void check(bool cond, const std::string& label) {
  if (!cond) {
    std::cerr << "FAIL: " << label << "\n";
    throw std::runtime_error(label);
  }
}

void test_parse_video_work_valid() {
  using namespace umminfer;
  Value::Map inputs;
  inputs["prompt"] = Value("  A paper boat floating on a calm pond  ");
  inputs["size"] = Value("704x480");
  inputs["seconds"] = Value(static_cast<int64_t>(2));
  inputs["fps"] = Value(static_cast<int64_t>(16));
  inputs["seed"] = Value(static_cast<int64_t>(20261122));
  inputs["output_format"] = Value("mp4");
  inputs["image"] = Value("data:image/png;base64,iVBORw0KGgo=");

  const stage::VideoWork w = stage::parse_video_work(Value(std::move(inputs)));
  check(w.prompt == "A paper boat floating on a calm pond", "prompt trimmed");
  check(w.width == 704 && w.height == 480, "aligned size parsed");
  check(w.seconds == 2 && w.fps == 16, "seconds and fps parsed");
  check(w.num_frames == 29, "num_frames aligned to 4*7+1=29");
  check(w.has_seed && w.seed == 20261122, "seed parsed");
  check(!w.teacache_thresh.has_value(), "omitted teacache_thresh stays unset");
  check(w.output_format == "mp4", "output_format == mp4");
  check(w.init_image_bytes.size() == 8 &&
            w.init_image_bytes.compare(0, 4, "\x89PNG") == 0,
        "data URL image decoded to raw bytes");

  for (double valid_tc : {0.0, 0.05, 0.25}) {
    Value::Map tc_map;
    tc_map["prompt"] = Value("fast waves");
    tc_map["teacache_thresh"] = Value(valid_tc);
    const stage::VideoWork tc_work =
        stage::parse_video_work(Value(std::move(tc_map)));
    check(tc_work.teacache_thresh.has_value() &&
              std::fabs(*tc_work.teacache_thresh - static_cast<float>(valid_tc)) < 1e-6f,
          "valid teacache_thresh parsed");
  }

  Value::Map raw;
  raw["prompt"] = Value("ok");
  raw["image"] = Value::bin(std::string("\x89PNG\x0d\x0a", 6));
  const stage::VideoWork raw_work = stage::parse_video_work(Value(std::move(raw)));
  check(raw_work.init_image_bytes.size() == 6,
        "binary image bytes forwarded unchanged");
}

void test_parse_video_work_anchor_binding() {
  using namespace umminfer;
  const char source_literal[] = "encoded-source-image\x00\xff";
  const std::string source(source_literal, sizeof(source_literal) - 1);
  const std::string digest = util::sha256_hex(source);
  Value::Map valid;
  valid["prompt"] = Value("animate the source");
  valid["image"] = Value::bin(source);
  valid["anchor_explicit"] = Value(true);
  valid["anchor_sha256"] = Value(digest);
  const stage::VideoWork accepted =
      stage::parse_video_work(Value(std::move(valid)));
  check(accepted.init_image_bytes == source,
        "exact source bytes accepted with matching anchor SHA-256");

  auto expect_rejected = [&](Value::Map inputs, const std::string& label) {
    bool threw = false;
    try {
      (void)stage::parse_video_work(Value(std::move(inputs)));
    } catch (const ProtocolError&) {
      threw = true;
    }
    check(threw, label);
  };
  Value::Map wrong;
  wrong["prompt"] = Value("animate the source");
  wrong["image"] = Value::bin(source);
  wrong["anchor_explicit"] = Value(true);
  wrong["anchor_sha256"] = Value(std::string(64, '0'));
  expect_rejected(std::move(wrong), "wrong anchor SHA-256 rejected");

  Value::Map tampered;
  tampered["prompt"] = Value("animate the source");
  tampered["image"] = Value::bin(source + "tampered");
  tampered["anchor_explicit"] = Value(true);
  tampered["anchor_sha256"] = Value(digest);
  expect_rejected(std::move(tampered), "tampered source rejected against frozen digest");

  Value::Map hash_without_image;
  hash_without_image["prompt"] = Value("animate the source");
  hash_without_image["anchor_sha256"] = Value(digest);
  expect_rejected(std::move(hash_without_image), "anchor SHA-256 without image rejected");

  Value::Map explicit_without_image;
  explicit_without_image["prompt"] = Value("animate the source");
  explicit_without_image["anchor_explicit"] = Value(true);
  expect_rejected(std::move(explicit_without_image),
                  "explicit anchor without image rejected");

  Value::Map malformed_flag;
  malformed_flag["prompt"] = Value("animate the source");
  malformed_flag["image"] = Value::bin(source);
  malformed_flag["anchor_explicit"] = Value("true");
  expect_rejected(std::move(malformed_flag), "non-boolean anchor_explicit rejected");
}

void test_parse_video_work_rejects_invalid() {
  using namespace umminfer;
  auto expect_protocol_error = [](Value::Map inputs, const std::string& label) {
    bool threw = false;
    try {
      (void)stage::parse_video_work(Value(std::move(inputs)));
    } catch (const ProtocolError&) {
      threw = true;
    }
    check(threw, label);
  };

  {
    Value::Map m;
    m["prompt"] = Value("   ");
    expect_protocol_error(std::move(m), "whitespace prompt rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["size"] = Value("500x480");
    expect_protocol_error(std::move(m), "unaligned 500x480 rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["output_format"] = Value("webm");
    expect_protocol_error(std::move(m), "non-mp4 output_format rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["num_frames"] = Value(static_cast<int64_t>(14));
    expect_protocol_error(std::move(m), "non-(4k+1) num_frames rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["image"] = Value("not-a-data-url");
    expect_protocol_error(std::move(m), "non-data-URL image rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["image"] = Value(static_cast<int64_t>(7));
    expect_protocol_error(std::move(m), "non-string image rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["teacache_thresh"] = Value(-0.01);
    expect_protocol_error(std::move(m), "negative teacache_thresh rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["teacache_thresh"] = Value(0.26);
    expect_protocol_error(std::move(m), "teacache_thresh > 0.25 rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["teacache_thresh"] = Value(std::numeric_limits<double>::quiet_NaN());
    expect_protocol_error(std::move(m), "NaN teacache_thresh rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["teacache_thresh"] = Value("0.05");
    expect_protocol_error(std::move(m), "string teacache_thresh rejected");
  }
  {
    Value::Map m;
    m["prompt"] = Value("ok");
    m["teacache_thresh"] = Value(true);
    expect_protocol_error(std::move(m), "bool teacache_thresh rejected");
  }
}

void test_video_result_artifact_shape() {
  using namespace umminfer;
  std::string fake_mp4("\x00\x00\x00\x18" "ftypisom\x00\x00\x02\x00", 16);
  const Value res = stage::video_result(
      fake_mp4, 480, 320, 16, 13, 0.8125, 42, 4, "Wan2.2-TI2V-5B", "UNKNOWN",
      "native-cpp-contract-stub", 12.5, 45.0);
  check(res.is_map(), "video_result returns map");
  check(res.find("media_type")->as_str() == "video", "media_type == video");
  check(res.find("mime_type")->as_str() == "video/mp4", "mime_type == video/mp4");
  check(res.find("format")->as_str() == "mp4", "format == mp4");
  check(res.find("size")->as_str() == "480x320", "size == 480x320");
  check(res.find("data")->is_bin() && res.find("data")->as_bin() == fake_mp4,
        "data bin preserved");
  check(res.find("backend")->as_str() == "native-cpp-contract-stub",
        "backend preserved");

  bool bad_mp4_rejected = false;
  try {
    (void)stage::video_result("not-an-mp4", 480, 320, 16, 13, 0.8, 0, 4, "m",
                              "r", "b");
  } catch (const ProtocolError&) {
    bad_mp4_rejected = true;
  }
  check(bad_mp4_rejected, "non-ftyp payload rejected");
}

}  // namespace

int main() {
  test_parse_video_work_valid();
  test_parse_video_work_anchor_binding();
  test_parse_video_work_rejects_invalid();
  test_video_result_artifact_shape();
  std::cout << "stage_video_common_test: 4 passed, 0 failed\n";
  return 0;
}
