// SPDX-License-Identifier: Apache-2.0
// Host contract test for stage_common video wire parsing and artifact shaping.
#include <iostream>
#include <stdexcept>
#include <string>

#include "stage_common.hpp"

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

  const stage::VideoWork w = stage::parse_video_work(Value(std::move(inputs)));
  check(w.prompt == "A paper boat floating on a calm pond", "prompt trimmed");
  check(w.width == 704 && w.height == 480, "aligned size parsed");
  check(w.seconds == 2 && w.fps == 16, "seconds and fps parsed");
  check(w.num_frames == 29, "num_frames aligned to 4*7+1=29");
  check(w.has_seed && w.seed == 20261122, "seed parsed");
  check(w.output_format == "mp4", "output_format == mp4");
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
  test_parse_video_work_rejects_invalid();
  test_video_result_artifact_shape();
  std::cout << "stage_video_common_test: 3 passed, 0 failed\n";
  return 0;
}
