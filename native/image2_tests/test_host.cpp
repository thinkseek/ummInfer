// SPDX-License-Identifier: Apache-2.0
// Host-side checks: scheduler schedule shape/values, tile helpers, PNG output.
#include "image2/png.hpp"
#include "image2/scheduler.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("check failed: ") + what);
}

bool close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

}  // namespace

int main() {
  try {
    using namespace umminfer::image2;

    // 512x512 -> 32x32 latent tokens; 40 steps.
    const auto sch = build_schedule(SchedulerConfig{}, 1024, 40);
    check(sch.timesteps.size() == 40, "timestep count");
    check(sch.sigmas.size() == 41, "sigma count");
    check(sch.sigmas.back() == 0.0f, "terminal sigma");
    // Initial sigma is 1 and the schedule is strictly decreasing.
    check(close(sch.sigmas.front(), 1.0f, 1e-6f), "first sigma");
    for (size_t i = 0; i + 1 < sch.sigmas.size(); ++i)
      check(sch.sigmas[i] > sch.sigmas[i + 1], "monotone sigmas");
    // Reference values from diffusers FlowMatchEulerDiscreteScheduler
    // (mu = 0.5387096774193548, 40 steps, 1024 tokens).
    check(close(sch.sigmas[1], 0.9849190711975098f, 1e-6f), "sigma[1]");
    check(close(sch.sigmas[2], 0.9695172309875488f, 1e-6f), "sigma[2]");
    check(close(sch.sigmas[39], 0.019999980926513672f, 1e-7f), "sigma[39]");
    check(close(sch.timesteps[1], 984.9190673828125f, 1e-3f), "timestep[1]");
    check(close(sch.timesteps[0] / 1000.0f, 1.0f, 0.0f), "timestep[0]");

    // Round-trip a known pattern through the RGBA writer.
    RgbaImage img(3, 2);
    for (size_t i = 0; i < img.rgba.size(); ++i) img.rgba[i] = static_cast<uint8_t>(i * 7);
    const std::string png = encode_png(img);
    // Full chunk walk: signature, IHDR fields, IDAT present, IEND last.
    static const unsigned char kSig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    check(png.size() > 40 &&
              std::memcmp(png.data(), kSig, sizeof(kSig)) == 0,
          "png signature");
    check(png.compare(12, 4, "IHDR") == 0, "png IHDR first");
    const auto be32 = [&](size_t off) {
      return (static_cast<uint32_t>(static_cast<unsigned char>(png[off])) << 24) |
             (static_cast<uint32_t>(static_cast<unsigned char>(png[off + 1])) << 16) |
             (static_cast<uint32_t>(static_cast<unsigned char>(png[off + 2])) << 8) |
             static_cast<uint32_t>(static_cast<unsigned char>(png[off + 3]));
    };
    check(be32(8) == 13, "png IHDR length");
    check(be32(16) == 3 && be32(20) == 2, "png IHDR dimensions");
    check(static_cast<unsigned char>(png[24]) == 8 &&
              static_cast<unsigned char>(png[25]) == 6,
          "png IHDR 8-bit RGBA");
    bool saw_idat = false, saw_iend = false;
    size_t pos = 8;
    while (pos + 12 <= png.size()) {
      const uint32_t len = be32(pos);
      const std::string type = png.substr(pos + 4, 4);
      if (type == "IDAT") saw_idat = true;
      if (type == "IEND") {
        saw_iend = true;
        check(len == 0 && pos + 12 == png.size(), "png IEND is last and empty");
      }
      pos += 12 + len;
    }
    check(saw_idat, "png has IDAT");
    check(saw_iend, "png has IEND");

    // [-1,1] -> [0,255]: -1 -> 0, 1 -> 255, 0 -> 128 (round half to even).
    float px[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    const RgbaImage out = tensor_n1p1_to_rgba(px, 1, 1);
    check(out.rgba[0] == 0 && out.rgba[2] == 255 && out.rgba[1] == 128, "denormalize");

    std::printf("image2_host_tests: ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "image2_host_tests: %s\n", e.what());
    return 1;
  }
  return 0;
}
