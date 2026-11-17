// SPDX-License-Identifier: Apache-2.0
// Stage parity against diffusers reference dumps.
//
//   image2_gpu_tests MODEL_DIR GOLDEN_DIR
//
// Golden files (all little-endian):
//   ids.bin    int32 n, int32 ids[n]
//   drop.bin   int32 drop
//   te.bin     int32 rows, int32 dim, bf16 data   (post-drop embeddings)
//   dit_in.bin int32 txt_len, int32 gh, int32 gw, bf16 lat[gh*gw*64], bf16 txt[txt_len*4096]
//   dit0.bin   int32 gh, int32 gw, bf16 velocity[gh*gw*64]
//   vae_in.bin int32 h, int32 w, bf16 nchw[64*h*w] (de-normalized)
//   vae_out.bin int32 H, int32 W, bf16 nchw[4*H*W]
//
// Any missing pair prints skip for that stage. Exit code is non-zero when a
// stage that ran misses its gate.
#include "image2/dit.hpp"
#include "image2/text_encoder.hpp"
#include "image2/tensor.hpp"
#include "image2/vae.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace umminfer::image2;

namespace {

bool read_raw(const std::string& path, void* dst, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}

struct BfBlob {
  std::vector<int32_t> dims;
  std::vector<uint16_t> data;
  bool ok = false;
};

BfBlob read_bf(const std::string& path, size_t ndims) {
  BfBlob b;
  std::ifstream f(path, std::ios::binary);
  if (!f) return b;
  b.dims.resize(ndims);
  f.read(reinterpret_cast<char*>(b.dims.data()), ndims * sizeof(int32_t));
  size_t n = 1;
  for (int32_t d : b.dims) n *= static_cast<size_t>(d);
  b.data.resize(n);
  f.read(reinterpret_cast<char*>(b.data.data()), static_cast<std::streamsize>(n * 2));
  b.ok = static_cast<bool>(f);
  return b;
}

double cosine(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = bits_to_bf16_float(a[i]), y = bits_to_bf16_float(b[i]);
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / std::sqrt(na * nb + 1e-30);
}

double rel_l2(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = bits_to_bf16_float(a[i]), y = bits_to_bf16_float(b[i]);
    num += (x - y) * (x - y);
    den += y * y;
  }
  return std::sqrt(num / (den + 1e-30));
}

double psnr(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double mse = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = bits_to_bf16_float(a[i]) - bits_to_bf16_float(b[i]);
    mse += d * d;
  }
  mse /= static_cast<double>(a.size());
  return 10.0 * std::log10(4.0 / (mse + 1e-30));  // data range [-1, 1]
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: image2_gpu_tests MODEL_DIR GOLDEN_DIR\n");
    return 2;
  }
  const std::string model = argv[1];
  const std::string golden = argv[2];
  const std::string only = argc > 3 ? argv[3] : "";
  int failures = 0;
  try {
    CudaRuntime rt(0);

    // ---- text encoder ------------------------------------------------------
    int32_t n_ids = 0, drop = 0;
    std::vector<int32_t> ids;
    BfBlob te = read_bf(golden + "/te.bin", 2);
    if ((only.empty() || only == "te") && read_raw(golden + "/ids.bin", &n_ids, 4) && n_ids > 0 &&
        read_raw(golden + "/drop.bin", &drop, 4) && te.ok) {
      ids.resize(n_ids);
      std::ifstream fi(golden + "/ids.bin", std::ios::binary);
      fi.seekg(4);
      fi.read(reinterpret_cast<char*>(ids.data()), n_ids * 4);
      TextEncoder enc(rt, model + "/text_encoder");
      DeviceTensor hidden = enc.forward(std::vector<int>(ids.begin(), ids.end()));
      const int rows = static_cast<int>(te.dims[0]), dim = static_cast<int>(te.dims[1]);
      auto h = hidden.to_host_u16();
      std::vector<uint16_t> sliced(h.begin() + static_cast<size_t>(drop) * dim,
                                   h.begin() + static_cast<size_t>(drop + rows) * dim);
      const double c = cosine(sliced, te.data), r = rel_l2(sliced, te.data);
      std::printf("TE: cosine %.6f rel_l2 %.6f\n", c, r);
      if (c < 0.999) { std::printf("TE: FAIL\n"); ++failures; }
    } else {
      std::printf("TE: skip (no golden)\n");
    }

    // ---- single DiT step ---------------------------------------------------
    BfBlob d0 = read_bf(golden + "/dit0.bin", 3);
    std::vector<uint16_t> din_lat, din_txt;
    int32_t din_dims[3] = {0, 0, 0};
    bool have_din = false;
    {
      std::ifstream f(golden + "/dit_in.bin", std::ios::binary);
      if (f) {
        f.read(reinterpret_cast<char*>(din_dims), sizeof(din_dims));
        const int txt_len = din_dims[0], gh = din_dims[1], gw = din_dims[2];
        if (txt_len > 0 && gh > 0 && gw > 0) {
          din_lat.resize(static_cast<size_t>(gh) * gw * 64);
          din_txt.resize(static_cast<size_t>(txt_len) * 4096);
          f.read(reinterpret_cast<char*>(din_lat.data()),
                 static_cast<std::streamsize>(din_lat.size() * 2));
          f.read(reinterpret_cast<char*>(din_txt.data()),
                 static_cast<std::streamsize>(din_txt.size() * 2));
          have_din = static_cast<bool>(f);
        }
      }
    }
    if ((only.empty() || only == "dit") && have_din && d0.ok) {
      const int txt_len = din_dims[0], gh = din_dims[1], gw = din_dims[2];
      DeviceTensor lat({gh * gw, 64});
      lat.copy_from_host(din_lat.data());
      DeviceTensor txt({txt_len, 4096});
      txt.copy_from_host(din_txt.data());
      DiT dit(rt, model + "/transformer");
      // Reference timeline: step 0 timestep of the 40-step schedule.
      const float t0 = 1.0f;  // sigma_0 = 1 -> timestep 1000 -> scaled 1.0
      DeviceTensor vel = dit.forward(lat, txt, t0, gh, gw, txt_len);
      auto v = vel.to_host_u16();
      const double c = cosine(v, d0.data), r = rel_l2(v, d0.data);
      std::printf("DiT step0: cosine %.6f rel_l2 %.6f\n", c, r);
      if (c < 0.999) { std::printf("DiT: FAIL\n"); ++failures; }
    } else {
      std::printf("DiT: skip (no golden)\n");
    }

    // ---- VAE decode --------------------------------------------------------
    BfBlob vin = read_bf(golden + "/vae_in.bin", 3);
    BfBlob vout = read_bf(golden + "/vae_out.bin", 3);
    if ((only.empty() || only == "vae") && vin.ok && vout.ok) {
      const int h = vin.dims[0], w = vin.dims[1];
      // dims are [h, w, 64] / [H, W, 4]; payload matches the product.
      DeviceTensor lat({64, h, w});
      lat.copy_from_host(vin.data.data());
      VaeDecoder vae(rt, model + "/vae");
      DeviceTensor out = vae.decode(lat, h, w);
      auto o = out.to_host_u16();
      double mo = 0, so = 0, mg = 0, sg = 0;
      for (size_t i = 0; i < o.size(); ++i) {
        const double a = bits_to_bf16_float(o[i]), b = bits_to_bf16_float(vout.data[i]);
        mo += a; mg += b; so += a * a; sg += b * b;
      }
      mo /= o.size(); mg /= o.size();
      so = std::sqrt(so / o.size() - mo * mo);
      sg = std::sqrt(sg / o.size() - mg * mg);
      const double r = rel_l2(o, vout.data), p = psnr(o, vout.data);
      std::printf("VAE: rel_l2 %.6f psnr %.2f dB  ours mean/std %.4f/%.4f  ref %.4f/%.4f\n",
                  r, p, mo, so, mg, sg);
      if (r > 0.06) { std::printf("VAE: FAIL\n"); ++failures; }
    } else {
      std::printf("VAE: skip (no golden)\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "image2_gpu_tests: %s\n", e.what());
    return 1;
  }
  std::printf(failures ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", failures);
  return failures ? 1 : 0;
}
