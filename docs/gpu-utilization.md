# GPU utilization of the native executors — H20B, 2026-11-17

Baseline commit `31cf1fd`. Measurements and changes were made on GPUs 6 and 7
of the shared H20 host; GPUs 0/1/3/4/5 were in use by other jobs and were not
touched. Every run used for a reported number was gated on the GPU being
exclusive (no other compute process resident).

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H20, 97871 MiB |
| Driver / container | 535.161.08 / H20B GPU container |
| Toolchain | CUDA 12.4 (`nvcc`), CMake 3.31, arch `sm_90` |
| SMs / clock / L2 | 78 / 1.98 GHz / 60 MiB (`cudaDeviceGetAttribute`) |
| Profilers | Nsight Systems 2023.4 (works); Nsight Compute 13.0 (unusable) |
| Models | `qwen3.8-27b`, `qwen3-omni` (30B-A3B), `qwen-image-2.1`, `qwen3-tts` |

### Measurement limits

Per-kernel SM occupancy and DRAM-throughput percentages need Nsight Compute.
`ncu` is installed in the container but rejects every run:

```
==ERROR== Cuda driver is not compatible with Nsight Compute.
==ERROR== ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA
          GPU Performance Counters on the target device 0.
```

`nsys --gpu-metrics-device` is the other route to SM activity. It also fails,
because the sampler does not support this GPU:

```
Possible --gpu-metrics-device values are:
Some GPUs are not supported: NVIDIA H20 PCI[0000:65:02.0] ...
```

So this document reports GPU busy fraction, idle gaps and launch counts from
the CUDA kernel trace, plus achieved bandwidth. **Measured SM occupancy is
absent from every table.** What is given instead is a theoretical,
register-limited occupancy computed from `cuobjdump -res-usage` and the device
limits (65,536 registers/SM, 2,048 threads/SM, 32 blocks/SM):

| Kernel | Registers/thread | Block | Register-limited warps/SM | Theoretical occupancy (not measured) |
| --- | ---: | ---: | ---: | ---: |
| qwen38 `gemv_kernel<bf16>` | 32 | 128 | 64 | 100% |
| qwen38 `rms_norm_kernel` | 18 | 256 | 64 (capped) | 100% |
| qwen38 `attention_kernel` | 31 | 128 | 64 (capped) | 100% |
| qwen38 `gated_delta_rule_kernel` | 40 | 128 | 51 | 80% |
| qwen38 `gemm_mma_kernel` (prefill) | 128 | 256 | 16 | 25% |
| omni `moe_route_kernel` | 35 | 128 | 58 | 91% |
| omni `gqa_attn_kernel` | 29 | 128 | 64 (capped) | 100% |
| omni `gemv_grouped_kernel` | 32 | 128 | 64 (capped) | 100% |
| tts `gemm_kernel<bf16>` | 24 | 256 | 64 (capped) | 100% |

Achieving measured occupancy needs either a driver that matches Nsight Compute
13 or a container that grants `ERR_NVGPUCTRPERM`. That is an operator decision.

### Measured streaming ceiling

A micro-benchmark that reads 100 MiB weight matrices with the same access shape
as the decode GEMV sustained:

| Pattern | Bandwidth |
| --- | ---: |
| one warp per row, 16-byte lane loads | 3.24–3.28 TB/s |
| flat grid-stride `uint4` read | 3.12 TB/s |

The usable read bandwidth on this H20 is about **3.3 TB/s**, not the 4.0 TB/s
paper number. Percentages of peak below use 3.3 TB/s.

## Baseline

All runs on GPU 6, exclusive.

| Engine | Request | Latency | GPU busy | Idle | Kernels | Launches/token | Peak memory | Power mean / p95 / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B decode | 27 tokens, text | 660.9 ms (24.48 ms/token, 40.8 tok/s) | 93.7% | 1.54 ms/token | 33,480 | 1,240 | 55,769 MiB | 136.8 / 286.2 / 357.7 W |
| Qwen3-Omni decode | 32 tokens, text | 578.1 ms (18.06 ms/token) | 56.9% | 7.78 ms/token | 58,624 | 1,832 | 78,751 MiB | 124.7 / 151.8 / 186.1 W |
| Qwen-Image-2.1 | 512×512, 40 steps | 68.05 s (denoise 59.26 s) | 98.1% | 1.3 s | 25,678 | 642/step | 14,053 MiB | 328.5 / 356.2 / 403.8 W |
| Qwen3-TTS | 4.24 s audio | 29.1 s | 93.5% | 1.64 s | 113,344 | — | 4,707 MiB | 289.0 / 391.1 / 400.3 W |

Power is the whole-process window sampled at 5 Hz with `nvidia-smi power.draw`
(mean, p95, max). For Qwen3.8 and Omni that window includes the 9–17 s weight
load, so the mean is below the inference draw; p95 and max show it.

### Qwen3.8-27B — top 10 kernels (27 decode steps)

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `gemv_kernel<bf16>` | 462.97 ms | 60.0% | 13,392 | 34.6 µs |
| 2 | `gemm_mma_kernel` (prefill only) | 140.53 ms | 18.2% | 496 | 283.3 µs |
| 3 | `rms_norm_kernel<zero-centred>` | 49.20 ms | 6.4% | 4,508 | 10.9 µs |
| 4 | `gated_delta_rule_kernel` | 36.18 ms | 4.7% | 1,344 | 26.9 µs |
| 5 | `attention_kernel` | 31.27 ms | 4.1% | 448 | 69.8 µs |
| 6 | `gemv_kernel<float>` (lm_head) | 18.71 ms | 2.4% | 28 | 668.4 µs |
| 7 | `residual_kernel` | 7.49 ms | 1.0% | 3,584 | 2.09 µs |
| 8 | `silu_mul_kernel` | 4.91 ms | 0.6% | 1,792 | 2.74 µs |
| 9 | `conv1d_causal_silu_kernel` | 4.26 ms | 0.6% | 1,344 | 3.17 µs |
| 10 | `beta_decay_kernel` | 3.19 ms | 0.4% | 1,344 | 2.37 µs |

Decode reads about 51.2 GB of text-decoder weights per token. The GEMV kernels
take 17.15 ms per token: **2.98 TB/s, 91% of the measured 3.28 TB/s ceiling**.
The CPU is not the limiter — 1,240 launches × 3.2 µs median = 4.0 ms per token
against 24.5 ms of GPU work.

### Qwen3-Omni — top 10 kernels (33 steps)

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `moe_route_kernel` | 80.75 ms | 22.0% | 1,584 | 51.0 µs |
| 2 | `gqa_attn_kernel` | 66.29 ms | 18.0% | 1,584 | 41.8 µs |
| 3 | `gemv_kernel<bf16>` | 38.36 ms | 10.4% | 6,144 | 6.2 µs |
| 4 | `gemv_grouped_kernel` | 37.34 ms | 10.2% | 3,072 | 12.2 µs |
| 5 | `rms_norm_kernel` | 25.39 ms | 6.9% | 6,369 | 4.0 µs |
| 6 | `moe_gemm_kernel<true>` | 24.72 ms | 6.7% | 96 | 257.5 µs |
| 7 | `gemv_grouped_f32act_kernel` | 20.09 ms | 5.5% | 1,536 | 13.1 µs |
| 8 | `moe_down_kernel` | 17.33 ms | 4.7% | 48 | 361.0 µs |
| 9 | `gemm_mma_kernel` (prefill) | 17.23 ms | 4.7% | 192 | 89.7 µs |
| 10 | `gemv_kernel<float>` | 12.24 ms | 3.3% | 1,569 | 7.8 µs |

### Qwen-Image-2.1 — top 10 kernels

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `flash_attn_k<128,16>` | 51.24 s | 83.6% | 2,596 | 19.7 ms |
| 2 | `cutlass bf16 tensorop GEMM` | 7.36 s | 12.0% | 8,960 | 0.82 ms |
| 3 | `conv3x3_k` | 1.88 s | 3.1% | 40 | 46.9 ms |
| 4 | `conv1x1_k` | 0.31 s | 0.5% | 6 | 52.1 ms |
| 5 | `rms_norm_lastdim_k` | 0.15 s | 0.3% | 2,704 | 57.0 µs |
| 6 | `silu_mul_k` | 0.09 s | 0.1% | 1,316 | 69.8 µs |
| 7 | `ln_mod_select_k` | 0.06 s | 0.1% | 2,560 | 21.8 µs |
| 8 | `rope_complex_k` | 0.06 s | 0.1% | 2,560 | 21.8 µs |
| 9 | `gated_tanh_k` | 0.05 s | 0.1% | 2,560 | 21.1 µs |
| 10 | `rms_nchw_k` | 0.02 s | 0.0% | 36 | 476.1 µs |

`flash_attn_k` is bimodal (median 32 µs, maximum 40 ms); the full-resolution
self-attention calls dominate the mean.

### Qwen3-TTS — top 10 kernels

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `conv1d_kernel` | 19.80 s | 84.6% | 31 | 639 ms |
| 2 | `transconv_kernel` | 2.22 s | 9.5% | 6 | 370 ms |
| 3 | `gemm_kernel<bf16>` | 0.80 s | 3.4% | 38,431 | 20.8 µs |
| 4 | `snake_kernel` | 0.24 s | 1.0% | 29 | 8.4 ms |
| 5 | `gqa_kernel<bf16>` | 0.17 s | 0.7% | 5,487 | 31.4 µs |
| 6 | `rms_kernel<bf16>` | 0.05 s | 0.2% | 22,797 | 2.1 µs |
| 7 | `gemm_kernel<float>` | 0.03 s | 0.1% | 62 | 553.8 µs |
| 8 | `rope_kernel<bf16>` | 0.03 s | 0.1% | 10,974 | 2.4 µs |
| 9 | `residual_kernel<bf16>` | 0.02 s | 0.1% | 11,837 | 1.5 µs |
| 10 | `gemv_kernel<bf16>` | 0.02 s | 0.1% | 849 | 19.8 µs |

## Changes

Three changes. Qwen3.8 and Qwen-Image are untouched.

### 1. Qwen3-Omni: resolve expert pointers once

`text_layer` resolved 3 × 128 expert weight pointers per layer per token through
`std::string` concatenation and an `unordered_map` lookup, and allocated two
vectors for the route staging. `EngineWeights::cache_expert_pointers()` resolves
all 48 × 128 × 3 pointers once after load; the staging buffers moved to
`EngineScratch`.

### 2. Qwen3-Omni: compile-time top-k in `moe_route_kernel`

`moe_route_kernel` took `top_k` as a runtime argument. The compiler therefore
kept `vals[8]` and `idxs[8]` in local memory, and the 128 × 8 insertion scan cost
~51 µs per call — 22% of Omni GPU time. The kernel is now templated on `TK`, and
the launcher requires `top_k == 8`. The same experts are visited in the same
order with the same strict `>` comparison, so ties keep the lower index.

Measured in isolation (2,000 launches, one random 128-expert logit row):
**51.56 → 26.26 µs per call (1.96×)**, with `top_idx` identical and `top_w`
bitwise identical.

### 3. Qwen3-TTS: one thread per output element in the codec convolutions

`conv1d_kernel` and `transconv_kernel` were launched with one block per output
element, but the body has no thread indexing: all 256 threads of the block ran
the same serial reduction and wrote the same value. Both now take a flat index,
one thread per output element, with the same loop nest and accumulation order.

## Before and after

### Qwen3-Omni (64 generated tokens, text prompt, GPU 6)

| Metric | Baseline | After both changes |
| --- | ---: | ---: |
| `infer_ms` | 1,214.7 (1211/1221/1212) | 899.3 (923/895/880) |
| Change | — | **1.35× faster** |
| Per token | 18.06 ms | 13.50 ms |
| Busy time per token | 10.28 ms | 9.10 ms |
| Idle per token | 7.78 ms | 4.39 ms |
| GPU busy | 56.9% | **67.4%** |
| Gaps > 20 µs | 1,542 | 307 |
| Peak memory | 78,751 MiB | 78,751 MiB |
| Power mean / p95 / max | 124.7 / 151.8 / 186.1 W | 123.0 / 156.9 / 189.2 W |
| Generated text | — | byte-identical |

The host-path change alone measured 984.3 ms `infer_ms`; the top-k change
brought it to 899.3 ms.

### Qwen3-TTS (same text, GPU 6)

| Metric | Baseline | After |
| --- | ---: | ---: |
| Wall clock | 29.1 s (29210/29017 ms) | 7.2 s (7213/7267 ms) |
| Change | — | **4.0× faster** |
| `conv1d_kernel` total | 19.80 s | 0.116 s (170×) |
| Kernel-busy time | 23.70 s | 1.82 s |
| Absolute idle | 1.64 s | 1.69 s |
| GPU busy | 93.5% | 51.9% |
| Peak memory | 4,707 MiB | 4,707 MiB |
| Power mean / p95 / max | 289.0 / 391.1 / 400.3 W | 153.7 / 262.0 / 269.6 W |
| Process energy (mean × wall) | 8.4 kJ | 1.1 kJ |

**The TTS busy percentage falls, and that is not an improvement claim.** It is
busy/span. The span fell 25.34 → 3.51 s and the busy time fell 23.70 → 1.82 s,
while absolute idle stayed at 1.6–1.7 s. The engine is now host-bound: 1.69 s of
the 3.51 s trace has no kernel resident, because the autoregressive talker loop
does 16 host-synchronised `argmax` calls and 15 host-side embedding gathers per
audio frame. Raising the busy percentage from here means cutting host latency,
not GPU work. Throughput and energy improved; the utilization ratio did not.

### Qwen3.8-27B and Qwen-Image-2.1

Not modified; their files are absent from this commit. Qwen3.8 measures
41.2–41.5 tok/s before and after, with byte-identical `gen_ids.i32`.

## H20 hardware features

| Feature | Status | Evidence |
| --- | --- | --- |
| BF16 tensor cores | **Used** for prefill and image2 | `HMMA.16816.F32` in SASS: 128 sites in `umm-qwen38-native`, 192 in `umm-omni-native`; image2 links `cutlass_80_tensorop_bf16_s16816gemm` |
| BF16 tensor cores in decode | **Not used, deliberately** | Decode is `m = 1`, memory-bound at 2.98 TB/s = 91% of the measured ceiling; tensor cores cannot raise a bandwidth-bound kernel |
| L2 persistence | **Not used** | L2 is 60 MiB; one decode token streams 51.2 GB of weights. No reuse at L2 scale. A 2-way prefetch with `__ldcs` hints measured 41.4 → 37.8 tok/s and was reverted |
| NVLink | **Not applicable** | Every engine is one process on one GPU. The host reports 5 NVLink links per GPU; no measured workload moves data between GPUs |

## Not done, and why

* **CUDA Graph over the Qwen3.8 decode loop.** Ceiling is 6.3%: the GPU is
  already 93.7% busy and the CPU keeps the queue ahead. A graph would recover
  part of the 1.54 ms/token of inter-kernel gaps, but needs per-token kernel
  arguments (KV row offsets, `history`) moved to device memory.
* **CUDA Graph over the Omni decode loop.** The idle was a per-layer host
  synchronisation, not launch latency. Removing the host work cut idle from 7.78
  to 4.39 ms/token; a graph now buys little.
* **Continuous batching or multi-stream concurrency.** Qwen3.8 is 93.7% busy and
  91% of the streaming ceiling; Image2 is 98.1% busy. A second stream cannot
  raise aggregate throughput on a bandwidth-saturated kernel. Omni and TTS have
  headroom, but batching needs per-sequence state (`rec_state`, `conv_state`,
  MoE workspaces are single-sequence today), which is a larger refactor.
* **Qwen3.8 GEMV kernel tuning.** A 2-way software-pipelined load loop, with and
  without `__ldcs`, measured 37.2–37.8 tok/s against 41.4 tok/s. Reverted; the
  kernel is within 9% of the measured streaming rate.
* **TTS host loop.** 1.69 s of absolute idle remains in the talker loop. It is
  the next target, but it is a sequencing change in the frame loop, not a
  launch-overhead fix, and it needs its own parity gate.

## Reproduction

```bash
# Inside the H20B GPU container, from <repo-root>; set UMM_QWEN38_MODEL to your local weights path
cmake -S native -B build-q38 -DUMM_QWEN38_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/omni -B build-omni -DUMM_OMNI_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/image2 -B build-img2 -DUMM_IMAGE2_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/tts -B build-tts -DUMM_TTS_CUDA=ON -DCMAKE_BUILD_TYPE=Release

CUDA_VISIBLE_DEVICES=6 nsys profile -o prof/q38 --trace=cuda \
  build-q38/umm-qwen38-native --model "${UMM_QWEN38_MODEL:-/opt/umminfer/models/qwen3.8-27b}" \
  --prompt "用一句话介绍你自己。" --gpu 0 --max-new-tokens 128
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace prof/q38.nsys-rep
```
