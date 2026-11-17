# GPU utilization of the native executors — H20B, 2026-11-17

Baseline commit `31cf1fd`. Measurements and changes in this document were made
on GPUs 6 and 7 of the shared H20 host; GPUs 0/1/3/4/5 were in use by other
jobs and were not touched.

The goal was to raise GPU utilization. Every number below comes from a recorded
run. Nothing in this document is estimated from a paper spec.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H20, 97871 MiB |
| Driver / container | 535.161.08 / `linyu-dev` |
| Toolchain | CUDA 12.4 (`nvcc`), CMake 3.31, arch `sm_90` |
| SMs / clock / L2 | 78 / 1.98 GHz / 60 MiB (`cudaDeviceGetAttribute`) |
| Profilers | Nsight Systems 2023.4 (works); Nsight Compute 13.0 (unusable, see below) |
| Models | `qwen3.8-27b`, `qwen3-omni` (30B-A3B), `qwen-image-2.1`, `qwen3-tts` |

### What could not be measured, and why

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

So this document reports:

* GPU busy fraction and idle gaps from the CUDA kernel trace (Nsight Systems).
* Achieved memory bandwidth, computed from bytes read and kernel time.
* The same bandwidth figure expressed against a **measured** streaming ceiling
  on this card, not against the 4.0 TB/s paper number.

SM occupancy as a measured per-kernel percentage is absent. What the binaries
do allow is a **theoretical, register-limited occupancy**, computed from
`cuobjdump -res-usage` and the device limits (65,536 registers/SM, 2,048
threads/SM, 32 blocks/SM, 233,472 B shared/SM):

| Kernel | Registers/thread | Block | Register-limited warps/SM | Theoretical occupancy |
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

The 25% figure for the prefill tensor-core GEMM is a deliberate trade: it holds
128 registers per thread and a 72 KiB shared tile. Decode kernels are at or
near full occupancy, which is consistent with the measured 91% of streaming
bandwidth.

Measured occupancy still needs Nsight Compute. Enabling it needs either a
driver that matches Nsight Compute 13 or a container that grants
`ERR_NVGPUCTRPERM`. That is an operator decision, not a code change.

### Measured streaming ceiling

A micro-benchmark that reads 100 MiB weight matrices with the same access shape
as the decode GEMV sustained:

| Pattern | Bandwidth |
| --- | ---: |
| one warp per row, 16-byte lane loads | 3.24–3.28 TB/s |
| flat grid-stride `uint4` read | 3.12 TB/s |

The usable read bandwidth on this H20 is therefore about **3.3 TB/s**, not
4.0 TB/s. All "percentage of peak" figures below use 3.3 TB/s.

## Baseline

All runs on GPU 6, exclusive (the harness waits until no other compute process
holds the GPU). Latency is the engine's own decode/infer timer.

| Engine | Request | Latency | GPU busy | Idle | Kernels | Launches per token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B decode | 27 tokens, text | 660.9 ms decode (24.48 ms/token, 40.8 tok/s) | 93.7% | 1.54 ms/token | 33,480 | 1,240 |
| Qwen3-Omni decode | 32 tokens, text | 578.1 ms (18.06 ms/token) | 56.9% | 7.78 ms/token | 58,624 | 1,832 |
| Qwen-Image-2.1 | 512×512, 40 steps | 68.05 s (denoise 59.26 s) | 98.1% | 1.3 s total | 25,678 | 642 per step |
| Qwen3-TTS | 4.24 s audio | 29.1 s wall (trace span 25.34 s) | 93.5% | 1.64 s total | 113,344 | — |

"Idle" is the sum of gaps between consecutive kernels on the stream. It counts
time when the GPU has no kernel resident.

### Qwen3.8-27B, top kernels (27 decode steps)

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `gemv_kernel<bf16>` | 462.97 ms | 60.0% | 13,392 | 34.6 µs |
| 2 | `gemm_mma_kernel` (prefill only) | 140.53 ms | 18.2% | 496 | 283.3 µs |
| 3 | `rms_norm_kernel<zero-centred>` | 49.20 ms | 6.4% | 4,508 | 10.9 µs |
| 4 | `gated_delta_rule_kernel` | 36.18 ms | 4.7% | 1,344 | 26.9 µs |
| 5 | `attention_kernel` (full attn) | 31.27 ms | 4.1% | 448 | 69.8 µs |
| 6 | `gemv_kernel<float>` (lm_head) | 18.71 ms | 2.4% | 28 | 668.4 µs |
| 7 | `residual_kernel` | 7.49 ms | 1.0% | 3,584 | 2.09 µs |
| 8 | `silu_mul_kernel` | 4.91 ms | 0.6% | 1,792 | 2.74 µs |
| 9 | `conv1d_causal_silu_kernel` | 4.26 ms | 0.6% | 1,344 | 3.17 µs |
| 10 | `beta_decay_kernel` | 3.19 ms | 0.4% | 1,344 | 2.37 µs |

Decode reads about 51.2 GB of text-decoder weights per token. The GEMV kernels
take 17.15 ms per token, which is **2.98 TB/s — 91% of the measured 3.28 TB/s
ceiling**. The remaining 7 ms per token is the non-GEMV work listed above.

The CPU is not the limiter: 1,240 launches × 3.2 µs median = 4.0 ms per token,
against 24.5 ms of GPU work.

### Qwen3-Omni, top kernels (33 steps)

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `moe_route_kernel` | 80.75 ms | 22.0% | 1,584 | 51.0 µs |
| 2 | `gqa_attn_kernel` | 66.29 ms | 18.0% | 1,584 | 41.8 µs |
| 3 | `gemv_kernel<bf16>` | 38.36 ms | 10.4% | 6,144 | 6.2 µs |
| 4 | `gemv_grouped_kernel` | 37.34 ms | 10.2% | 3,072 | 12.2 µs |
| 5 | `rms_norm_kernel` | 25.39 ms | 6.9% | 6,369 | 4.0 µs |
| 6 | `moe_gemm_kernel` | 24.72 ms | 6.7% | 96 | 257.5 µs |

### Qwen-Image-2.1, top kernels

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `flash_attn_k<128,16>` | 51.24 s | 83.6% | 2,596 | 19.7 ms |
| 2 | `cutlass bf16 tensorop GEMM` | 7.36 s | 12.0% | 8,960 | 0.82 ms |
| 3 | `conv3x3_k` | 1.88 s | 3.1% | 40 | 46.9 ms |
| 4 | `conv1x1_k` | 0.31 s | 0.5% | 6 | 52.1 ms |
| 5 | `rms_norm_lastdim_k` | 0.15 s | 0.3% | 2,704 | 57.0 µs |
| 6 | `silu_mul_k` | 0.09 s | 0.1% | 1,316 | 69.8 µs |
| 7 | `ln_mod_select_k` | 0.06 s | 0.1% | 2,560 | 21.8 µs |

`flash_attn_k` has a bimodal distribution (median 32 µs, maximum 40 ms); the
self-attention calls at full 64×64 latent resolution dominate the mean.

### Qwen3-TTS, top kernels

| # | Kernel | Time | Share | Calls | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `conv1d_kernel` | 19.80 s | 84.6% | 31 | 639 ms |
| 2 | `transconv_kernel` | 2.22 s | 9.5% | 6 | 370 ms |
| 3 | `gemm_kernel<bf16>` | 0.80 s | 3.4% | 38,431 | 20.8 µs |
| 4 | `snake_kernel` | 0.25 s | 1.0% | 29 | 8.4 ms |
| 5 | `gqa_kernel` | 0.17 s | 0.7% | 5,487 | 31.4 µs |

## Changes

Two engines were changed. Qwen3.8 and Qwen-Image were left untouched.

### 1. Qwen3-Omni: remove the per-layer host round trip in the MoE block

`text_layer` resolved 3 × 128 expert weight pointers per layer per token
through `std::string` concatenation and an `unordered_map` lookup, and
allocated two `std::vector`s for the routen staging. At 48 layers and 1832
kernels per token, that host work was the reason the GPU sat idle 43% of the
time.

`EngineWeights::cache_expert_pointers()` resolves all 48 × 128 × 3 pointers once
after the checkpoint is loaded; the layer indexes them by `layer * 128`. The
route staging buffers moved to `EngineScratch` and are reused.

No arithmetic changed: the same pointers, the same plan, the same kernels.

### 2. Qwen3-TTS: one thread per output element in the codec convolutions

`conv1d_kernel` and `transconv_kernel` were launched with one block per output
element (`grid = (L, co)`), but the kernel body has no thread indexing: all 256
threads of the block ran the same serial reduction over the same inputs and
wrote the same value. The GPU did 256× the necessary instruction work.

Both kernels now take a flat index, one thread per output element, with the
same loop nest and the same accumulation order. `conv1d` went from 639 ms per
call to 3.75 ms.

## Before and after

### Qwen3-Omni (64 generated tokens, text prompt, GPU 6, three runs each)

| Metric | Baseline | After |
| --- | ---: | ---: |
| `infer_ms` | 1,214.7 (1211/1221/1212) | 984.3 (987/986/980) |
| Change | — | **1.23× faster** |
| Decode per token | 18.06 ms | 14.79 ms |
| GPU busy | 56.9% | **69.6%** |
| Idle per token | 7.78 ms | 4.49 ms |
| Gaps > 20 µs | 1,542 | **91** |
| Generated text | — | byte-identical |

### Qwen3-TTS (same text, GPU 6, two runs each)

| Metric | Baseline | After |
| --- | ---: | ---: |
| Wall clock | 29.1 s (29210/29017 ms) | 7.2 s (7213/7267 ms) |
| Change | — | **4.0× faster** |
| `conv1d_kernel` total | 19.80 s | 0.116 s (170×) |
| GPU busy | 93.5% | 51.9% |

GPU busy percentage falls because the wall clock shrank 7× while absolute idle
stayed at 1.6–1.7 s. The remaining TTS wall time is host-bound; the talker loop
still spends ~1.7 s with no kernel resident.

### Qwen3.8-27B and Qwen-Image-2.1

Not modified. Their files are absent from this commit's diff; Qwen3.8 still
measures 41.3–41.5 tok/s and produces the same token ids as before
(`gen_ids.i32` byte-identical across repeated runs).

## H20 hardware features

| Feature | Status | Evidence |
| --- | --- | --- |
| BF16 tensor cores | **Used** for prefill and image2 | `HMMA.16816.F32` in SASS: 128 sites in `umm-qwen38-native`, 192 in `umm-omni-native`; image2 links `cutlass_80_tensorop_bf16_s16816gemm` |
| BF16 tensor cores in decode | **Not used, deliberately** | Decode is `m = 1`, memory-bound at 2.98 TB/s = 91% of the measured ceiling. Tensor cores cannot raise a bandwidth-bound kernel |
| L2 persistence | **Not used** | L2 is 60 MiB; one decode token streams 51.2 GB of weights. There is no reuse at L2 scale, so an access-policy window cannot help. A 2-way prefetch experiment with `__ldcs` caching hints made decode slower (41.4 → 37.8 tok/s) and was reverted |
| NVLink | **Not applicable** | Every engine here is one process on one GPU. The host reports 5 NVLink links per GPU, but no measured workload moves data between GPUs |

## Not done, and why

* **CUDA Graph over the Qwen3.8 decode loop.** The measured ceiling is 6.3%:
  the GPU is already busy 93.7%, and the CPU keeps the queue ahead (4.0 ms of
  launch time per token against 24.5 ms of GPU work). A graph would recover part
  of the 1.54 ms/token of inter-kernel gaps. It was not implemented because the
  gain is bounded at 6% and the change needs per-token kernel arguments
  (`history`, KV row offsets) moved to device memory.
* **CUDA Graph over the Omni decode loop.** The idle was not launch latency but
  a per-layer host synchronisation for MoE routing. Fixing the host path
  removed 3.3 ms per token; the remaining 4.49 ms is spread over many small
  gaps, so a graph would now buy little.
* **Continuous batching or multi-stream concurrency.** Qwen3.8 is 93.7% busy and
  91% of the streaming bandwidth ceiling; Image2 is 98.1% busy. A second stream
  cannot raise aggregate throughput on a bandwidth-saturated kernel. Omni and
  TTS have headroom, but batching them needs per-sequence state: `rec_state`,
  `conv_state` and the MoE workspaces are single-sequence today. That is a
  larger refactor than this change set.
* **Qwen3.8 GEMV kernel tuning.** A 2-way software-pipelined load loop, with and
  without `__ldcs`, measured 37.2–37.8 tok/s against 41.4 tok/s baseline. It
  was reverted; the existing kernel is within 9% of the hardware's measured
  streaming rate.
* **Omni `moe_route_kernel`.** It takes 51 µs per call and 22% of Omni GPU time
  for a 128-expert top-8 over one token. It is the largest remaining kernel
  target, but changing it can change expert tie-breaking, so it was left for a
  separate change with its own parity gate.

## Reproduction

```bash
# Container linyu-dev, source synced to /root/ckl17zq8m/src
cmake -S native -B build-q38 -DUMM_QWEN38_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/omni -B build-omni -DUMM_OMNI_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/image2 -B build-img2 -DUMM_IMAGE2_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S native/tts -B build-tts -DUMM_TTS_CUDA=ON -DCMAKE_BUILD_TYPE=Release

CUDA_VISIBLE_DEVICES=6 nsys profile -o prof/q38 --trace=cuda \
  build-q38/umm-qwen38-native --model /root/models/qwen3.8-27b \
  --prompt "用一句话介绍你自己。" --gpu 0 --max-new-tokens 128
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace prof/q38.nsys-rep
```
