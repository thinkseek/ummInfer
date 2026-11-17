# Native C++/CUDA Qwen3-Omni thinker/VLM vertical slice

A from-scratch inference engine for the **thinker** of the Qwen3-Omni
checkpoint: one image plus a text prompt in, greedy text answer out, with no
Python / PyTorch / vLLM / SGLang / cuBLAS in the runtime. The audio tower,
talker and code2wav are not loaded and audio/video inputs are unsupported.

## Layout

| Path | Contents |
|---|---|
| `native/omni/` | Engine: multi-shard mmap loader, image preprocessing, Qwen2 BPE tokenizer, CUDA kernels, vision tower, MoE text decoder, engine |
| `native/omni_cli/` | The `umm-omni-native` command line |
| `native/omni_tests/` | Host (CPU) unit tests, Unicode table generator, golden fixtures |
| `native/OMNI_NATIVE.md` | This document |

The pre-existing `native/umm_asr_native/` ASR slice is untouched. This build is
a separate CMake project rooted at `native/omni/CMakeLists.txt`.

## What is implemented

- **Image preprocessing** (`image_preprocess.*`): stb_image decode to RGB,
  Qwen2-VL `smart_resize` (factor 32, pixel bounds), antialias bicubic resize
  (Keys cubic a=-0.5, half-pixel coordinates, downscale support widening),
  `/127.5 - 1` normalization, temporal-2 patch reorder into 864 x 1536.
- **Tokenizer** (`tokenizer.*`, `unicode_ranges.hpp`): full Qwen2
  byte-level BPE encode plus decode, the six-alternative pre-tokenization
  regex over generated Unicode property tables, GPT-2 byte mapping, ranked
  merges with leftmost tie handling, and 33 protected added tokens.
- **Multi-shard loader** (`safetensors_mmap.*`): read-only mmap of all 15
  safetensors shards via `model.safetensors.index.json`; only the
  `thinker.visual.*`, `thinker.model.*` and `thinker.lm_head.*` BF16 tensors
  are copied to the GPU (~58 GiB resident). Audio/talker/code2wav are skipped.
- **Vision tower** (`vision_forward.cu`, `merger.cu`): conv3d-as-linear patch
  embed, bilinear-interpolated absolute position embedding, 27 bidirectional
  MHA blocks with vision rotary embeddings (theta 1e4) and gelu_tanh MLP,
  three deepstack mergers after blocks 8/16/24 and the final 2x2 patch merger
  producing the 216 x 2048 pooler.
- **MoE text decoder** (`text_decoder.cu`, `omni_engine.cu`): 48 layers,
  128 experts with top-8 fp32 softmax routing and probability renormalization,
  GQA 32q/4kv head_dim 128, per-head q/k RMSNorm, interleaved mRoPE
  (theta 1e6, T/H/W sections 24/20/20), bounded KV cache, and the deepstack
  feature injection after text layers 0/1/2 at the image positions.
- **Greedy generation**: stop on `<|im_end|>` (151645) or `<|endoftext|>`
  (151643), no sampling or temperature; decode positions continue from the
  prefill mRoPE maximum without delta bookkeeping.

## Build and run

Host (Mac or Linux, no GPU):

```
cmake -S native/omni -B native/omni/build
cmake --build native/omni/build -j
ctest --test-dir native/omni/build
```

GPU (the reference H20 host, CUDA 12.4, compute capability 9.0):

```
cmake -S native/omni -B build -DUMM_OMNI_CUDA=ON
cmake --build build -j
./build/omni_cli/umm-omni-native \
  --model /path/to/omni --prompt "..." [--image pic.jpg] \
  --gpu 0 --max-new-tokens 32
```

The runtime has no framework dependencies; `ldd` shows only the CUDA runtime
and system libraries.

## Matmul kernels

`omni/omni_gemm.cu` holds every matmul in the slice. Four paths:

| Path | Used by | Shape strategy |
|---|---|---|
| `gemm_bf16_mma` | prefill and the vision tower | 128x128 CTA tile, 64-deep K stages, `cp.async` double buffering, `ldmatrix` operand loads, warp tile 64x32 from 16 `mma.sync.m16n8k16` ops per K step |
| `gemv_bf16_f32/bf16` | batch-1 decode projections, router, lm_head | one warp per output row, 16-byte vector loads, warp-shuffle reduction |
| `moe_gemm_grouped_f32` | MoE gate and up | same tile as `gemm_bf16_mma`, driven by a host-built CTA list so one launch covers all 128 experts |
| `moe_down_f32` / `gemv_grouped_f32act` | MoE down projection | CUDA-core FP32 FMA, 16-pair x 128-output tile, FP32 activation operand |

The down projection deliberately does **not** use the tensor-core tile: the
reference multiplies the expert activation in FP32, and an `mma.sync` needs
BF16 operands. Rounding the activation to BF16 would change the numeric
semantics (up to 2^-7 relative per element) rather than only the reduction
order, so that projection runs as FP32 FMA over upcast BF16 weights. Products
are then bit-identical to the reference and only the accumulation order
differs. `omni_tests/test_moe_down.cu` is a GPU test that fails if a BF16
activation is reintroduced: it builds activations that BF16 cannot represent,
and asserts the kernel is within FP32 accumulation noise of an FP32-activation
reference (measured 4e-7) and 2.7e-3 away from a BF16-activation one.

The reason for the grouped launch: the router already round-trips to the host
once per layer (the reference implementation resolves experts there), so the
(token, expert) pairs are sorted by ascending expert id on the host. Each
expert's pairs then form one contiguous row block, and the expert weights are
streamed exactly once per prefill pass instead of once per pair. Pair outputs
are kept in fp32 and combined in the reference order -- ascending expert id,
BF16 rounding after each add -- so the combination step is bit-identical to
the previous per-expert accumulator chain.

Measured instruction ceilings on the target H20 (78 SM, 1.98 GHz, 60 MB L2),
from a standalone probe: `mma.sync.m16n8k16` peak 95.1 TFLOPS (flat across
warp/accumulator/occupancy configurations), HBM read 3.63 TB/s. The model
needs 2.1 TFLOP and 62 GB of weight reads per image prefill, so the roofline
is ~18 ms of memory and ~22 ms of tensor core. Measured prefill is 233 ms
(195 ms before the down projection moved from a BF16 tensor-core tile to
FP32 FMA, which costs ~38 ms and is the only place precision was bought
back).

Attention was measured before rewriting it and left alone: the two-pass exact
softmax GQA kernel is 0.2% of prefill and 0.1% of decode at this model size,
so a FlashAttention-style rewrite would add numeric risk -- a different
online-softmax reduction order -- for no measurable gain. The naive GEMM, the
32x32 tiled GEMM, the block-per-row GEMV and the single-block expert GEMV
accumulator were deleted.

## Benchmark

`umm-omni-bench` runs the acceptance inputs with fixed arguments: the real
`man_ironing.png` fixture and the accepted prompt, once with the image and
once text-only. It prints phase timings, decode throughput and the answer.

```
cmake -S native/omni -B build -DUMM_OMNI_CUDA=ON
cmake --build build -j
CUDA_VISIBLE_DEVICES=6 ./build/omni_cli/umm-omni-bench \
  --model /path/to/omni --image /tmp/man_ironing.png --gpu 0
```

## Tests and golden data

`omni_host_tests` covers smart_resize, image error paths, bicubic resize
against a torchvision dump, the full pixel buffer against the HF
`AutoImageProcessor` output, synthetic and real-tokenizer BPE golden vectors,
and decode round-trips. Golden data under `omni_tests/golden/` is downloaded
from the GPU reference host and is git-ignored; the dump scripts live outside
the repository. On the GPU host, vision-tower intermediates (pooler,
deepstack, q pre/post rope, position embeddings) were compared against the
real HF `Qwen3OmniMoeVisionEncoder` and match at BF16 tolerance
(pooler cosine ~0.91 across 27 blocks; early tensors cosine > 0.9999).

`omni_moe_down_test` runs on the GPU and guards the FP32 expert activation
(see "Matmul kernels"). `omni_tests/evidence/` holds the before/after runs as
committed text: `run_measurement.sh host|gpu` regenerates them, recording each
command line, its real exit code, and its full stdout/stderr in files that are
hard-capped at 64 KiB each. No weights, audio, images or build outputs are
stored there.

## Known limitations

- Single image, single sequence, batch 1, greedy only. Multi-image packing and
  video (t>1) are explicitly out of scope; audio input is rejected by design.
- Tokenizer NFC is skipped (prompts are NFC); see `omni/NOTICE`.
- JPEG decode uses stb_image, whose IDCT differs from libjpeg-turbo by a few
  LSB on ~1.5% of pixels; the bicubic resize fed identical decoded bytes
  matches torchvision to within 2 LSB almost everywhere.
- Text prefill and decode are shaped for batch 1 + one prefill chunk. There is
  no continuous batching and no multi-request scheduler.
- Decode is launch-latency bound, not bandwidth bound: 48 layers issue about 20
  short kernels each, so ~960 launches per token dominate. Its bandwidth floor
  is ~1.7 ms/token (6.1 GB of weights at 3.6 TB/s); measured is ~32 ms/token
  (31 tok/s on the image case, 59 tok/s on the 23-token text case).
  A CUDA-graph capture of one decode step is the documented next step if
  decode latency matters more than the current 15x margin over the target.
- Numerics follow BF16 storage with fp32 accumulation at the explicitly
  upcast points (norms, softmax, router, RoPE). Small last-ulp differences
  from the HF/SGLang tensor-core path are expected; greedy answers match.
- Numeric semantics are preserved everywhere except the reduction order.
  Every matmul accumulates in FP32 and every operand keeps its reference
  precision: the MoE activation feeding the down projection stays FP32 (see
  "Matmul kernels" above), so no activation is rounded to BF16.
