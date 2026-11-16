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

## Known limitations

- Single image, single sequence, batch 1, greedy only. Multi-image packing and
  video (t>1) are explicitly out of scope; audio input is rejected by design.
- Tokenizer NFC is skipped (prompts are NFC); see `omni/NOTICE`.
- JPEG decode uses stb_image, whose IDCT differs from libjpeg-turbo by a few
  LSB on ~1.5% of pixels; the bicubic resize fed identical decoded bytes
  matches torchvision to within 2 LSB almost everywhere.
- MoE dispatch uses per-expert GEMVs with one host-side routing sync per
  layer. Correctness-first; prefill of a 241-token image prompt takes roughly
  4 minutes on one H20. A grouped/tiled grouped GEMM is the documented
  throughput upgrade.
- Numerics follow BF16 storage with fp32 accumulation at the explicitly
  upcast points (norms, softmax, router, RoPE). Small last-ulp differences
  from the HF/SGLang tensor-core path are expected; greedy answers match.
