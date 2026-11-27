# Qwen-Image 2.1 native executor

Native C++/CUDA text-to-image executor for `Qwen/Qwen-Image-2.1`. No Python,
PyTorch, diffusers or SGLang is loaded at run time.

## Scope

Implemented: text-to-image at 512x512 and other multiples of 32, 40-step
default schedule, RGBA PNG output. Not implemented: condition-image editing,
true CFG, KV-cache reuse, tiled VAE decode, batched prompts.

## Components

| Component | Class | Shape |
|---|---|---|
| Text encoder | `Qwen3VLForConditionalGeneration` text_config | 36 layers, hidden 4096, 32 q / 8 kv heads, head_dim 128, SwiGLU 12288, vocab 151936, rope theta 5e6 |
| Transformer | `QwenImage21Transformer2DModel` | 32 single-stream blocks, inner 4096, 32 heads x 128, SwiGLU 12288, in/out 64 channels, patch 1, mlp_ratio 3 |
| VAE | `AutoencoderKLQwenImage21` | 64 latent channels, decoder base 144, dim_mult [1,2,4,8,8], 16x spatial |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | dynamic exponential shift, shift_terminal 0.02, 40 steps |

### Text encoder

Qwen3-VL text decoder. For a text-only prompt all three mRoPE axes carry the
same position, so the interleaved layout reduces to standard 1-D rotate-half
RoPE. Per-head q/k RMSNorm (eps 1e-6), causal GQA attention, pre-norm
residual blocks. The engine returns the last decoder layer's output *before*
the final RMSNorm: the diffusers pipeline deliberately disables that norm
(`language_model.norm` forward hook) because the transformer was trained on
the pre-norm states.

Prompt construction copies `QwenImage21Pipeline`:
`<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
The first 14 tokens (the system block) are dropped after encoding, matching
`pipeline._drop_idx`.

### Transformer

Single-stream joint sequence `[text; target image]`.

- `causal_condition=true`: the model embeds `t` and a second row at `t=0`.
  Text tokens use the `t=0` row for every modulation scale/gate; target-image
  tokens use the sampled timestep. This is why text activations are
  step-independent in the reference and why it can cache their K/V.
- Block-causal attention: text queries attend causally to text keys only;
  target-image queries attend the whole sequence, bidirectionally inside
  their own block. Exact for a single target image; the reference's general
  form also isolates multiple condition images.
- RoPE: complex-pair rotation over `axes_dims_rope = [16, 56, 56]`
  (theta 10000). Text rows use their index on all three axes; image rows
  freeze the frame axis at the text length and use centred height/width
  coordinates `[-h/2, h/2)`.
- One shared `modulation` projection (`SiLU` then Linear to `4*inner`) feeds
  every block; blocks have no modulation parameters of their own.
- Block order: `LN -> scale -> qkv -> qk-norm -> rope -> attention -> out ->
  +tanh(gate1)*attn`, then `LN -> scale -> SwiGLU -> +tanh(gate2)*mlp`.
- Final `norm_out` is LayerNorm with a scale-only AdaLN (`1 + scale`), applied
  to target tokens only, then `proj_out` to 64 channels.

### VAE decoder

`QwenImage21CausalConv3d` inherits `nn.Conv2d`: its "temporal" kernel is a
plain 2D conv with explicit padding. For a single frame the temporal branches
collapse:

- every residual conv is a 3x3 (or 1x1) 2D conv;
- `upsample3d` skips its `time_conv` on the first chunk (the `"Rep"` branch),
  leaving nearest-exact 2x + 3x3 conv;
- `DupUp3D` shortcuts become a 2x2 pixel shuffle over channels; with
  `first_chunk=True` and factor_t=2 the kept temporal slice is index 1.

Decoder width progression: 64 -> 1152 -> (3x1152) -> 1152 -> 576 -> 288 ->
144 -> 4, with four 2x stages and one spatial attention block at latent
resolution (single head, 1152 channels). Output is clamped to [-1, 1] and
written as RGBA PNG; the fourth channel is the checkpoint's alpha channel.

## Differences from the 2512 executor (`native/image`)

| | 2512 | 2.1 |
|---|---|---|
| Transformer layers | 60 | 32 |
| Latent channels | 16 | 64 |
| Text encoder | Qwen2.5-VL, 28 layers, hidden 3584, 28/4 heads | Qwen3-VL, 36 layers, hidden 4096, 32/8 heads, qk-norm |
| Text conditioning | separate text stream with its own modulation | one joint stream, one shared modulation |
| Attention | non-causal over `[text; image]` | block-causal |
| Timestep conditioning | one modulation row | two rows (`t`, `t=0`) selected per token |
| VAE | 16-channel, 8x spatial, 3D causal blocks | 64-channel, 16x spatial, 2D-collapsed decoder |
| Postprocess | RGB | RGBA |

## Precision strategy

All weights and activations are bf16, matching the reference run. Norms,
softmax, RoPE angles and the Euler sampler accumulate in fp32 and round once,
the same convention PyTorch uses for bf16 elementwise kernels. The Euler step
keeps the sample in fp32 and rounds back to bf16 each step, as
`FlowMatchEulerDiscreteScheduler.step` does. The timestep mirrors the
reference's two bf16 roundings (timestep -> bf16, divide by 1000 in bf16)
before the model multiplies it back by 1000 in fp32.

## Numerical alignment

Reference dumps come from diffusers 0.43.0 + transformers 5.6.0 in bf16 with
the same prompt, seed and schedule. Errors are measured on the tensors the
stages exchange.

| Stage | Metric | Value | Gate |
|---|---|---|---|
| Text encoder (post-drop, 24x4096) | cosine / rel_l2 | 0.999478 / 0.032321 | cos >= 0.999 |
| DiT step 0 velocity (32x32x64) | cosine / rel_l2 | 0.999975 / 0.007087 | cos >= 0.999 |
| VAE decode (4x512x512) | rel_l2 / PSNR | 0.036840 / 38.55 dB | rel_l2 <= 0.06 |

## Measured performance

NVIDIA Hopper (SM90) GPU 7 (`CUDA_VISIBLE_DEVICES=7`), 512x512, 40 steps, bf16,
two runs (English prompt seed 1234, Chinese prompt seed 2024). Encode, denoise
and VAE times differ by at most 0.4% between runs; load time differs by 232 ms
(4.9%).

| Phase | Time |
|---|---|
| Weight load | 4926 ms / 4694 ms |
| Text encode | 32 ms |
| Denoise (40 steps) | 59740 ms / 59954 ms |
| VAE decode | 3628 ms / 3632 ms |
| Total | 69778 ms / 69774 ms |
| Peak GPU memory (sampled) | 14763.2 MiB |

Denoise is 85.6% of total. Per step: 59.74 s / 40 = 1.49 s for a 1024-token
text + 1024-token image joint sequence at 64 latent channels. The reference
diffusers run, with KV cache and cuDNN/flash kernels, takes 6.04 s for the same
40 steps; the native path is about 10x slower and is not optimized yet.

Content check: the local Qwen3-VL (same checkpoint as the text encoder) captions
the English output as a red fox sitting upright in a snowy landscape, and the
Chinese output as a watercolor ginger cat on a windowsill looking at a city
street. Both match their prompts.

## Usage

```
umm-image2-native --model /opt/umminfer/models/qwen-image-2.1 --gpu 0 \
  --prompt "A red fox in a snowy forest at dawn" --size 512x512 \
  --seed 1234 --steps 40 --out out.png
```

Build:

```
cmake -S native/image2 -B build-image2 -DUMM_IMAGE2_CUDA=ON
cmake --build build-image2 -j
ctest --test-dir build-image2
```

`image2_gpu_tests MODEL_DIR GOLDEN_DIR` ran the three stage comparisons when
the development-time golden dumps were present. `native/image2_tests/` was
removed in `ab51cf5`, so the target no longer builds. `native/image2/CMakeLists.txt`
still names the removed directory; that is a known build break outside this
document.

## Request-owned allocator telemetry

The native stage build routes every explicit image2 `cudaMalloc`/`cudaFree`
through `native/common/cuda_request_allocator.hpp`, including BF16/FP32 tensor
owners, text input ids, and DiT target masks. With the public
`telemetry.native_allocator=true` opt-in, each serial request creates a fresh
ownership epoch; late frees remain assigned to their original owner. The 2512
backend is not instrumented and reports `NOT_COVERED` with null byte fields.

The standalone `umm-image2-native` CLI intentionally keeps direct CUDA calls and
emits no request telemetry. Pre-request resident weights/workspaces, host memory,
CUDA context/driver activity, cuBLAS/library internals, and unsupported batched
or overlapping attribution are excluded. A warm zero-event epoch is valid only
when no direct request allocation occurs; normal image2 generation allocates
per-request tensors and must not inherit the TTS/ASR zero-event convention.

Task #69 passed the real serial SM90 GPU 1 public JSON/SSE/PNG gate with warm
reuse, failure/cancellation/recovery, independent fresh-process verification,
and separate whole-device NVML samples. Exact hashes and limits are in
[docs/native-service.md](../../docs/native-service.md#sm90-qwen-image-21-validation-gate--pass-2026-11-27).
This is allocator/protocol evidence, not an image-quality or deployment claim.

## Known limitations

- Text-to-image only. Condition images, editing and true CFG are absent.
- No KV cache: text tokens are recomputed every step. Correct, but it leaves
  the reference's main speedup on the table.
- VAE attention materializes an hw x hw fp32 score matrix. At 512x512 that is
  4 MiB; at 1024x1024 it is 64 MiB per pass and the naive conv kernels get
  slow. 512 is the validated size.
- The native RNG is Philox + Box-Muller. It is deterministic for a seed but
  does not reproduce `torch.randn` bit-for-bit, so a native image and a
  diffusers image from the same seed are different samples, not different
  renders of the same sample. Stage parity uses dumped reference tensors.
