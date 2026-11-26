# Native Qwen3.8-27B executor

Self-contained C++20/CUDA inference path for `Qwen/Qwen3.8-27B`
(`Qwen3_5ForConditionalGeneration`). It loads the real BF16 safetensors
checkpoint, runs the 27-layer vision tower and the 64-layer hybrid text
decoder, and generates text greedily. Runtime dependencies: `libcudart` only.
No PyTorch, SGLang, vLLM, cuBLAS or cuDNN.

## Model structure

From the checkpoint config (not assumed):

| item | value |
|---|---|
| parameters | 27,781,427,952 (BF16, 55.6 GB across 18 shards) |
| text layers | 64, hidden 5120, intermediate 17408, vocab 248320 |
| layer pattern | 3 × `linear_attention` + 1 × `full_attention`, `full_attention_interval=4` |
| full attention | 24 query heads / 4 KV heads, head_dim 256, scaling 1/16 |
| q projection | emits query **and** a per-head output gate (`attn_output_gate`) |
| q/k norm | zero-centred RMSNorm per head (`(1 + weight)`) |
| RoPE | partial 64 of 256 channels, theta 1e7, mRoPE sections [11, 11, 10] |
| linear attention | 16 key heads / 48 value heads, key/value head_dim 128, conv kernel 4, fp32 state |
| MLP | dense SwiGLU (`silu(gate) * up`, then down) |
| vision tower | 27 blocks, hidden 1152, 16 heads, head_dim 72, patch 16, temporal 2, merge 2, output 5120 |
| vision position | learned 48×48 table, bilinear `align_corners=True` resampling, axial 2D RoPE |
| vision merger | LayerNorm(1152) → 4608 → GELU(erf) → 5120 |

The `mtp.*` tensors in the checkpoint are not loaded; plain autoregressive
decoding does not use them.

## Execution

Per decoder layer, with `h` the residual stream:

```
u = (1 + w_in) * rms_norm(h)                 # zero-centred RMSNorm
if layer is linear_attention:
    qkv = u @ W_qkv^T                        # 10240
    z   = u @ W_z^T                          # 6144
    b   = u @ W_b^T ; a = u @ W_a^T          # 48 each
    qkv = silu(depthwise_causal_conv4(qkv))  # state = last 3 inputs
    beta = sigmoid(b)                        # bf16-rounded
    g    = -exp(A_log) * softplus(a + dt_bias)
    q, k = l2norm(q), l2norm(k); q *= 1/sqrt(128)
    D    = exp(g) * S_{t-1}                  # state decay, fp32
    delta = (v - D^T k) * beta
    S_t  = D + k ⊗ delta                     # fp32 state, 48 × 128 × 128
    o    = S_t^T q
    o    = rms_norm(o) * w_norm * silu(z)    # plain weight, gate in fp32
    mixer = o @ W_out^T
else:
    qg = u @ W_q^T                           # 12288 = 24 × (query 256 + gate 256)
    k, v = u @ W_k^T, u @ W_v^T              # 4 × 256 each
    q = rms_norm_0c(q, w_qn); k = rms_norm_0c(k, w_kn)
    rope_partial(q, k, 64 of 256, mRoPE)
    a = causal_gqa(q, kv_cache) * sigmoid(gate)
    mixer = a @ W_o^T
h += mixer
n = (1 + w_post) * rms_norm(h)
h += (silu(n @ W_gate^T) * (n @ W_up^T)) @ W_down^T
```

Prefill processes the whole prompt in one pass (`m` rows); decode runs
`m = 1` through the same code and reuses the KV cache (16 full layers) and the
per-layer recurrent/conv states.

## Linear attention: implementation choice

The reference (`transformers/models/qwen3_5`) uses the **chunked** gated delta
rule for prefill and the **recurrent** form for decode. Both compute the same
recurrence. This executor uses the recurrent form for both:

- One CUDA block per value head (48 blocks), 128 threads, the 128×128 state
  held in shared memory. Per token it is exact in the same sense as the
  reference recurrent path: fp32 decay, fp32 `kv_mem`, fp32 update, fp32 read.
- Prefill at ~300 tokens costs 48 launches per linear layer, not one launch
  per token, and each launch is memory-light (64 KiB state tile).
- Decode needs no separate kernel: `m = 1` updates the same state in place.

This avoids implementing the UT triangular solve and keeps one numerical path
for prefill and decode; the price is that prefill does not parallelise across
the sequence dimension. At the measured 512×512 workload the linear layers are
not the bottleneck (see measurements).

## Precision

- Weights BF16; GEMM uses `mma.sync` BF16 × BF16 → FP32 accumulate, output
  BF16. Decode uses a warp GEMV with FP32 accumulation.
- Norms, softmax, mRoPE tables and the delta-rule state are FP32.
- Rounding points follow the reference: `sigmoid(gate)` is rounded to BF16
  before multiplying the attention output; SiLU in the MLP is rounded to BF16
  before multiplying `up`; the conv output is rounded to BF16 before SiLU; the
  gated RMSNorm rounds `weight * norm` to BF16 before the FP32 gate multiply.
- mRoPE cos/sin tables are rounded to BF16 (the reference casts them to the
  hidden dtype). Vision RoPE stays FP32 (the reference upcasts q/k/cos/sin).

## Build

```sh
cmake -S native -B native/build -DUMM_QWEN38_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j
```

Targets: `umm-qwen38-native`, `qwen38_host_tests`, `qwen38_kernel_tests`,
`qwen38_consistency_tests`, `qwen38_tokenize`, `qwen38_preprocess`.

## Run

```sh
CUDA_VISIBLE_DEVICES=6 native/build/umm-qwen38-native \
  --model /opt/umminfer/models/qwen3.8-27b --gpu 0 \
  --image /opt/umminfer/qwen38-test.png \
  --prompt "Describe the shapes and colors in this image in one sentence." \
  --max-new-tokens 64
```

The answer goes to stdout. Timings (`load_ms`, `vision_ms`, `prefill_ms`,
`decode_ms`, `decode_tok_per_s`, `gpu_used_mb`) go to stderr as measured
values; nothing is fitted.

`--thinking` renders the prompt with `enable_thinking=true`; the default is
false, which makes the assistant answer immediately.

## Reference parity harness (development only)

RETIRED. `native/qwen38_tests/` was removed in `ab51cf5`. The scripts and
commands below are a historical record and do not run on the current source.

`reference.py` runs the same prompt and image through transformers on one GPU
(or CPU for the FP32 anchor) and dumps the embedding, every layer/mixer
output, the vision tower and the generated ids.  `compare_layers.py` compares
the BF16 dumps; `op_parity.py` tests the linear-attention operator on
identical inputs.

```sh
# BF16 reference and native run (dumps)
python native/qwen38_tests/reference.py --model DIR --prompt TEXT \
    [--image PATH] --max-new-tokens N --out /opt/umminfer/ref
native/build/umm-qwen38-native --model DIR --prompt TEXT [--image PATH] \
    --max-new-tokens N --dump-dir /opt/umminfer/nat

# FP32 anchor (CPU; one GPU cannot hold 112 GB of FP32 weights)
python native/qwen38_tests/fp32_dump.py --model DIR --prompt TEXT \
    [--image PATH] --out /opt/umminfer/ref-fp32

# End-to-end comparison
python native/qwen38_tests/compare_layers.py --ref /opt/umminfer/ref \
    --native /opt/umminfer/nat --fp32-ref /opt/umminfer/ref-fp32

# Per-layer linear-attention operator (same input to both implementations)
python native/qwen38_tests/op_parity.py --model DIR \
    --native /opt/umminfer/nat --gpu 0
```

Criteria, identical to the constants in the scripts:

- `op_parity.py`: same-input relative RMS <= `1e-2` for all 48 linear layers.
- `compare_layers.py`: `rel_rms(native, fp32) <= 1.5 * rel_rms(hf_bf16, fp32)
  + 1e-3`.  Both sides are BF16 approximations of the same FP32 model; the
  native path must not sit more than 1.5x further from FP32 than the
  reference's own BF16 path.
- Functional: prompt ids, first-token argmax and generated ids equal.

## Measurements (Linux SM90 GPU host, GPU 6)

`CUDA_VISIBLE_DEVICES=6`, weights in the page cache, greedy decoding, one
process per request. `load_ms` includes the 55.6 GB H2D copy; `prefill_ms`
excludes the vision tower.

| case | prompt tokens | image tokens | load | vision | prefill | decode | GPU |
|---|---|---|---|---|---|---|---|
| text-only | 24 | 0 | 9264 ms | — | 159 ms | 41.7 tok/s (1 step) | 55,769 MB |
| 512×512 image + text | 282 | 256 | 9155 ms | 588 ms | 466 ms | 29.1 tok/s (15 steps) | 55,769 MB |

Image prefill including the vision tower is 1054 ms, below the 30 s budget.
Decode is above the 5 tok/s floor in both cases. The text-only case stops after
one token because the model emits EOS.

Outputs (identical to the HF reference, `skip_special_tokens=True`):

- text: `Paris`
- image: `There is a red circle and a blue square on a light gray background.`

## Parity results

**Functional parity: exact in both cases.**

| case | prompt ids | first-token argmax | generated ids | text |
|---|---|---|---|---|
| text | 24/24 identical | 57590 = 57590 | 1/1 identical | equal |
| image | 282/282 identical | 3733 = 3733 | 15/15 identical | equal |

**Per-layer linear-attention operator parity (same input to both).**  The input
is the native trajectory's pre-layer hidden state after the layer's input
RMSNorm; the native kernel output (dumped mixer) is compared with the
transformers module on exactly that input.

| case | layers | max same-input rel_rms | gate |
|---|---|---|---|
| text | 48 | 7.34e-3 (layer 21) | 1e-2 |
| image | 48 | 1.03e-3 (layer 30) | 1e-2 |

All 48 layers pass in both cases.  This isolates the operator: for the worst
end-to-end tensor (text `mixer_42`), HF's own module fed the native input
reproduces the native output to `4.36e-3`; the remaining difference is the
input, not the operator.

**End-to-end parity (FP32 anchor).**  All tensors pass in the image case
(worst: pooler, native 3.30e-2 vs HF-BF16 2.56e-2, ratio 1.29).  In the text
case every tensor passes except `mixer_42` (native_vs_fp32 3.53e-2 vs band
2.04e-2, ratio 1.73).  Its input `layer_41` passes at ratio 1.03, so the
excess comes from the layer's small-norm output reacting to the already
inherited input difference, not from the operator (which matches to 4.36e-3).

Representative FP32-anchor numbers:

| tensor | HF BF16 vs FP32 | native vs FP32 | ratio |
|---|---|---|---|
| text layer_63 | 4.10e-1 | 4.08e-1 | 1.00 |
| text mixer_42 | 2.04e-2 | 3.53e-2 | 1.73 |
| image pooler | 2.56e-2 | 3.30e-2 | 1.29 |
| image mixer_58 | 4.37e-1 | 5.00e-1 | 1.14 |
| image layer_63 | 4.10e-1 | 4.08e-1 | 1.00 |

The magnitude of the deep-layer numbers is the model's BF16 path itself: the
reference, run in BF16, is 41% away from its own FP32 result at layer 63.

**Array-level attribution (the worst tensors).**

text `mixer_42`, max difference at token 17, channel 3456:

```
input layer_41  ref[17,3456:3464] = [10.5625, 0.55469, 0.40039, 0.04492, -0.23730, 0.5, -0.18945, -0.51562]
input layer_41  nat[17,3456:3464] = [ 9.75,   0.57422, 0.37109, 0.04199, -0.19824, 0.48438, -0.19141, -0.52344]
mixer ref[17,3456:3464] = [24.125, -0.01495, -0.25391, 0.24707, -0.49609, 0.83984, -0.01697, 0.40625]
mixer nat[17,3456:3464] = [22.375, -0.03613, -0.25781, 0.23047, -0.43750, 0.78906, -0.00497, 0.34766]
HF module on the native input = [22.5, -0.03369, -0.25977, 0.22754, -0.43750, 0.79297, -0.00620, 0.34961]
```

At the max-difference element the input differs by 0.8125; the operator adds
only 0.125 (one BF16 step at 22.5).  Weights at that layer:
`linear_attn.out_proj.weight[0,0:6] = [-0.00909, -0.00897, 0.00922, -0.00745,
-0.00720, -0.01477]`, `in_proj_qkv.weight[0,0:6] = [0.03247, -0.01794,
0.00705, -0.00149, -0.00229, 0.00038]`.

image `mixer_58`, max difference at token 96, channel 3994:

```
input layer_57  ref[96,3994:4002] = [157.0,  -3.6875, -2.15625, -2.9375, -4.4375, 0.89062, -1.57812, -0.92578]
input layer_57  nat[96,3994:4002] = [314.0,  -0.44922, -0.54688, 0.55469, -1.25781, 1.75, 3.6875, 0.71484]
mixer ref[96,3994:4002] = [-16.125, 0.06445, -0.08008, -0.41602, 0.23047, -0.46680, 0.07715, 0.46680]
mixer nat[96,3994:4002] = [ 55.0,  -0.08057, 0.01105, -0.08838, 0.06641, 0.00299, -0.04785, -0.23145]
HF module on the native input = [55.0, -0.08057, 0.01105, -0.08838, 0.06641, 0.00299, -0.04785, -0.23145]
```

The input channel is 157 vs 314 after 57 layers; the operator output is
bit-identical on the same input.  The whole 71.125 difference is inherited.

**State-path checks.**  `qwen38_consistency_tests` verifies that the conv
state and the FP32 gated-delta-rule state produce bit-identical outputs and
state whether the sequence is processed in one prefill call or token by token
(decode).  Vision order checks: `vpatch` differs from the FP32 reference by
2.1e-3 and the pooler by 3.3e-2, while changing only the reference's attention
kernel (sdpa -> eager) moves its own pooler by 2.6e-2; a wrong merger order or
cross-block state would be O(1).

## Native video input selection

The stage accepts one `video_url` instead of one still image. Default and
explicit uniform-8 sampling retain the historical FFmpeg command, 8 RGB frames,
temporal-2 packing, and synthetic slice timestamps. Optional
`video_url.frame_selection.max_frames` accepts exactly integer 4, 8, or 16;
omission stays at 8. Uniform 4/16 use deterministic rounded source indices with
first/last anchors and source PTS, so repeated pixels cannot obscure the source
mapping and the historical 0.25-fps floor cannot clamp 16 to 8.

The opt-in `policy="temporal"` path decodes at most 64 candidates, ranks
deduplicated scene boundaries and motion on a fixed 16×16 luma grid, and selects
4/8/16 frames with first/last anchors and uniform fallback. Ties use the earliest
decode index; short/static/repeated clips always have a bounded valid result.
Selection stays in decode order and preserves finite source PTS, including
non-monotonic PTS. Pair-first source timestamps feed the existing video prompt
and 3D MRoPE; an odd short decode duplicates only its final temporal-2 pixel
pair. Both policies use `preprocess_video_frames`, the same batched multi-slice
ViT, and the same serial/batched engine calls; selected frames are never
submitted as independent still images. The public budget is validated before
output allocation; candidate (64) and final patch (3584) ceilings cannot be
raised through the API. See `video_sampling.*`, `video_input.*`, and
`native/qwen38_tests/test_video_input.cpp`.

## Known limitations

- Sampling supports `temperature`, `top_p`, and signed `seed`; no `top_k` knob.
- One image or one video per request; image and video inputs are mutually
  exclusive. Video frames use native temporal merge, not independent images.
- The vision workspace is capped at 4096 pre-merge patch tokens (1024×1024);
  the video preprocessor retains its tighter 3584-patch budget.
- `max_seq` is 8192 (prompt + generated).
- Attention is a two-pass FP32 softmax with per-(head, token) blocks; it is
  O(n²) and not a FlashAttention kernel. Fine for the measured prompt sizes.
- The tokenizer implements the Qwen2 byte-level BPE with the Qwen3.8
  pretokenizer; Unicode NFC normalisation and combining-mark (M) grouping are
  not implemented. ASCII, Chinese and the tested prompts match the HF ids
  exactly.
- No weight quantisation or multi-GPU. Decode continuous batching is available
  through the native stage `--batch-size N`; the standalone CLI is serial.
- Load time is dominated by reading 55.6 GB from disk and copying it over PCIe;
  it is reported as `load_ms` and excluded from `prefill_ms`.

## License

Apache-2.0. See `NOTICE` for derivation and upstream references.
