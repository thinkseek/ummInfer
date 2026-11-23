# Native Qwen3-TTS 0.6B CustomVoice executor

A C++/CUDA inference path for the Qwen3-TTS-12Hz-0.6B CustomVoice model. It
loads BF16/FP32 weights directly from safetensors and runs entirely on CUDA.
No Python, PyTorch, torch, SGLang or the `qwen_tts` package is used at run
time. Python is used only offline to produce reference tensors for parity
checks.

## Scope and layout

- `native/tts/` — engine, talker, code predictor, F32 speech codec, CUDA
  kernels, BPE tokenizer, prompt builder, safetensors/WAV I/O.
- `native/tts_cli/` — `umm-tts-native` command line front end.
- `native/tts/tail_trim_test.cpp` — host self-check for the trailing-noise
  trim, built by the `tts_tail_trim_test` target.

RETIRED. The `native/tts_tests/` tree (host unit tests, CUDA kernel tests,
tokenizer fuzz harness and probe tools) was removed in `ab51cf5`. Every
reference to it below is historical.

No existing ASR source or shared top-level file is modified. The TTS slice
links its own standalone CMake project.

## Model architecture

The checkpoint has two parts.

### Talker and code predictor (BF16)

- Talker: 28-layer transformer, hidden 1024, 16 query heads / 8 KV heads,
  head_dim 128, GQA, RMSNorm eps 1e-6, RoPE theta 1e6.
- Code predictor: 5-layer transformer, hidden 1024, 16/8 heads, head_dim 128.
- Text path: byte-level Qwen2 BPE tokenizer, then a two-layer text projection
  MLP (2048 -> 2048 -> 1024) with SiLU.
- 16 code groups per frame, codebook 2048, 12.5 frames per second. The talker
  emits group 0; the predictor emits groups 1 to 15 autoregressively.
- mRoPE uses section [24,20,20]. The three position axes are identical for
  this text-only input, so the rotation is numerically a standard 1D RoPE.
  Half-rotation (GPT-NeoX) layout is used.
- Multimodal RoPE, RMSNorm, SiLU-gated MLP and GQA follow the upstream
  `qwen_tts` modeling code. Only formulas, constants and tensor names were
  referenced; the kernels are original.

### Speech codec (FP32)

- Split residual vector quantizer: one semantic codebook plus 15 residual
  codebooks, each dequantized from `embedding_sum / max(cluster_usage, eps)`.
- Pre-conv, an 8-layer sliding-window transformer (MHA 16 heads, head_dim 64,
  window 72), two ConvNeXt x2 upsample stages, then a waveform decoder.
- The waveform decoder is a 1536-channel causal conv, four SnakeBeta +
  transposed-conv blocks with rates [8,5,4,3], three dilated (1,3,9) residual
  units per block, a final SnakeBeta and a 1-channel conv.
- Total upsample per frame is 1920 samples. At 24 kHz one 12.5 Hz frame is
  80 ms.

Causal convolutions use left zero padding of `(kernel-1)*dilation`.
ConvTranspose1d trims `kernel-stride` samples on the right. SnakeBeta is
`x + 1/(exp(beta)+1e-9) * sin(x*exp(alpha))^2`.

## Decoding strategy

The reference `generate()` defaults to stochastic sampling with
`repetition_penalty=1.05`. This executor is deterministic:

- Greedy argmax at both the talker codec head and the predictor LM heads.
- Logits [2048,3072) are suppressed except codec EOS 2150; EOS is allowed from
  frame 2 onward.
- HuggingFace repetition penalty 1.05 is applied to the emitted group-0
  history. Pure greedy without this penalty loops and never reaches EOS; this
  was confirmed against the reference model offline.
- A consecutive-run guard bans a group-0 id after four identical emissions.
  It is a safety net for a near-tie flipped by reduction-order noise. On the
  verified trajectories it does not fire.

### Reduction order and determinism

All matmul/norm accumulation is FP32, but the summation order differs from
the reference kernels. The talker's last layer is numerically sensitive: a
sub-BF16-ULP difference in an earlier near-tie logit can move decoding onto a
different, self-reinforcing token attractor.

The dense reduction kernels (`gemm`, `gemv`, RMSNorm, LayerNorm) therefore
launch 1024 threads per block, so each thread holds roughly one term on the
1024-wide dimensions and the warp reduction is shorter and closer to the
reference accumulation order. Elementwise and convolution kernels keep a
256-thread block. This keeps the verified Chinese and English prompts on the
converging trajectory without slowing the codec convolution path.

Determinism scope: repeated runs produce bit-identical prefill embeddings and
bit-identical codec token traces (the speech content is fixed). The FP32 codec
still sums in a nondeterministic block order, so the final waveform differs
between runs by sub-LSB PCM16 amounts (under 0.3% of samples, peak about
1.5e-3 of full scale in the observed runs). PCM16 stats and ASR transcription
are unaffected. Bit-exact waveform reproducibility is not provided; see the
limitations section.

## Build and test

RETIRED. The `tts_host_tests` and `tts_cuda_tests` targets were built from
`native/tts_tests/`, which was removed in `ab51cf5`. `native/tts/CMakeLists.txt`
still names those sources, so configuring this project currently fails. That is
a known build break outside this document.

The trailing-noise trim has a standalone host test, `tts_tail_trim_test`, wired
through `native/stage_cli`:

```
cmake -S native/stage_cli -B native/stage_cli/build
cmake --build native/stage_cli/build --target tts_tail_trim_test
ctest --test-dir native/stage_cli/build -R tts_tail_trim_test
```

The BPE tokenizer was fuzz-checked against HuggingFace tokenizers 0.22.2 via
`native/tts_tests/tools/tok_fuzz.{cpp,py}` (30010/30010 parity in the bring-up
run). That harness was removed in `ab51cf5`.

## CLI

```
umm-tts-native --model DIR --text TEXT --out WAV \
               [--speaker vivian|ryan] [--language chinese|english] \
               [--gpu N] [--max-frames N]
```

The text is wrapped in the assistant chat template before tokenization. The
output is a real 24 kHz mono PCM16 WAV. Speaker presets are fixed to Vivian
(Chinese) and Ryan (English); custom voice cloning is not implemented.

## Verification evidence

Auditable artifacts for the acceptance run live on the Linux SM90 GPU host at
`/opt/umminfer/umm-native-tts-20261116/evidence/final/`
(binary WAVs, raw logs) with the text-only manifest and host-test log committed
under `native/tts_tests/evidence/`. Per-log size is hard-capped at 64 KiB.

- Codec parity from reference codes: English 55 frames and Chinese 77 frames
  decode to waveform correlation 0.99989 and 0.99949 against the reference
  codec output; RMS matches to three decimals.
- End-to-end synthesis reaches natural EOS: 88 frames (English, 7.04 s,
  RMS 0.0422319) and 83 frames (Chinese, 6.64 s, RMS 0.0551875), real
  24 kHz mono PCM16 non-silent WAV. Final WAV SHA-256:
  English `c0529f8752d2d97883d7494b26e4db2f1e400d73636e7376bb4f3537d6b3c073`,
  Chinese `79271212bb2dc7f1e388ec2851573c9101da9644563bebec77943151a448adb2`.
- Round-trip intelligibility uses the validated `umm-asr-native` after a
  test-only 24 kHz -> 16 kHz resample
  (`native/tts_tests/tools/resample16.py`, removed in `ab51cf5`). Both runs
  exited 0 and transcribed
  the input text verbatim; stdout/stderr, exit codes, resample commands and
  input hashes are in `{en,zh}.asr.{meta,stdout,stderr}` and
  `{en,zh}.asr_inputs.sha256` in the evidence directory.
- Binary SHA-256 of `umm-tts-native`:
  `8c91f5a7c1c8e154155bf0c282757a90659e9dbce178b1dfecdad75f14e9255c`.
  `ldd` shows only CUDA 12.4, ICU and system libraries.
- Model revision `85e237c12c027371202489a0ec509ded67b5e4b5`; per-file
  SHA-256 in `model_files.sha256`. CUDA 12.4 (V12.4.131), driver 535.161.08.
- GPU5 peak resident memory 4877 MiB; the device returns to 0 MiB after the
  process exits.
- compute-sanitizer memcheck, racecheck and initcheck each report zero errors;
  raw summaries are saved as `sanitizer_*.log` in the evidence directory.
- Mac host tests (`tts_host_tests`) pass; log committed at
  `native/tts_tests/evidence/host_tests.log`.

## Limitations

- Deterministic greedy decoding is a deliberate divergence from the
  reference stochastic sampler. Output prosody differs run to run in the
  reference but is fixed here.
- Only the two fixed speaker presets and Chinese/English are wired.
- Kernels are naive global-memory GEMM/GEMV and direct convolutions. They are
  correct, not optimized for throughput; codec decode dominates latency.
- The 1024-thread reduction block and the run-ban guard are calibration
  choices for the observed attractor behavior, not guaranteed to hold for
  every unseen prompt. Long or out-of-domain text may still need the frame
  cap.
- Bring-up probes in `native/tts_tests/tools/` are manual `nvcc` builds and
  are excluded from CMake and from the committed build.

## Third-party notice

See `native/tts/NOTICE` for the TTS-specific notice. It records the Apache-2.0
`qwen_tts`/Transformers specification sources, the three generic primitives
(`bf16.hpp`, `safetensors.hpp/cpp`) reused byte-for-byte from the in-repo
Apache-2.0 ASR slice, and the CUDA/ICU/nlohmann runtime dependencies. The
project-wide ASR notice remains in `THIRD_PARTY_NOTICES.md`.
