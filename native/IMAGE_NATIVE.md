# Native Qwen-Image-2512 vertical slice

A from-scratch C++/CUDA text-to-image engine for Qwen-Image-2512. At runtime it
loads the real BF16 checkpoint and runs the full pipeline — byte-level BPE
tokenization, Qwen2.5-VL language model, 60-block MMDiT diffusion transformer,
FlowMatch Euler scheduler, and the Wan-derived 3D VAE decoder — with no Python,
PyTorch, diffusers, or transformers dependency. Python is used only offline to
produce reference golden tensors for numerical verification.

## Layout

- `image/` — engine sources and its self-contained CMake project
  (`cmake -S image -B build -DUMM_IMAGE_CUDA=ON`). Nothing under here modifies
  the sibling Qwen3-ASR build in `native/CMakeLists.txt`.
  - host-only (builds on macOS, no GPU): `scheduler`, `png` (zlib),
    `safetensors_shard` (mmap, no whole-file load), `tokenizer` (Qwen2 BPE),
    `packing`.
  - CUDA: `tensor`/`cuda_kernels` (cuBLASLt row-major linears, flash attention,
    GQA causal attention, RoPE, norms, 2D convs, Philox RNG), `text_encoder`,
    `dit`, `vae`, `engine` orchestration.
- `image_cli/` — `umm-image-native` CLI.
- `image_tests/` — REMOVED in `ab51cf5`; host tests, tokenizer parity dump and
  GPU golden parity tests no longer exist.

## Pipeline facts implemented (all traced to diffusers 0.36.0 source)

- Fixed prompt template; the first 34 template tokens are dropped after the
  text encoder; retained sequence capped at 512. True-CFG uses two transformer
  forwards and the per-token norm renormalization `neg+s*(cond-neg)` rescaled
  by `||cond||/||comb||`.
- Scheduler: dynamic exponential sigma shift, terminal stretch to 0.02, Euler
  steps. GEMMs use fp32 accumulation through cuBLASLt; norms and softmax keep
  fp32 statistics; BF16 is the storage/transport dtype.
- MMDiT: affine-free LayerNorm + six AdaLN modulations per block, GELU tanh
  (not GEGLU) MLPs, complex-polar RoPE with centered grid coordinates, joint
  `[text,image]` attention with the padding mask intentionally ignored by the
  reference.
- VAE: single-frame decode reduces every causal Conv3d to a 2D convolution on
  the last time slice; `time_conv` upsampling branches are dead code at T=1;
  RMS norms are gamma-only; output clamped to [-1,1].

## Deviations from the official defaults

- Default render size is 512x512 (official default 1024x1024), the minimum
  size whose latent dimensions are all divisible by the required factors. The
  CLI still accepts `--size 1024x1024`.
- RNG is a self-contained Philox4x32 Box-Muller sampler. Fixed seeds are
  self-reproducible, but its bit sequence is not claimed to match
  `torch.randn`; cross-engine numerical checks use dumped golden latent blobs
  so RNG parity is never conflated with operator parity.
- The tokenizer does not run full Unicode NFC normalization. The reference
  fast tokenizer produced identical ids for NFD and NFC inputs across 10k+
  composed-character probes on this vocab, so precomposed UTF-8 prompts (the
  normal IME/terminal output, including the required Chinese prompt) are
  covered; exotic decomposed combining-mark sequences are the known ceiling.

## Third-party code and licenses

No third-party source is vendored. zlib (zlib license) is linked for PNG
deflate; cuBLAS/cuBLASLt and the CUDA runtime are NVIDIA system libraries.
All engine code is original, derived from reading the Apache-2.0 diffusers and
Apache-2.0 transformers source. See `THIRD_PARTY_NOTICES.md`.

## Verification

All GPU numbers come from offline diffusers 0.36 bf16 golden blobs on H20B
GPU4. Deep bf16 networks diverge in elementwise max from fp32 reduction-order
differences, so the deep-net gates use relative L2 and cosine similarity
(content preservation), not max-abs.

- Host (`image_host_tests`, also builds/runs on macOS without CUDA): scheduler
  arrays vs independently computed tables; PNG chunk/CRC and inflate round-trip;
  sharded safetensors mmap; latent pack/unpack; BPE on a synthetic vocab.
- Tokenizer: `tok_dump` compared id-for-id against the HuggingFace Qwen2
  tokenizer on English, Chinese, emoji, math symbols, contractions and
  whitespace edge cases (14 prompts, zero mismatches).
- GPU (`image_gpu_tests`): scheduler arrays; known-answer row-major GEMM
  (exact); text encoder en/zh (cosine 0.999, rel_l2 ~0.04); full 60-block
  MMDiT velocity (cosine 0.99997, rel_l2 0.008); full VAE decoder on a fixed
  latent (rel_l2 0.033, PSNR 44 dB — matches diffusers-torch bf16 on the same
  input, the residual being bf16 online-softmax/NCHW accumulation).
- Kernel known-answer tests vs brute-force fp32 host references:
  `flash_known` (non-causal D=128), `flash384_known` (VAE D=384 path),
  `gqa_known` (causal GQA group 7), `conv1x1_known` (NCHW 1x1 conv). All
  max_abs below 5e-4. `compute-sanitizer` racecheck reports 0 hazards.
- Determinism: the flash online-softmax `block_sum` has an entry barrier that
  closes a shared-memory back-edge race (a fast thread could overwrite
  `warp_sums[0]` while a slow thread still read it across tiles). Two
  same-seed 512px renders now produce byte-identical PNGs.
- End-to-end: `umm-image-native` renders both required prompts (English fox,
  Chinese space-cat) at 512x512, non-monochrome, prompt-relevant, with recorded
  timing/VRAM and PNG SHA-256.

