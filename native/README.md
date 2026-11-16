# Native Qwen3-ASR vertical slice

A self-contained C++20/CUDA inference path for the Qwen3-ASR 0.6B model that
loads the real BF16 safetensors checkpoint and transcribes 16 kHz mono WAV
files. No Python, SGLang, vLLM, PyTorch, cuDNN or cuBLAS at runtime — only the
CUDA runtime plus hand-written kernels.

## Layout

- `umm_asr_native/wav.*` — PCM16 RIFF WAV parser and validation.
- `umm_asr_native/feature_frontend.*` — Whisper-compatible log-mel frontend
  (128 mels, 3000 frames, Slaney filter bank, reflect-pad STFT, log floor).
- `umm_asr_native/safetensors.*` — safetensors header/BF16 tensor reader.
- `umm_asr_native/bf16.hpp` — host-side BF16 scalar used by host tests.
- `umm_asr_native/tokenizer.*` — Qwen2 byte-level vocab decode and fixed ASR
  prompt construction (encoding needs only constant special-token ids).
- `umm_asr_native/cuda_kernels.*` — RMS/LayerNorm, RoPE, GQA attention with
  KV-cache, naive BF16 GEMM/GEMV, conv3x3/s2, GELU/SiLU, embeddings, argmax.
- `umm_asr_native/engine.*` — weight loading, audio tower, 28-layer text
  decoder and greedy decode loop.
- `cli/main.cpp` — `umm-asr-native <model_dir> <wav> [gpu_id]`.
- `tests/` — host unit tests and a GPU GQA attention test.

## Build

```sh
cmake -S . -B build -DUMM_ASR_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Host-only tools (no GPU) build without `-DUMM_ASR_CUDA`:

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

## Run

```sh
CUDA_VISIBLE_DEVICES=2 ./build/umm-asr-native /path/to/asr /path/to/input.wav 0
```

The transcript is written to stdout; timing and GPU memory to stderr.

## Scope and limitations

- Greedy decoding only; no sampling, temperature or beam search.
- Fixed KV capacity of 1024 tokens; sufficient for 30 s clips (the
  preprocessor pads every input to 30 s, longer audio is rejected upstream).
- Naive global-memory GEMM/GEMV and O(n^2) attention. Correctness-first;
  throughput is adequate at the 0.6B scale but not optimized for long audio.
- Only the exact checkpoint dimensions in `config.json` are accepted; a
  mismatch fails loudly rather than running on wrong shapes.
- 16 kHz mono PCM16 WAV only; other sample rates and formats are rejected.
