# Third-party notices — native Qwen3-ASR slice

## Copied source code

None. The CUDA kernels, audio frontend, safetensors/BF16 loader, tokenizer
and greedy decoder in `umm_asr_native/` were written from scratch for this
slice. No source files were copied from other projects into this directory.

## Specifications implemented from upstream references (read-only, not copied)

The math and tensor layouts were derived by reading these sources on the
build host and cross-checking numeric outputs against them:

- HuggingFace Transformers 5.6, Apache-2.0
  - `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
    (Whisper-style audio encoder, Qwen3 text decoder, RoPE, RMS/LayerNorm)
  - `transformers/audio_utils.py` (Slaney mel filter bank, STFT framing)
  - `transformers/processing_qwen3_omni_moe.py` (audio chunking, prompt tokens)
- SGLang, Apache-2.0
  - `sglang/srt/multimodal/processors/qwen3_asr.py` (DEFAULT_ASR_PROMPT)
  - `sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`
    (language-prefix handling, max_new_tokens)
- safetensors format specification, Apache-2.0 (header + data_offsets layout)

Only formulas, constants and tensor name/shape conventions were taken; the
C++/CUDA implementation is original.

## Runtime dependencies (linked, not embedded)

- NVIDIA CUDA 12.4 runtime (`libcudart`) — EULA, system-provided on the host.
- nlohmann/json — MIT, header-only, resolved from the system include path;
  no third-party source is vendored in this repository.
- C++20 standard library and GCC/libstdc++.

No inference engine (PyTorch, vLLM, SGLang, TensorRT, cuDNN, cuBLAS) is linked
by the final `umm-asr-native` binary. cuBLAS was evaluated and removed because
the host's cuBLAS 12.4 cannot serve the model's GEMM shapes under driver
535.161.08; all GEMM/GEMV and attention are self-contained CUDA kernels.

---

# Third-party notices — native Qwen-Image-2512 slice

## Copied source code

None. Every engine source under `native/image/` and test under
`native/image_tests/` was written from scratch. No flash-attention, vLLM or
SuperInfer source was vendored; the online-softmax flash kernel is an original
implementation validated against brute-force fp32 host references.

## Specifications implemented from upstream references (read-only, not copied)

Tensor layouts, modulation order, RoPE conventions and scheduler math were
derived by reading these packages on the build host and cross-checking numeric
outputs against bf16 golden blobs they produce:

- HuggingFace Transformers (Qwen2.5-VL text/timestep encoder, RoPE, RMSNorm),
  Apache-2.0.
- diffusers 0.36 (`QwenImagePipeline`, `QwenImageTransformer2DModel`,
  `QwenImageTransformerBlock`, `AutoencoderKLQwenImage`,
  `FlowMatchEulerDiscreteScheduler`, AdaLayerNormContinuous), Apache-2.0.
- Qwen byte-level BPE tokenizer and the safetensors format specification,
  Apache-2.0.

Only formulas, constants and tensor name/shape conventions were taken; the
C++/CUDA implementation is original.

## Runtime dependencies (linked, not embedded)

- NVIDIA CUDA 12.4 runtime, cuBLAS and cuBLASLt (`libcudart`, `libcublas`,
  `libcublasLt`) — NVIDIA EULA, system-provided on the host. The dense BF16
  GEMMs go through cuBLASLt; attention, norms, convs, RoPE, modulation and the
  RNG are self-contained CUDA kernels.
- zlib (`libz`) — zlib license, system-provided; used only for PNG encoding.
- nlohmann/json — MIT, header-only, resolved from the system include path.
- C++20 standard library and GCC/libstdc++.

No PyTorch, vLLM, SGLang, TensorRT or cuDNN runtime is linked by
`umm-image-native`; Python/diffusers is used only offline to generate golden
reference tensors.


---

# Third-party notices — native Qwen-Image-2.1 slice

## Copied source code

None. The engine under `native/image2/`, the CLI under `native/image2_cli/`
and the tests under `native/image2_tests/` were written from scratch. The
tensor, safetensors, PNG and kernel files are adapted from the
`native/image/` slice in this repository, also original code.

## Specifications implemented from upstream references (read-only, not copied)

Tensor layouts, block-causal attention, `causal_condition` modulation, mRoPE
indexing, the VAE decoder and the scheduler math were derived by reading these
packages on the development host and cross-checking numeric output against
reference blobs they produce:

- HuggingFace Transformers 5.6 (Qwen3-VL text encoder, mRoPE, per-head q/k
  RMSNorm), Apache-2.0.
- diffusers 0.43 (`QwenImage21Pipeline`, `QwenImage21Transformer2DModel`,
  `AutoencoderKLQwenImage21`, `FlowMatchEulerDiscreteScheduler`,
  AdaLayerNormContinuous), Apache-2.0.
- Qwen byte-level BPE tokenizer and the safetensors format specification,
  Apache-2.0.

Only formulas, constants and tensor name/shape conventions were taken; the
C++/CUDA implementation is original. Python and diffusers are used offline
only to produce golden reference tensors and are not runtime dependencies.

## Runtime dependencies (linked, not embedded)

Same set as the Qwen-Image-2512 slice: CUDA 12.4 runtime, cuBLAS and cuBLASLt,
zlib, nlohmann/json and the C++20 standard library. `umm-image2-native` links
no PyTorch, vLLM, SGLang, TensorRT or cuDNN runtime.
