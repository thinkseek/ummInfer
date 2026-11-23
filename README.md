# ummInfer

ummInfer is an inference runtime for unified multimodal models. A C++20 control
plane serves an OpenAI-compatible HTTP/SSE surface plus a WebSocket realtime
transcription/voice surface (`/v1/realtime`), and drives one native C++/CUDA
model executor per GPU as an external stage process. No Python, PyTorch, vLLM
or SGLang is loaded on the serving path; `ldd` on every stage binary lists the
CUDA runtime, libzmq, libmsgpackc and system libraries only.

## Architecture

```mermaid
flowchart TB
  CLIENT["HTTP / SSE / WebSocket client<br/>curl · umm-infer CLI · C++ client · macOS ummmlx · web"]
  SERVER["<b>umm-infer serve</b><br/>HTTP + SSE + WS · Coordinator · UnifiedEngine<br/>request ids · abort · /health /ready /metrics"]
  ZMQ["ZeroMQ + MessagePack control edges<br/>POSIX shared-memory payload relay"]

  ASRG["<b>umm-native-stage-asr</b><br/>GPU 2 · native/umm_asr_native"]
  VLM["<b>umm-native-stage-qwen38</b><br/>GPU 3 · native/qwen38 (VLM leg)"]
  OMNI["<b>umm-native-stage-omni</b><br/>GPU 3, optional VLM · native/omni"]
  IMGG["<b>umm-native-stage-image</b><br/>GPU 4 · native/image (2512) / native/image2 (2.1)"]
  TTSG["<b>umm-native-stage-tts</b><br/>GPU 5 · native/tts"]

  CLIENT --> SERVER
  SERVER --> ZMQ
  ZMQ --> ASRG
  ZMQ --> VLM
  ZMQ -.-> OMNI
  ZMQ --> IMGG
  ZMQ --> TTSG
```

One request runs an ordered plan: optional ASR, then the VLM leg, then the
optional image and TTS output legs. Every stage registers with the existing
`Stage::Config` contract (`recv`, `coordinator`, `abort`, `announce_ready`).
Each process loads one model on one GPU; stages default to one request at a
time, while `qwen38` supports opt-in decode continuous batching via
`--batch-size N` (set to `8` in `service/native-service.json`).
Exactly one VLM stage runs at a time: `vlm_stage` in the service config picks
`qwen38` (default) or `omni`.

| Path | Stages | Terminal |
| --- | --- | --- |
| Text chat | VLM | `complete`; both `qwen38` and `omni` emit `vlm_delta` per token |
| Audio -> text | ASR -> VLM | `asr_final`, then VLM deltas, then `complete` |
| Image + text -> text | VLM | `complete` |
| Text -> speech | VLM -> TTS | `complete` with one `artifact` (WAV), or streamed PCM16 `artifact` chunks when `audio.stream=true` |
| Text -> image | VLM -> image | `complete` with one `artifact` (PNG) |
| Text + image + audio -> text + image + audio | ASR -> VLM -> image ∥ TTS | `complete` with two `artifact`s |

## Quick start

Build the server (Zig 0.16.0). The native stage executables are a separate
CMake project and need a CUDA host.

```sh
zig build -Doptimize=ReleaseFast \
  -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl

cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DCMAKE_PREFIX_PATH=/path/to/deps` when zmq, msgpack, nlohmann-json or ICU
are outside the default include and library paths.

Start the server and one stage per model. The server queues requests until a
stage announces readiness; `/ready` answers 200 after the entry and VLM stages
are ready, and the optional image/TTS legs never gate it.

```sh
IPC=$(mktemp -d /tmp/umminfer-native.XXXXXX)

./zig-out/bin/umm-infer serve --host 127.0.0.1 --port 43301 --model qwen3-native \
  --entry asr --entry-endpoint "ipc://$IPC/asr" \
  --vlm-entry qwen38 --vlm-entry-endpoint "ipc://$IPC/qwen38" \
  --image-entry image --image-entry-endpoint "ipc://$IPC/image" \
  --tts-entry tts --tts-entry-endpoint "ipc://$IPC/tts" \
  --completion "ipc://$IPC/completion" --abort "ipc://$IPC/abort"

build/umm-native-stage-asr --kind asr --name asr --model "$ASR_MODEL" --gpu 2 \
  --recv-endpoint "ipc://$IPC/asr" --coordinator-endpoint "ipc://$IPC/completion" \
  --abort-endpoint "ipc://$IPC/abort"

build/umm-native-stage-qwen38 --kind qwen38 --name qwen38 \
  --model "$QWEN38_MODEL" --gpu 3 --max-new-tokens 96 \
  --recv-endpoint "ipc://$IPC/qwen38" --coordinator-endpoint "ipc://$IPC/completion" \
  --abort-endpoint "ipc://$IPC/abort"

# Optional output legs.
build/umm-native-stage-image --kind image --name image --model "$IMAGE_MODEL" --gpu 4 \
  --recv-endpoint "ipc://$IPC/image" --coordinator-endpoint "ipc://$IPC/completion" \
  --abort-endpoint "ipc://$IPC/abort"
build/umm-native-stage-tts --kind tts --name tts --model "$TTS_MODEL" --gpu 5 \
  --recv-endpoint "ipc://$IPC/tts" --coordinator-endpoint "ipc://$IPC/completion" \
  --abort-endpoint "ipc://$IPC/abort"
```

One request:

```sh
curl -sS http://127.0.0.1:43301/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","messages":[{"role":"user",
       "content":"What is the sea?"}]}'
```

Recorded response from the native Omni stage (abridged; HTTP 200 in 38.2 s).
The Qwen3.8-27B stage returns the same JSON shape with a different answer:

```json
{"choices":[{"finish_reason":"stop","index":0,"message":{
  "content":"The sea is a vast, ever-changing expanse of saltwater that stretches to the horizon.",
  "role":"assistant"}}],"id":"req-2","model":"qwen3-native",
 "object":"chat.completion"}
```

Transcription:

```sh
curl -sS -F model=qwen3-native -F file=@input_16k.wav \
  http://127.0.0.1:43301/v1/audio/transcriptions
```

On SM90 the same layout is launched by
`service/native-service.sh start|stop|status`, which reads the
`UMM_NATIVE_*` environment variables. The supervised path is
`python3 -I python/umminfer_service.py --config service/native-service.json
start`, which takes the same layout from that file and restarts unhealthy
children. Set `UMM_NATIVE_OMNI=1` to serve Omni instead of Qwen3.8-27B as the
VLM leg. Both paths need `zig-out/bin/umm-infer` and `build/umm-native-stage-*`
installed in the same directory; see [the SM90 runbook](docs/native-service.md).

## Supported models

Measured on SM90, BF16, greedy decoding, one request in flight. Each engine's
own document has the full conditions and limits.

| Model | Native implementation | GPU | Measured |
| --- | --- | --- | --- |
| Qwen3-ASR 0.6B | `native/umm_asr_native` | 2 | 15.05 s clip -> 46 tokens in ~1.0 s after model load (4.8 s) |
| Qwen3.8-27B (hybrid linear/full attention) | `native/qwen38` | 3 | text: 0.16 s prefill, 42.0 tok/s; 512x512 image: 1.05 s prefill, 29.2 tok/s |
| Qwen3-Omni thinker (MoE, 57.87 GiB), optional | `native/omni` | 3 | image case: vision 0.363 s, prefill 0.231 s, decode 30.6 tok/s; text case 14.79 ms/token after the MoE host-path fix |
| Qwen3-TTS 0.6B CustomVoice | `native/tts` | 5 | 4.24 s speech in 7.2 s after the codec-conv fix (29.1 s before) |
| Qwen-Image-2512 | `native/image` | 4 | 512x512, 50 steps: 228.8 s (runs: 227.5-229.5 s) |
| Qwen-Image-2.1 | `native/image2` | 7 | 512x512, 40 steps: 69.8 s, peak 14,763 MiB |

Qwen3.8-27B is the default VLM stage. It has a text decoder and a vision tower
but no audio tower: audio and video parts, more than one image, `http(s)` image
URLs and `modalities` containing `audio` are rejected with an error that names
the field. ASR, Omni and TTS answers are greedy and deterministic.

## Layout

```text
build.zig                       server, client, CLI, C++ contract tests
include/umminfer/               public C++ headers (client, proto, relay, serve)
src/                            server: coordinator, unified engine, HTTP/SSE/WS, CLI, metrics
native/umm_asr_native/          Qwen3-ASR engine + CLI
native/omni/  native/omni_cli/  Qwen3-Omni thinker engine + CLI
native/qwen38/                  Qwen3.8-27B executor + CLI
native/tts/   native/tts_cli/   Qwen3-TTS engine + CLI
native/image/ native/image_cli/ Qwen-Image-2512 engine + CLI
native/image2/                  Qwen-Image-2.1 engine + CLI
native/stage_cli/               the five umm-native-stage-* binaries
clients/ummmlx/                 native macOS SwiftUI client + Swift unit tests
web/                            single-file browser client (index.html)
benchmarks/                     multimodal benchmark manifest (70 cases) and data fetcher
service/                        SM90 launcher, concurrency stack, supervisor config
tests/                          C++ contract tests, Python E2E, live/benchmark harnesses
tools/                          feature-map anchor checker, audio analysis, release publisher
release/                        public release templates and verify script
python/                         process supervisor (umminfer_service.py), legacy MLX/TinyASR
                                stages, and offline RL tooling (not on the native serving path)
docs/                           feature map, roadmap, contracts, benchmarks, validation, runbooks
```

`native/omni`, `native/tts`, `native/image`, `native/image2` and
`native/stage_cli` each have their own CMake project. Qwen3.8-27B builds from
`native/CMakeLists.txt` with `-DUMM_QWEN38_CUDA=ON`. `native/stage_cli` links
the accepted engine trees without modifying them.

## Tests and validation

Server and control plane (CPU, no model, no CUDA):

```sh
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

This builds and runs `tests/unified_contract_test.cpp`, the end-to-end contract
suite. Unit-level suites (`self_check`, `wire_compat`, `runtime_compat`,
`omni_transport_test`, `omni_serve_test`, `omni_client_test`) were removed in
`ab51cf5`. A non-zero exit is a real failure.

Host tests that need no weights and no GPU:

```sh
cmake -S native/omni -B native/omni/build && cmake --build native/omni/build -j
ctest --test-dir native/omni/build

cmake -S native/stage_cli -B build-host && cmake --build build-host
ctest --test-dir build-host
```

`native/tts`, `native/image`, `native/image2` and the Qwen3.8-27B project have
their own host targets with the same shape. GPU parity and kernel tests run on
the CUDA host, one project at a time, as `ctest` in each build directory. Gate
values, golden-data provenance and known gaps are in each executor's document.

The live smoke harness `native/stage_tests/native_stage_smoke.py` and its
committed evidence were removed in `ab51cf5`. Use the case reproduction below.

Full request-level reproduction of a four-stage deployment:

```sh
TARGET=<gpu-host> CONTAINER=<gpu-container> sh docs/cases/0001_repro.sh
```

Case record, per-leg timings and hashes: [docs/cases/0001-end-to-end-native.md](docs/cases/0001-end-to-end-native.md).

CUDA IPC remains separately testable:

```sh
zig build test -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew
zig build cuda-bench -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew -Doptimize=ReleaseFast
```

## Known limitations

- Stages default to one request at a time (`SimpleScheduler` `kSerial`). Opt-in
  decode continuous batching is implemented only for `qwen38` (`--batch-size N`,
  see [docs/batching.md](docs/batching.md)). There is no multi-GPU model split.
- `max_new_tokens` reaches the `qwen38` and `omni` stages (clamped to
  `max_seq`; default 96 in the SM90 launcher/config), and `audio.stream` reaches
  the `tts` stage. `temperature`, sampling, beam search and repetition controls
  are ignored: all native executors are greedy and deterministic.
- Both `qwen38` and `omni` stages stream UTF-8-safe `vlm_delta` chunks per
  generated token and keep the aggregated terminal result; `omni` also supports
  pure-text prefix KV reuse (`UMM_OMNI_PREFIX_CACHE`).
- ASR accepts 16 kHz mono PCM16 only, at most 30 s per request.
- Qwen3.8-27B accepts one still image per request and `max_seq` 8192.
- Image output is 512x512 PNG only (`--image-backend 2512|2.1`). Qwen-Image-2512
  defaults to 512x512 (official default 1024x1024); Qwen-Image-2.1 has no true
  CFG, no KV cache and no condition-image editing.
- TTS has two fixed voice presets (`vivian`, `ryan`), Chinese and English.
  Repeated runs differ by sub-LSB PCM16 amounts; waveforms are not bit-exact.
- Mixed output requires `stream=true`. A non-streaming request with `image` or
  `audio` output is rejected with HTTP 400.
- The image stage loads its weights on the first request after start (about
  9.5 s, +11 GiB), so the first render is slower than steady state.
- Model revisions are `UNKNOWN`; the local model directories carry no `.git`.
- CUDA IPC carries host-serialized payload bytes, not device tensors. CUDA relay
  under real model load, zero-copy transfer and multi-host IPC are `NOT_COVERED`.
- Native VLM stages reject `http(s)` image URLs; they accept `data:` URLs and
  local paths only.

## Documentation

### Core index, architecture and contracts

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Authoritative system architecture, call chains (sync/stream/cancel/error/lifecycle), local verification matrix, and tracked technical debt |
| [docs/feature-map.md](docs/feature-map.md) | Code-anchored feature map across interfaces, engines, stages, clients, benchmarks, tests and deployment (verified by `tools/check_feature_map.py`) |
| [docs/model-support.md](docs/model-support.md) | Model support matrix, Hugging Face license audit, P0/P1/P2 selection roadmap, and real-model verification gates |
| [docs/roadmap.md](docs/roadmap.md) | Phases, acceptance gates, model support plan, verification split |
| [docs/mixed-output-contract.md](docs/mixed-output-contract.md) | Frozen text+image+audio request, streaming and artifact contract |
| [docs/full-duplex.md](docs/full-duplex.md) | Full-duplex voice L1/L2: streaming input, server VAD, endpointing, barge-in |
| [docs/external_stage.md](docs/external_stage.md) | Stage protocol, C++ stage host and legacy external Python stage path |
| [docs/cuda-ipc.md](docs/cuda-ipc.md) | CUDA IPC build, relay and fallback rules |

### Native executors

| Document | Contents |
| --- | --- |
| [native/STAGE_NATIVE.md](native/STAGE_NATIVE.md) | The five `umm-native-stage-*` binaries, wire shapes, streaming, cancellation |
| [native/README.md](native/README.md) | Native Qwen3-ASR 0.6B vertical slice |
| [native/qwen38/README.md](native/qwen38/README.md) | Qwen3.8-27B architecture, parity harness, limits (default VLM) |
| [native/OMNI_NATIVE.md](native/OMNI_NATIVE.md) | Qwen3-Omni thinker engine, matmul kernels, prefill/decode measurements (optional VLM) |
| [native/TTS_NATIVE.md](native/TTS_NATIVE.md) | Qwen3-TTS 0.6B CustomVoice engine, codec, determinism and evidence |
| [native/IMAGE_NATIVE.md](native/IMAGE_NATIVE.md) | Qwen-Image-2512 pipeline and verification |
| [native/image2/IMAGE2_NATIVE.md](native/image2/IMAGE2_NATIVE.md) | Qwen-Image-2.1 pipeline, parity, performance |

### Deployment, operations and clients

| Document | Contents |
| --- | --- |
| [docs/native-service.md](docs/native-service.md) | SM90 native resident service runbook, topology, supervisor (`umminfer_service.py`) |
| [docs/observability.md](docs/observability.md) | Prometheus `/metrics`, NVML GPU gauges, `UMM_TRACE=1` JSONL request tracing |
| [docs/release.md](docs/release.md) | Release packaging (`tools/publish-release.sh`), verification (`release/tools/verify.sh`), rollback |
| [docs/client.md](docs/client.md) | C++ client library and `umm-infer` CLI |
| [clients/ummmlx/README.md](clients/ummmlx/README.md) | Native macOS SwiftUI client (`ummmlx`), local speech input, fault gate |
| [web/README.md](web/README.md) | Single-file browser client (`web/index.html`) |

### Benchmarks, performance and validation

| Document | Contents |
| --- | --- |
| [docs/benchmark-system.md](docs/benchmark-system.md) | Multimodal benchmark datasets (70 cases), metrics, stability gate |
| [docs/multimodal-benchmark-results.md](docs/multimodal-benchmark-results.md) | SM90 real-model multimodal benchmark results (English/Mandarin ASR, GSM8K, VQA, SpokenCOCO) |
| [docs/batching.md](docs/batching.md) | Qwen3.8 decode continuous batching design and SM90 measurements |
| [docs/concurrency.md](docs/concurrency.md) | Admission control (`--max-in-flight`), 503 shedding and capacity measurements |
| [docs/gpu-utilization.md](docs/gpu-utilization.md) | Measured GPU busy fraction, bandwidth, kernel hotspots |
| [docs/soak.md](docs/soak.md) | Soak and fault recovery on the native SM90 service |
| [docs/cases/0001-end-to-end-native.md](docs/cases/0001-end-to-end-native.md) | One recorded four-stage end-to-end request with hashes and timings |
| [docs/validation.md](docs/validation.md) | User journeys, SM90/Mac runs, what remains `NOT_COVERED` |
| [docs/mac-unified.md](docs/mac-unified.md) | Apple Silicon low-resource unified validation |

### Historical baselines, legacy backends and RL

| Document | Contents |
| --- | --- |
| [docs/performance-baseline.md](docs/performance-baseline.md) | Historical SGLang-backed unified performance baseline (2026-11-13) |
| [docs/joint-optimization.md](docs/joint-optimization.md) | Historical worker-concurrency optimization record (2026-11-13) |
| [docs/mlx.md](docs/mlx.md) | MLX-LM text backend on Apple Silicon (legacy local path) |
| [docs/rl.md](docs/rl.md) | RL architecture phase 1: rollout, trajectory export, reward boundary |
| [docs/rl-phase2.md](docs/rl-phase2.md) | RL phase 2: real optimizer-step validation on SM90 |

Licensed under Apache-2.0. Third-party derivations are listed in
[native/THIRD_PARTY_NOTICES.md](native/THIRD_PARTY_NOTICES.md) and each
executor's `NOTICE`.
