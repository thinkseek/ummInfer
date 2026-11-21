# Native stages behind the UnifiedEngine

The five accepted native executors run as external stage processes. At
runtime the deployment loads no Python, PyTorch or SGLang: the stage binary
holds the engine, the server holds the `Coordinator` and `UnifiedEngine`.

```
HTTP / SSE client -> umm-infer serve (Coordinator + UnifiedEngine)
                        |  ZeroMQ control edges + SHM payload relay
        +---------------+-----------------+----------------+----------------+
        |               |                 |                |                |
  umm-native-stage-asr  umm-native-stage-qwen38  umm-native-stage-image  umm-native-stage-tts
  (GPU 2)               (GPU 3)                   (GPU 4)                 (GPU 5)
        |               |                 |                |                |
  native/umm_asr_native native/qwen38     native/image      native/tts
```

Each process loads one model on one GPU and registers with the existing
`Stage::Config` contract (`recv`, `coordinator`, `abort`, `announce_ready`).
No wire message, endpoint type or payload schema is new.

`umm-native-stage-omni` is still built and runs on the same GPU 3, but the
default deployment does not start it. Exactly one VLM stage runs at a time.
`vlm_stage` in the service config selects it.

## Binaries

| Binary | `--kind` | Engine | Default `max_new_tokens` / frames |
| --- | --- | --- | --- |
| `umm-native-stage-asr` | `asr` | `native::AsrEngine` | 256 |
| `umm-native-stage-omni` | `omni` | `omni::OmniEngine` | 512 (`--max-new-tokens`); the H20B launcher sets 96 |
| `umm-native-stage-qwen38` | `qwen38` | `qwen38::Engine` | 256 (`--max-new-tokens`); the H20B launcher sets 96 |
| `umm-native-stage-tts` | `tts` | `native::TtsEngine` | 8192 (`--max-frames`) |
| `umm-native-stage-image` | `image` | `image::ImageEngine` (`2512`, default) / `image2::Image2Engine` (`2.1`) via `--image-backend` | 50 (`2512`) / 40 (`2.1`) (`--steps`) |

The compute function runs on the `Stage` scheduler thread, whose CUDA current
device defaults to 0. Every kind calls `cudaSetDevice(--gpu)` at the start of
each request; the engines set the device only in their constructors.

Measured on H20B: after the MoE host-path fix ([OMNI_NATIVE.md](OMNI_NATIVE.md)),
the Omni engine decodes at 30.6 tok/s on the image case and 14.79 ms/token on
the text case (earlier pre-fix runs decoded at ~2 s/token, which is why the
H20B launcher defaulted to 96 tokens).
The Qwen3.8-27B engine decodes at 42.0 tok/s for text and 29.2 tok/s for an
image prompt, with 0.16 s and 1.05 s prefill. The ASR stage returns a 15 s
clip in about 1 s after model load.

One executable per kind, from one `main.cpp`. `native/umm_asr_native` and
`native/tts` both define `umminfer::native::SafeTensors` and `TensorInfo`, so a
single binary cannot link both without editing the accepted trees. `--kind` is
still accepted and must match the compiled kind; a mismatch exits 2.

Common flags: `--model DIR --gpu N --recv-endpoint EP --coordinator-endpoint EP
--abort-endpoint EP [--name NAME] [--next name=EP] [--peer name=EP]`.

The model is loaded before `Stage::start()`, so `stage_ready` means "weights
resident", not "process alive".

## Wire shapes

ASR (`inputs` of the entry payload):

```
{"audio": <Bin PCM16 16 kHz mono>}        # or a 16 kHz mono PCM16 WAV
-> {"text", "generated_token_count", "finish_reason"}
```

The HTTP boundary already decodes `input_audio` WAV to PCM16, so the raw path is
the one in use. A WAV container is also accepted for direct stage clients.

Omni (the optional VLM leg, after `UnifiedEngine::build_vlm_inputs`):

```
{"messages": [{"role": "user", "content": "text" | [text part, image_url part]}],
 "modalities": ["text", "image"]}
-> {"content", "generated_token_count", "finish_reason"}
```

One `image_url` part is accepted. Its `url` must be a `data:` URL or a local
path; `http(s)` is rejected with an explicit error instead of being fetched.
Prompt rendering and the `image_pad` expansion copy `native/omni_cli`.

Qwen38 (the default VLM leg, after `UnifiedEngine::build_vlm_inputs`):

```
{"messages": [{"role": "user", "content": "text" | [text part, image_url part]}],
 "modalities": ["text"] | ["text", "image"]}
-> {"content", "generated_token_count", "finish_reason"}
```

The request and result shapes are identical to Omni, so the unified engine and
the HTTP layer need no branch. Prompt rendering and the `image_pad` expansion
copy `native/qwen38`; thinking is disabled, matching the accepted CLI default.
Image preprocessing uses the same bounds as `qwen38_cli` (65536 to 16777216
pixels, factor 32).

Qwen3.8-27B has a text decoder and a vision tower, but no audio tower. Every
field the engine cannot consume is an error, never a silently dropped input:

| Input | Result |
| --- | --- |
| content part `input_audio`, `audio_url`, `video_url` or any other type | error naming the part type; no generation runs |
| `modalities` containing `audio` or `video` | error naming the modality |
| an `audio` byte field in the inputs | error: audio goes through the ASR stage |
| more than one `image_url` part | error: one image per request |
| an `image_url` with an `http(s)` URL | error; send a `data:` URL or a local path |
| an image with `grid_t != 1` | error: still images only |

`native/stage_tests/test_host.cpp` covered each row. It was removed in
`ab51cf5`, together with the rest of `native/stage_tests/`; no automated host
test for this table remains. The VLM stage is also reachable directly, without
the unified engine in front of it, so any future check cannot rely on the server
having already removed these fields.

`max_new_tokens` from request `params` is read via `bound_request_params()`
and clamped to `max_seq` (falling back to `--max-new-tokens` when omitted);
`temperature` and sampling controls are ignored (greedy decoding).

TTS (the audio leg, from `UnifiedEngine::build_tts_inputs`):

```
{"text", "voice", "format": "wav", "sample_rate": 24000}
-> {"media_type": "audio", "mime_type": "audio/wav", "format": "wav",
    "sample_rate": 24000, "channels": 1, "voice": "<used>", "data": <Bin WAV>}
```

`voice` accepts `vivian` and `ryan` (case-insensitive). Any other non-empty
label falls back to `vivian` with a warning, and the descriptor reports the
voice actually used. The codec language id is chosen from the text script
(CJK -> chinese, otherwise english).

Image (the image leg, from `UnifiedEngine::build_image_inputs`):

```
{"prompt", "negative_prompt"?, "size": "512x512", "seed"?, "steps"?,
 "guidance"?, "output_format": "png"}
-> {"media_type": "image", "mime_type": "image/png", "format": "png",
    "size": "512x512", "width": 512, "height": 512, "seed": N,
    "steps": N, "data": <Bin PNG>}
```

The accepted image slice currently supports only `512x512` PNG. Other sizes
and JPEG are rejected explicitly. `steps` defaults to 50 for `2512` and 40 for
`2.1`; zero selects the process default.

## Streaming deltas

A stage compute function is one call that returns one result, so it cannot
stream through its return value. It calls `umminfer::emit_stage_stream(Value
chunk)` instead. `Stage` turns each chunk into the existing `StreamMessage`,
the `UnifiedEngine` accumulates VLM chunks into `vlm_text` and forwards them as
`vlm_delta` events, and the HTTP layer writes them as SSE `delta.content`. No
wire message is new.

The scheduler binds the emitter to the request id for the duration of that
compute call. A chunk can never be attributed to another request, and a call
outside a compute function is dropped.

Both `umm-native-stage-qwen38` and `umm-native-stage-omni` emit one chunk per
generated token as `{"text": "<new suffix>"}`. Both buffer incomplete trailing
UTF-8 bytes in O(1) per token (`utf8_complete_prefix`) so a multi-byte character
split across two tokens is never sent as half a codepoint. The aggregate result
still arrives at completion, which keeps non-streaming requests and the
TTS/image legs unchanged. `omni` also supports pure-text prefix KV reuse
(`UMM_OMNI_PREFIX_CACHE`, enabled by default).

`umm-native-stage-tts` emits raw PCM16 chunks (`format: "pcm16"`, 24 kHz mono)
when `inputs.stream=true`, and checks bit-for-bit equality against the terminal
waveform (`stream_matches`). The `asr` and `image` stages do not emit chunks;
their outputs arrive once in the completed result.

## Cancellation

`POST /abort` broadcasts the existing `AbortMessage`; the wire protocol is
unchanged. The `Stage` scheduler binds the abort flag to the exact request id
and passes it to the compute function (`CancellableComputeFn`). Each engine
checks it at its own boundary and raises `RequestAborted`:

| Engine | Boundaries |
| --- | --- |
| ASR | every audio-tower layer, every prefill layer, every generated token |
| Omni | every vision block, every prefill layer, every generated token |
| Qwen38 | before the vision tower, every prefill layer (64), every decode step |
| TTS | every generated codec frame; inside the waveform decoder: after the pre-conv, every pre-transformer layer, every upsample stage, every decoder block and every residual unit |
| Image | after model load and text encoding, every denoise step, after VAE decode |

The exception takes the existing abort path: the Coordinator has already
settled the request as `cancelled`, the scheduler consumes the tombstone and
emits no result. The check is bound to one request id, so an abort for another
id, or a late abort for a finished request, cannot stop running work. The stage
logs the boundary it stopped at plus `abort_epoch_ms` and `exit_epoch_ms`.

The boundaries run on the host between kernel launches, so they change no
numeric result. An empty check keeps the original execution bit for bit.

## Audio understanding: ASR -> VLM text

The Omni stage could answer from audio inside the VLM stage. Qwen3.8 cannot.
Audio still works end to end because the unified engine already runs the ASR
stage first and appends the transcript to the VLM prompt:

| before | now |
| --- | --- |
| audio -> Omni stage (audio tower + text) | audio -> ASR stage -> transcript -> Qwen3.8 text |

API impact:

- The public request shape does not change. `input_audio` parts,
  `/v1/audio/transcriptions`, and the `asr_partial` / `asr_final` events keep
  their contract.
- The answer now depends on a transcript, not on the acoustic signal. Tone,
  speaker identity, emotion and non-speech sounds are lost.
- ASR word errors and truncation propagate into the answer.
- An audio turn includes the ASR pass (about 1 s for a 15 s clip) before the
  VLM starts.
- Audio sent directly to the qwen38 stage, bypassing the unified engine, is
  rejected with an explicit error instead of being answered from the text half.

## Behaviour differences from the Python stages

- Request `params`: `max_new_tokens` reaches `qwen38` and `omni` via
  `bound_request_params()` (clamped to `max_seq`), and `inputs.stream` reaches
  `tts`. `temperature`, sampling controls and `asr_partials` are ignored by the
  native stages; the ASR leg delivers its final transcript (realtime previews
  are driven by the server's 200 ms preview loop in `realtime_api.cpp`).
- Stages default to one request at a time (`SimpleScheduler` `kSerial`);
  `qwen38` supports opt-in decode continuous batching via `--batch-size N`
  (`kThreadSafe`, see [docs/batching.md](../docs/batching.md)).
- ASR is 16 kHz mono only, at most 30 s per request, inherited from the
  accepted frontend.

## Build

Host (`tts_tail_trim_test` only, no CUDA):

```sh
cmake -S native/stage_cli -B build-host -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build-host && ctest --test-dir build-host
```

GPU host:

```sh
cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/root/umm-c6rt-ckl/deps
cmake --build build -j
```

## Run

```sh
service/h20b-native-service.sh start     # server + four stages
service/h20b-native-service.sh stop
```

The live smoke harness `native/stage_tests/native_stage_smoke.py` was removed in
`ab51cf5`. For a request-level end-to-end run use `docs/cases/0001_repro.sh`.

The supervisor (`python/umminfer_service.py`) reads the same selection from
`vlm_stage` in its config. Stages listed under `optional_stages` are validated
but not started, which is how the omni entry is kept available.

The committed evidence under `native/stage_tests/evidence/` was removed in
`ab51cf5` with the harness.
