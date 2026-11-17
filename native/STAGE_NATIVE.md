# Native stages behind the UnifiedEngine

The four accepted native executors run as external stage processes. At
runtime the deployment loads no Python, PyTorch or SGLang: the stage binary
holds the engine, the server holds the `Coordinator` and `UnifiedEngine`.

```
HTTP / SSE client -> umm-infer serve (Coordinator + UnifiedEngine)
                        |  ZeroMQ control edges + SHM payload relay
        +---------------+----------------+----------------+----------------+
        |               |                |                |                |
  umm-native-stage-asr  umm-native-stage-omni  umm-native-stage-image  umm-native-stage-tts
  (GPU 2)               (GPU 3)                (GPU 4)                 (GPU 5)
        |               |                |                |                |
  native/umm_asr_native native/omni      native/image      native/tts
```

Each process loads one model on one GPU and registers with the existing
`Stage::Config` contract (`recv`, `coordinator`, `abort`, `announce_ready`). No
wire message, endpoint type or payload schema is new.

## Binaries

| Binary | `--kind` | Engine | Default `max_new_tokens` / frames |
| --- | --- | --- | --- |
| `umm-native-stage-asr` | `asr` | `native::AsrEngine` | 256 |
| `umm-native-stage-omni` | `omni` | `omni::OmniEngine` | 512 (`--max-new-tokens`); the H20B launcher sets 96 |
| `umm-native-stage-tts` | `tts` | `native::TtsEngine` | 8192 (`--max-frames`) |
| `umm-native-stage-image` | `image` | `image::ImageEngine` | 50 (`--steps`) |

The compute function runs on the `Stage` scheduler thread, whose CUDA current
device defaults to 0. Every kind calls `cudaSetDevice(--gpu)` at the start of
each request; the engines set the device only in their constructors.

Measured on H20B GPU3: the accepted Omni engine decodes about 2 s per token
(one image or text answer of 96 tokens takes roughly 3 minutes). The launcher
caps the answer length for that reason. The ASR stage returns a 15 s clip in
about 1 s after model load.

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

Omni (the VLM leg, after `UnifiedEngine::build_vlm_inputs`):

```
{"messages": [{"role": "user", "content": "text" | [text part, image_url part]}],
 "modalities": ["text", "image"]}
-> {"content", "generated_token_count", "finish_reason"}
```

One `image_url` part is accepted. Its `url` must be a `data:` URL or a local
path; `http(s)` is rejected with an explicit error instead of being fetched.
Prompt rendering and the `image_pad` expansion copy `native/omni_cli`.

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
and JPEG are rejected explicitly. `steps` defaults to 50; zero selects the
process default.

## Cancellation

`POST /abort` broadcasts the existing `AbortMessage`; the wire protocol is
unchanged. The `Stage` scheduler binds the abort flag to the exact request id
and passes it to the compute function (`CancellableComputeFn`). Each engine
checks it at its own boundary and raises `RequestAborted`:

| Engine | Boundaries |
| --- | --- |
| ASR | every audio-tower layer, every prefill layer, every generated token |
| Omni | every vision block, every prefill layer, every generated token |
| TTS | every generated codec frame; inside the waveform decoder: after the pre-conv, every pre-transformer layer, every upsample stage, every decoder block and every residual unit |
| Image | after model load and text encoding, every denoise step, after VAE decode |

The exception takes the existing abort path: the Coordinator has already
settled the request as `cancelled`, the scheduler consumes the tombstone and
emits no result. The check is bound to one request id, so an abort for another
id, or a late abort for a finished request, cannot stop running work. The stage
logs the boundary it stopped at plus `abort_epoch_ms` and `exit_epoch_ms`.

The decoder boundaries run on the host between kernel launches, so they change
no numeric result. An empty check keeps the original execution bit for bit.

## Behaviour differences from the Python stages

- Request `params` (`max_new_tokens`, `stream`, `temperature`) do not reach a
  `ComputeFn`: `Stage` hands the function only the payload data or the request
  inputs. The stage uses the defaults above. `asr_partials` therefore produces
  no partial events; the ASR leg still delivers its final transcript.
- One request at a time per stage (the default `SimpleScheduler` concurrency).
- ASR is 16 kHz mono only, at most 30 s per request, inherited from the
  accepted frontend.

## Build

Host (parsing tests only, no CUDA):

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
python3 -I native/stage_tests/native_stage_smoke.py \
  --base-url http://172.17.0.2:43301 --wav asr_en_pcm16.wav \
  --image man_ironing.png --out-dir native/stage_tests/evidence/live
service/h20b-native-service.sh stop
```

Evidence from the acceptance run is in `native/stage_tests/evidence/`.
