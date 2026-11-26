# C++ clients

The `umm-infer` executable provides OpenAI-compatible chat and Omni ASR clients.
The public library API is in `include/umminfer/client/client.hpp`.

## Dependency

The client uses libcurl for HTTP(S), multipart forms, SSE, and WebSocket
transport. TLS certificate and hostname verification are enabled. The installed
libcurl must support `ws`; use **libcurl 8.13.0 or newer** for realtime. Earlier
releases (including the locally tested 8.11.1) misreport continuation flags and
fail the fragmented-message E2E. The realtime client opens only `ws://` on
loopback; remote `wss://` is not supported. Apple's system libcurl 8.7.1 does
not support WebSocket; use Homebrew libcurl and pass its prefix explicitly. The
realtime server also requires libwebsockets.

On Linux, check `curl-config --protocols` for `ws` and `wss`. Ubuntu 24.04
ships libcurl 8.5.0 without them. Build curl 8.13.0 or newer with
`--enable-websockets` into a separate prefix and pass it with `-Dcurl-prefix`;
do not replace the system libcurl. [Validation](validation.md) section 3a gives the commands.

The build needs the same dependencies as the root README:

```sh
brew install zig libzmq msgpack cpp-httplib nlohmann-json curl libwebsockets
python3 -m venv .venv
.venv/bin/python -m pip install mlx-lm pyzmq msgpack websockets
zig build -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

## Chat

```sh
export UMMINFER_API_KEY='...'
./zig-out/bin/umm-infer chat \
  --url https://api.openai.com/v1 --model MODEL --prompt 'Hello'

./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:30000/v1 --model omni --prompt 'Hello' --stream
```

`UMMINFER_API_KEY` takes precedence over `OPENAI_API_KEY`. The CLI also reads
`OPENAI_BASE_URL` and `OPENAI_MODEL`. Keys are not accepted as command-line
arguments. Server error text is redacted before it is truncated or printed.

Plain HTTP is allowed only for a loopback host. Other OpenAI endpoints must use
HTTPS. Redirects are disabled.

Options include `--system`, `--max-tokens`, `--temperature`, `--top-p`,
`--seed`, `--timeout-ms`, `--first-token-ms`, `--stall-ms`, and `--max-attempts`.
Retries use exponential backoff with equal jitter: the delay is drawn from
`[cap/2, cap)`, and `cap` doubles from `retry_base` up to `retry_max`.
`Retry-After` and `Retry-After-Ms` are honored; a wait above `max_retry_after`
is reported as an error. The server wait is not shortened by the attempt
timeout. A retry never starts after a text delta was delivered.

The route accepts an explicit allowlist of top-level chat fields: `model`,
`messages`, `stream`, `temperature`, `max_tokens`, `max_completion_tokens`,
`top_p`, `seed`, `stop`, `response_format`, `stream_options`, `session_id`,
`session_history`, `modalities`, `image`, `audio`, `video`, and `narrative`.
The mixed-output and narrative fields require unified mode. Any other non-null
top-level field returns HTTP 400 `unsupported field: NAME`, whether or not this
server knows the name. A null value is treated as absent. A non-empty
`stop` is rejected, as is `response_format` other than type `text`,
`stream_options` other than `include_usage: false`, and message content that is
not a text string. `top_p` must be a number in `(0, 1]`; `seed` must be a signed
64-bit integer. Both are forwarded to the generation stage and are absent from
the request when the caller omits them. A field with the wrong JSON type returns
400, not 500. The client sends `top_p` and `seed` only when the caller sets them.
It never adds `stream_options`, and it never drops a caller field and retries: an
unsupported field is reported as an error.

### Native video input

`--video-input PATH` attaches a local MP4, WebM, or QuickTime video to a unified
chat turn as a `video_url` content part:

```sh
./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:43301/v1 --model qwen3-native \
  --prompt 'Describe the camera motion and key events' \
  --video-input /tmp/input.mp4
```

The CLI reads at most 32 MiB, rejects an empty file, and sends
`data:video/mp4;base64,...`, `data:video/webm;base64,...`, or
`data:video/quicktime;base64,...` according to the filename extension. Video
input is mutually exclusive with `--image-url` and `--video-url`; generated
image/video output options are independent. `ContentPart::Kind::kVideoUrl`
provides the same library wire representation. The C++ client validates native
data URLs, including their decoded 32 MiB limit, and rejects any chat request
that combines image and video inputs before sending it. The existing remote
`--video-url` option remains available for compatible OpenAI endpoints; unified
ummInfer requires a native data URL.

The default sampler is deterministic uniform selection of at most 8 decoded RGB
frames. It is unchanged when `frame_selection` is absent. Native Qwen3.8 also
accepts an explicit bounded temporal policy, either through the CLI:

```sh
./zig-out/bin/umm-infer chat \
  --url "$UMMINFER_URL" --model qwen3-native \
  --prompt 'Identify the first scene change and describe the motion' \
  --video-input /tmp/input.mp4 \
  --video-frame-policy temporal \
  --video-motion-threshold 0.02 --video-scene-threshold 0.25
```

or on the `video_url` object:

```json
{
  "type": "video_url",
  "video_url": {
    "url": "data:video/mp4;base64,...",
    "frame_selection": {
      "policy": "temporal",
      "motion_threshold": 0.02,
      "scene_threshold": 0.25
    }
  }
}
```

`policy` is `uniform` or `temporal`; both thresholds are finite numbers in
`[0,1]`. `{"policy":"uniform"}` is equivalent to omission. Uniform mode rejects
threshold fields rather than silently ignoring them, and unknown fields
(including attempts to raise frame limits) return HTTP 400. Temporal mode
analyzes at most 64 candidates and selects at most 8 frames; these hard limits
are not API-configurable. Adjacent scene detections are deduplicated, motion and
scene ties use the earliest decode index, and static/repeated/high-threshold
clips fill unused slots by deterministic uniform coverage. First/last candidate
anchors and short-decode fallback always leave a valid bounded video. Selected
frames stay in decode order with their source presentation timestamps;
non-monotonic timestamps are not sorted. The existing temporal-2 patch packer,
video prompt, 3D MRoPE, and serial/batched engine paths are shared with uniform
mode.

For a later real-model A/B run, `python3 -I tests/video_temporal_live.py
--self-check` validates the request fixture offline. The live mode requires
`UMMINFER_LIVE_OPT_IN=1`, `--url`, `--video`, `--question`, and `--out-dir`, and
records answers, TTFT, and total latency for both policies. CPU/offline runs
report `REAL_MODEL_QUALITY: NOT_RUN` and `LIVE_LATENCY: NOT_RUN`.

#### Web UI and macOS ummmlx

Both graphical clients expose **Input frames** under their existing Options
panel. **Uniform** is the default and omits `frame_selection`, preserving the
exact legacy `video_url` request object. Selecting **Temporal** enables Motion
and Scene threshold fields (defaults `0.02` and `0.25`) and sends:

```json
{
  "type": "video_url",
  "video_url": {
    "url": "data:video/mp4;base64,...",
    "frame_selection": {
      "policy": "temporal",
      "motion_threshold": 0.02,
      "scene_threshold": 0.25
    }
  }
}
```

Both clients require the two thresholds to be finite numbers in the inclusive
range `[0,1]` before starting transport. Empty, malformed, non-finite, and
out-of-range values leave the attachment intact and produce no HTTP request.
Returning to Uniform removes the field again, including after a new chat or
reconnect. These controls apply only to attached video understanding; the
existing generated-video size/frame/step/CFG controls and opt-in **Fast video**
(`video.teacache_thresh`) remain independent, and image/video input exclusivity,
the 32 MiB upload budget, model selection, chat history, and realtime voice are
unchanged. Physical browser/macOS layout and real temporal-model quality remain
`NOT_RUN` on a CPU-only host.

### Token usage in a stream

The last `chat.completion.chunk` of a streaming chat response, the one that
carries `finish_reason`, also carries a `usage` object. It arrives before
`data: [DONE]`:

```text
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],...,
       "usage":{"completion_tokens":17}}
```

- Every value is the count the stage reported in its terminal result. The
  server never estimates a count from characters or words and never writes a
  zero for an unobserved count.
- `completion_tokens` is the stage's `generated_token_count`. It is present
  whenever the stage counted its output.
- `prompt_tokens` and `total_tokens` are present only when the stage also
  reported `prompt_tokens`. The bundled MLX text stage
  (`python/umminfer_mlx_stage.py`) reports the count MLX-LM measured for the
  tokenized prompt, so its requests carry all three. The native text stages do
  not report it yet, so their requests carry `completion_tokens` alone.
  `total_tokens` is emitted only as the sum of both reported halves, never as a
  partial sum.
- Image and audio legs add no field. They have no comparable token count.
- The key rides on an existing chunk, so a client that does not know it
  ignores it. It is sent regardless of `stream_options`; the field
  `stream_options.include_usage: true` is still rejected with HTTP 400.
- The non-streaming response body is unchanged.

## Mixed output (generated image, audio and video)

A unified server can return generated image, audio and video artifacts alongside
the text answer. The request opts in with `modalities` plus an `image`, `audio`
and/or `video` object. Non-narrative requests work with both `stream=true` (SSE)
and `stream=false` (non-streaming JSON with `artifacts` and `legs`).
`audio.stream=true` and enabled narrative each require HTTP `stream=true`.

```sh
./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:43301/v1 --model qwen3-native --prompt 'Draw this' \
  --image-out /tmp/picture.png --image-prompt 'a red cube' \
  --image-size 1024x1024 --image-steps 4 --image-seed 7 --image-format png

./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:43301/v1 --model qwen3-native --prompt 'Edit this' \
  --image-out /tmp/edited.png --image-ref /tmp/picture.png \
  --image-prompt 'turn it into blue glass' --image-steps 4

./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:43301/v1 --model qwen3-native --prompt 'Say hello' \
  --audio-out /tmp/hello.wav --voice vivian [--ref-audio /tmp/ref24k.wav]

./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:43301/v1 --model qwen3-native --prompt 'Animate this' \
  --video-out /tmp/clip.mp4 --video-prompt 'ocean waves at sunset' \
  --video-image /tmp/picture.png \
  --video-size 704x480 --video-fps 16 --video-frames 29 --video-steps 20 \
  --video-guidance 5.0 --video-seed 11
```

- `image` accepts `prompt` (absent means the full aggregated text answer is
  used), `size` (default `512x512`), optional `steps` (`1..100`), optional
  `image` (`data:image/...;base64,...`, `--image-ref PATH` in CLI) for editing,
  integer `seed`, and optional `output_format="png"` (the default; other values
  are rejected).
- `audio` requires `voice`, accepts `format=wav`, `sample_rate=24000`
  (WAV, PCM16, mono), and optional `ref_audio` (base64/data-URL WAV,
  `--ref-audio PATH` in CLI) for `Qwen3-TTS-12Hz-1.7B-Base` voice cloning. The
  CLI does not expose `audio.stream`, so it always receives one complete WAV
  and writes it after the request finishes.
- `video` (`VideoOutput` and CLI `--video-out`) accepts `prompt`
  (absent falls back to the aggregated text answer), `size` (default `1280x720`),
  optional positive `seconds`, `fps`, `num_frames` (`1..241`, `--video-frames`),
  `steps` (`1..100`, `--video-steps`), `guidance_scale` (`0..30`,
  `--video-guidance`), optional non-negative `seed`, and `output_format="mp4"`.
  Optional `image` is a `data:image/...;base64,...` URL (max 16 MiB decoded);
  `--video-image PATH` encodes a local PNG, JPEG, or WebP file as this URL.
  `Wan2.2-TI2V-5B` uses it as the first frame for native I2V animation. Decoded
  `video/mp4` bytes are returned in `ChatResponse::artifacts`.

### Interleaved narrative and global streaming voice

The wire API accepts an optional top-level `narrative` object for a streaming
mixed-output request:

```json
{
  "model": "qwen3-native",
  "stream": true,
  "max_tokens": 256,
  "messages": [{"role": "user", "content": "Tell a two-scene story about a red lantern."}],
  "modalities": ["text", "image", "audio", "video"],
  "audio": {"voice": "vivian", "format": "wav", "stream": true},
  "narrative": {"enabled": true, "visual_mode": "auto", "max_slots": 2}
}
```

- `enabled` is a boolean and defaults to `false`. `visual_mode` is `auto`
  (default), `image`, or `video`. `max_slots` is an integer in `1..6` and
  defaults to `3`.
- Enabled narrative requires `stream=true` and at least one visual modality.
  An image-only or video-only mode also requires that modality in `modalities`.
  Invalid fields or mode/modality combinations return HTTP 400.
- `audio.stream=true` is now allowed alongside image, video, and narrative when
  HTTP streaming is enabled. It still requires `stream=true`. Global TTS starts
  from the first clean-prose sentence and can overlap VLM generation and visual
  rendering. Each sentence arrives as an independent `audio/pcm` artifact with
  `format=pcm16`, `sample_rate`, `channels`, and `stream_index`; no final WAV is
  appended.
- The server strips `[[image: ...]]`, `[[video: ...]]`, and `[[visual: ...]]`
  directives from text deltas, terminal text, history, and TTS input. A client
  consumes the resulting prose directly and does not parse these markers.

A `visual_slot` event announces a visual immediately when its directive closes,
rather than after the VLM finishes:

```json
{
  "id": "http-1",
  "object": "chat.completion.chunk",
  "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
  "unified": {
    "seq": 42,
    "kind": "visual_slot",
    "request_id": "http-1",
    "session_id": "...",
    "stage": "image",
    "slot_id": "slot-0",
    "slot_index": 0,
    "media_type": "image",
    "prompt": "a glowing red lantern over a misty mountain lake",
    "text_offset": 128
  }
}
```

Create the UI block from this event and retain both identifiers. Every chunk of
the resulting visual `artifact` carries matching `slot_id` and `slot_index` in
`unified.artifact`; verify the final chunk's decoded `bytes` and `sha256` after
reassembly. A visual `leg_error` carries both fields in `unified`. Terminal
`complete`, `cancelled`, and `error` events include each visual slot leg in
`unified.legs[]` with `stage`, `slot_id`, `state`, and optional `error`.
Sequence numbers remain gap-free from zero, with one terminal event and one
`[DONE]`.

Narrative slot child request ids are `<request_id>:slot:<slot_index>`.
Cancellation and disconnect abort all active VLM, TTS, and slot children. If the
VLM emits no directive, the server creates one fallback `slot-0` from the clean
answer.

The bundled C++ `ChatRequest` API and CLI expose no narrative controls. Use the
JSON wire fields above when calling a server that supports them.

#### Showcase starter presets and live per-turn telemetry (Web UI and macOS ummmlx)

Both the Web UI (`web/index.html`) and the native macOS client (`clients/ummmlx`)
display three curated North-Star starter preset cards in the empty conversation
state and hide them automatically once the conversation contains any turns:

| Preset ID | Title | Subtitle | Applied Controls |
| --- | --- | --- | --- |
| `documentary` | **Live Documentary** | Interleaved prose, diagram & voice | Narrative `auto`, `allow_text_only=true`, continuity `off`, empty `style_anchor`, Fast Video on, Voice/audio output on |
| `storyboard_i2v` | **Storyboard I2V Chain** | Slot 0 keyframe -> Slot 1 anchored I2V | Narrative `auto`, `allow_text_only=false`, continuity `i2v_chain`, `style_anchor="cinematic 35mm film grain, warm lantern twilight, smooth camera motion"`, Fast Video on, Voice/audio output on |
| `smart_text_only` | **Smart Text-Only** | Autonomous zero-visual abstention | Narrative `auto`, `allow_text_only=true`, continuity `off`, empty `style_anchor`, Fast Video off, Voice/audio output off |

Clicking a preset card updates the reactive client controls, populates the
composer prompt, and focuses the prompt input without auto-sending a network
request so the user can inspect or edit before sending. Existing default control
values are untouched when no preset is selected.

During streaming, each assistant turn tracks real client-observed timings
(`performance.now()` in Web UI; `ContinuousClock` in `ummmlx`) and renders a
compact pill strip beneath the turn bubble (`formatTurnTelemetry` /
`TurnTelemetry.formattedPills`). Unobserved fields are omitted rather than
estimated, and the strip stays hidden until at least one real metric arrives:

- `TTFT <dur>`: elapsed time from request send to the first non-empty text delta.
- `TTFA <dur>`: elapsed time from request send to the first audio chunk or WAV artifact.
- `<N> visual(s) (<dur>)`: count of settled visual slots/artifacts plus elapsed time to the first settled visual.
- `I2V`: shown when any `visual_slot` carries `anchor_slot_index` I2V continuity metadata.
- `Total <dur>`: elapsed time from request send to stream completion or stop.

Sub-second durations format as `<N> ms` (e.g. `265 ms`) and values at or above
`1000 ms` format as `<X.XX> s` (e.g. `1.42 s`). Both clients adapt the cards and
telemetry pills cleanly across light and dark themes.

## Direct Speech (`POST /v1/audio/speech`)

`OpenAIClient::speech(const SpeechRequest&)` and the `umm-infer speech`
subcommand synthesize WAV or raw 24 kHz mono PCM16 directly without a VLM turn:

```sh
./zig-out/bin/umm-infer speech \
  --url http://127.0.0.1:43301/v1 --model qwen3-native \
  --text 'Hello world.' --voice vivian --format wav --out /tmp/out.wav
```

Pass `--ref-audio PATH` (24 kHz mono PCM16 WAV) when targeting a
`Qwen3-TTS-12Hz-1.7B-Base` stage. Requests are sent once with no retries.
- A client that wants audio before synthesis ends sends `"stream": true` in the
  `audio` object (see the mixed output contract). The server then delivers
  `mime_type: audio/pcm` artifacts, one per finished sentence, and sends no WAV
  at the end. The `ummmlx` macOS client does this and plays each chunk as it
  arrives; its first chunk arrived 0.36 s into a 9.1 s request on SM90.
- A modality whose stage is not registered on the server is rejected with 400
  before submission: `image output is not enabled on this server` /
  `audio output is not enabled on this server` /
  `video output is not enabled on this server`.

Artifacts arrive as base64 `artifact` SSE events in independently decodable
chunks whose decoded target is 2 MiB. The C++ client reassembles them in
order, verifies the final-chunk `sha256` against the decoded bytes, and rejects
a mismatch, out-of-order chunks, non-aligned base64, or a truncated stream.
Decoded limits are 48 MiB per image, 48 MiB per audio, 48 MiB per video, and
60 MiB aggregated per request; one SSE wire frame is capped at 3 MiB (the
parser ceiling is 4 MiB). The C++ client keeps a 64 MiB cap for text-only
streams and uses a 96 MiB wire/aggregated stream cap for mixed output. The CLI
writes each output through an atomic part file and refuses to overwrite an
existing path. Session history stores text only; artifact bytes are never
persisted in history, and logs never contain prompts, TTS text, base64 or Bin
payloads.

## Direct visual render (`POST /v1/visual/render`)

This endpoint renders one image or video without a VLM turn. To swap a
narrative slot in place, send the selected slot's prompt with the opposite
`kind`, keep its local `slot_id`, and replace the displayed artifact with the
response bytes. The request has no `slot_id` field.

```json
{
  "kind": "video",
  "prompt": "the lantern drifts across the lake",
  "size": "704x480",
  "steps": 20,
  "num_frames": 29,
  "fps": 16,
  "guidance_scale": 5.0,
  "seed": 42,
  "image": "data:image/png;base64,..."
}
```

`kind` (`image` or `video`) and a non-empty string `prompt` are required. Both
kinds accept optional `size`, positive integer `steps`, signed 64-bit integer
`seed`, and an `image` data URL (max 16 MiB decoded). Image uses `image` for
FLUX.2 image-to-image editing. Video with `kind: "video"` uses it as the first
frame for `Wan2.2-TI2V-5B` native I2V. Video also accepts positive integers
`num_frames`, `fps`, `seconds`, and non-negative number `guidance_scale`.

HTTP 200 returns raw `image/png` or `video/mp4` with a matching `Content-Type`
and `X-UmmInfer-Kind: image` or `video`. Invalid JSON, unknown or invalid fields,
and an empty prompt return 400. An unavailable stage or fixed-upstream mode
returns 501. Overload returns 503 with `Retry-After: 1`. The endpoint performs
one synchronous stage request; it emits no SSE events.

## Final ASR

```sh
./zig-out/bin/umm-infer transcribe \
  --url http://127.0.0.1:8000 --model MODEL --file audio.wav

./zig-out/bin/umm-infer transcribe \
  --url http://127.0.0.1:8000 --model MODEL --file audio.pcm --format pcm
```

The endpoint is `POST /v1/audio/transcriptions`. `ASRClient::transcribe` and
the CLI read raw PCM, convert it to WAV with `wav_from_pcm16`, and upload that
WAV as the multipart `file` part. The client always sends `stream=true` and
`response_format=json`, so it parses SSE only; it never reads a plain JSON body.
WAV input to the route must be PCM16, mono, 16 kHz, with one data chunk. Raw PCM
input must contain complete 16-bit samples.

The route accepts `stream=true|false` and `response_format=json`.
A request without `stream`, or with `stream=false`, returns
`{"text": "..."}` as `application/json`. Only `stream=true` returns SSE
(`transcript.text.done` followed by `[DONE]`). Any other `response_format` is
rejected with HTTP 400.

`model` must equal the model the server serves; any other value returns 400.
Multipart parts are checked in two namespaces. The field namespace accepts only
`model`, `stream`, `response_format`, `include_generation_metadata`, `language`,
`prompt`, `max_new_tokens` and `temperature`; any other field name returns 400
`unsupported multipart field: NAME`. The file namespace accepts exactly one part
named `file`; a part with any other name returns 400
`unsupported multipart file part: NAME`, so a scalar field cannot bypass the
field checks by resending itself with a filename. A repeated field name returns
400 `duplicate multipart field: NAME`.

`language` and `prompt` are rejected by presence, including an empty value:
this server does not use them, so accepting them would silently ignore the
caller's intent. The same rule rejects `audio_layout`, `stop_at_end_of_text` and
`stop_on_token_loop` through the field allowlist. An explicit `temperature` is
accepted only as `0`, the only value the entry stage uses. Invalid UTF-8 in a
multipart value returns 400 with a valid JSON body, not 500.

The route reports `generation_metadata` only from values the stage sent in its
`complete.result`: `generated_token_count` and `finish_reason`. The MLX stage
uses the `finish_reason` and cumulative `generation_tokens` from the last
MLX-LM generation response. The TinyASR stage records them from the CTC token
sequence before text stripping. `finish_reason` is `length` only when the
generation limit stopped the run, and `stop` when the model stopped on its
own, including when it stopped exactly at the limit. The field is omitted when the
stage reports neither value. A request with `include_generation_metadata=true`
is answered with HTTP 500 when the stage did not report them; the route never
substitutes a zero count or a default finish reason. The bundled client sends
`include_generation_metadata=true`, so a stage that cannot count tokens makes
the request fail instead of returning an invented `generated_tokens=0`.
`finish_reason=unknown` is reserved for a stage that reports the metadata but
cannot name its stop condition.

Both ASR listeners enforce `Authorization: Bearer KEY` when the server runs with
`--api-key KEY`. The HTTP route, the realtime WebSocket handshake, `/health` and
`/ready` all require it. A missing or wrong key returns HTTP 401
`{"error":{"message":"invalid api key","type":"auth_error"}}`; the WebSocket
handshake is rejected with an HTTP 401 status line. No response or server log
echoes the key. The client reads the key from `UMMINFER_API_KEY`, then
`OPENAI_API_KEY`; the realtime Final request uses the same key.

ASR endpoints are restricted to loopback HTTP. Proxy use is disabled. Use
`--timeout-ms` for the total request deadline and `--stall-ms` for the maximum
interval without response data.

### Real ummInfer ASR server

The HTTP transcription route and realtime WebSocket route submit audio to the
same C++ Coordinator. The external Python stage then runs the model. The HTTP
and WebSocket listeners use separate ports.

The deterministic tiny CTC model uses an FFT, a log-mel filterbank, two dense
layers, and greedy CTC decoding. It is an initialized inference test model, not
a trained speech recognizer. Its output has no recognition-quality guarantee.

Generate its configuration and safetensors weights. The output path must not
exist:

```sh
WORKDIR=$(mktemp -d /tmp/umminfer-asr.XXXXXX)
mkdir "$WORKDIR/ipc"
python3 -I python/umminfer_tiny_asr_stage.py generate \
  --output "$WORKDIR/model"
```

Start the server in one terminal, with the same `WORKDIR`:

```sh
./zig-out/bin/umm-infer serve --host 127.0.0.1 \
  --port 55271 --realtime-port 55272 --model tiny-ctc --entry asr \
  --entry-endpoint "ipc://$WORKDIR/ipc/entry" \
  --completion "ipc://$WORKDIR/ipc/completion" \
  --abort "ipc://$WORKDIR/ipc/abort"
```

Start the real MLX stage in another terminal:

```sh
python3 -I python/umminfer_tiny_asr_stage.py serve --name asr \
  --recv "ipc://$WORKDIR/ipc/entry" \
  --coordinator "ipc://$WORKDIR/ipc/completion" \
  --abort "ipc://$WORKDIR/ipc/abort" --model "$WORKDIR/model"
```

The stage sends `stage_ready` with its `from_stage` name after model loading and
control-plane initialization. The server queues early requests until this event;
an early abort remains terminal and its queued Submit is removed. Start the
server and stage in either order. The CLI startup path requires this protocol and
does not fall back to a timer for older stages.

`GET /ready` reports startup admission for that entry stage. With an external
`--entry-endpoint`, it returns HTTP 503 `{"status":"starting"}` until the entry
stage sends `stage_ready`, then HTTP 200 `{"status":"ready"}`. The in-process
`--demo` path keeps its previous always-200 behavior. The route does not check
downstream stages, model health or per-request liveness.

Use HTTP port 55271 for `transcribe` and WebSocket port 55272 for `realtime`.
The realtime `--final-url` must name the HTTP listener, not the WebSocket
listener:

```sh
./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:55272 \
  --final-url http://127.0.0.1:55271 \
  --model tiny-ctc --file audio.wav --chunk-ms 100
```

Omitting `--final-url` selects the base address given by `--url`; it does not
disable the Final request. Use `--final-url` when the WebSocket listener and the
HTTP listener use different ports.

## Realtime ASR

```sh
./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:8001 --final-url http://127.0.0.1:8000 \
  --model MODEL --file audio.wav --chunk-ms 100

./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:8001 --final-url http://127.0.0.1:8000 \
  --model MODEL --file audio.wav --live-result
```

Realtime transport uses loopback `ws://` and `/v1/realtime`. This protocol is
private to this server. It is not compatible with the OpenAI Realtime API. The
client sends `session.update` and base64 PCM16 append messages. Only
`--live-result` sends `transcription.done`; the default `finish()` closes the
WebSocket and goes straight to HTTP Final.

The `session.update` allowlist is `input_audio_format`, `turn_detection`
and `max_new_tokens`. `language` and `prompt` are rejected by presence,
including an empty string or `null`, because the entry stage does not read
them; accepting them would report a configuration that is not in effect.
The same request sent to the HTTP Final route returns 400, so both paths
refuse the field. `transcription_session.updated` and `transcription.completed`
therefore carry no language. Any other session field returns an `error` event.

`turn_detection` accepts `null` (no server endpointing) or
`{"type": "server_vad", "silence_ms": N}` with N between 100 and 5000, plus an
optional boolean `echo_suppression` (default `true`, omitted on the wire when
enabled so default payloads stay unchanged; explicit `false` opts into the A/B
baseline without server-side echo suppression). While a turn or spoken response
is active, a narrow `session.update` carrying only `turn_detection` with
`type="server_vad"`, unchanged `silence_ms`, and boolean `echo_suppression`
toggles echo suppression live without resetting `playback_ack` or response
ownership. See [full duplex](full-duplex.md) for the measured behavior and its
limits.

#### Web UI and macOS ummmlx full-duplex voice controls

Both the Web UI (`web/index.html`) and macOS `ummmlx` expose an **Echo
suppression** toggle (default ON) alongside their full-duplex voice controls.
Toggling it before connecting sets the initial `session.update` payload (omitting
`echo_suppression` when ON and sending `"echo_suppression": false` when OFF);
toggling it while connected or mid-response sends the narrow active
`turn_detection`-only `session.update` with the negotiated `silence_ms`. When a voice session is active, both clients render a live status strip showing
a real PCM16 microphone input level meter (`computePcm16RmsLevel` in Web UI and
`RealtimeAudioLevel.rmsLevel(pcm16:)` in `ummmlx`, computing normalized
`[0.0, 1.0]` RMS directly from captured little-endian 16-bit PCM frames with no
synthetic animation and resetting strictly to `0.0` on stop, disconnect, or
error), the server-confirmed echo-suppression state (`Echo suppression: On` or
`Echo suppression: Off (A/B)`, omitted until confirmed by
`transcription_session.updated`), and the real observed barge-in count
(`Barge-ins: N`, counting `response.cancelled` events in the current session and
resetting on disconnect or reconnect).

When `response.cancelled` interrupts an active Live Voice assistant turn, both
clients attach an inline cut-point badge driven strictly by real
server-acknowledged UTF-8 byte offsets from `response.playback_ack` and
`response.cancelled` (`text_cursor`): `Interrupted · <N> B spoken` when `N > 0`,
and `Interrupted before audio` when cancelled with `0` acknowledged bytes.
Normally completed turns display no interrupted badge.

The bundled server submits a preview request for every 200 ms of new audio.
Each completed preview sends one `transcription.segment` with `is_final: false`
and the cumulative text of the turn. The client groups segments by
`event_index` and `segment_id` and passes non-final text to `on_partial`. With
`server_vad`, the server also sends `input_audio_buffer.speech_started` and
`input_audio_buffer.speech_stopped`; the stop event commits the turn, and the
WebSocket stays open for the next turn. A new utterance aborts a computing
turn. Without `turn_detection`, `transcription.done` commits the whole turn,
the server sends one final segment and one `transcription.completed`, then
closes the WebSocket. The bundled CLI client still treats the first
`transcription.completed` as terminal.

By default, `finish()` sends no `transcription.done`. It closes the WebSocket
and performs one HTTP Final ASR request with the retained PCM. Only the PCM is
kept for that request; the preview segments are not used. `--live-result`
instead sends `transcription.done`, accepts `transcription.completed` as the
result, and waits for it up to `--final-timeout-ms`. When that event does not
arrive, when the live path fails, or when the wait times out, the client still
performs the HTTP Final request.

Incoming WebSocket messages are limited to 1 MiB. The client retains at most
1 GiB of realtime audio. The bundled realtime server keeps at most 64 MiB per
session and rejects an append past that limit. SIGINT and SIGTERM cancel active
client operations.

## Full-duplex voice (`voice`)

The `voice` subcommand is the opt-in realtime microphone + spoken-response
client. It is separate from the existing file-based `realtime` command, whose
behavior is unchanged:

```sh
./zig-out/bin/umm-infer voice \
  --url http://127.0.0.1:55272 --model qwen3-unified \
  --voice vivian --silence-ms 400 --max-new-tokens 256
```

`--url` names the realtime listener's HTTP base, so use the separate
`--realtime-port` address when HTTP and WebSocket are on different ports. The
server must have unified ASR, VLM and TTS stages. Start it with
`--session-history` when interrupted turns should retain exactly the
acknowledged assistant prefix. Options are `--no-echo-suppression` (opt-in A/B
disable for server-side echo suppression; default ON omits `echo_suppression` on
the wire so the initial `turn_detection` payload stays byte-identical),
`--connect-timeout-ms` (10000), `--reconnect-attempts` (2),
`--reconnect-delay-ms` (250), and Linux ALSA `--input-device` /
`--output-device` names (default `default`).

The client sends `session.update` with server VAD, `input_audio_format=pcm16`,
and `audio_output.playback_ack=true`, and refuses to start capture until the
server echoes that negotiation. When `--no-echo-suppression` is passed, the
initial `session.update` emits `"echo_suppression": false` inside
`turn_detection`, and the client fails cleanly if the server does not confirm
`echo_suppression=false` in `transcription_session.updated`. Programmatic
callers can also call `VoiceClient::set_echo_suppression(bool)` while connected
or mid-response to send the narrow `turn_detection`-only `session.update` with
the server-negotiated `silence_ms` without resetting active response state. On
session ready, the CLI prints the server-confirmed echo suppression state
(`on`/`off`), counts real observed `response.cancelled` barge-in events during
the session (resetting on reconnect), and prints `barge_ins=<N>` in the stop
summary when at least one barge-in occurred. Microphone frames are exact little-endian
16 kHz mono PCM16 (20 ms per ALSA read), sent as `input_audio_buffer.append`;
WAV headers are not added and audio is not retained for a Final HTTP request.
Spoken replies use an isolated output queue. An
`output_audio_buffer.playback_ack` is written only after the output device
reports complete consumption of the corresponding PCM, with strictly
non-decreasing cursors. `audio.done` stores the final cursor until all queued
PCM drains; its final ACK may equal the last delta cursor and is sent once.
`response.completed` alone is never a playback acknowledgement.

`input_audio_buffer.speech_started` immediately invalidates and stops the old
local response, before the server's `response.cancelled` arrives. Both that
event and repeated cancellation drop old queued audio, text, final cursors and
late device callbacks. A new response cannot overlap a cancelled one.

Type `stop`, `stop voice`, `/stop`, `/stop-voice`, `quit`, or `exit` and press
Enter to **Stop Voice**. Stdin EOF, SIGINT and SIGTERM take the same bounded
cleanup path: stop capture, interrupt playback, close the socket and join the
client threads. Other stdin text is not sent as a chat turn. A transport break
closes all old audio resources, waits for the configured delay, and opens a
fresh negotiated session. Captured audio and old ACKs are never replayed. A
reconnect does not resume the old server-side realtime session history.

Linux loads `libasound.so.2` at runtime; no ALSA development package or new
build-time library is needed. ALSA capture requests 16 kHz mono PCM16 with
software resampling disabled and fails explicitly if the selected device does
not support it. Playback uses hardware-delay accounting to ACK consumed
frames. macOS uses AudioQueue completion callbacks and the default input and
output devices; named device selection is not supported there. The terminal
application needs microphone permission. Use headphones: the CLI does not add
a software acoustic echo canceller, so speaker output can trigger VAD.

API keys still come only from `UMMINFER_API_KEY` or `OPENAI_API_KEY` and are
sent in the WebSocket `Authorization` header. They are never placed in the URL
or voice logs. Realtime remains loopback-only `ws://`; remote `wss://` is not
implemented.

Offline verification uses deterministic fake sockets, microphones and output
completion callbacks:

```sh
umm-test-local voice-test
# Equivalent direct build step with explicit prefixes:
zig build voice-test -Ddeps-prefix=/path/to/deps -Dcurl-prefix=/path/to/curl-ws
```

The suite covers cursor monotonicity, delayed/final ACKs, partial consumption,
cancellation races, repeated cancellation, stale callbacks, reconnect,
`echo_suppression` wire omission/negotiation/mid-response toggling, observed
`barge_in_count` telemetry, and microphone/playback cleanup on Stop Voice,
external cancellation and errors. These are not physical-device or live-model
results.

### Manual physical / live checklist

On a machine with a microphone, speaker and a real unified ASR+VLM+TTS server,
use an ephemeral loopback realtime listener (55272 above) and run the command
interactively. Then:

1. Speak a complete question and verify the transcript, answer text and audible
   PCM reply. Confirm no append occurs before negotiation.
2. Speak over a multi-sentence answer. Verify playback stops at the speech-start
   edge, no old suffix resumes, and the next response retains only the prefix
   whose ACK was actually sent.
3. Let an answer finish naturally and verify the final ACK is delayed until the
   speaker queue drains, including the equal-final-cursor case.
4. Stop with the typed command; repeat with stdin EOF and Ctrl-C. Each path must
   release microphone and speaker devices without a hang or a late ACK.
5. With `--reconnect-attempts 1 --reconnect-delay-ms 100`, restart the local
   listener. Verify a fresh negotiation, no replay of pre-disconnect audio, and
   no old playback after the new session starts.

Record server/stage commits, model revisions, device names, audible observations,
ACK logs and interrupt latency as run evidence. In the current headless task
environment this checklist, physical listening, a real local model server, and
macOS AudioQueue execution are all **NOT_RUN**. Device nodes alone are not
acceptance evidence.

## Library use

Link the client sources with the rest of `umm_core` and libcurl. Construct
`OpenAIClient` or `ASRClient` with an explicit configuration. The library does
not read environment variables itself.

```cpp
#include <umminfer/client/client.hpp>

using namespace umminfer::client;

OpenAIConfig config;
config.base_url = "http://127.0.0.1:30000/v1";
config.model = "omni";
OpenAIClient client(config);

ChatRequest request;
request.messages.push_back({"user", "Hello"});
request.stream = true;
request.on_delta = [](const std::string& delta) { /* consume delta */ };
const ChatResponse response = client.chat(request);
```

Pass a `std::atomic<bool>` through `ChatRequest::cancel`,
`TranscribeRequest::cancel`, or `RealtimeRequest::cancel` to cancel an operation.
All client failures derive from `umminfer::client::Error`; `code()` and
`retryable()` expose the failure classification.

## Verification

Run the stdlib-only narrative contract check offline:

```sh
python3 -I tests/narrative_live_e2e.py --self-check
```

Run it against a real interleaved narrative server and save the visual, PCM,
swapped visual, and JSON timing report:

```sh
python3 -I tests/narrative_live_e2e.py \
  --url http://127.0.0.1:43301 --model qwen3-native --out-dir /tmp/narrative-live
```

Live mode reports TTFT, first PCM audio time, first visual slot versus VLM
finish, per-slot render latency, total wall time, and the clean-prose check. An
image-to-video swap is checked with `ffprobe` for H.264; a video-to-image swap
is checked for a PNG header. The final `/health` check requires `in_flight=0`.
`ffprobe` is needed only when the live swap targets video.

Date-specific binary hashes, log paths and latency measurements are recorded in
[validation.md](validation.md). They are snapshots of one run, not a contract.
