# Mixed output contract (text + image + audio + video)

Frozen 2026-11-16 for the `mixed-output/cpp` work and extended to unified
4-modality (`text`, `image`, `audio`, `video`) orchestration. The engine stays
one `UnifiedEngine` over one `Coordinator`:
`[optional ASR] -> VLM -> (image || audio/tts || video)`.
No new proto, wire, comm or Coordinator message type is introduced, and no
second control plane is created for video.

## Public request

Same endpoint: `POST /v1/chat/completions`. New keys are opt-in; a request
without them is byte-identical to the current text path.

- `modalities`: array of strings, allowed values `text`, `image`, `audio`,
  `video`.
- `image`: object, optional.
  - `prompt`: optional non-empty string. When absent the image prompt is the
    full aggregated VLM answer. Never a transcript or a streaming delta.
  - `size`: string, default `"512x512"` (matching the native `umm-native-stage-image` `--image-backend 2512|2.1` stages; `FLUX.2` supports e.g. `"1024x1024"`).
  - `steps`: optional integer in `[1, 100]`.
  - `seed`: optional integer.
  - `image`: optional `data:image/...;base64,...` data URL (max 16 MiB decoded) for image-to-image editing (supported by `flux2`; native `2512`/`2.1` stages reject it explicitly).
  - `output_format`: only `"png"` (default). The image stage emits PNG only,
    so any other value is rejected.
- `audio`: object, required when `modalities` includes `audio`.
  - `voice`: non-empty string.
  - `format`: only `"wav"`.
  - `sample_rate`: integer, default and contract value `24000`.
  - `ref_audio`: optional base64 or `data:audio/wav;base64,...` PCM16 WAV (max 16 MiB decoded) for `Qwen3-TTS-12Hz-1.7B-Base` zero-shot voice cloning (also accepted on `POST /v1/audio/speech`).
  - `stream`: optional boolean, default `false`. When `true`, the TTS leg
    synthesizes sentence by sentence and each finished sentence is delivered
    immediately as an `artifact` event with `mime_type: "audio/pcm"`,
    `format: "pcm16"`, `sample_rate` and `channels` propagated dynamically from
    the stage chunk (defaulting to `24000` / `1`), and a
    `stream_index` that counts the sentence. The stage marks its result
    `"streamed": true`, so the engine does not append the whole WAV afterwards:
    a streamed request yields only `audio/pcm` artifacts. When `false`, the
    request yields exactly one `audio/wav` artifact after synthesis completes.
    With HTTP `stream=true`, `audio.stream=true` is allowed together with
    `image`, `video`, and `narrative`; TTS starts from the first sentence and
    runs concurrently with VLM generation and visual slots. Its input is only
    clean prose with visual directives stripped. Measured on SM90 for the
    non-narrative path: first `audio/pcm` chunk at 0.36-0.39 s of a 9.1 s
    request.
- `video`: object, optional.
  - `prompt`: optional non-empty string. When absent the video prompt is the
    full aggregated VLM answer.
  - `size`: optional non-empty string, default `"1280x720"`.
  - `seconds`: optional positive integer.
  - `fps`: optional positive integer.
  - `num_frames`: optional positive integer in `[1, 241]`.
  - `steps`: optional integer in `[1, 100]`.
  - `guidance_scale`: optional finite number in `[0, 30]`.
  - `seed`: optional integer.
  - `output_format`: optional string, only `"mp4"` (default `"mp4"`).
- `narrative`: optional non-null object. It enables interleaved visual slots in
  a streaming answer:

  ```json
  "narrative": {
    "enabled": true,
    "visual_mode": "auto",
    "max_slots": 3
  }
  ```

  - `enabled`: optional boolean, default `false`.
  - `visual_mode`: optional `"auto"` (default), `"image"`, or `"video"`.
  - `max_slots`: optional integer in `[1, 6]`, default `3`.
  - When enabled, HTTP `stream` must be `true`, and `modalities` must contain
    `image` or `video`. `visual_mode="image"` requires `image` in `modalities`;
    `visual_mode="video"` requires `video` in `modalities`.
  - A non-object `narrative`, an incorrectly typed field, an unknown
    `visual_mode`, or an out-of-range `max_slots` returns HTTP 400.
  - An absent `narrative` or `enabled=false` preserves the non-narrative request
    and event behavior.
- Non-narrative requests with `image`, `audio` or `video` output work with
  `stream=true` (SSE) and `stream=false` (one JSON body, see "Non-streaming
  mixed output"), except that `audio.stream=true` requires SSE. Enabled
  narrative always requires SSE.
- `audio.stream=true` with `stream=false` is rejected with 400:
  `audio.stream=true requires stream=true`. Streamed PCM chunks have no place
  in one JSON body.
- When the matching optional stage is not registered in this process, the
  requested modality is rejected at the HTTP layer with 400 before submit:
  `image output is not enabled on this server` /
  `audio output is not enabled on this server` /
  `video output is not enabled on this server`.

## Engine inputs

The HTTP boundary translates the public keys into engine-only inputs that are
stripped before the VLM worker sees them (same mechanism as `asr_partials`):

- `output_modalities`: array of `image` / `audio` / `video`.
- `image_options`: map `{prompt?, size, steps?, seed?, image?, output_format}`.
- `audio_options`: map `{voice, format, sample_rate, stream?, ref_audio?}`.
- `video_options`: map `{prompt?, size, seconds?, fps?, num_frames?, steps?, guidance_scale?, seed?, output_format}`.

Leg worker inputs:

- image leg: `{prompt, size, steps?, seed?, image?, output_format}`.
- tts leg: `{text, voice, format, sample_rate, stream?, ref_audio?}`. With
  `audio.stream=false`, `text` is the aggregated `record.vlm_text`. With
  `audio.stream=true`, each leg receives one clean-prose sentence or the final
  clean tail as soon as it is available. `ref_audio` is decoded binary WAV bytes.
- video leg: `{prompt, size, seconds?, fps?, num_frames?, steps?, guidance_scale?, seed?, output_format}` where
  `prompt` falls back to `record.vlm_text` when omitted.

### Narrative parsing and visual dispatch

When `narrative.enabled=true`, the engine prepends a director system message.
The VLM writes coherent prose in the user's language and may insert up to
`max_slots` standalone directives at narrative beats:

```text
[[image: vivid concrete visual scene description in English]]
[[video: vivid concrete motion/camera scene description in English]]
```

The streaming parser recognizes `image`, `video`, and `visual` tags
case-insensitively across arbitrary split UTF-8 chunks. It strips every
directive from `vlm_delta`, aggregated VLM text, session history, and streaming
TTS input. Empty prompts and directives beyond `max_slots` are dropped and
stripped.

`visual_mode=image` and `video` force that kind. Under `auto`, the requested
kind is used when enabled; otherwise the parser falls back to the other enabled
visual modality. A generic `[[visual: ...]]` follows the same rules. When the
VLM finishes with no dispatched slot, the engine dispatches one fallback
`slot-0` using the clean VLM text. It chooses image unless only video is enabled
or `visual_mode=video`.

Each accepted directive dispatches its visual stage immediately when the
closing `]]` arrives. Visual generation does not wait for VLM completion. A
narrative visual leg uses its directive prompt rather than the full aggregated
answer.

## Child ids, stages and unified leg lifecycle

- Public stage ids in events: `asr`, `vlm`, `image`, `tts`, `video`.
- Child request ids: `<unified_id>:vlm`, `<unified_id>:image`,
  `<unified_id>:tts`, `<unified_id>:video`. The suffixes do not advance the
  Coordinator counter. Narrative visual slots instead use
  `<request_id>:slot:<slot_index>`.
- Every leg shares one `LegState` lifecycle (`pending` → `running` →
  `completed` | `failed` | `aborted`), observable via
  `UnifiedEngine::legs_snapshot(request_id)`. Non-narrative image/video legs and
  non-streaming TTS start `pending` until the VLM barrier submits them
  (`running`). Streaming TTS becomes `running` when the first clean-prose
  sentence is available. Narrative slots start when their directives close.
- For non-narrative requests, the VLM barrier remains the only visual fan-out
  point. Narrative visual slots fan out mid-stream, and streaming TTS can start
  from sentence 1 without waiting for the VLM or visual slots.
- Cancellation by `POST /abort`, session cancellation, or client disconnect
  broadcasts abort to every active child: VLM, TTS, non-narrative image/video,
  and every narrative slot child id. Unsubmitted legs become `aborted`, and
  `release()` cleans up every child id.

## Events

Existing six event kinds keep their names and order. New kinds:

- `artifact`: one chunk of one generated artifact.
- `leg_error`: an optional leg or narrative visual slot failed; not terminal.

Sequence numbers stay gap-free from 0 across every event kind of one request.
There is exactly one terminal event (`complete`, `error` or `cancelled`) and
exactly one `data: [DONE]`.

### Interleaved `visual_slot`

A narrative emits `kind: "visual_slot"` immediately when a directive closes,
before that slot's artifact and while later `vlm_delta` and `audio/pcm` events
may continue:

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

`text_offset` is the clean-prose offset at which the directive closed. Clients
use `slot_id` and `slot_index` to associate later results; they never need to
parse visual directives from prose.

`artifact` event (unified envelope plus `artifact` object):

```json
{"kind":"artifact","stage":"image","seq":7,"session_id":"...","is_final":true,
 "text":"",
 "artifact":{"artifact_id":"art-0","media_type":"image","mime_type":"image/png",
             "encoding":"base64","data":"<independently decodable base64>",
             "index":2,"total":3,"bytes":184320,
             "sha256":"<hex of full decoded artifact>",
             "format":"png","size":"512x512","seed":42}}
```

- Every chunk's `data` is independently legal base64 (multiple of 4 chars).
- Only the final chunk carries `sha256` (of the full decoded artifact) and the
  full decoded `bytes`.
- Non-streaming TTS emits WAV. With `audio.stream=true`, each sentence emits an
  independent `audio/pcm`, PCM16 artifact instead; its descriptor carries
  dynamic `sample_rate` and `channels` (defaulting to 24000 and 1), plus
  `stream_index`. No final WAV is appended.
- Video artifacts are MP4 (`media_type: "video"`, `mime_type: "video/mp4"`);
  their descriptor carries `format`, `size`, and optional `fps`/`seconds`/`seed`.
- `artifact_id` is `art-<n>` in artifact delivery order.
- A narrative visual artifact also carries `slot_id` and `slot_index` in its
  `artifact` object. They match the earlier `visual_slot`; every chunk of one
  artifact carries the same attribution.
- A narrative visual `leg_error` carries the same `slot_id` and `slot_index` at
  the top level of `unified`, alongside `stage` and `error`.
- Terminal `complete`, `cancelled`, and `error` events include each visual slot
  in `unified.legs[]` as `{"stage":"image"|"video", "slot_id":"slot-N",
  "state":"completed"|"failed"|"aborted", "error"?: "..."}`. The `tts` leg is
  included when enabled. Slot leg identity is `slot_id`; `slot_index` is carried
  by `visual_slot`, `artifact`, and `leg_error`.

### Partial failure

- For non-narrative requests, ASR or VLM failure/cancellation keeps the current
  single `error` / `cancelled` terminal and prevents output legs from starting.
  A narrative primary failure or cancellation aborts any slots already active.
- A failed or aborted narrative slot or other optional leg emits `leg_error`;
  the primary VLM can continue. Already delivered text and artifacts stay valid,
  and a request whose primary VLM succeeds still ends with exactly one
  `complete`. The terminal `legs` names every optional leg's final state
  (`completed`, `failed`, `aborted`), with `slot_id` on visual slot legs. The
  `legs` key is absent when no optional leg ran.
- All artifact and leg_error events are enqueued before the terminal.

## Direct single-stage visual render

`POST /v1/visual/render` renders one visual without a VLM turn. A client uses it
to replace a narrative slot in place: keep the slot identity in the UI, send its
prompt with the new `kind`, and replace the local artifact with the returned
bytes. The endpoint does not take `slot_id`; the client owns that association.

Request JSON:

```json
{
  "kind": "image",
  "prompt": "a red lantern over a mountain lake",
  "size": "512x512",
  "steps": 4,
  "seed": 42,
  "image": "data:image/..."
}
```

- `kind` is required and must be `image` or `video`; `prompt` is required and
  must be a non-empty string.
- `size` is optional. `steps` is an optional positive integer. `num_frames`,
  `fps`, and `seconds` are optional positive integers for video;
  `guidance_scale` is an optional non-negative number for video. `seed` is an
  optional signed 64-bit integer. `image` is an optional image data URL for
  image editing.
- HTTP 200 returns raw `image/png` or `video/mp4` bytes, matching `kind`, with
  `Content-Type` and `X-UmmInfer-Kind: image` or `video` headers.
- Invalid JSON, unknown fields, invalid field values, or an empty prompt return
  HTTP 400. An unregistered stage or fixed-upstream mode returns 501. Admission
  overload returns 503 with `Retry-After: 1`.
- Rendering uses the same admission guard and synchronous Coordinator request
  path as direct speech. It does not create a chat request or an SSE stream.

## Non-streaming mixed output

`stream=false` with `image`, `audio` and/or `video` answers one
`chat.completion` JSON body instead of SSE. The server collects the same
unified events the SSE path forwards, reassembles each artifact, and writes the
response only at the single terminal event.

Text-only requests without optional output keys are unchanged: no `artifacts`,
no `legs` and no `session_id` key appears.

```json
{"id":"http-7","object":"chat.completion","created":1763900000,
 "model":"qwen3-unified","session_id":"sess-1",
 "choices":[{"index":0,
             "message":{"role":"assistant","content":"<VLM answer>"},
             "finish_reason":"stop"}],
 "artifacts":[{"artifact_id":"art-0","media_type":"image",
               "mime_type":"image/png","encoding":"base64",
               "data":"<whole base64 artifact>","bytes":184320,
               "sha256":"<hex of decoded artifact>",
               "format":"png","size":"512x512","seed":42}],
 "legs":[{"stage":"image","state":"completed"},
         {"stage":"tts","state":"failed","error":"tts unavailable"}]}
```

- `artifacts` carries one whole decoded artifact per successful optional leg:
  no `index`/`total` chunk fields, one `data` blob, plus `bytes` and `sha256`
  of the whole decoded artifact. It is an empty array when no leg succeeded.
- `legs` is the terminal legs array, identical to the SSE terminal
  `unified.legs`: `[{stage, state, error?}, ...]` with `state` one of
  `completed`, `failed`, `aborted`.
- `session_id` echoes the request label, or the generated label when the
  request had none.
- Partial failure: a failed optional leg keeps HTTP 200 with the VLM text, the
  artifacts that did succeed, and the failure named in `legs`. An ASR or VLM
  failure answers the same HTTP error shape the text-only `stream=false` path
  uses (HTTP 500 `generation_error`).
- A client disconnect cancels and releases the request before the response is
  written.

## Byte limits

Frozen values, enforced at the HTTP boundary and mirrored by the C++ client:

- one decoded image: at most 48 MiB;
- one decoded audio: at most 48 MiB;
- one decoded video: at most 48 MiB;
- decoded artifact bytes per request aggregated: at most 60 MiB;
- one SSE frame on the wire: at most 3 MiB (the parser hard limit is 4 MiB);
- the C++ client keeps its 64 MiB hard cap for text-only responses and uses a
  96 MiB wire/aggregated cap (`kMixedStreamWireBytes`) for mixed output, on both
  the SSE stream and the non-streaming JSON body (giving base64 + envelope
  headroom for the 60 MiB decoded aggregate), failing deterministically rather
  than truncating.

Worker -> engine transport is one msgpack Bin in the existing
`CompleteMessage.result`; no new top-level message type. The result map is
`{"media_type":"image|audio|video","mime_type":"...","data": <Bin>, ...descriptor}`.
Base64 encoding and chunking run on the HTTP provider thread, never inside a
Coordinator pump callback.

## History, logging, readiness

- Session history stores text only (user text and `vlm_text`). Artifact bytes
  and references are never stored.
- Logs never contain prompts, TTS text, base64 or Bin payloads.
  `--log-requests` style logging, where present, records length and sha256 only.
- Optional stages do not enter the `/ready` gate. A process with no optional
  worker registered is a healthy text server.
