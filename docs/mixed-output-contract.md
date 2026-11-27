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
  - `image`: optional `data:image/...;base64,...` data URL (max 16 MiB decoded)
    for `Wan2.2-TI2V-5B` native image-to-video (I2V). The decoded image supplies
    the first frame to animate.
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
  - `continuity_mode`: optional `"off"` (default), `"prompt_anchor"`, or
    `"i2v_chain"`. `"prompt_anchor"` carries opening-visual/style text into
    later prompts. `"i2v_chain"` additionally binds an earlier image slot to a
    later video slot for native I2V.
  - `style_anchor`: optional string, trimmed and limited to 256 UTF-8 bytes. If
    present without `continuity_mode`, it selects `"prompt_anchor"`.
  - When enabled, HTTP `stream` must be `true`, and `modalities` must contain
    `image` or `video`. `visual_mode="image"` requires `image` in `modalities`;
    `visual_mode="video"` requires `video` in `modalities`.
  - A non-object `narrative`, an incorrectly typed field, an unknown
    `visual_mode`/`continuity_mode`, an oversized `style_anchor`, or an
    out-of-range `max_slots` returns HTTP 400.
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
- `video_options`: map `{prompt?, size, seconds?, fps?, num_frames?, steps?, guidance_scale?, seed?, image?, output_format}`.

Leg worker inputs:

- image leg: `{prompt, size, steps?, seed?, image?, output_format}`.
- tts leg: `{text, voice, format, sample_rate, stream?, ref_audio?}`. With
  `audio.stream=false`, `text` is the aggregated `record.vlm_text`. With
  `audio.stream=true`, each leg receives one clean-prose sentence or bounded final
  span as soon as it is available. Supported Markdown presentation delimiters are
  removed only from native speech input; public prose and playback cursors retain
  the original UTF-8 byte ranges. At most 32 native TTS children are active, and a
  completed child releases capacity for the next lossless buffered span.
  `ref_audio` is decoded binary WAV bytes.
- video leg: `{prompt, size, seconds?, fps?, num_frames?, steps?, guidance_scale?, seed?, image?, output_format}` where
  `prompt` falls back to `record.vlm_text` when omitted. `image` remains the
  validated data URL and selects native I2V on `Wan2.2-TI2V-5B`.

### Narrative parsing and visual dispatch

When `narrative.enabled=true`, the engine prepends a director system message.
The VLM writes coherent prose in the user's language and may insert up to
`max_slots` standalone directives at narrative beats:

```text
[[image: vivid concrete visual scene description in English]]
[[video: vivid concrete motion/camera scene description in English]]
```

With `continuity_mode="i2v_chain"`, an image or video directive may explicitly
select an earlier image slot by its existing public slot id:

```text
[[image: the same character and style at night | source=slot-0]]
[[video: the character turns as the camera orbits | source=slot-1]]
```

An image dependent performs a real FLUX.2 edit using the selected image; a video
dependent performs native Wan I2V. The suffix is reserved only after `| source`;
other `|` text remains prompt text. The parser removes the suffix from the render
prompt and from clean prose/TTS. Duplicate, malformed, missing, self/forward,
cross-session, wrong-modality, and wrong-target references do not silently become
unconditioned generation. An explicit reference also rejects a conflicting
request-level `image.image` or `video.image`, respectively, and requires
`continuity_mode="i2v_chain"`; there is no implicit "previous image"
interpretation of `source=`.

The streaming parser recognizes `image`, `video`, and `visual` tags
case-insensitively across arbitrary split UTF-8 chunks. It strips every
directive from `vlm_delta`, aggregated VLM text, session history, and streaming
TTS input. Empty prompts and directives beyond `max_slots` are dropped and
stripped. At EOF, a recognized reserved prefix or unterminated directive is also
removed as directive bytes rather than leaking into public prose, history, or TTS;
literal non-directive bracket text remains unchanged.

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

Without an explicit suffix, `i2v_chain` preserves its original latest-image
selection and graceful T2V fallback when the anchor fails. With `source=slot-N`,
the named slot must be an earlier image in the same unified request. The
dependent image/video leg stays `pending` and contributes to the terminal join
until that exact image reaches `completed`; a failed/aborted source fails/aborts
the dependent instead of dispatching unconditioned generation, and transitive
dependents finalize rather than remaining pending. Before dispatch, the engine
recomputes SHA-256 over the immutable settled image bytes and verifies the
request-local content-addressed binding. The worker receives those exact bytes
(binary `inputs.image`), the existing prompt/size/seed/step settings (plus
frame/guidance for video), and out-of-band `anchor_slot_index`, `anchor_sha256`,
and `anchor_explicit=true`. Only Wan video receives
`anchor_crop="center_cover"`; FLUX edits do not claim that video crop operation.
The FLUX.2 image stage and native/Python video stages independently recompute the
digest before generation. The external `/v1/videos` backend rejects I2V
explicitly rather than returning unconditioned T2V with anchor metadata. No media
bytes appear in directive text, `visual_slot`, terminal legs, or trajectory
JSONL.

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

Continuity is additive and omitted by default. A chained `visual_slot` carries
`continuity_mode: "i2v_chain"` and the zero-based `anchor_slot_index`. An
explicit `source=slot-N` directive additionally carries `anchor_explicit: true`.
It never carries `anchor_sha256`: the source may still be rendering, so including
a digest here would make immediate and delayed completion produce different
metadata. The resolved image/video artifact and matching terminal leg carry
`anchor_slot_index`, `anchor_explicit`, and the verified `anchor_sha256`. Failed
explicit legs may carry the index without a digest. Mainline implicit fallback
artifacts may have an anchor index without a digest.

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

Trajectory schema v2 projects `continuity_mode`, `anchor_slot_index`, and
`anchor_explicit` only when present, keeps the resolved `anchor_sha256` on
artifact/terminal metadata, and stores media only in the existing CAS. Validation
accepts unresolved failed/cancelled references and the mainline implicit T2V
fallback, but every resolved link must select an earlier image whose completed
CAS artifact has exactly the recorded digest. Validators also accept the exact
legacy v2 slot/leg-error projections when raw events already carry continuity
fields. Re-recording the same stream is byte-deterministic; SFT reconstruction
restores `| source=slot-N` only for a resolved explicit image-edit or I2V slot
and skips a failed empty malformed marker without triggering fallback.

The native Wan Stage independently enforces the same content binding before
execution: an explicit anchor requires image bytes, and any supplied
`anchor_sha256` must equal SHA-256 of the exact binary `image`. FLUX.2 applies
the same explicit-image/digest check before invoking Diffusers and additionally
requires a non-negative integer source index; legacy data-URL edits without
anchor metadata are unchanged. These are internal Stage-input checks, not new
public HTTP fields. Direct
`/v1/visual/render` continues to reject unknown anchor metadata, and omitted
anchor metadata preserves legacy direct T2V/I2V behavior. The Task #62 real
FLUX.2/native-Wan gate and negative controls are recorded in
[flux2-i2v-continuity.md](flux2-i2v-continuity.md). The Task #65 real public
VLM narrative, exact per-leg speech join, cancellation regression, and fresh
replay evidence are recorded in
[public-narrative-i2v.md](public-narrative-i2v.md). The Task #76 three-slot
image-establishment → exact image-edit → exact edited-image-I2V chain, native
negative controls, cancellation/retry, and raw-evidence verification are recorded
in [cross-slot-visual-continuity.md](cross-slot-visual-continuity.md).

### Partial failure

- For non-narrative requests, ASR or VLM failure/cancellation keeps the current
  single `error` / `cancelled` terminal and prevents output legs from starting.
  A narrative primary failure or cancellation aborts any slots already active.
- A failed or aborted narrative slot or other optional leg emits `leg_error`;
  the primary VLM can continue. Already delivered text and artifacts stay valid,
  and a request whose primary VLM succeeds still ends with exactly one
  `complete`. The terminal `legs` names every optional leg's final state
  (`completed`, `failed`, `aborted`), with `slot_id` on visual slot legs. On
  cancellation, both pending and running narrative slots are finalized as
  `aborted`; late source bytes or VLM/directive events cannot revive the
  request. The `legs` key is absent when no optional leg ran.
- All artifact and leg_error events are enqueued before the terminal.

## Direct single-stage visual render

`POST /v1/visual/render` renders one visual without a VLM turn. A client uses it
to replace a narrative slot in place: keep the slot identity in the UI, send its
prompt with the new `kind`, and replace the local artifact with the returned
bytes. The endpoint does not take `slot_id`; the client owns that association.

Request JSON:

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

- `kind` is required and must be `image` or `video`; `prompt` is required and
  must be a non-empty string.
- `size` is optional. `steps` is an optional positive integer. `num_frames`,
  `fps`, and `seconds` are optional positive integers for video;
  `guidance_scale` is an optional non-negative number for video. `seed` is an
  optional signed 64-bit integer. `image` is an optional
  `data:image/...;base64,...` URL for both kinds (max 16 MiB decoded). Image
  uses it for FLUX.2 image-to-image editing. Video uses it as the first frame
  for `Wan2.2-TI2V-5B` native I2V.
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
- decoded terminal/aggregate artifact bytes per request: at most 60 MiB.
  Independently consumed streamed `audio/pcm` chunks do not accumulate against this
  terminal aggregation budget; each remains bounded by the 48 MiB audio limit and
  by existing lossless transport backpressure. This is not permission to drop or
  truncate a long narration;
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

Task #81's deterministic red/green and real SM90 Before/After evidence for these
lifecycle boundaries is in [ns01-tts-lifecycle.md](ns01-tts-lifecycle.md).
