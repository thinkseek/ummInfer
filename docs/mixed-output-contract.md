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
  - `size`: string, default `"512x512"` (matching the native `umm-native-stage-image` `--image-backend 2512|2.1` stages).
  - `seed`: optional integer.
  - `output_format`: `"png"` (default) or `"jpeg"`.
- `audio`: object, required when `modalities` includes `audio`.
  - `voice`: non-empty string.
  - `format`: only `"wav"`.
  - `sample_rate`: integer, default and contract value `24000`.
  - `stream`: optional boolean, default `false`. When `true`, the TTS leg
    synthesizes sentence by sentence and each finished sentence is delivered
    immediately as an `artifact` event with `mime_type: "audio/pcm"`,
    `format: "pcm16"`, `sample_rate: 24000`, `channels: 1`, and a
    `stream_index` that counts the sentence. The stage marks its result
    `"streamed": true`, so the engine does not append the whole WAV afterwards:
    a streamed request yields only `audio/pcm` artifacts. When `false`, the
    request yields exactly one `audio/wav` artifact after synthesis completes.
    Measured on H20B: first `audio/pcm` chunk at 0.36-0.39 s of a 9.1 s request.
- `video`: object, optional.
  - `prompt`: optional non-empty string. When absent the video prompt is the
    full aggregated VLM answer.
  - `size`: optional non-empty string, default `"1280x720"`.
  - `seconds`: optional positive integer.
  - `fps`: optional positive integer.
  - `seed`: optional integer.
  - `output_format`: optional string, only `"mp4"` (default `"mp4"`).
- Any request with `image`, `audio` or `video` output must set `stream=true` or
  the server answers 400. Non-streaming mixed output is not implemented.
- When the matching optional stage is not registered in this process, the
  requested modality is rejected at the HTTP layer with 400 before submit:
  `image output is not enabled on this server` /
  `audio output is not enabled on this server` /
  `video output is not enabled on this server`.

## Engine inputs

The HTTP boundary translates the public keys into engine-only inputs that are
stripped before the VLM worker sees them (same mechanism as `asr_partials`):

- `output_modalities`: array of `image` / `audio` / `video`.
- `image_options`: map `{prompt?, size, seed?, output_format}`.
- `audio_options`: map `{voice, format, sample_rate}`.
- `video_options`: map `{prompt?, size, seconds?, fps?, seed?, output_format}`.

Leg worker inputs:

- image leg: `{prompt, size, seed?, output_format}`.
- tts leg: `{text, voice, format, sample_rate}` where `text` is always the
  aggregated `record.vlm_text`.
- video leg: `{prompt, size, seconds?, fps?, seed?, output_format}` where
  `prompt` falls back to `record.vlm_text` when omitted.

## Child ids, stages and unified leg lifecycle

- Public stage ids in events: `asr`, `vlm`, `image`, `tts`, `video`.
- Child request ids: `<unified_id>:vlm`, `<unified_id>:image`,
  `<unified_id>:tts`, `<unified_id>:video`. The suffixes do not advance the
  Coordinator counter.
- Every leg shares one `LegState` lifecycle (`pending` → `running` →
  `completed` | `failed` | `aborted`), observable via
  `UnifiedEngine::legs_snapshot(request_id)`. Optional output legs start
  `pending` until the VLM barrier submits them (`running`), then settle to
  `completed`, `failed`, or `aborted`.
- The VLM barrier is the only fan-out point. Image, tts and video are submitted
  under the engine mutex after the VLM completion, each only when requested.
- `cancel()` / `cancel_for_session()` aborts only already-submitted non-empty
  child ids (including `:video`). Legs not yet submitted are blocked by the
  engine `cancel_requested` flag and transition to `aborted`; the code never
  relies on a Coordinator broadcast for an unknown id.

## Events

Existing six event kinds keep their names and order. New kinds:

- `artifact`: one chunk of one generated artifact.
- `leg_error`: an optional leg failed after the VLM succeeded; not terminal.

Sequence numbers stay gap-free from 0 across every event kind of one request.
There is exactly one terminal event (`complete`, `error` or `cancelled`) and
exactly one `data: [DONE]`.

`artifact` event (unified envelope plus `artifact` object):

```json
{"kind":"artifact","stage":"image","seq":7,"session_id":"...","is_final":true,
 "text":"",
 "artifact":{"artifact_id":"art-0","media_type":"image","mime_type":"image/png",
             "encoding":"base64","data":"<independently decodable base64>",
             "index":0,"total":3,"bytes":184320,
             "sha256":"<hex of full decoded artifact>",
             "format":"png","size":"512x512","seed":42}}
```

- Every chunk's `data` is independently legal base64 (multiple of 4 chars).
- Only the final chunk carries `sha256` (of the full decoded artifact) and the
  full decoded `bytes`.
- Audio artifacts are WAV, PCM16, mono, 24 kHz; their descriptor carries
  `sample_rate` and `channels` instead of image fields.
- Video artifacts are MP4 (`media_type: "video"`, `mime_type: "video/mp4"`);
  their descriptor carries `format`, `size`, and optional `fps`/`seconds`/`seed`.
- `artifact_id` is `art-<n>` in artifact delivery order.

### Partial failure

- ASR or VLM failure/cancellation keeps the current single `error` /
  `cancelled` terminal; output legs never start.
- After a successful VLM, a failed or aborted optional leg emits `leg_error`,
  already delivered text and artifacts stay valid, and the request still ends
  with exactly one `complete`. The terminal carries
  `legs: [{"stage":"image","state":"failed","error":"..."}, ...]` naming every
  optional leg's final state (`completed`, `failed`, `aborted`). The
  `legs` key is absent when no optional leg ran.
- All artifact and leg_error events are enqueued before the terminal.

## Byte limits

Frozen values, enforced at the HTTP boundary and mirrored by the C++ client:

- one decoded image: at most 48 MiB;
- one decoded audio: at most 48 MiB;
- one decoded video: at most 48 MiB;
- decoded artifact bytes per request aggregated: at most 60 MiB;
- one SSE frame on the wire: at most 3 MiB (the parser hard limit is 4 MiB);
- the C++ client keeps its 64 MiB hard cap for text-only streams and uses a
  96 MiB wire/aggregated stream cap (`kMixedStreamWireBytes`) for mixed output
  (giving base64 + SSE/JSON envelope headroom for the 60 MiB decoded aggregate),
  failing deterministically rather than truncating.

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
