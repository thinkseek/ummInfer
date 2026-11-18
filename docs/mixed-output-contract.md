# Mixed output contract (text + image + audio)

Frozen 2026-11-16 for the `mixed-output/cpp` work. The engine stays one
`UnifiedEngine` over one `Coordinator`: `[optional ASR] -> VLM -> (image || tts)`.
No new proto, wire, comm or Coordinator message type is introduced.

## Public request

Same endpoint: `POST /v1/chat/completions`. New keys are opt-in; a request
without them is byte-identical to the current text path.

- `modalities`: array of strings, allowed values `text`, `image`, `audio`.
- `image`: object, optional.
  - `prompt`: optional non-empty string. When absent the image prompt is the
    full aggregated VLM answer. Never a transcript or a streaming delta.
  - `size`: string, default `"1024x1024"`.
  - `seed`: optional integer.
  - `output_format`: `"png"` (default) or `"jpeg"`.
- `audio`: object, required when `modalities` includes `audio`.
  - `voice`: non-empty string.
  - `format`: only `"wav"`.
  - `sample_rate`: integer, default and contract value `24000`.
- Any request with `image` or `audio` output must set `stream=true` or the
  server answers 400. Non-streaming mixed output is not implemented.
- When the matching optional stage is not registered in this process, the
  requested modality is rejected at the HTTP layer with 400 before submit:
  `image output is not enabled on this server` /
  `audio output is not enabled on this server`.

## Engine inputs

The HTTP boundary translates the public keys into engine-only inputs that are
stripped before the VLM worker sees them (same mechanism as `asr_partials`):

- `output_modalities`: array of `image` / `audio`.
- `image_options`: map `{prompt?, size, seed?, output_format}`.
- `audio_options`: map `{voice, format, sample_rate}`.

Leg worker inputs:

- image leg: `{prompt, size, seed?, output_format}`.
- tts leg: `{text, voice, format, sample_rate}` where `text` is always the
  aggregated `record.vlm_text`.

## Child ids and stages

- Public stage ids in events: `asr`, `vlm`, `image`, `tts`.
- Child request ids: `<unified_id>:vlm`, `<unified_id>:image`,
  `<unified_id>:tts`. The suffixes do not advance the Coordinator counter.
- The VLM barrier is the only fan-out point. Image and tts are submitted under
  the engine mutex after the VLM completion, each only when requested.
- `cancel()` aborts only already-submitted non-empty child ids. Legs not yet
  submitted are blocked by the engine `cancel_requested` flag; the code never
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
             "format":"png","size":"1024x1024","seed":42}}
```

- Every chunk's `data` is independently legal base64 (multiple of 4 chars).
- Only the final chunk carries `sha256` (of the full decoded artifact) and the
  full decoded `bytes`.
- Audio artifacts are WAV, PCM16, mono, 24 kHz; their descriptor carries
  `sample_rate` and `channels` instead of image fields.
- `artifact_id` is `art-<n>` in artifact delivery order (image before audio).

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
- decoded artifact bytes per request aggregated: at most 60 MiB;
- one SSE frame on the wire: at most 3 MiB (the parser hard limit is 4 MiB);
- the C++ client keeps its existing 64 MiB aggregate hard limit and must fail
  deterministically rather than truncate.

Worker -> engine transport is one msgpack Bin in the existing
`CompleteMessage.result`; no new top-level message type. The result map is
`{"media_type":"image|audio","mime_type":"...","data": <Bin>, ...descriptor}`.
Base64 encoding and chunking run on the HTTP provider thread, never inside a
Coordinator pump callback.

## History, logging, readiness

- Session history stores text only (user text and `vlm_text`). Artifact bytes
  and references are never stored.
- Logs never contain prompts, TTS text, base64 or Bin payloads.
  `--log-requests` style logging, where present, records length and sha256 only.
- Optional stages do not enter the `/ready` gate. A process with no optional
  worker registered is a healthy text server.
