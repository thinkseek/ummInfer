# web

A single-file browser client for the ummInfer unified chat stream. Open
`index.html` directly, or serve it from any static server. No build step, no
package manager, no CDN, no external font. The page works offline except for
the chat request itself.

## Run

Open `web/index.html` in a browser, set **Base URL** and **Model**, press
**Send**. Settings persist in `localStorage`. For full-duplex voice, enable
**Realtime**; when the server uses a separate `--realtime-port`, set **Realtime
URL** under Options to that listener's HTTP base (for example `/v1` on its own
origin).

The browser sends one `POST <base>/chat/completions` with `stream: true`,
reads the SSE body with `fetch` + `ReadableStream`, and renders text, images
and streamed PCM audio as they arrive.

`file://` needs the endpoint to answer with `Access-Control-Allow-Origin`,
because a `file://` page has origin `null`. The ummInfer HTTP server sets no
CORS headers today, so opening the file directly works only against an
endpoint that does (for example a local mock). Against a stock ummInfer
server, serve the page from the same origin instead of opening the file.

## Features

| Area | Behaviour |
| --- | --- |
| Text | `choices[0].delta.content` appended per event, blinking caret while streaming |
| Image | `artifact` chunks reassembled by `artifact_id`, then `Blob` + `objectURL`, click opens full size |
| Audio | `audio/pcm` chunks decoded to `AudioBuffer` and chained on one `AudioContext` cursor |
| Realtime voice | Default-off **Realtime** toggle opens `/realtime`, streams 16 kHz PCM16 microphone frames, plays spoken replies, and acknowledges only consumed playback |
| Barge-in | `response.cancelled` stops scheduled sources immediately and drops every later frame for that `response_id` |
| Errors | HTTP status, `unified.kind == error`, `leg_error`, `[DONE]`-missing, seq gap |
| Stop | aborts the fetch and closes the AudioContext |
| Keyboard shortcuts | Pure client-side global shortcuts: `Cmd/Ctrl+E` (Export JSON), `Cmd/Ctrl+Shift+E` (Export Markdown), `Cmd/Ctrl+O` (Import JSON), `Alt+N` / `Cmd/Ctrl+Shift+K` (New chat, avoiding browser `Cmd/Ctrl+N`), `Esc` (Stop active generation), plus `Cmd/Ctrl+Enter` in the prompt composer to Send |

Settings: base URL, model, image output, audio output, voice. Image size is
fixed at `512x512`, PNG. Theme follows `prefers-color-scheme` and can be
overridden with the System / Light / Dark selector.

## Native video input frames

For **+ Vid** attachments, **Input frames: Uniform** and **Frame budget: Default
(8, omitted)** preserve the legacy `video_url` object with no `frame_selection`.
Temporal enables validated Motion/Scene thresholds. Explicit **4**, **8**, or
**16** emits exactly that integer in `frame_selection.max_frames` under either
policy; explicit 8 is distinct from omission. The selector describes
lower/higher selection, prefill, and ViT cost, not answer quality.

New chats and Showcase preset changes restore default frame controls. Invalid
persisted values fail before fetch instead of being clamped. JSON session
exports carry optional root `video_frame_selection` only when non-default;
legacy imports reset it to default, explicit 4/8/16 configurations round-trip,
and malformed values (including `4.0` and `4e0`) reject without network traffic
or mutation. Run `python3 -I tests/web_ui_contract_test.py` for the offline wire,
reset, reconnect, and session contract gates.

## Protocol source

Every field sent or read has a source in this repository:

| Wire element | Source |
| --- | --- |
| `POST <base>/chat/completions`, path normalisation | `clients/ummmlx/Sources/ummmlx/ChatClient.swift:200-226` |
| `model`, `messages`, `max_tokens: 64`, `temperature: 0`, `stream: true`, `session_id` | `ChatClient.swift:264-276` |
| `modalities`, `image{size,output_format}`, `audio{voice,format,stream}` | `ChatClient.swift:277-297` |
| `audio.stream` omitted when an image is requested | `ChatClient.swift:293-295` |
| SSE framing, `data:` prefix, `[DONE]` required | `ChatClient.swift:73-154`, `493-508` |
| Unified envelope `kind/seq/session_id/stage/is_final/text` | `UnifiedStream.swift:126-164` |
| Event kinds and terminal set | `UnifiedStream.swift:9-20` |
| Artifact chunk fields, order, `bytes`, `sha256` | `UnifiedStream.swift:166-198`, `247-340` |
| `audio/pcm` descriptor: `format`, `sample_rate`, `channels` | `UnifiedStream.swift:259-272` |
| Jitter buffer and back-to-back scheduling | `clients/ummmlx/Sources/ummmlx/AudioPlayback.swift:116-235` |
| Colours, radii, font sizes | `clients/ummmlx/Sources/ummmlx/Theme.swift:13-32` |
| Contract for the event sequence | `docs/mixed-output-contract.md` |
| Realtime session, PCM16 events, playback ACK cursors, barge-in | `docs/full-duplex.md` (Realtime spoken response, Stage B) |

## Verification (mock, not a live model)

Run against a stdlib mock server that replays a fixed contract-shaped stream.
The mock is not committed; it lives outside the repository.

| Path | Evidence |
| --- | --- |
| Text stream | 4 `vlm_delta` events render as `Streaming text from the mock server.` |
| Image artifact | 2-chunk `image/png`, sha256 verified, `<img>` decoded at 512x512 from a `blob:` URL |
| PCM audio | 3 x 0.6 s chunks scheduled at t=0.601 / 1.201 / 1.801 s, wave indicator on during playback |
| WAV artifact | image+audio request returns one `audio/wav`, rendered as an `<audio controls>` element |
| `error` | HTTP 400 body shown in the red banner, status `Error` |
| `leg_error` | text kept, `image output: ...` line shown, status `Completed` |
| Stop | status `Stopped`, partial text kept, Send button restored |
| Themes | light `#FFFFFF` on `#111111`, dark `#151515` on `#F2F2F2`, accent `#0D9384` / `#2BB3A3` |
| No external dependency | network log during load: one request, `http://127.0.0.1:30000/` |

Not verified: a real ummInfer server, a real TTS voice, real image generation,
and audible output (headless Chrome schedules the buffers but has no speaker).

The offline contract suite also checks JavaScript syntax and executes the
realtime player against controllable Web Audio nodes:

```sh
python3 -I tests/web_ui_contract_test.py
python3 -I tests/narrative_live_e2e.py --self-check
```

It verifies that ACKs wait for `onended` and final queue drain, cursors never
decrease, and cancelled sources or stale callbacks cannot acknowledge audio.
This is not a physical microphone, speaker, or live-server validation.

## Known limits

- No sha256 replay or saved copy of streamed PCM; live playback only.
- Realtime has no automatic reconnect after a broken WebSocket.
- Browser WebSocket cannot send the server's `Authorization` header directly;
  API-key-protected realtime requires an authenticating same-origin proxy. The
  page rejects this unsupported configuration rather than exposing a key in the URL.
- No reconnect or retry after a broken stream.
- One assistant turn per request; server-side session history is preferred over
  replaying history, matching `ChatClient.swift:341-346`.
