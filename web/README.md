# web

A single-file browser client for the ummInfer unified chat stream. Open
`index.html` directly, or serve it from any static server. No build step, no
package manager, no CDN, no external font. The page works offline except for
the chat request itself.

## Run

Open `web/index.html` in a browser, set **Base URL** and **Model**, press
**Send**. Settings persist in `localStorage`.

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
| Errors | HTTP status, `unified.kind == error`, `leg_error`, `[DONE]`-missing, seq gap |
| Stop | aborts the fetch and closes the AudioContext |

Settings: base URL, model, image output, audio output, voice. Image size is
fixed at `512x512`, PNG. Theme follows `prefers-color-scheme` and can be
overridden with the System / Light / Dark selector.

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

## Known limits

- No sha256 replay or saved copy of streamed PCM; live playback only.
- No microphone input, so `asr_partial` / `asr_final` are logged, not rendered.
- No reconnect or retry after a broken stream.
- One assistant turn per request; server-side session history is preferred over
  replaying history, matching `ChatClient.swift:341-346`.
