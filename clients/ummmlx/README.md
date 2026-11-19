# ummmlx

A native macOS SwiftUI client for a local or remote ummInfer unified server.
No third-party runtime dependencies.

## Features

- Text chat over `POST /chat/completions` with incremental SSE output.
- Parsed `unified` events: `asr_final`, `vlm_delta`, `artifact`, `leg_error`,
  `complete`, `error`, `cancelled`.
- Strict `unified.seq`: must start at 0 and advance by 1. A gap or repeat fails
  the request; the client never continues silently.
- Generated PNG/JPEG artifacts show as thumbnails; click opens the full image.
- Audio turns enable `audio.stream: true`, schedule verified `audio/pcm`
  (PCM16) chunks onto `AVAudioPlayerNode` as they arrive, and coalesce the turn
  into one inline WAV card with Play, Stop, and duration for replay. Image +
  audio multimodal turns continue to accept whole `audio/wav` artifacts.
- Artifact chunks are merged by `artifact_id`, then checked against the final
  chunk's `bytes` and `sha256`. A mismatch fails the request.
- Local voice input with Apple's on-device speech model: microphone and audio
  file, partial transcript while recording, editable text on completion.
- The server session id is remembered and the first eight characters are shown.
- Multi-turn history: the client replays complete exchanges. When a unified
  server rejects replayed text ("at most one text part"), it retries once with
  the last user message plus `session_history: true`.
- Light, dark, and system appearance.

## Local speech input

Both paths use `SFSpeechRecognizer` with `requiresOnDeviceRecognition = true`:

- microphone: `SFSpeechAudioBufferRecognitionRequest` + `AVAudioEngine`;
- audio file: `SFSpeechURLRecognitionRequest`.

Audio, partial transcripts, and the recognition request never leave this Mac
and are never sent to an umm server. The UI states this next to the composer.

Requirements:

- The app bundle must carry `NSMicrophoneUsageDescription` and
  `NSSpeechRecognitionUsageDescription` (`Support/Info.plist`).
- Dictation must be enabled in System Settings > Keyboard > Dictation. Without
  it, macOS returns "Siri and Dictation are disabled".
- The first microphone or file run asks for Speech Recognition permission;
  microphone input also asks for Microphone permission.
- `Scripts/build.sh` ad-hoc signs the bundle, which TCC requires before it
  considers speech or microphone access. The ad-hoc signature is cdhash-only,
  so every rebuild changes the identity and macOS may ask for the grant again.

The recognizer is behind `protocol SpeechTranscribing`; unit tests use a fake
and need neither a microphone nor permissions. The default language is en-US.

## Endpoint policy

`https` is accepted for any host. Plain `http` is accepted only for:

- loopback (`127.0.0.0/8`, `::1`, `localhost`);
- private literal addresses (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`,
  IPv6 `fc00::/7`);
- hosts listed in `UMMMLX_HTTP_HOSTS` (comma-separated).

Credentials in the URL, a query, a fragment, and redirects are rejected.
`http://172.17.0.2:43301/v1` is accepted because it is a private literal
address. This Mac cannot reach the container bridge directly; use the existing
SSH tunnel and the loopback address on this side:

```sh
/usr/bin/ssh -N -L 43301:172.17.0.2:43301 h20b
# then use http://127.0.0.1:43301/v1 in the app
```

## Limits

- One SSE line: 4 MiB; one SSE event: 4 MiB; a stream: 64 MiB.
- A request must end with `data: [DONE]`; a truncated stream is an error.
- Per artifact: 48 MiB decoded (image and audio); per request: 60 MiB decoded.
- The client timeout is 600 s per request and 900 s per resource. Image
  generation measured about 225 s on the H20B native chain.

## Build and test

```sh
cd clients/ummmlx
swift test
Scripts/build.sh release
open build/ummmlx.app
```

`build/ummmlx.app` is unsigned development output apart from the ad-hoc
signature used for TCC. No notarization, sandbox, or DMG.

## Automation

Static capture (no server contact, no persisted preferences):

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation \
  --output=/tmp/ummmlx-static.json --screenshots=/tmp/ummmlx-static
```

Text conversation, Stop, same-session second turn, and a visible server error:

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation --stop-first \
  --endpoint=http://127.0.0.1:43301/v1 --model=qwen3-native \
  --prompt="Reply with one short sentence." \
  --followup-prompt="Reply with one short sentence about the previous reply." \
  --error-model=qwen3-native-not-served \
  --output=/tmp/ummmlx-text.json --screenshots=/tmp/ummmlx-text
```

One multimodal request (text + image + audio), then light and dark screenshots:

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation --multimodal \
  --endpoint=http://127.0.0.1:43301/v1 --model=qwen3-native \
  --prompt="PROMPT_REDACTED" --image-size=512x512 --voice=vivian \
  --output=/tmp/ummmlx-multimodal.json --screenshots=/tmp/ummmlx-multimodal
```

Local file recognition through `SFSpeechURLRecognitionRequest`:

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation \
  --asr-file=/path/to/16k-mono.wav \
  --output=/tmp/ummmlx-asr.json --screenshots=/tmp/ummmlx-asr
```

Every run writes `automation.json`. Report values include `asr_text`,
`asr_on_device`, `artifacts` (id, MIME type, byte count, SHA256, never bytes),
and `session_id`. The report never carries prompt text, audio bytes, or base64.
Screenshots are window captures and show the conversation, so pass
`--prompt=PROMPT_REDACTED` when the screenshots are shared as evidence.

`Scripts/real_e2e.py` remains the frozen local MLX gate. It starts an exclusive
server and stage, drives the GUI, and checks request history, Stop, and a real
400. It needs `UMMMLX_SERVER` (the frozen binary) and `UMMMLX_MODEL`.

### Fault regression (no server)

`--automation --faults` drives the real `ChatModel`, `ChatClient`, `URLSession`
and `LiveAudioStreamPlayer` through a scripted transport inside the app. It
opens no socket, needs no ummInfer server, and asks for no microphone
permission:

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation --faults \
  --output=/tmp/ummmlx-faults.json
```

It exits 0 only when every check passes and writes the same checks to
`--output`.

| Check | Path |
| --- | --- |
| `audio_cancel`, `audio_cancel_next_request` | Stop during streamed PCM, then a clean request in the same chat |
| `network_drop_visible`, `network_drop_recovery` | Connection lost after a delta, then a successful request |
| `server_error_visible`, `server_error_recovery` | HTTP 503, then a successful request |
| `asr_failure_visible` | Speech permission denied |
| `asr_ends_without_final` | Recognition ends without a final result |
| `jitter_burst`, `jitter_slow`, `jitter_reset` | Burst delivery, a 2.5 s gap, and the stream after a slow one |

The plain `--automation` runs above remain the live-server GUI gate; they are
the only path that contacts a server. The fault gate is the unattended
regression for the failure paths that the live gate cannot inject. `swift test`
covers the same fault paths without the GUI, plus the parser and artifact
checks.

## Voxt reuse

Voxt commit `5645daf0480e9bf572b45a2bba8b8e2f1e642daa` supplied the SSE/error
handling structure and the neutral/teal design tokens. Modified Swift ports and
unmodified reference sources are identified in `ThirdParty/README.md`. Apache
license and NOTICE files are retained.

## NOT_COVERED

- Realtime WebSocket, streaming ASR from the server, video, and tools.
- Model downloads, accounts, cloud provider presets, and HTTPS E2E.
- Persistent history across launches, multiple saved sessions, and retries.
- Immediate cancellation inside a model call on the server.
- Code signing with a certificate, notarization, sandboxing, installer, DMG.
