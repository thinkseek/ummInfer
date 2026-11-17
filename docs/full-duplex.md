# Full duplex, L1: streaming input, server endpointing, interruption

This page defines the first full-duplex layer, records what was measured on
H20B, and lists what L2 and L3 still need. It does not claim any behavior
that was not run.

## Definition of L1

A client streams PCM16 to the realtime WebSocket while the server transcribes.
The server, not the client, decides when a turn ended, and a new utterance
stops the previous turn's computation at a computation boundary.

L1 is complete when all five statements hold:

1. Audio appended in chunks reaches the ASR stage before `transcription.done`;
   the first provisional transcript arrives in less than one second for a real
   utterance.
2. `turn_detection` accepts a server configuration. Digital silence never
   starts a turn.
3. A pause longer than `silence_ms` commits the turn without
   `transcription.done`.
4. A new utterance while an earlier request computes aborts that request, and
   its output stops within one second.
5. The local suites and the real native ASR path pass.

## What the server does now

`src/serve/realtime_api.cpp`:

- **Incremental submission.** Every 200 ms (`kPreviewWindowBytes`, 6400 bytes)
  of new audio starts a preview request over the turn's audio so far, if no
  request is in flight. A slower stage coalesces windows instead of queueing
  them. Preview requests use the id space `realtime-preview-N`; turn requests
  keep `realtime-N`.
- **Cumulative previews.** Each completed preview emits one
  `transcription.segment` with `event_index: 0`, the turn's `segment_id`, the
  cumulative text and `is_final: false`. A preview failure is dropped and does
  not fail the turn; the next window retries.
- **Server VAD.** `turn_detection` accepts `null` (disabled, the previous
  behavior) or `{"type": "server_vad", "silence_ms": N}` with N in
  `[100, 5000]`, default 500. `src/serve/energy_vad.hpp` is a 10 ms-frame
  energy detector with an adaptive noise floor; it has no new dependency.
- **Endpointing.** `input_audio_buffer.speech_started` and
  `input_audio_buffer.speech_stopped` are emitted on VAD transitions. The stop
  event submits the turn's final request without `transcription.done`. Under
  server VAD the WebSocket stays open after `transcription.completed`, and the
  next utterance becomes a new turn with `segment_id + 1`.
- **Interruption (barge-in).** A VAD speech start while a request is in flight
  calls `Coordinator::abort`, releases the record, clears the provisional
  segment for the client, and starts the new turn at the onset frame. The
  stage stops at its existing abort check: the native ASR engine checks every
  audio-tower layer, every prefill layer and every generated token, and the
  Python stage checks between yields.
- **Compatibility.** Without `turn_detection`, `transcription.done` still
  submits the whole turn and the socket still closes after the result. The CLI
  client is unchanged.

The native ASR engine already had an abort hook (`native/umm_asr_native/engine.hpp`,
`AbortCheck`), so L1 uses it; no new checkpoint was needed in the stage.

## Measured on H20B, 2026-11-17

Environment: container `linyu-dev`, native ASR on GPU 2, model
`/root/omniinfer-model-validation-20261111-k3/models/asr`, real speech
`asr_en_pcm16.wav` from `/root/umm-native-asr-20261116/inputs/`, server and
stage built from this commit. The server used ephemeral ports; the resident
services on 43300 and 43301 and GPU 0, 1, 4, 5 were not touched.

Command, inside the container, using the committed script
`tests/full_duplex_live.py`:

```sh
python3 -I tests/full_duplex_live.py \
  --bin-dir /root/umm-duplex-20261117/bin \
  --model /root/omniinfer-model-validation-20261111-k3/models/asr \
  --audio /root/umm-native-asr-20261116/inputs/asr_en_pcm16.wav \
  --out /root/umm-duplex-20261117/evidence-repo --gpu 2 --host 172.17.0.2
```

Raw evidence: `/root/umm-duplex-20261117/evidence-repo/live-duplex-evidence.json`,
`server.log`, `stage.log`. It is not a committed artifact. The script exits 0
only when every check below passes and the barge-in interrupt was observed.

| Check | Result |
| --- | --- |
| `session.update` echo | `{"turn_detection":{"type":"server_vad","silence_ms":400}}` |
| Silence does not trigger | 500 ms lead-in plus 500 ms wait: 0 events, 0 turns |
| First provisional transcript | 601.4 ms after the first speech chunk (2 s of speech sent in 100 ms chunks at 100 ms pacing); text `嗯。` |
| Server-side first preview | `realtime first_partial_ms=1601 segment=0`, measured from the first byte of the session, so it includes the 1.0 s of leading silence |
| Endpoint commit | 559.5 ms from the first 100 ms silence chunk to `transcription.completed` (400 ms `silence_ms`, 100 ms pacing, the rest is final compute) |
| Preview cadence | 9 previews for a 2 s turn; stage compute 149-219 ms for 0.5-1.4 s of audio |
| Barge-in interrupt latency | New speech sent at `t_barge`; server logged `realtime interrupt reason=barge_in request=realtime-2 epoch_ms=...` 201 ms later |
| Abort to stage exit | Stage logged `aborted at token 0 ... abort_epoch_ms=... exit_epoch_ms=...`, 19 ms later (24 ms in an earlier run) |
| Old output after the barge-in | 0 segments and 0 `transcription.completed` from the aborted turn (`segment_id` 1) |
| New turn after the barge-in | First preview (`segment_id` 2) at 430 ms, first non-empty text and `transcription.completed` at 984 ms, text `嗯。` |

The 201 ms interrupt delay is two 100 ms chunk boundaries plus scheduling: the
VAD can only see speech once a chunk carrying it arrives.

Residuals observed in the same run:

- The first previews of a new turn can carry empty text (430/560/701 ms)
  because the ASR stage needs more than a short prefix.
- The aborted turn's provisional text stays on screen until the new turn's
  first preview replaces it; the server sends an empty provisional segment at
  the barge-in, and the client replaces it at 430 ms.
- The bundled CLI client treats the first `transcription.completed` as
  terminal. Multi-turn reuse of one WebSocket is a server capability that the
  CLI does not yet exercise.

## Known limits of L1

- Energy VAD only. Steady noise above the adaptive floor by 6 dB is classified
  as speech. The floor starts at -60 dBFS and learns only from non-speech
  frames.
- One request per session at a time. Previews re-encode the whole turn prefix,
  so cost grows with the square of turn length. The 200 ms window is a floor on
  latency, not a throughput claim.
- No echo cancellation. A speaker playing through the client's own output can
  start a turn.
- The preview window is not configurable on the wire.

## What L2 still needs

**Concurrent encoding and generation.** L1 uses one serial stage: a preview and
a turn request never run together, and each request re-encodes all audio so
far. L2 needs an incremental audio encoder with cached state and an incremental
decoder that emits tokens for new audio only. That removes the O(n²) cost and
lowers the first-token latency below the current ~600 ms. It also needs a
replacement policy for the in-flight request: today it is aborted and
restarted, which discards its work.

**TTS frame streaming.** The native TTS stage returns one WAV artifact. L2
needs the codec to emit frames as they decode, a wire event that carries those
frames, and a client that plays them while the LLM still generates. The TTS
codec already checks abort between decoder blocks, so barge-in can stop it at
frame boundaries.

**Client session lifecycle.** The bundled client must keep one socket open
across turns, handle several `transcription.completed` events, and stop
playback on `speech_started`. The server side is ready; the client is not.

## What L3 still needs

L3 is turn semantics learned from data. L1 endpoints on silence, so a thinking
pause ends a turn. L3 needs:

- a semantic end-of-turn model over partial transcripts and audio, not only
  energy;
- overlap handling: both sides speak, and the system decides whether to yield;
- a training corpus of real multi-turn conversations with barge-in labels;
- evaluation of turn-taking latency and false cut-ins, not only WER.

Nothing in this commit trains or evaluates such a model.
