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

## L2: streaming output, sentence pipeline, prefix reuse

L2 closes the two product gaps that L1 left: the answer waited for the whole
text before audio started, and every turn re-prefilled the whole prompt.

### Streaming TTS

`TtsEngine::synthesize` takes an optional `StreamSink`. The talker emits its
first chunk after 2 codec frames (160 ms of audio) and later chunks every 8
frames (640 ms). A chunk decodes the frame prefix generated so far; only the
samples past the frames already emitted are kept, and they are appended to the
one vector the engine returns. The codec is causal, so a prefix decode has the
same mathematical value as the head of the full decode.

The TTS stage runs this path when the request sets `stream=true` and emits one
stage-stream chunk per codec boundary:
`{"audio": <PCM16>, "index": i, "first_frame": f, "frames": n,
"sample_rate": 24000, "channels": 1, "format": "pcm16"}`. The terminal result
still carries the complete WAV and the phase times.

Within one streamed request the concatenated chunks and the terminal WAV come
from the same buffer, so they are equal bit for bit. The stage checks this on
every streamed request (`tts stream_check identical=1`) and raises a protocol
error if it ever fails, so a regression cannot pass silently.

Across requests the F32 codec is not bit-repeatable. Two identical CLI runs of
the same text produce waveforms that first differ at sample 19422 with a
maximum difference of 9 PCM16 units (2.7e-4 of full scale); a repeat decode of
the same frames inside one process also differs (`tts repeat_check`). The
implementation therefore decodes each frame prefix once and never replays a
decode. No cross-run bit equality is claimed, and no listening test was run.

### Sentence pipeline

`include/umminfer/pipeline/sentence_split.hpp` cuts the VLM delta stream:

- strong terminators end a sentence: `。 ！ ？ ! ? ； ;`, newline and `…`;
- closing quotes/brackets and repeated terminators stay with the sentence;
- `.` ends a sentence only when the next codepoint is whitespace, end of text
  or CJK, and the previous codepoint is not a digit, so `3.14` stays whole;
- weak breaks are `， , 、 : ：` and space;
- with no terminator, a 48-codepoint window cuts at a weak break in its second
  half, otherwise at the window edge;
- the first TTS leg may cut at a weak break after 8 codepoints, so a long
  opening sentence does not delay all audio;
- a cut never splits a UTF-8 sequence.

Uncertain boundaries are not handled: abbreviations (`Mr. Smith`, `e.g.`) cut
after the period, `U.S. Army` becomes `U.S.` plus ` Army`, and a comma inside a
number is a weak break only at the length window. This is punctuation, not a
semantic end-of-turn model (that is L3).

The unified engine submits one TTS child leg per cut, `<id>:tts:0`,
`<id>:tts:1`, ... in answer order. A non-streaming request keeps the frozen
single `<id>:tts` leg and the one-shot artifact. A request that asks for image
and audio together keeps one leg too, because the frozen contract orders image
artifacts before audio. The request completes only when the VLM leg is
terminal and every output leg is terminal.

Each streamed chunk becomes one `artifact` event with `media_type: "audio"`,
`format: "pcm16"`, `mime_type: "audio/pcm"`, `sample_rate: 24000`,
`channels: 1` and `index: 0, total: 1`. The SSE frame carries the PCM16 base64.
The terminal WAV is not re-emitted.

### Realtime spoken response

`session.update` accepts `audio_output: {"voice": "vivian"}`. It needs a
unified engine with a registered TTS stage and server VAD; otherwise the
server rejects it. After `transcription.completed`, the listener submits a
VLM+TTS request for the transcript and forwards its events on the same socket:

- `response.text.delta` (`text`);
- `audio.delta` (`audio` base64 PCM16, `sample_rate`, `channels`, `format`);
- `audio.done`;
- `response.completed` or `response.failed`.

A new utterance cancels the response request through the unified engine, so
barge-in stops both the VLM and the TTS stage at their existing checkpoints.
Events that arrive after the cancel are dropped by the cleared request id.
The spoken path does not opt into session history: the prompt would grow with
every answer and first-audio latency with it. The session id still keys the
VLM prefill cache.

The turn buffer now drops exactly the committed turn on reset. Before this
fix, `reset_turn` erased `turn_audio_start` bytes, which is 0 on a normal turn,
so the audio of every turn accumulated. Multi-turn ASR then saw the whole
session and could return empty text.

### Incremental prefill

The omni stage keeps one prefix cache per process: the session id, the prompt
ids of the last request, and how many of its KV slots came from prefill. A
request in the same session reuses `min(longest common prefix, cached prefill
length)` tokens, embeds only the suffix, and prefills it with `history=prefix`
and the position ids sliced from the full prompt. Only prefill-written slots
are reusable; slots written by decode are not, because decode uses a different
reduction path.

The cache is on by default; `UMM_OMNI_PREFIX_CACHE=0` disables it. The stage
renders every message of the request, so `session_history` is part of the
prompt instead of being dropped. The stage reports `prefix_reused`,
`prefill_tokens` and `prefill_ms` per request.

## Measured on H20B, 2026-11-17 (L2)

Environment: container `linyu-dev`, native ASR on GPU 2, native Omni thinker on
GPU 7, native Qwen3-TTS on GPU 5, model directories
`/root/omniinfer-model-validation-20261111-k3/models/{asr,omni}` and
`/root/umm-mixed-output-20261116/models/qwen3-tts`, real speech
`asr_en_pcm16.wav`, server and stages built from this commit. Ports were
ephemeral; the resident services and GPU 0/1 were not touched.

Command, inside the container:

```sh
python3 -I tests/full_duplex_l2_live.py --mode all \
  --bin-dir bin --asr-model .../models/asr --omni-model .../models/omni \
  --tts-model .../qwen3-tts --audio .../asr_en_pcm16.wav \
  --out evidence/full --log-dir logs/full \
  --gpu-asr 2 --gpu-vlm 7 --gpu-tts 5 --turns 20
```

Raw evidence: `evidence/rt-fix/l2-evidence.json` for the realtime mode,
`evidence/fix2/l2-evidence.json` for the pipeline mode,
`evidence/prefix/l2-evidence.json` for the prefill rerun (the first-turn prompt
is longer there, so the shared prefix is measurable), and the stage logs next to
them. They are not committed artifacts. Each mode was rerun after the streaming
waveform change with the same command and `--mode realtime`, `--mode pipeline`
or `--mode prefix`.

### First audio, 20 spoken turns over one WebSocket

Each turn streams 2 s of real speech in 100 ms chunks, then 600 ms of silence;
`silence_ms` is 400. The measurement starts at the last speech chunk and ends
at the first `audio.delta`.

| Metric | Value |
| --- | --- |
| samples | 20 |
| p50 | 771.9 ms |
| p95 | 954.3 ms |
| min / max | 716.8 / 1033.6 ms |

The same run produced 157 streamed TTS legs. Every one logged
`tts stream_check identical=1`: the chunks concatenated in order equaled the
terminal WAV bit for bit, with no tolerance.

### Sentence pipeline advance

Same prompt, two HTTP requests with `stream=true` and audio output. In
`sentence_stream` the request sets `audio.stream=true`; `single_leg` omits it,
so one TTS leg starts after the VLM answer ends.

| Metric | sentence_stream | single_leg |
| --- | --- | --- |
| submit to first audio | 302.1 ms | 23091.0 ms |
| last VLM delta to first audio | -317.8 ms | 22507.9 ms |
| first audio before last VLM delta | true | false |
| audio chunks | 45 | 1 |

End-to-end first-audio advance: **22825.7 ms**.

The single-leg baseline synthesizes the whole answer before its first sample,
by construction. In this run the one-shot leg produced 512 codec frames (the
stage cap) for the same text, so its 22.5 s is what that path does for this
input, not a typical answer length. The comparison is end-to-end and includes
that cost. The streaming run's first audio arrived 317.8 ms before the last VLM
delta.

### Incremental prefill, two-turn session with `session_history`

The same two-turn conversation runs once with `UMM_OMNI_PREFIX_CACHE=1` and
once with `=0`. Stage-reported numbers for the second turn:

| Metric | cache off | cache on |
| --- | --- | --- |
| second-turn prefill | 127.9 ms | 88.4 ms |
| reused prompt tokens | 0 | 71 |
| answers identical | - | true |

Prefill time reduction: **30.9%**. The cached and
uncached answers are compared byte for byte and are equal.

## What L2 still needs

**Incremental ASR encoding.** The realtime path still re-encodes the whole turn
for each preview and each final, so ASR cost grows with the square of turn
length. An encoder with cached state would remove that, and a replacement
policy would keep the work of an in-flight request instead of aborting it.

**Client playback.** The server sends `audio.delta`; the bundled CLI client
does not play it or stop playback on `speech_started` yet. A client must keep
one socket open across turns and play PCM16 at 24 kHz.

**More than one cached session.** The omni stage caches one session's prefix.
Concurrent sessions each need their own KV snapshot.

## What L3 still needs

L3 is turn semantics learned from data. L1 endpoints on silence, so a thinking
pause ends a turn. L3 needs:

- a semantic end-of-turn model over partial transcripts and audio, not only
  energy;
- overlap handling: both sides speak, and the system decides whether to yield;
- a training corpus of real multi-turn conversations with barge-in labels;
- evaluation of turn-taking latency and false cut-ins, not only WER.

Nothing in this commit trains or evaluates such a model.
