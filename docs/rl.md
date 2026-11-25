# RL architecture — multimodal trajectory schema v2

The RL boundary remains outside serving. The recorder is a normal public
HTTP/SSE client; the reward reader and training worker are separate offline
concerns. No C++ serving file imports the recorder, trainer, optimizer, torch,
or transformers, and no second inference engine exists.

| File | Responsibility |
| --- | --- |
| `python/umminfer_rl_trajectory.py` | Out-of-process recording, narrative normalization, media verification, CAS export |
| `python/umminfer_rl_reward.py` | Export-only reward interface and plugin loader |
| `python/umminfer_rl_worker.py` | Independent v1/v2 validation, NDJSON worker protocol, checkpoint hash chain |
| `tests/narrative_rl_fixtures.py` | Deterministic interleaved narrative, failed-leg, and cancelled-response fixtures |
| `tests/rl_architecture_test.py` | Deterministic schema-v2 recorder/validator/reward contracts |
| `tests/rl_phase2_test.py` | Worker/checkpoint/protocol contracts; does not execute an optimizer |

The recorder captures only traffic driven through it. A passive sidecar that
captures other clients is not implemented.

## Compatibility decision

New recordings have `schema_version: 2`. The worker accepts both schema 1 and
schema 2, including a mixed-version JSONL export. This is backward-compatible
read support, not an up-conversion: v1 records keep their original fields and
validation semantics, and the reader does not invent empty narrative fields for
them. Unsupported, missing, boolean, or non-integer versions are rejected.

V2 retains the phase-1 keys used by the GSM8K trainer (`output_text`,
`output_sha256`, `events`, `request_params`, `inputs`, `terminal_state`, reward
blocks, and the later `messages`/`prompt_id` enrichment). The phase-2 REINFORCE
consumer therefore reads either version. Token ids are still absent; the
boundary-token limitation in [rl-phase2.md](rl-phase2.md) is unchanged.

## V2 record

One compact JSON object per line represents one terminal request. A shortened
record looks like:

```json
{
  "schema_version": 2,
  "session_id": "sess-1",
  "request_id": "http-1",
  "turn_index": 0,
  "endpoint": "/v1/chat/completions",
  "inputs": [{"kind":"text","ref":null,"media_type":null,"sha256":"…","text":"explain"}],
  "request_params": {"model":"…","stream":true},
  "clean_text": "At dawn, a lantern rises.",
  "clean_text_sha256": "…",
  "output_text": "At dawn, a lantern rises.",
  "output_sha256": "…",
  "visual_slots": [{"seq":1,"server_seq":1,"slot_id":"slot-0","slot_index":0,
                    "stage":"image","media_type":"image","prompt":"a lantern",
                    "text_offset":9}],
  "events": [
    {"seq":0,"server_seq":0,"type":"vlm_delta","text":"At dawn, "},
    {"seq":1,"server_seq":1,"type":"visual_slot","slot_id":"slot-0","slot_index":0},
    {"seq":2,"server_seq":2,"type":"artifact","artifact":{
      "artifact_id":"art-0","index":0,"total":1,"chunk_bytes":123,
      "chunk_sha256":"…","bytes":123,"sha256":"…","ref":"<sha256>.png"}},
    {"seq":3,"server_seq":3,"type":"complete","text":"At dawn, a lantern rises.","legs":[]}
  ],
  "artifacts": [{"artifact_id":"art-0","media_type":"image","mime_type":"image/png",
                 "ref":"<sha256>.png","sha256":"…","bytes":123,
                 "chunk_count":1,"chunks_sha256":"…"}],
  "leg_errors": [],
  "terminal_state": "completed",
  "terminal_event_seq": 3,
  "terminal_legs": [],
  "error": null
}
```

`clean_text` is the concatenation of clean VLM/OpenAI deltas, excluding ASR
transcripts, visual markers, terminal duplicate text, and media. For a
non-streaming assistant response it is `message.content`. `output_text` keeps
the v1 meaning: final ASR text plus VLM/OpenAI deltas, or the non-streaming
assistant content. Each has its own SHA-256.

`visual_slots` and `leg_errors` are projections of their corresponding events;
the worker recomputes and compares them. `terminal_event_seq` identifies exactly
one final `complete`, `error`, or `cancelled` event, and `terminal_legs` must
match that event's `legs` exactly. A plain OpenAI stream gets a normalized
synthetic `complete` event after `[DONE]`, so v2 consumers have one terminal
contract across plain and unified responses.

## Media and content-addressed storage

JSONL contains no embedded image, audio, or video bytes.

- Text inputs are inlined and hashed over UTF-8.
- Message WAV/image bytes are decoded and stored under `<sha256>.<ext>`.
- Inline `image.image`, `video.image`, and `audio.ref_audio` request options are
  also replaced by reference objects and added to `inputs`; leaving those data
  URLs in `request_params` would bypass the no-media rule.
- Remote image URLs remain unfetched references with `sha256: null`. Their
  reproducibility still depends on the URL serving unchanged bytes.
- The default artifact directory is `artifacts/` beside the JSONL file. A custom
  recorder directory must also be supplied as the worker request's
  `artifact_dir`, or directly to `validate_trajectory(..., artifact_dir=...)`.
- Existing CAS files are re-hashed before reuse. New files are verified before
  atomic publication. Refs are relative basenames bound to their SHA-256.

For each SSE artifact chunk, the event retains `index`, `total`, `chunk_bytes`,
and `chunk_sha256`, but not `data`. Chunks may be interleaved across artifact
ids. The recorder requires stable descriptor metadata, rejects duplicate or
invalid indices, reassembles by index, and checks total byte count and final
SHA-256 before CAS storage. `artifacts[]` contains only complete verified
outputs with `ref`, `bytes`, `sha256`, `chunk_count`, and `chunks_sha256` (a
digest over the ordered chunk hashes).

A completed turn with a missing chunk is a failed recording and publishes no
partial output for that artifact. A cancelled or failed turn may retain
incomplete chunk metadata, but partial bytes are not published. Complete groups
in such turns are still verified and exported. A failed visual leg need not
produce an artifact; its `leg_error` and terminal failed leg remain available.

The worker independently validates event and server sequences, slot/index
uniqueness, slot/artifact/leg binding, chunk completeness and hash binding,
text hashes, terminal consistency, output ordering, and every referenced CAS
byte string. Embedding media or changing a slot, hash, terminal, or chunk field
is rejected rather than silently repaired.

Non-streaming mixed `artifacts[]` responses are treated as one-chunk artifacts
and verified by the same path. Full visual-slot decisions require the server's
`visual_slot` events; a non-streaming response that omits those events cannot
supply decisions that were not observed.

## Cancellation and realtime

`record_chat_completion(..., on_request_id=...)` exposes the server request id
so another thread can call `POST /abort`. HTTP 200 means the coordinator
accepted the abort, not that a model forward had already stopped. Only a final
unified `cancelled` event makes the record cancelled. Completed and failed legs
remain in `terminal_legs`; interrupted visual/audio groups follow the partial
chunk rule above.

`record_realtime()` still accepts explicitly observed inbound/outbound events
without importing a WebSocket stack. A live realtime adapter is `NOT_COVERED`.
Transport failures and EOF without `[DONE]` remain failed records; the EOF text
is `stream ended before [DONE]`. A previously observed unified terminal keeps
its authoritative state.

## Determinism

There are no timestamps, UUIDs, host names, or random ids. JSON is key-sorted,
compact UTF-8 with `allow_nan=False`. Turn indexes start at zero per recorder
and session. One JSONL file has one writer. Two recordings of the same request
and canned response bytes are asserted byte-identical, including the CAS file
names and bytes.

The export stores normalized events, not raw SSE bytes. Byte-level raw-stream
replay therefore remains unimplemented, and the roadmap Replay gate is not
claimed. Per-event timestamps also remain absent because wall-clock values
would break byte determinism.

## Rewards

`Reward.score(record)` reads only the exported JSON object. It needs no serving
memory, worker memory, model process, or optimizer state. The architecture test
computes a narrative reward from clean text, slot decisions, output descriptors,
and terminal state directly from the export. `NullReward` continues to write
`score: null`, preserving the distinction between missing and zero reward.
`train_step` requires a finite numeric JSON score and rejects bools, strings,
NaN, Infinity, and non-finite/oversized numbers before importing the trainer.

## Worker and optimizer status

```text
python3 -I python/umminfer_rl_worker.py --state-dir <dir>
```

The NDJSON protocol remains version 1. Unknown optional fields are ignored.

| Request | Result |
| --- | --- |
| `{"type":"ping"}` | `pong` |
| `{"type":"train","task_id":"…","trajectory_path":"…","artifact_dir":"…"}` | Metadata-only `train_result`, `optimizer_status: "NOT_RUN"` |
| `{"type":"train_step","task_id":"…","trajectory_path":"…","model_path":"…","hyperparameters":{...}}` | Real optimizer result only after execution, `optimizer_status: "EXECUTED"` |

`artifact_dir` is optional when the default sibling directory is used. Invalid
exports and unscored `train_step` requests consume no checkpoint sequence and
do not import torch. Metadata-only checkpoints cannot claim an algorithm or
weights. A real `train_step` fills those fields only after the trainer ran and
weights were saved. If execution completed but a later file operation failed,
the error response reports `EXECUTED`, not `NOT_RUN`.

Checkpoint files remain `checkpoints/ckpt-<sequence:08d>.json` plus `.sha256`.
Sequence and filename must agree, every sidecar is recomputed on open, and
`base_checkpoint_sha256` links each manifest to its parent. The phase-2
REINFORCE verb, residency rule, and real historical evidence are documented in
[rl-phase2.md](rl-phase2.md). PPO, DPO, value-head/KL GRPO, and business reward
functions remain unbuilt.

## Tests and current limits

```text
python3 -I tests/rl_architecture_test.py
python3 -I tests/rl_phase2_test.py
python3 -I tests/narrative_live_e2e.py --self-check
python3 -I tests/multimodal_benchmark_test.py
```

The first two suites are deterministic and CPU-only. They use canned contract
fixtures, never real-model evidence. They cover byte reproducibility, CAS
reassembly and verification, no-media JSONL, leg errors, terminal legs,
cancellation, tamper rejection, v1 reads, export-only rewards, NDJSON,
checkpoint reopen/tamper checks, and the no-optimizer gate. The phase-2 suite
prints `REAL OPTIMIZER: NOT_RUN`; no new real optimizer or model-quality result
is claimed by this schema work.
