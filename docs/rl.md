# RL architecture — multimodal trajectory schema v2

The RL boundary remains outside serving. The recorder is a normal public
HTTP/SSE client; the reward reader and training worker are separate offline
concerns. No C++ serving file imports the recorder, trainer, optimizer, torch,
or transformers, and no second inference engine exists.

| File | Responsibility |
| --- | --- |
| `python/umminfer_rl_trajectory.py` | Out-of-process recording, narrative normalization, media verification, CAS export |
| `python/umminfer_sse_replay.py` | Exact raw-SSE/chunk capture, tick provenance, replay validation CLI, byte-identical offline replay |
| `python/umminfer_token_export.py` | Optional tokenizer-backed token ids, UTF-8/source spans, SFT masks, and independent validation |
| `python/umminfer_rl_reward.py` | Export-only reward interface, deterministic composite UMM reward, and plugin loader |
| `python/umminfer_rl_worker.py` | Independent v1/v2 validation, NDJSON worker protocol, checkpoint hash chain |
| `tests/narrative_rl_fixtures.py` | Deterministic interleaved narrative, explicit-slot, failed-leg, and cancelled-response fixtures |
| `tests/sse_replay_test.py` | Loopback live/replay byte identity, framing, timing, CAS, chunked HTTP, and tamper contracts |
| `tests/sse_replay_live.py` | Opt-in isolated real public-SSE capture, cancellation, fresh-process replay, framing, and tamper gate |
| `docs/sse-replay-live.md` | SM90 live hashes, contract-media boundary, replay evidence, and cleanup record |
| `tests/token_export_test.py` | Multilingual/UTF-8, visual-anchor, cancellation, mask-equivalence, malformed-input, and optional real-tokenizer contracts |
| `tests/narrative_slot_reference_live.py` | Explicit `source=slot-N` offline self-check and deferred real-model gate |
| `tests/rl_architecture_test.py` | Deterministic schema-v2 recorder/validator/reward contracts |
| `tests/rl_phase2_test.py` | Worker/checkpoint/protocol contracts; does not execute an optimizer |

The recorder captures only traffic driven through it. A passive sidecar that
captures other clients is not implemented.

The Stage C single-variable narrative-routing runner builds on this export
boundary. Its four existing policy overlays, fixed-request fingerprints,
label-gated metrics, deterministic contract replay, and explicit non-claims are
specified in [narrative-routing-ablation.md](narrative-routing-ablation.md).
The raw recorder adds no latency observations or ablation labels to trajectory
schema v2. Offline analysis may enrich the exported JSON object with the
optional `ablation` metadata documented below; the existing narrative-routing
runner continues to keep its own latency observations outside its trajectories.

## Compatibility decision

New recordings have `schema_version: 2`. The worker accepts both schema 1 and
schema 2, including a mixed-version JSONL export. This is backward-compatible
read support, not an up-conversion: v1 records keep their original fields and
validation semantics, and the reader does not invent empty narrative fields for
them. Unsupported, missing, boolean, or non-integer versions are rejected.

Raw replay is an **additive schema-v2 extension**, not a v3 migration. Records
without `sse_replay` retain their exact old validation behavior. Replay-enabled
records still have `schema_version: 2`; their sidecar declares
`format: "umminfer-raw-sse-replay-v1"` and `format_version: 1`, so a future
incompatible sidecar can be rejected explicitly without invalidating old JSONL.
Source-binding fields on normalized events are likewise additive. The replay
CLI requires replay-enabled v2 records; it does not invent bytes for old ones.

Token/span export follows the same additive rule. `token_export` is optional,
has its own `format: "umminfer-trajectory-token-export-v1"` and
`format_version: 1`, and never changes `schema_version`. V2 records without it
and v1 records retain their old fields and validation behavior; a v1 record may
not carry the block. The offline file exporter leaves v1 records untouched.

V2 retains the phase-1 keys used by the GSM8K trainer (`output_text`,
`output_sha256`, `events`, `request_params`, `inputs`, `terminal_state`, reward
blocks, and the later `messages`/`prompt_id` enrichment). The phase-2 REINFORCE
consumer therefore reads either version. Optional post-hoc token ids are now
available through `token_export`; they are re-encoded training/export ids, not
the model's sampled decode ids. The phase-2 boundary-token limitation in
[rl-phase2.md](rl-phase2.md) is unchanged.

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

## Raw SSE byte replay (sidecar format v1)

Enable raw capture for a normal recorder call with
`record_chat_completion(..., capture_replay=True)`. Stage C
`sse_stream_to_v2_record()` enables it by default, so `--capture-live` also
writes replayable streams. The recorder observes the public HTTP response only;
it neither imports nor instruments the serving process.

The exact response byte string is stored as `artifacts/<raw-sha256>.sse`. This
sidecar necessarily contains the original base64 artifact frames, which is why
it is separate from JSONL. Decoded media continue to use their existing CAS
files and the JSONL still contains no media `data` payload. A replay read copies
verified input/output CAS objects and the raw sidecar into the destination and
refuses conflicting destination bytes.

The additive `sse_replay` object contains:

- `raw_ref`, `raw_bytes`, and `raw_sha256`, binding the complete sidecar;
- `chunks[]`: zero-based `index`, contiguous `offset`, `bytes`, `sha256`, and
  arrival `tick` for every actual response read;
- `frames[]`: the same identity fields plus `kind` (`event`, `done`, `comment`,
  `keepalive`, or `control`) and `data_sha256` for data frames. Frames cover the
  raw byte string exactly, retain LF/CRLF spelling and multi-line data, and the
  final frame must be `[DONE]`;
- `clock`, `done_frame`, `event_count`, `normalized_events_sha256`,
  `terminal_state`, and `terminal_event_seq`.

Every normalized event adds `source_event_index`, `source_event_sha256`, and
`source_event_tick`. Multiple normalized events may bind one data frame (for
example, content plus finish); a synthetic plain-OpenAI terminal binds the final
`[DONE]` frame. Source indices and ticks are nondecreasing, server sequences
must be gap-free from zero, and exactly one final normalized terminal is
required. `normalized_events_sha256` is over the canonical event array including
these bindings. Worker, Stage C, replay, and reward validation reject modified
frame/chunk hashes, event bindings, timing, sequences, or terminal state rather
than repairing them. Whenever `sse_replay` is present its raw sidecar is
required, re-hashed, and re-normalized without writes for comparison with the
source-bound event array. Records without replay retain the old behavior,
including reward scoring with CAS hidden; replay-bearing records never silently
drop that byte-level verification.

### Clock and provenance contract

Ticks are relative observations, not UTC timestamps. Two explicit clock kinds
are supported:

| `clock.kind` | Source / unit | Meaning |
| --- | --- | --- |
| `monotonic_ns` | `python.time.perf_counter_ns`, `ns` | Observed chunk-arrival tick relative to `capture_start`; this is the live default |
| `logical_tick` | `caller_supplied_logical_ticks`, `tick` | Deterministic test/fixture ordering only; `observation` is `deterministic_fixture_not_measured_time` |

Both use `origin: "capture_start"`. Chunk ticks are sampled before incremental
frame parsing and a completed frame inherits the current chunk's tick.
Coalesced events may share a tick. Replay never calls a clock: it reuses the
recorded descriptors, so replayed JSONL is byte-identical. No wall-clock value,
hostname, UUID, or random id is introduced. Stage C aligns its live monotonic
origin with request start and derives TTFT/TTFA from source arrival ticks, not
from the later normalization pass.

### Offline replay

```text
python3 -I python/umminfer_sse_replay.py \
  path/to/trajectories.jsonl --out replayed/trajectories.jsonl
```

The input's sibling `artifacts/` is used by default; `--artifact-dir` selects
another source CAS. The output path must be new and different from the input.
Success requires byte-identical canonical JSONL and exact raw-SSE reproduction,
and prints `RAW_SSE_REPLAY: PASS`. Malformed JSON/SSE, truncated frames, missing
`[DONE]`, bytes after `[DONE]`, invalid UTF-8 data, duplicate/out-of-order server
sequences, hash/timing/terminal tampering, and CAS mismatches produce a concise
`replay error:` and nonzero exit. Cancellation and incomplete artifact groups
follow the existing v2 cancellation rules and replay identically.

## Optional token-id and source-span export (format v1)

`python/umminfer_token_export.py` adds exact tokenizer-backed supervision data
after recording. It is an offline utility, not a recorder hook: serving and raw
capture do not construct a tokenizer, and the module imports no serving,
trainer, torch, numpy, or transformers code on its structural path. Only an
explicit real-tokenizer CLI invocation lazily loads `transformers` with
`local_files_only=True`, `use_fast=True`, and `trust_remote_code=False`. No model
is loaded and no inference runs.

### Opt-in identity and file provenance

Export requires all four identity options together:

```text
python3 -I python/umminfer_token_export.py trajectories.jsonl \
  --out trajectories.tokens.jsonl \
  --tokenizer-dir /path/to/local/tokenizer \
  --tokenizer-id org/model-or-fixture-id \
  --tokenizer-revision immutable-revision-or-local-fixture-version \
  --tokenization-version transformers-and-tokenization-version
```

The output must be new; input JSONL and its CAS are read-only. The additive
`token_export.tokenizer` object records the explicit id, revision, tokenization
version, complete token-id-space size, unknown-token id (or null), and a sorted
`available_files` list containing every regular file below the selected
directory, each with relative POSIX path, exact bytes, and SHA-256. Absolute
paths and symlinks are not exported. `available_files_sha256` binds the complete
list. `config_sha256` binds identity plus policy, `source_record_sha256` binds
the entire record except `token_export`, and `export_sha256` binds the rest of
the export block. Apply reward/ablation enrichment before token export; later
source-record changes intentionally make the block stale.

### Streams, offsets, spans, and masks

Two streams are always present for an exported v2 record:

- `clean_text`: the recorded clean assistant text, including cancelled prefixes;
- `sft_target`: the canonical director target with visual markers reconstructed
  exactly as in the SFT dataset builder, including an explicit
  `| source=slot-N` suffix when its verified artifact binding is present.

Each stream stores `text_bytes`, `text_sha256`, ordered `token_ids`, and one
entry per token with `index`, `token_id`, half-open UTF-8 byte `start`/`end`,
codepoint `char_start`/`char_end`, `piece_sha256`, `source_event_indices`, and
`source_frame_indices`. Byte offsets start at zero, are contiguous, non-empty,
and cover the complete stream exactly. They may not split a UTF-8 codepoint.
This covers route headers such as `[[image: `, visual-prompt bodies, closing
`]]`, multilingual text, emoji, combining characters, and spoken spans.

Source events are zero-based normalized event indices. When raw replay metadata
is present, source frames are the corresponding zero-based raw SSE frame
indices. A token crossing coalesced clean deltas can name several events/frames.
A synthesized route header, prompt body, explicit-anchor suffix, or closing
delimiter maps to its originating `visual_slot` event/frame. Records without raw
capture use empty `source_frame_indices` but still have exact normalized source
events and byte offsets. The validator recomputes these mappings rather than
accepting caller annotations.

`spans[]` uses the existing Stage C types `narrative_prose`,
`visual_route_tag`, `visual_prompt_body`, and `spoken_style`, with byte and
codepoint ranges, token ranges, modality, and text SHA-256. Spans are contiguous,
non-overlapping, and cover the target. `sft_mask` records the `joint_full` span
weights and per-token positive weights. A token crossing span types receives the
same codepoint-overlap-weighted mean as the real SFT adapter; byte offsets never
change that established weighting rule. Tests compare ids, labels, weights,
span types, and closing-delimiter annotations against
`umminfer_joint_sft_live._token_loss_mask`. This format fixes the mask contract
to `joint_full`; other ablation profiles require a later versioned extension.

The fixed policy records `add_special_tokens=false`, `padding="none"`,
`truncation="none"`, `unknown_tokens="reject"`, exact contiguous UTF-8 offsets,
and per-token decode with special-token skipping and cleanup disabled. Unknown,
negative, boolean, out-of-range, zero-width, gap/overlap, normalizing, or
non-byte-exact tokenizer output is rejected rather than repaired. The
`decoded_sha256` binds concatenated per-token decode output to the original
UTF-8 byte string.

### Independent validation and compatibility

The worker and SFT integrity verifier automatically validate `token_export`
whenever present and otherwise preserve legacy behavior. Structural validation
needs no tokenizer or ML package: it recomputes source-record/config/export
hashes, target reconstruction, ids/ranges, source mappings, typed spans, masks,
and all stream digests. It rejects invalid ids, unknown tokens, offset/span gaps,
overlaps and out-of-range values, noncanonical concatenation, changed policy or
identity fields, stale records, and hash tampering. Structural validation alone
does not prove that ids belong to the declared tokenizer.

For that stronger check, pass the same explicitly identified local tokenizer to
`validate_record_token_export(..., tokenizer=...)`, or run:

```text
python3 -I python/umminfer_token_export.py trajectories.tokens.jsonl \
  --validate-only \
  --tokenizer-dir /path/to/local/tokenizer \
  --tokenizer-id org/model-or-fixture-id \
  --tokenizer-revision immutable-revision-or-local-fixture-version \
  --tokenization-version transformers-and-tokenization-version
```

Strong validation rehashes every available file, verifies the expected explicit
id/revision/version and token-id space, re-encodes both streams, compares every
id and byte offset, decodes every token independently, and reconstructs both
UTF-8 texts byte-for-byte. `--validate-only` without tokenizer options is the
stdlib structural mode and reports `tokenizer_reencoded: 0`.

`tests/token_export_test.py` exercises deterministic re-export, independent
fixture-tokenizer decode, multilingual/emoji/combining input, visual anchors,
source-frame mapping, cancellation, SFT mask equivalence, legacy absence, CLI
validation, and malformed/tampered inputs. Its optional local real-fast-
tokenizer fixture gate requires four `--real-tokenizer-*` options and prints
`REAL_TOKENIZER_FIXTURE: NOT_RUN` when unavailable. No real tokenizer, model
quality, sampled-token identity, or optimizer outcome is claimed by the default
CPU run.

## Determinism

JSON is key-sorted, compact UTF-8 with `allow_nan=False`. Turn indexes start at
zero per recorder and session, or retain the recorded index during replay. One
JSONL file has one writer. Legacy recordings without replay remain deterministic
for identical canned response bytes. Replay-enabled recordings are deterministic
when supplied the same logical tick sequence; independently observed live
arrival ticks naturally differ, while replay of one capture is byte-identical,
including CAS names and bytes.

This closes the **recorded transport** portion of the roadmap Replay gate. It
does not make model inference deterministic and does not add sampled token ids.
The isolated SM90 live gate has now passed for real text-only, interleaved
visual-contract, and in-flight-cancelled public streams with observed
`monotonic_ns` ticks. Fresh-process replay was byte-identical, raw/media CAS
checks passed, and altered frame/tick cases were rejected. Visual bytes are
explicit contract fixtures, not model-quality evidence. See
[sse-replay-live.md](sse-replay-live.md) for exact hashes, scope, and cleanup.
Model quality and generalized performance remain `NOT_RUN`.

## Rewards

`Reward.score(record)` reads only the exported JSON object. It needs no serving
memory, worker memory, model process, or optimizer state. The architecture test
computes a narrative reward from clean text, slot decisions, output descriptors,
and terminal state directly from the export. `NullReward` continues to write
`score: null`, preserving the distinction between missing and zero reward.
`train_step` requires a finite numeric JSON score and rejects bools, strings,
NaN, Infinity, and non-finite/oversized numbers before importing the trainer.

### Composite multi-objective UMM reward

`CompositeUmmReward` is a deterministic experiment signal computed only from
one exported schema-v2 JSON object. It requires labels and at least one latency
observation inside that same object:

```json
"ablation": {
  "expected_visual_modalities": ["image", "video"],
  "latency": {"ttft_ms": 500, "ttfa_ms": 900, "terminal_ms": 15000}
}
```

These fields are offline analysis metadata, not new serving response fields and
not measurements invented by the reward. Latency keys are `ttft_ms`, `ttfa_ms`,
`terminal_ms`, and `barge_in_ms`; values must be finite and non-negative. Raw
replay ticks are transport ordering/timing evidence, not automatically an
ablation label or budget measurement; an enrichment step must copy explicit
observations into the JSONL before scoring. Missing labels or latency coverage
produce `score: null`. Schema v1, malformed labels, malformed events, and
invalid observations are rejected rather than silently scored.

The default formula is:

```text
0.4 * routing_f1
+ 0.3 * prose_visual_balance
+ 0.3 * barge_in_stability
- 0.2 * latency_penalty
```

- `routing_f1` is the image/video set F1 between `visual_slots` and the labels.
- `prose_visual_balance` is the harmonic mean of non-empty `clean_text`
  presence and `min(predicted_slots, expected_slots) /
  max(predicted_slots, expected_slots)`, with 0/0 visual counts defined as a
  perfect count match. It penalizes missing prose and under/over-generation.
- `latency_penalty` is the mean of observed `latency / budget` ratios, each
  clamped to 1. Default budgets are 1000 ms TTFT, 1500 ms TTFA, 30000 ms
  terminal, and 250 ms barge-in. Constructor arguments can override weights
  and budgets; all values are validated and emitted with the components.
- Without a speech-start event, `barge_in_stability` is 1 for a completed turn
  and 0 for a failed or unexplained cancelled turn. With an observed
  `input_audio_buffer.speech_started`/`speech_started`, it is 1 only when the
  exported terminal state is cancelled with no error event or failed leg.
  This checks recorded cancellation consistency, not unrecorded playback
  acknowledgment or audio quality.

`apply_reward` writes the component values, observed latency, budgets, and
weights under `reward.components`, making every score independently
recomputable from the JSONL. Reapplying the same reward is byte-deterministic.
This is an ablation/training signal, not evidence of model-quality improvement,
an optimizer run, or real full-duplex performance.

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
[rl-phase2.md](rl-phase2.md). PPO, value-head/KL GRPO, and business reward
functions remain unbuilt.  Bounded Stage C DPO is implemented by a separate
adapter, not a new worker verb; its measured scope and checkpoint interface are
in [joint-training.md](joint-training.md#bounded-real-joint-dpo-optimizer-gate).

## Tests and current limits

```text
python3 -I tests/ablation_runner_test.py
python3 -I tests/rl_architecture_test.py
python3 -I tests/rl_phase2_test.py
python3 -I tests/sse_replay_test.py
python3 -I tests/token_export_test.py
python3 -I tests/joint_trainer_test.py
python3 -I tests/joint_sft_live_test.py
python3 -I tests/joint_dpo_live_test.py
python3 -I tests/narrative_live_e2e.py --self-check
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/multimodal_benchmark_test.py
```

These suites are deterministic and CPU-only. They use canned contract
fixtures, never real-model evidence. They cover raw/live replay byte identity,
split/coalesced framing, tick provenance, CAS reassembly and verification,
no-media JSONL, explicit slot anchors, leg errors, terminal legs,
cancellation, optional exact token/span export, SFT mask equivalence, tokenizer
and hash tamper rejection, v1 reads, export-only rewards, NDJSON, checkpoint
reopen/tamper checks, and the no-optimizer gate. The phase-2 suite
prints `REAL OPTIMIZER: NOT_RUN`; no new real optimizer or model-quality result
is claimed by this schema work.  The separately bounded Stage C joint-SFT and
joint-DPO optimizer gates and their resume/invalid-input contracts are specified
in [joint-training.md](joint-training.md).  They do not claim another schema-v2
REINFORCE run, GRPO, model quality, generalized improvement, or production-model
training.
