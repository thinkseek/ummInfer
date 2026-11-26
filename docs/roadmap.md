# Roadmap

## Goal

One engine runs VLM and ASR workloads together, on one control plane and one
data plane: a session shared by text, speech, and image turns; one orchestration
path that can plan several model workers for one request; one streaming event
contract; one cancellation path; backpressure at every worker boundary; one
error-propagation contract; and output coordination when a request produces more
than one artifact.

Image and video are optional modalities. They attach as image-understanding,
image-generation, or video workers when a use case needs them, and they do not
block the VLM+ASR main path.

The engine is the existing runtime, extended, not a rewrite. The C++ Coordinator,
ZeroMQ/MessagePack control messages, shared-memory relay, external Python stage
protocol, HTTP/SSE/WebSocket adapters, CLI, and clients stay. A model runs in its
own worker process behind that control plane; the control plane must not learn
model internals.

The architecture figure lives in [README](../README.md#architecture), and the
code-anchored implemented-versus-unverified/planned split lives in
[feature-map.md](feature-map.md). The North-Star end-to-end UMM vision and joint
training/ablation blueprint live in [vision.md](vision.md); this file holds the
phases, their gates, the model plan, and the verification split.

## Phases

Each phase lists a goal, deliverables, and an acceptance gate. A phase is
complete only when every gate row passes on the named hardware. A partial or
unrun case is recorded as `BLOCKED`, `NOT_RUN`, or `NOT_COVERED`, never as
passed. `NOT_RUN` means no run was attempted; `BLOCKED` means a required
external dependency, such as hardware access or credentials, is missing.
[Validation](validation.md) uses `BLOCKED` and `NOT_COVERED` for the same
purpose.

### Current integration status — 2026-11-14

Optional multi-turn session history landed on 2026-11-14. `session_id` remains
a grouping label by default; with `--session-history` and the per-request
`session_history=true` opt-in, prior completed turns of the same session are
prepended to the next VLM prompt. The store bounds, eviction, HTTP 400 contracts,
and the full rule set are tracked in [feature-map.md](feature-map.md#5-会话);
deterministic contract coverage and the real SM90 validation are in
[validation 5d](validation.md#5d-optional-multi-turn-session-history). SM90
multi-turn validation passed on 2026-11-14, including cancel/error exclusion and
the default-off gate; evidence is in
`/tmp/umminfer-phase5-final-sm90-evidence/live/`. Remote `http(s)` image URLs
were validated in the same window with the repository image served from a local
HTTP server.

### Integration status — 2026-11-13 (historical snapshot)

Phases 0–3 were **partial**, not complete, as of this date. The integrated
branch exposed the existing public chat route through `UnifiedEngine` to real
Qwen ASR and Qwen Omni workers. On SM90, public HTTP/SSE and the real CLI
completed ASR -> VLM with real Chinese and English recordings. The recorded run
also passed a cancel after ASR final with no VLM execution, an ASR failure with
no VLM execution, and a successful later request. Event order, parent/child
request ids, one terminal, and `in_flight=0` are in
`/tmp/umminfer-integration-logs/sm90-final/`. A later run cancelled while the
VLM was streaming and failed the ASR leg by stopping its real backend; evidence
is in `/tmp/umminfer-phase4-cancel-failure-evidence/`.

Realtime streaming added on 2026-11-13: the Qwen ASR stage requests SGLang
streaming, the realtime route forwards cumulative partials, the real client
sends PING control frames during the stream, and cancellation asks SGLang to
abort and records its `finish_reason=abort`. On 2026-11-13 the unified chat
route also passed opt-in ASR partials and a VLM-leg GPU abort; that evidence is
in `/tmp/umminfer-phase5-streaming-abort-evidence/` and [validation
5c](validation.md#5c-unified-asr-partials-and-vlm-gpu-abort).

Later runs extended this record and are documented in their single fact entries,
not restated here: scheduler-counter and GPU-hardware decode-stop evidence
reached PASS on 2026-11-14 ([validation
5e](validation.md#5e-backend-decode-stop-scheduler-sse-and-gpu-hardware-evidence));
a real decode NaN injection passed the same day ([validation
5f](validation.md#5f-real-model-compute-nan-failure-recorded-on-2026-11-14));
CUDA build, IPC, sanitizer, and stress passed on dedicated GPUs 2 and 3 (see
[SM90 status](#sm90-status)).

As of this 2026-11-13 snapshot the full Phase 3 Mac profile, weights hashes, full
resource attribution, and output reproducibility were not a complete record. The
Mac low-resource profile later passed on 2026-11-13 ([validation
5b](validation.md#5b-mac-unified-low-resource-profile)); model revisions stay
`UNKNOWN`. No full Phase 2 or Phase 3 gate claim is made.

### Phase 0 — Contract and orchestration

**Goal.** Define the worker and session contract before adding model code.
Existing chat and ASR paths keep working through it.

**Deliverables.**

- A session contract: session id, ordered turns, per-turn modality, and the
  owner of session state. The control plane owns the session; workers own model
  state only.
- A worker contract: capability declaration, input and output schema,
  streaming event types, cancel signal, backpressure signal, and terminal error
  shape. It rides on the existing MessagePack control messages; it does not add
  a second protocol stack.
- An orchestration plan: a request is a list of ordered worker steps with input
  and output binding. One step is the current chat or ASR case; several steps
  are the new case. The fixed-upstream proxy is not a plan step: it serves one
  route from one upstream and has no Coordinator, session, or orchestration.
- One event stream per request that multiplexes step events in plan order.
- Cancel, backpressure, and error propagation rules for a multi-step plan,
  including which step observes a cancel first and how a downstream step is
  informed.
- Contract tests. A two-worker plan runs with two conforming test workers, one
  of which fails; the test checks event order, cancel, error shape, and session
  isolation.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Existing behavior | Chat JSON and chat SSE still pass the default `zig build test`, which needs `-Ddeps-prefix` and `-Dcurl-prefix`. Transcription and realtime have no C++ case in that suite; they must still pass their hand-run Python E2E: `tests/client_boundary_e2e.py` for transcription and chat boundaries, and `tests/realtime_contract_e2e.py` plus `tests/realtime_client_contract_e2e.py` for realtime, each with a real TinyCTC model. |
| Plan execution | A two-step plan with test workers returns ordered stream events and one terminal event. |
| Cancel | Cancelling a running two-step plan reaches both workers and leaves `in_flight` at zero. |
| Backpressure | A slow consumer bounds worker output without unbounded buffer growth; the test measures high-water bytes. |
| Error propagation | A failing middle step yields one terminal error with the failing step id, and later steps do not run. |
| Session isolation | Two concurrent sessions do not share turns or output events. |
| No model internals | The contract carries no model-specific field; a test worker and a real worker use the same schema. |

### Phase 1 — Real model workers

**Goal.** Replace test compute with real models behind the Phase 0 contract, one
model family per worker process.

**Deliverables.**

- An ASR worker that runs a trained model and reports real transcription, with a
  declared audio format contract and resampling at the worker boundary where the
  model needs it.
- A VLM worker that accepts text plus image input and produces text or streamed
  text. The worker declares image input in its capability record.
- Worker startup, readiness, health, and shutdown under the existing stage
  protocol.
- Model identity in every response: model id, revision, precision, and
  quantization method, so a later comparison can attribute a result.
- Per-worker resource accounting: peak accelerator memory and host RAM.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Real ASR output | A real audio file produces text through CLI and HTTP, with model id and revision recorded, and the text is checked by a human and a reference transcript. |
| Real VLM output | A real image plus a prompt produces a decoded text answer through CLI and HTTP. HTTP 200 alone is not acceptance. |
| Streaming | Both workers stream partial output through the Phase 0 event stream. |
| Cancel | Cancel reaches both worker types and the worker process stays usable for the next request. |
| Readiness | `/ready` answers 503 before worker `stage_ready` and 200 after. |
| No simulator | No test substitutes a fake model, a handwritten response, or a generated artifact for model output. |

### Phase 2 — ASR to VLM orchestration

**Goal.** Run one request through ASR then VLM in one session, with one event
stream and one terminal result.

**Deliverables.**

- A plan that binds ASR output to VLM input.
- Ordered streaming: ASR progress and final text arrive before VLM tokens.
- Output coordination: the client receives both the transcript and the answer
  with clear step attribution.
- Error and cancel semantics for the two-step plan, including a cancel between
  steps.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Joint output | One request with real audio and a real image returns a transcript and an answer that uses both inputs. |
| Event order | Stream events show ASR completion before the first VLM token. |
| Cancel between steps | A cancel after ASR and before VLM leaves no VLM work running and `in_flight` at zero. |
| Failure isolation | An ASR failure returns one terminal error and does not start the VLM step. |
| Session reuse | A second turn in the same session sees the first turn's transcript and answer. |

### Phase 3 — Unified multimodal end-to-end acceptance

**Goal.** Accept the unified engine through the public interfaces on real
models, on the named hardware, with the Phase 0 contract in force.

**Deliverables.**

- One acceptance script that drives the real CLI or public C++ API through a
  real server, real workers, and real models, then decodes the real artifacts.
- A recorded run: model ids and revisions, weights hashes, precision,
  quantization method, parameters, seeds, artifact SHA256, wall time, output
  length, peak accelerator memory, host RAM.
- The Mac profile and the SM90 profile, recorded separately.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Path | CLI or public API to ummInfer server to real workers to real models to decoded artifact. A direct backend run is supporting evidence only. |
| Mac profile | Low-resource real-model run completes and its artifact decodes. |
| SM90 profile | High-precision run completes and its artifact decodes. The 2026-11-13 audio-only run is evidence for the public path, but it does not complete the full profile or its record requirements. |
| Reproducibility | The recorded parameters reproduce the recorded artifact hash, or the difference is explained. |
| Status honesty | Any unrun modality or model is recorded `BLOCKED` or `NOT_RUN`. |

The recorded-run fields and the anti-simulator rules are in
[validation section 7](validation.md#7-future-model-acceptance-gates).

### Phase 4 — Optional image and video

**Goal.** Attach image understanding, image generation, and video workers when a
use case needs them, without blocking the VLM+ASR path.

**Deliverables.**

- A capability record and a plan step per optional worker, on the same Phase 0
  contract.
- Image input to the VLM worker through the existing content part schema, and
  image output coordination when a plan produces an image artifact.
- Video as a separate worker with its own job lifecycle, if a use case requires
  local video.
- A configuration path that runs the engine with zero optional workers.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Main path unaffected | The unified VLM+ASR plan passes its gate with every optional worker absent. |
| Image in | A real image reaches the VLM worker and the answer references its content. |
| Image out | A generated image is decoded by an image decoder; format, dimensions, color mode, and SHA256 are recorded. |
| Video | Unrun cases stay `BLOCKED` or `NOT_RUN`; no claim of local 2K or untested video output. |
| Contract reuse | Optional workers use the same event, cancel, backpressure, and error schema; no worker-specific event channel. |

### Phase 5 — Unified performance baseline and joint optimization

**Goal.** Establish one measurable baseline for the unified engine, then
optimize it without changing output.

**Deliverables.**

- A fixed configuration per path: model snapshot, prompt, `max_tokens`,
  `temperature=0`, sample count, exclusive measurement window.
- Reported metrics per path: first-event latency, total latency, output tokens
  per second, peak memory, cancellation latency, queue depth.
- Attribution per level: direct model, worker, control plane, HTTP/SSE.
- Artifact hashes and binary, stage, harness, model, and weights hashes in every
  report.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Baseline recorded | One report per hardware profile with every path present; no skipped path. |
| Output identity | One output hash per path, equal across baseline and candidate runs. |
| No regression | Candidate first-event latency, total latency, and peak memory are not above baseline by more than 5%, or the difference is reported as unchanged. |
| Stability | The coefficient of variation is reported; `cv > 0.10` means the window was not quiet and the run repeats. |
| Attribution | The report separates model time from control-plane time. |

The existing MLX text-chat optimization record keeps its own convention; it is
not a unified-engine baseline. [docs/mlx.md](mlx.md) gives the
number and its conditions.

The first real SM90 unified baseline is recorded in
[performance-baseline.md](performance-baseline.md): 10 scenario blocks, 200
successful requests, and unified concurrency 1/2/4. In that pre-change baseline
one serial Omni worker delivered about 20 output tokens/s and concurrency 2/4
added queueing rather than throughput. The joint worker-concurrency change
removed the queueing ([joint-optimization.md](joint-optimization.md): at
concurrency 4 total p95 fell from 55.2 s to 14.9 s, aggregate throughput 3.6×).
A complete Phase 5 Mac unified baseline remains unmeasured; low-resource
latency and RSS are recorded in [mac-unified.md](mac-unified.md). Per-worker
model-time attribution and cancellation latency remain unmeasured.

### Phase 6 — RL rollout, trajectory, reward, and training

**Goal.** Expose the unified engine as a rollout target for reinforcement
learning, without putting training inside the serving process.

**Deliverables.**

- A rollout record per request: session id, ordered turns, per-step inputs and
  outputs, event timestamps, cancel and failure events, terminal status.
- A trajectory export in a documented format, written by the control plane, not
  by a worker.
- A reward hook at the control-plane boundary: reward is computed from recorded
  trajectories, and a trainer reads the export.
- Independent training as a separate process and repository concern. The
  serving engine does not load optimizer state.
- Determinism controls documented per worker: seed handling, sampling
  parameters, and what cannot be made deterministic.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Export | A run produces a trajectory file that parses and contains every turn of the session. |
| Replay | Replaying a recorded trajectory reproduces the recorded outputs within the stated precision, or the difference is explained. |
| Reward boundary | Reward computation reads the export only and does not need access to worker memory. |
| Isolation | Training load does not degrade serving latency in the measured window, or the degradation is measured and reported. |
| No unverified claim | No phase claims an RL result that has not run on real models and recorded hardware. |

**Phase 6 step 2 — real optimizer-step validation (2026-11-14).** The
architecture above was exercised end to end on SM90 GPU 4 with a real 0.5B
causal LM, real GSM8K rollouts, a verifiable exact-match reward, and real
REINFORCE/AdamW updates in the separate worker process. This validates the
training architecture, not a business model-training result. The measured
evidence, scope, limits, and replay precision boundary are in
[rl-phase2](rl-phase2.md).

**Phase 6 recorded raw-SSE replay (2026-11-26).** Schema-v2 exports can now bind
normalized events to exact source-frame hashes and monotonic arrival/logical
ticks, with content-addressed raw `.sse` bytes and transport chunk/frame
boundaries outside JSONL. Deterministic CPU loopback fixtures prove
live-versus-replay byte identity, source hashes, cancellation, explicit
`source=slot-N` anchors, framing variants, and tamper rejection; schema v1 and
older v2 records remain readable. The isolated SM90 GPU 0 live gate also
passed real public text-only, interleaved visual-contract, and in-flight
cancelled captures with observed `monotonic_ns` ticks, byte-identical
fresh-process JSONL/raw/media replay, split/coalesced reads, and altered
frame/tick rejection. Visual bytes are explicitly marked contract fixtures.
This satisfies the recorded-transport replay contract only; exact hashes and
cleanup are in [sse-replay-live.md](sse-replay-live.md). Model quality,
generalized performance, and sampled token-id recovery remain unclaimed. The
phase-2 boundary-token precision limit is unchanged.

**Phase 6 real local Qwen token-export gate (2026-11-26).** The accepted strict
token-export v1 remains byte-for-byte pinned. An explicit format-v2 extension
now handles the designated `Qwen/Qwen3.8-27B` fast tokenizer's legitimate NFC
composition and byte fallback without weakening v1: exact normalized token bytes,
original UTF-8 source units, raw-offset SFT mask equivalence, complete file
hashes, and fail-closed compatibility checks. The CPU-only local-file gate passed
multilingual, emoji, combining-mark, spoken, visual-slot, explicit-anchor, and
cancelled records, plus independent fresh-process export and strong validation
with `transformers 5.6.0` / `tokenizers 0.22.2`. Exact identities, hashes, and
limitations are in [token-export-real-qwen](token-export-real-qwen.md). No GPU,
model inference, sampled decode-id recovery, serving, online rollout, or model
quality is claimed.

**Phase 6 exported-token joint-SFT gate (2026-11-26).** The bounded SFT adapter
now has an explicit `--token-source token_export` mode with strong local
v1/v2 tokenizer/file/source validation and fail-closed mixed-batch behavior;
default legacy re-encoding and no-export pack bytes are unchanged. On isolated
SM90 GPU 7, the same frozen multilingual/NFC/emoji/spoken and explicit-anchor
visual examples produced exact-equal ids, labels, masks, losses, gradients, and
updates through legacy and export-driven paths, plus byte-identical
fresh-process export checkpoint resume. Exported post-hoc ids are not sampled
online behavior ids. Commands, hashes, rejection evidence, cleanup, and scope
are in [exported-token-sft-live](exported-token-sft-live.md).

The Stage C joint-SFT, joint-DPO, and joint-GRPO optimizer gates are separately
bounded to deterministic schema-v2 text/visual-slot fixtures and one small local
Qwen2 model. Their commands, fail-closed input/mask contracts, checkpoint
equivalence procedures, and measured SM90 status are in
[joint-training](joint-training.md). The DPO gate records real sigmoid-DPO
policy/reference log-probabilities. The GRPO gate records complete export-bound
groups, normalized advantages, clipped policy ratios, Schulman KL, and two finite
AdamW updates on GPU 7. These architecture gates do not satisfy or substitute for
online rollout generation, model-quality, generalized-improvement, or
production-model training gates.

**Phase 6 online-rollout contract (2026-11-26).** A separate trainer-side
orchestrator now defines deterministic online GRPO plans, seeds, group/candidate
order, exact request and behavior-policy/model/config/sampling hashes, required
endpoint attestation, raw-SSE/cancellation handling, bounded retry/no-retry,
complete group validation, and fresh-process byte-identical replay. Deterministic
fake-endpoint tests cover interleaved visual text and malformed/drifting groups
without ML or serving imports. The accepted frozen-offline GRPO adapter rejects
online-provenance groups. Real endpoint generation and online optimizer/quality
outcomes remain `NOT_RUN`; behavior log-probabilities/masks are `NOT_COVERED`
unless exposed with complete evidence by an endpoint. The current public server
lacks the required response attestation, and the later isolated live gate is in
[joint-training](joint-training.md#online-grpo-rollout-generation-and-validation-contract).

## Model support plan

These entries are not current deliverables. Their acceptance procedure,
precision rules, artifact checks, and the quantized-versus-official distinction
are in [validation section 7](validation.md#7-future-model-acceptance-gates).

| Model | Status | Target and constraints |
| --- | --- | --- |
| `Qwen3.5-Omni` | **PLANNED** | Authorized official API only. The documented model IDs are `qwen3.5-omni-plus` and `qwen3.5-omni-flash`; their documented interface accepts text, image, audio, and video input and returns text or streaming speech. Authorized access is unverified, so every execution case is `NOT_COVERED`. Do not claim image or video generation. |
| `MiniMaxAI/MiniMax-H3` | **IN_PROGRESS — integration planning** | Full-stack BF16 baseline on SM90. The open H3-Base path is limited to 768p; 2K needs hosted Context-IR/Regenerate-2K modules that are not fully open. |
| `MiniMaxAI/MiniMax-H3-Image` | **IN_PROGRESS — integration planning** | Full-stack BF16 baseline on SM90. Text-to-image and editing with up to 16 reference images. |

The BF16 scope, the 144.05 GB / 135.38 GB disk-size figures (not VRAM), the
no-official-quantized-checkpoint rule, and the final-acceptance path all live in
[validation section 7](validation.md#7-future-model-acceptance-gates).

## Verification split

Two hardware profiles, with different jobs:

| Profile | Job | Limits |
| --- | --- | --- |
| Apple Silicon Mac | Low-consumption development. Control-plane tests, stage contract tests, small real-model chat and audio runs, CLI and client checks. | No CUDA. No H3 or H3-Image acceptance. No performance commitment. A model that cannot run inside the Mac limits stays `NOT_COVERED` on Mac. |
| NVIDIA Hopper (SM90) GPU | High-precision acceptance and quality/resource comparison. BF16 baseline, CUDA build and IPC runs, sanitizer and stress runs. | Only one owner runs builds and benchmarks at a time, in an agreed window. Do not run benchmarks concurrently with sanitizer or stress runs. |

Mac runs stay at low resource; generic MPS or offload examples do not establish
model acceptance, and a simulator never replaces the real model.

### SM90 status

**ACCESSIBLE; real Qwen worker and unified audio-only runs recorded.** The SM90
GPU container served `qwen3-asr` on GPU 1, port 31000, and `qwen3-omni`
on GPU 0, port 31001. The final path used public CLI/HTTP -> `umm-infer` ->
`UnifiedEngine` -> both external workers -> those SGLang backends. The final
snapshot used 88581 MiB on GPU 0 and 86167 MiB on GPU 1. Model revisions are
`UNKNOWN`; the Omni config declares `bfloat16`.

The 2026-11-13 evidence directory is
`/tmp/umminfer-integration-logs/sm90-final/`. It contains the topology, model
metadata, real audio hashes, transcripts, answers, event timings, Worker receive
logs, cancellation and failure checks, GPU snapshot, and `SHA256SUMS`. The run
is local evidence, not a committed result artifact. Within this audio-only
directory's scope it does not establish a full BF16 baseline. Joint image input
and same-session reuse were covered by the separate `phase4-joint` run later the
same date (see README), and the Mac low-resource profile by [validation
5b](validation.md#5b-mac-unified-low-resource-profile).

A separate window on 2026-11-14 covered the CUDA path in the same container.
It used `CUDA_VISIBLE_DEVICES=2,3`, so the SGLang workers on GPUs 0 and 1 were
not touched, and no benchmark ran at the same time. The CUDA build, IPC tests,
compute-sanitizer, and a 10/10 stress run all passed; the commands and result
table are in [validation section 8](validation.md#8-cuda-build-ipc-sanitizer-and-stress-on-sm90),
with the run record in README. The window needed two Linux build fixes (the
dependency prefix now precedes distro headers, and `src/client/openai.cpp`
names the `std::min` type). Evidence is in
`/tmp/umminfer-phase5-cuda-stress-evidence/`. CUDA relay under real model load,
zero-copy device-tensor transfer, and multi-host IPC remain `NOT_COVERED`.

A third window on 2026-11-14 covered realtime live in the same container. The
real CLI, realtime server and Qwen3-ASR stage passed incremental partials,
PING/PONG interleaving, cancel, GPU abort, and a later request. Evidence is in
`/tmp/umminfer-phase6-realtime-sm90-evidence/`; the libcurl build recipe and
result table are in [validation section 3a](validation.md#3a-sm90-realtime-live-run).

## Next priorities

1. Record model revisions, reproducibility, and complete Phase 3 resource
   attribution. Weights hashes and HF byte-identity are already recorded in
   [performance-baseline.md](performance-baseline.md#model-snapshot-and-provenance);
   the SGLang `revision` field stays `UNKNOWN`.
2. Repeat the SM90 CUDA window under real model load, and add zero-copy
   device-tensor transfer and multi-host IPC if a use case needs them.
3. Keep the fixed-upstream proxy independent of the local unified path.
4. Track remaining engineering polish in
   [architecture.md](architecture.md#4-已知技术债与治理边界): extended visual
   evaluation before retiring fallback `native/omni` / `native/image` stages
   and external authorized `/v1/videos` upstream testing (Wave 7–8 resolved
   three-layer control-flow/stage convergence, dynamic TTS chunk metadata,
   public HTTP media passthrough, exact per-latent-frame causal streaming
   Wan2.2 3D-VAE preventing 1280x704 77-frame OOM, 7-model SHA-256 provenance
   archival, and the live 5-stage SM90 `43301` rollout).

## Explicitly unverified or unsupported

- CUDA relay under real model load, zero-copy device-tensor transfer, and
  multi-host CUDA IPC. The build, IPC tests, compute-sanitizer, and stress pass
  are recorded for SM90 on 2026-11-14.
- The full Phase 3 Mac profile and trained-ASR recognition quality. The
  low-resource Mac unified profile passed with the TinyCTC inference test model
  ([validation 5b](validation.md#5b-mac-unified-low-resource-profile)); TinyCTC
  recognition quality stays `NOT_COVERED`.
- A first-party HTTP 429 with `Retry-After` from this server.
- A successful upstream forward through the proxy. The in-tree tests prove
  configuration, route gating, and a connect failure only.
- Multi-stage dynamic batching (opt-in decode continuous batching is implemented
  for `qwen38` only; see [batching.md](batching.md)), model usage accounting,
  and per-worker queue statistics.
- Full Phase 2/3 acceptance across every modality and hardware profile.
