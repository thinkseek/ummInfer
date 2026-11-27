# NS-03 autonomous modality routing benchmark (Task #78)

This is the executable four-class gate for
[north-star-cases.md](north-star-cases.md#ns-03). It evaluates natural input through
the default public client request path; it is not a training or post-training run.
Physical playback and perceptual quality are `NOT_RUN`.

## Frozen suite and populations

- Rubric commit: `91310b36184c86a1c147debb27db8643cc0cb3e4`.
- Benchmark freeze commit: `cdb18b834b7dbd6a8a7ca6f68544d26a99227fa7`.
- [suite.json](../benchmarks/ns03-routing/suite.json) contains 40 cases: five
  development and five held-out examples for each of text/proof/code, static
  spatial/structure, dynamic process, and contextual follow-up. English, Chinese,
  French, Japanese, Spanish, German, and Arabic are represented. Visual-noun
  adversarial cases discuss images/video without requesting generated media.
- Every follow-up executes a frozen setup turn first and then sends only the
  follow-up with `session_history=true` in the same real session. Setup and answer
  request/response/text hashes are separately bound in the scorecard.
- Canonical suite SHA-256:
  `35769692293fb99828ad23f40f1eaf3dd154b4c3c7e9b66b21d71ee27f61b097`;
  raw suite SHA-256:
  `b28004b52b9d44cdd8ee238a53c1c8329b9c8b3db6ab4c40ac3c0f07dafbcb9b`;
  raw preregistration SHA-256:
  `ccea4d085a743f8f1e412b43c9cdd22c1b1c3f64e6bb51d7fc66355fbfc20f8e`.
- [preregistration.json](../benchmarks/ns03-routing/preregistration.json) freezes
  cardinality, history, extraction, scoring, redundancy, statistical analysis, and
  promotion before model output. Its raw bytes are pinned again by the tests.

Development Before is captured first. One bounded candidate can then be frozen.
Only after the complete development Before and final candidate freeze may the
other three arms start. No model request is retried, and held-out answers are not
used for prompt or parser tuning.

## Exact default public input

The request builder mirrors a fresh default Web client state: `stream=true`,
`max_tokens=256`, `temperature=0`, omitted `top_p`/seed, auto narrative with
`max_slots=3` and `allow_text_only=true`, `512x512` PNG, `704x480` 29-frame MP4,
streaming `vivian` PCM voice, and modalities `[text,image,video,audio]`.
Continuity, style anchor, TeaCache, input-video frame controls, and manual modality
switches are absent. Follow-ups add only real session history.

The public SSE parser uses the outer OpenAI frame id as the authoritative request
identity; inner `request_id` is optional except on visual slots. It checks unique
request/session ownership, gap-free sequence, exactly one final complete terminal,
one `[DONE]`, clean text equality, canonical slots, complete artifact chunks,
content hashes, slot attribution, terminal leg state, decoded media, and PCM16
voice. Raw SSE and media remain external; compact reports aggregate audio rather
than duplicating thousands of PCM chunk rows.

Primary routing correctness is the exact frozen image/video slot cardinality plus
a clean complete protocol. End-to-end correctness additionally requires valid real
visual artifacts and voice. Public `leg_error` and media failures are retained
separately and listed as misses; they cannot be hidden by a successful slot or a
nonempty audio prefix. Redundant image and video diagnostics are separate from the
combined owner gate.

## Frozen candidate

[candidate.json](../benchmarks/ns03-routing/candidate.json) binds candidate
`auto-director-four-route-r1` to execution commit
`806b2e2033b3f63993f4c803a67334917eab03d9` and exact source hashes. The sole
runtime change is the default auto director prompt: explicit
TEXT_ONLY/STATIC_IMAGE/DYNAMIC_VIDEO decisions, one-visual cardinality, follow-up
and visual-noun abstention, concise output, and an explicit multi-beat exception.
Parser grammar, public protocol, clients, model, native backends, and frame routing
are unchanged. After reuses the exact Before native VLM/TTS/Wan binaries so only
the server binary differs.

`test_narrative_director_routing_prompt` fails against the unmodified director in
nine focused assertions and passes with the candidate. The initial raw analyzer
also has red/green coverage for explicit public leg-error retention. Capture and
verification hardening does not change the frozen suite or scoring formulas.

## Metrics and statistics

The scorecard reports per-case raw decisions, artifact hashes/counts, raw response
hashes, terminal/media counts, public errors, misses, and follow-up correctness.
It aggregates every class and population, including separate redundant image/video
rates. Paired analysis uses a two-sided exact McNemar sign test over discordant
cases and Wilson 95% intervals for each arm. The owner targets are routing accuracy
at least 80% and redundant visual rate below 5%; the rubric's 90% hard ideal is
reported separately. Promotion also requires a positive paired gain and held-out
non-regression.

## Reproduction and independent verification

With the exact source, model trees, dependencies, and an explicitly idle physical
GPU 1, the real entry point is:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I tests/ns03_routing_live.py \
  --population development --arm before \
  --base-commit 91310b36184c86a1c147debb27db8643cc0cb3e4 \
  --execution-commit 91310b36184c86a1c147debb27db8643cc0cb3e4 \
  --server-binary /path/to/before/umm-infer \
  --vlm-binary /path/to/umm-native-stage-qwen38 \
  --tts-binary /path/to/umm-native-stage-tts \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py --python /path/to/venv/bin/python \
  --vlm-model /path/to/qwen3.8-27b --flux-model /path/to/FLUX.2-klein-4B \
  --tts-model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir /path/to/external/development-before
```

Use `--population development --arm after`, `heldout/before`, and `heldout/after`
in the frozen order; After additionally requires `--candidate-manifest`. The
harness permits only GPU 1, starts an ephemeral loopback stack, never contacts the
resident service, stops only its own recorded processes, removes only its own IPC
runtime, and requires final `in_flight=0` plus idle GPU 1.

The independent verifier imports no producer module. It reconstructs every exact
request and raw public decision, decodes all media, rehashes binaries and model
files, derives same-session history, recomputes aggregates/statistics/thresholds,
and checks the complete external self-excluded manifest:

```sh
python3 -I tests/ns03_routing_verify.py /path/to/external/final --write-manifest
python3 -I tests/ns03_routing_verify.py /path/to/external/final \
  --out /path/to/external/independent-verification.json
```

The second command must be a fresh process and writes outside the frozen manifest
root. Synthetic contract fixtures supplement but never replace the real golden run;
a rehashed prompt-tamper control must fail exact request reconstruction.

Offline non-regression controls:

```sh
umm-test-local
umm-test-local -Doptimize=ReleaseFast
python3 -I tests/narrative_live_e2e.py --self-check
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/sse_replay_test.py
python3 -I tests/public_speech_pcm_test.py
python3 -I tests/showcase_test.py
python3 -I tests/i2v_geometry_latency_test.py
python3 -I tests/realtime_network_test.py
python3 -I tests/web_ui_contract_test.py
```

The machine-readable result is
[scorecard.json](../benchmarks/ns03-routing/scorecard.json).

## Paired result and final decision: NOT_PROMOTED

| Population | Before routing | After routing | Before redundant | After redundant | Before follow-up context | After follow-up context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 15/20 = 75% | 16/20 = 80% | 1/20 = 5% | 0/20 = 0% | 2/5 = 40% | 0/5 = 0% |
| Held-out | 17/20 = 85% | 17/20 = 85% | 0/20 = 0% | 0/20 = 0% | 2/5 = 40% | 1/5 = 20% |
| **All** | **32/40 = 80%** | **33/40 = 82.5%** | **1/40 = 2.5%** | **0/40 = 0%** | **4/10 = 40%** | **1/10 = 10%** |

End-to-end accuracy equals routing accuracy in every arm. After has no missing
visual slots on held-out cases and one development static miss; there are zero
wrong-modality decisions and zero media decode failures. The candidate fixes the
held-out dynamic misses, but three development and three held-out visual answers
contain a visual with no prose or voice. Those six responses fail the golden
text/voice requirement even when their visual modality is appropriate.

The owner routing/redundancy thresholds, positive overall paired gain, and held-out
routing non-regression are met, so the frozen scorecard's formula reports
`PROMOTION_ELIGIBLE`. The 90% hard ideal is not met. Paired improved/worsened counts
are 2/1 development, 2/2 held-out, and 4/3 overall; every exact McNemar p-value is
1.0, so the gain is not statistically robust. More importantly, follow-up
context correctness regresses by 30 percentage points overall and by 40 points on
development. Therefore the candidate is **NOT_PROMOTED** under the task's broader
non-regression requirement. `PROMOTION_ELIGIBLE` is not a claim that the candidate
was retained.

The final default `src/pipeline/narrative_prompt.cpp` is restored to official-main
SHA-256 `0c495a8800bdf2ca351b8862299eae3c3c5adbe45d8adfb2e213650e2892dc7b`;
`tests/unified_contract_test.cpp` is restored to
`fe72d9009005cb6ceac341d9c81d1b3e54516e7b131f1dc32f98c53aa8d5ab3d`.
The candidate-specific prompt test remains in the frozen candidate history, not in
the reverted default test runtime. No deployment or push is claimed.

## Retained negative evidence and open defects

- Public TTS leg errors decrease from 160 Before to 119 After but remain common;
  voice-missing primary cases increase from two to six. The benchmark does not
  convert a nonempty audio prefix into a claim of complete or perceptually valid
  speech. Native logs show bounded 8,192-frame generations and existing
  streamed-artifact-limit failures. Audio quality and physical playback are
  `NOT_RUN`.
- Real follow-up prompt sizes and answers expose an existing first-turn history
  defect: the default first turn omits `session_history`, while
  `UnifiedEngine::on_legs_drained_locked` records a completed turn only when that
  same request opted in. The follow-up therefore has no stored setup context.
  This task does not hide or rewrite that behavior with a second candidate.
- Initial development Before reports hit the harness's optional-inner-request-id
  bug. Original reports and all raw bytes are retained; corrected reconstruction
  changed no model output, prompt, route, or scoring rule. A later extractor fix
  retains slots even when empty/unclean text is itself a golden failure.
- Development After completed all 25 streams and passed final `in_flight=0`, but
  its reporter exited when a single post-cleanup NVML sample read 1% utilization
  at 3 MiB. No request was repeated. The recovered report explicitly marks the
  missing final process records; fresh GPU and no-owned-process checks prove the
  final cleanup. Later arms require three consecutive idle samples and append
  process start/stop events immediately.
- The first held-out Before attempt is excluded infrastructure evidence: a process
  event callback initialization defect interrupted the first Wan transition. Its
  raw files/logs are retained separately. Only task-owned processes were stopped;
  GPU 1 returned to 3 MiB/0%, and the complete r2 used identical frozen inputs,
  candidate state, and scoring. r1 is not included in any accepted denominator.

All errors, redundant visuals, misses, visual-only responses, and follow-up context
failures remain in the scorecard. No physical/perceptual quality claim is made.

## External evidence integrity and cleanup

The accepted external raw bundle is identified logically as `ns03-routing-task78`
and by the manifest digest below. Its complete self-excluded `manifest.json` covers
**408 files**. The raw evidence remains unmoved in external storage and is not
committed to Git; no machine-specific bundle path is part of the repository.

| Object | SHA-256 |
| --- | --- |
| External manifest | `cc5129a4501b06035950d1d676f8b081f2da3573481c083a06e7c279da7936da` |
| Paired scorecard | `2fa7e68d4635ffc12e4d7f5e4b1c517bd295b29cc24b7b7b9395210d374819df` |
| [Independent report](../benchmarks/ns03-routing/independent-verification.json) | `0171b509ab2a38f339ae0ee3c05b3136e1db13dba493dcf4ac9620ff2288dadf` |

The independent fresh process returned `PASS`, reconstructed 80 primary streams and
20 setup streams, decoded all retained media, rehashed all four model trees, checked
exact requests/sources/binaries/candidate ordering, reproduced every score,
statistic, threshold and miss, and found no task-owned process left. Every accepted
arm finished with `in_flight=0`; final GPU 1 samples were 3 MiB/0%, with three
consecutive idle samples on the two held-out arms. The resident service and GPUs
0/2–7 were not used or restarted. Perceptual/physical quality remains `NOT_RUN`.
