# NS-01 streaming opening and visual-admission diagnostic (Task #87)

## Result and promotion boundary

**NOT_PROMOTED; full NS-01 remains NOT_MET.** The single bounded auto-director
prompt candidate did not front-load the requested visuals, did not improve TTFA or
first-image latency, and did not preserve every content/lifecycle requirement. It
is absent from final runtime. Final production code under `src/`, `native/`, and
`include/` has zero diff from official main
`9c02225451252001c11021b389a7792caf03d503`; the only native delta is the
`stage_common_test.cpp` EOF-contract assertion described below.

Preregistration commit: `3c92f08262a54504f34004095139f459592bd8a0`.
Candidate runtime freeze: `dd7cabe0a5319478dec723be304ad4d9ea9c2dae`.
Candidate manifest: `312fbf91109d20afe580540d88a163fe8f181561`.
Final candidate-removal commit: `8b5b8863fb0a39e5b6ff834d5d1ce5563aa5d96e`.
The final director SHA-256 is again
`0c495a8800bdf2ca351b8862299eae3c3c5adbe45d8adfb2e213650e2892dc7b`.
No parser, scheduler, queue cap, public schema, native arithmetic, TeaCache,
training, optimizer, or client control changed. The compact machine-readable
result is [scorecard.json](../benchmarks/ns01-streaming-opening/scorecard.json),
SHA-256 `5d75972949a3a6f3a967176d3fedb8a7e21bb5e0df98a0870cb16001e8e90e9f`.

Human/perceptual evaluation, physical audio, deployment, and resident-service
contact remain **NOT_RUN / NOT_CONTACTED**.

## Profile and frozen protocol

The baseline traces separate four intervals:

1. The VLM starts clean prose at about 0.56 s and first native TTS submission at
   0.60–0.79 s.
2. No autonomous visual marker closes during the Chinese baseline. At VLM EOF,
   the existing fallback admits `slot-0` at about 15.67 s with the entire answer
   as its prompt; the public image completes at about 17.16 s.
3. The serial harness then stops FLUX and cold-starts native TTS. Public TTFA is
   therefore about 27.73 s, not a sentence-splitter or parser-buffering delay.
4. No video slot exists, so there is no I2V submit or playable artifact. The
   existing drain-TTS-before-Wan path is not the cause of that absence.

The candidate added one conditional prompt block for an explicit still-plus-
animation request, requiring a concise opening, an image, an anchored video, and
then the complete long-form answer. It was frozen before the first After or
English run and was never tuned on golden output.

[Preregistration](../benchmarks/ns01-streaming-opening/preregistration.json)
freezes the exact accepted Chinese black-hole request and an English companion,
768 tokens, greedy seed 42, two process-separated pairs per language, identical
models/settings/native binaries, automatic narrative, 4-step FLUX, 29-frame/20-
step Wan I2V, and streaming 24 kHz mono PCM16. The requests contain no hand-
authored visual directives or TeaCache field. All attempts, failures, and
outliers are retained.

## Paired measurements

Values below are milliseconds from public capture start. ASR is
`NOT_APPLICABLE_TEXT_INPUT`. `I2V` means the complete public/playable artifact,
not merely a visual-slot event.

| Frozen case | Arm | TTFT | First TTS submit | TTFA | First image | I2V |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chinese r1 | Before | 561.847 | 787.635 | 27,844.041 | 17,141.579 | absent |
| Chinese r1 | After | 741.103 | 876.081 | 28,297.359 | 17,663.842 | absent |
| Chinese r2 | Before | 559.705 | 787.885 | 27,619.264 | 17,181.711 | absent |
| Chinese r2 | After | 739.256 | 873.653 | 28,410.603 | 17,590.901 | absent |
| **Chinese median** | **Before** | **560.776** | **787.760** | **27,731.653** | **17,161.645** | **absent** |
| **Chinese median** | **After** | **740.179** | **874.867** | **28,353.981** | **17,627.372** | **absent** |
| English r1 | Before | 566.746 | 607.124 | 27,814.772 | 17,234.859 | absent |
| English r1 | After | 753.304 | 795.113 | 28,042.481 | 17,292.512 | absent |
| English r2 | Before | 559.309 | 601.302 | 27,621.413 | 17,214.047 | absent |
| English r2 | After | 745.377 | 785.669 | 27,398.799 | 17,204.480 | absent |
| **English median** | **Before** | **563.027** | **604.213** | **27,718.092** | **17,224.453** | **absent** |
| **English median** | **After** | **749.341** | **790.391** | **27,720.640** | **17,248.496** | **absent** |

Chinese TTFA changed by **−2.244%** (a regression) and first image by
**−2.714%** (a regression). English TTFA changed by only −0.0025 s and first
image by +0.024 s, noise rather than an engineering gain. No pair improves both
required metrics. These are paired engineering measurements, not a claim of
statistical model-quality equivalence.

Every After run still misses the hard TTFA, first-image, and I2V thresholds.
TTFT happens to remain below 800 ms in all eight arms, but the larger candidate
prompt moves it closer to that boundary.

## Content, media, and lifecycle integrity

- Chinese Before reproduces the accepted Task #81 text exactly: 3,611 bytes,
  SHA-256 `883802d889885bd032b18a3dc21c8f7e1ce77773f020ab0ee3e69a816ac8b572`,
  with 43/43 natural-EOS legs and the accepted 14,987,520-byte PCM hash
  `2ce2c53454349fd70edceab97e6afe6d93cf8ffc22e7bcdf497c303222d94f78`.
- Chinese After has a different 3,665-byte response and 43/43 natural-EOS legs,
  but still only the post-VLM fallback image at text offset zero. It has no
  concise pre-image opening, no video directive, and no I2V anchor.
- English Before has 97/97 natural-EOS legs. English After has 88/89: leg 62
  reaches 8,192 frames without natural EOS in both deterministic repetitions.
  Every completed native/public/terminal-WAV PCM join is still exact, but the
  non-EOS leg is a retained lifecycle failure.
- English After emits a real image directive only at sequence 648, text offset
  3,239, after multiple headings. It never emits the requested video. Its clean
  public text also ends at the 768-token VLM limit with a reserved visual tail
  suppressed at EOF. Parser suppression prevents tag leakage; it does not turn a
  truncated answer into a successful opening.
- All eight public streams have one terminal and one DONE, gap-free sequence,
  clean public/directive text, zero native Markdown presentation leakage, exact
  public/native PCM on completed legs, and final `in_flight=0`. All eight VLM
  generations end `finish_reason=length`, so none satisfies this task's natural-
  stop/completeness requirement.
- The scorecard retains **74** content/media/lifecycle failures across the
  matrix. Missing media remains null and `NOT_MET`; no placeholder artifact is
  called complete.

The corrected speech/prose comparison mirrors the accepted stateful single-
and double-emphasis normalization while preserving ordinary arithmetic and
intraword underscores. This analyzer correction changed no raw output. All
TTS text/prose comparisons now pass where the underlying content agrees.

## Promotion decision

The preregistered rule requires at least 25% Chinese median TTFA reduction, 50%
first-image reduction, improvement in every pair, exact After integrity, and all
controls. None of the latency, content, I2V, or natural-stop requirements is met.
The candidate is therefore **NOT_PROMOTED** and removed from final runtime. The
retained synthetic test still proves immediate opening TTS, fragmented parser
buffering, immediate image admission, deferred exact-byte I2V anchoring, clean
prose, and one terminal; a separate restoration guard proves the rejected prompt
is absent.

## Frozen NS-03 routing control: NOT_MET

The complete 40-case suite plus 20 real setup turns ran on GPU 7 with the frozen
candidate and identical native binaries. The
[NS-03 control scorecard](../benchmarks/ns01-streaming-opening/ns03-control-scorecard.json)
is byte-identical across two fresh scoring processes at
`5d18a0b606ca5b06ba7db3675584b4a3330a449219418e29650d80fce78ef0cd`.

| Metric | Accepted baseline | Frozen candidate |
| --- | ---: | ---: |
| Overall routing | 32/40 = 80% | **30/40 = 75%** |
| Development | 15/20 | **14/20** |
| Held-out | 17/20 | **16/20** |
| Text | 10/10 | **9/10** |
| Static | 6/10 | **5/10** |
| Dynamic | 6/10 | **7/10** |
| Follow-up routing | 10/10 | **9/10** |
| Follow-up context | 4/10 | **3/10** |
| Redundant visual cases | 1/40 = 2.5% | **1/40 = 2.5%** |
| Held-out redundant visual cases | 0/20 | **1/20** |
| Voice-missing cases | 2/40 | 2/40 |
| Public leg errors | historical 160 | 0 |

`dev-text-01` hit the frozen 1,200-second overall deadline. Held-out static
misses are `hold-static-03` and `hold-static-05`; `hold-dynamic-03` missed its
video; and `hold-followup-03` emitted a redundant image in a text-only follow-up.
`dev-dynamic-01` produced an image instead of a video. Dynamic overall improved
by one case, but that does not offset the text/static/follow-up regressions. All
14 valid visual artifacts decoded; there were zero media decode failures. The
terminal count is 39/40, with the deadline retained. The 90% hard ideal is also
NOT_MET.

The control has 11 explicit non-regression failures and `pass=false`. The
development SSH wrapper later returned 255 after its complete report and
`CAPTURED 20 cases` had already been written and GPU 7 released; no development
process remained. That wrapper anomaly is separate from the independently
verified raw report and does not erase any model outcome.

## Exact source, binary, and model identities

| Object | Before SHA-256 | After SHA-256 |
| --- | --- | --- |
| `umm-infer` | `9b4336af31511a7d44fb1090a946857726322458f69adef54e983b5834816a76` | `171a456a48e811139cd2d05961893d1348efc24078d6d5d20541cb8aeabe682f` |
| Qwen3.8 Stage | `1eb526171686bf8c2b90d568c5de4b5eb29dbaa7f3a4a34049e56c6832fa6339` | identical |
| Qwen3-TTS Stage | `ef8ee25611f540331499c371cd3538c7d777a2f24a3dc09c60bf596ddfa8422a` | identical |
| Wan Stage | `558959692895cadeaa732e1845d99898dfc5e1facb220cc8dbd7730508b2813e` | identical |

All model revisions remain `UNKNOWN`; exact local tree/config identities bind the
runs:

| Model | Files | Tree SHA-256 | Config/model-index SHA-256 |
| --- | ---: | --- | --- |
| Qwen3.8-27B | 29 | `22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| FLUX.2-klein-4B | 25 | `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997` | `51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0` |
| Qwen3-TTS 1.7B CustomVoice | 14 | `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d` | `17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9` |
| Wan2.2-TI2V-5B | 30 | `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616` | `6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7` |

Only explicitly idle physical GPU 7 was used. GPUs 0/1/2–6 and the resident
service were not used, contacted, restarted, or deployed to.

## Excluded infrastructure attempts

Three pre-acceptance harness failures are sealed under `negative/` and excluded
from all timing denominators: two Chinese Before attempts consumed the response
inside an assertion's eagerly evaluated error message, and one Chinese After
attempt followed a venv symlink to system Python before the public request. The
first two produced no public event; the third failed before request submission.
They caused no prompt or protocol tuning. The corrected accepted arms are each
captured once with identical frozen requests.

## Completed adjacent controls

- `umm-test-local` on the restored runtime: PASS, raw SHA-256
  `f7597603894e2903647959e273c3d479808e80a6caa5b584688960462b63b875`.
  ReleaseFast: PASS, `b016f3a37c53f9681d2a51e46fe593312e8eade502bcad5b635cd5e5771f05fa`.
- Native TTS host/config: 2/2 PASS,
  `72482d32bf31a90a96db0ff8cce7cdf8e4516c09c3844db107a26d97a41eabe4`.
  Real SM90 TTS CUDA determinism: 3/3 PASS,
  `1d3aef76c47599cc089584bb8be83043c81e4ddbee303f78d2735ade23fd4d13`.
  Applicable Stage CTest: 13/13 PASS,
  `97d7faa85b0fb69a44cf2f2356b4de4d77be3b2c2437490dfe4811fef8eaa994`.
  The original stale unclosed-directive assertion failed at
  `df779e157a71be777d0f60843c4ea0a661e19ddad52a323ae4a68418d9599a3d`;
  only the test was corrected to require accepted fail-closed EOF behavior. The
  two-device image2 allocator fixture was NOT_RUN under the GPU-7-only constraint.
- Candidate red/green: five focused director assertions failed on the unmodified
  baseline (`ba1de8c6843e52f0717370f08570f4e362a6c8a94a9fee20f1eaf46469d79216`);
  the frozen candidate passed the full local suite
  (`0753a21bde4b733138c1127d69ebf8f90ed9a0dc0e1ba9472c46fdb46aa3435b`).
  The final restoration guard and fragmented-opening lifecycle test also pass.
- Offline NS-01 request/projection controls: 4/4 PASS,
  `feabb4add9e2ca3374e538560634d1ec97dc4f4ce89aab12f3d726e36ea9b2f6`.
  Frozen NS-03 request/scoring controls: 12/12 PASS,
  `fc523972b6a1714a1d54810e5e0468e39c4bd3e3d5232a86ffd2e808fc829bd9`.
- Public narrative self-check, NS-06 slot-reference self-check, raw-SSE replay,
  public PCM, media/geometry, realtime-network, and Web contract controls all
  passed. Raw hashes respectively:
  `d372e72956ef90b949225a2166b005becfd8b5c8f275aa74cf68bf401a77f17e`,
  `67b629121baeb6cefb1365c87465adf2394a998aa7a2ef542fdd9b8b8593e512`,
  `c822a8a320a7cedfdc274de3a1d50dbb65289766bad5dd00e453c0d97df88275`,
  `326180ee8d8cca69d5e9188c5a565e031344c4fd6537440a49e966e55fae1e0d`,
  `a4c3352ba626bfd7c056ab4b7d7f35b98bd9ac851dad48df2cebe4c81fcab442`,
  `da7d6b12fb707746fd3ff74ec1aad6906e9e3919a304714fec909dc61e9e7d55`, and
  `c4003765af94d064155e04fb52b1ad4304c3a945773f82cf9a943e9578a3746f`.
- NS-06 focused cross-slot and accepted-live self-checks passed:
  `c8eafc153aa8e4c7e5489f765946d0ab808c7e0b01e1d709fbed4a3f54ff47f1` and
  `222990ab40a56d07d93322b4ece0cceb06c651a06d8419b5f75b28fc255b6375`.
  Human character/style judging remains NOT_RUN.
- NS-08 static export/import/reconnect/source-bound revision/cancel-retry metrics
  are all exactly 1. The independent static-arm verifier passed at
  `7b2a9864c0c83f89c4d1d71ea93ed11401cc2d09d73700fe7d8920c9e5b1d738`.
  The accepted current baseline still exposes two voice controls, so its
  zero-default-controls hard gate is false; this is not a Task #87 regression.
  Chromium, real macOS, physical audio, and human walkthrough remain NOT_RUN.
- All eight raw NS-01 arms independently redecoded and matched after the
  analyzer correction: `cacc78f1c2f4a155f0970250b1ba2b6a6a72477ea14f63af293e20a094db7318`.

## Independent whole-bundle verification and cleanup

The complete external bundle contains **16,059 regular files** and a
self-excluded `SHA256SUMS` at
`ff7a6173516af1bd6a9568d11198a87e5aab4f56573e0794f790b9246ec67044`.
It includes every accepted arm, all three infrastructure attempts, raw NS-03
cases, the complete NS-08 static arm, and all red/negative/control logs.

Two fresh independent `tests/ns01_streaming_opening_verify.py` processes produced
byte-identical PASS reports at
`41a3bf982f917d23afa4ed1a1277cb779c1c134b67a4c87f0d7e4d0de2b86cab`.
They redecoded raw SSE/CAS/MessagePack/WAV, recomputed public event order,
timings, content/media failures, per-leg PCM/EOS, pair aggregates, and promotion,
rehashed every binary and model file, and verified every manifest entry. The
[tracked report](../benchmarks/ns01-streaming-opening/independent-verification.json)
binds scorecard SHA-256
`5d75972949a3a6f3a967176d3fedb8a7e21bb5e0df98a0870cb16001e8e90e9f`,
`promotion_decision=NOT_PROMOTED`, and
`all_required_controls_pass=false`. PASS means evidence-integrity verification,
not candidate or full NS-01 success.

All task-owned model and test processes exited. GPU 7 returned to **3 MiB / 0%**
with zero compute processes. No other GPU or resident service was contacted or
restarted, and no deployment or push occurred.

## Reproduction

Use the exact local model trees and a fresh caller-selected external directory.
No tool has an absolute output default. The principal entry points are:

```sh
python3 -I tests/ns01_streaming_opening_test.py
python3 -I tests/ns01_streaming_opening_live.py --self-check
# Live: tests/ns01_streaming_opening_live.py with an explicitly idle GPU 7,
# frozen preregistration, exact binaries/models, and a new --out-dir.
python3 -I tests/ns01_streaming_opening_scorecard.py \
  --bundle <external-bundle> --preregistration <preregistration.json> \
  --candidate <candidate.json> --negative-root <external-bundle/negative> \
  --controls <controls.json> --out <new-scorecard.json>
python3 -I tests/ns01_streaming_opening_verify.py <external-bundle> --write-manifest
python3 -I tests/ns01_streaming_opening_verify.py <external-bundle> \
  --out <external-independent-report.json>
```

Raw SSE, MessagePack, media CAS, process logs, and per-event captures remain
external only. The tracked compact scorecard retains per-leg outcomes and raw
hashes, not large media or per-event blobs.
