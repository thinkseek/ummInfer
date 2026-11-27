# Post-determinism two-GPU R4 full-duplex gate (Task #73)

## Result and evidence classes

The corrected full R4 gate and the separately preregistered first-audio
candidate are **PASS** on the frozen digital-loopback workload. Serial native TTS
now fairly alternates first-audio legs with continuation/ordinary work. Two
independent cross-policy comparisons reduce concurrent-cold-short median
ASR-final → first public audio by **46.161%** and **45.921%**, with exact
transcript, text, prompt, cursor, and PCM identity. Same-policy A/A controls also
have zero identity mismatches.

This is **REAL_DIGITAL_LOOPBACK_NETWORK_AND_NATIVE_MODELS** evidence only.
Physical microphone/speaker playback, WAN, deployment, perceptual/acoustic
listening, generalized workloads, and a genuinely long real opening remain
**NOT_COVERED**. The alternate `en_16.wav` arm still produces the short opening
`Hello!`; its name is not evidence of long-opening coverage.

## Integration and preregistration

- Accepted Task #70 tree:
  `6e1d38b27f00dd3750e92989676e133d1cf3eb0d`.
- Official main `8e6ccfa1b6a0d888cf8e2bbb08e53efb9443ab2e`
  merged as `813f73cd1af9dacf669cf65231d5bdcf859a4e63`, without dropping the
  Task #59 native continuation fix or Task #70 transport-validity fix.
- ACK-anchored barge-in instrumentation: `a2f0335d48ccc09b8cdb4634d014f5833a002564`.
- New preregistration commit: `fea9a2b20f54ecaa04dc7a3f0b04b44a700228bc`;
  frozen [contract](../benchmarks/realtime-first-audio/post-determinism-preregistration.json)
  SHA-256 `333c836f998aa51d84b91ccaf80d9c5bd076814f2eceb8521727361eff07b355`.
- Execution source and candidate: `c63d4f3fdb07729162744cb58fa4fceaed9881db`.
  Later documentation/AOCI commits do not change runtime or analyzer bytes.

The fresh profile preceded the candidate. Its report/manifest SHA-256 values are
`57eabe226a44cdd4dfb06d8e00e61a4b785fded6e2eca5cb725d53be48190ba1` and
`887d17b3c6c035d6f7a17f5758346ce86aa3bf301766c2ab3339e108e3d8ec04`.
Concurrent-cold-short median TTFA was 18,067.374 ms; TTS submit → first native
PCM accounted for 13,598.932 ms, native first-PCM compute 71.878 ms, and native
PCM → public audio 5.653 ms. This reproduced continuation-first admission as the
large residual. The profile used the old contract and is not mixed into the new
A/B denominators.

The previously rejected Task #55 fair-priority candidate remains historical and
its latency numbers are not reused. This new experiment was preregistered only
after the Task #59 determinism correction and the fresh profile. Its no-fix path
still requires reversion on any correctness, exact-identity, or latency
guardrail failure.

## Bounded candidate

`native/stage_cli/main.cpp` installs an internal predicate only for serial native
TTS. Streamed `:tts:0` and one-shot `:tts` form the first-audio class; later
`:tts:N` and ordinary work remain in the other class. Each class is FIFO. When
both are queued, `MessageQueue` alternates them; already-running compute is never
preempted. Empty hooks preserve FIFO, and batched/concurrent schedulers reject
the hook.

No public request field, response deadline, queue limit, sampling default,
cancellation policy, PCM chunking rule, native model semantic, or UTF-8 cursor
rule changed. `native/stage_cli/first_audio_scheduler_test.cpp` failed to compile
before the implementation because `set_request_priority_fn` was absent, then
passed after it. It covers fair alternation, class FIFO, abort-before-submit,
default FIFO, and batched rejection.

## Frozen execution

SplitMix64 ordering seed `0x73059` selected the interleaved run order
`fair-r1 → fifo-r1 → fair-r2 → fifo-r2`. Every run executes all seven arms, all
81 measured responses, 96 total server responses, warmups, delayed barge-in,
echo-only protected/disabled A/B, and both double-talk controls. No slow or
failed response is omitted. The FIFO control differs only by disabling the TTS
predicate with `if (false && kind == "tts")`; its one-line diff SHA-256 is
`f243119b0fbb4687800289011915c1287f0f17f73a5ac04366712dd0e82bf268`.

Physical GPU 0 runs Qwen3-ASR-1.7B; physical GPU 1 runs Qwen3.8-27B and native
Qwen3-TTS-12Hz-1.7B-CustomVoice. GPU 7 and the resident service/devices are not
used. Model manifest SHA-256 values are respectively
`a16807fc65e05f36687484bdfb8cb9f103c3fb59f6787d1f904f3ffebf8a3005`,
`fb2a8751df1dfbcdf556988acd8e2ac7c6f7180c609797d5e9b3a8e3e40a1c7e`, and
`92db72616c9578462add19ed81720b7abf1c1e9c3f4d225877a1cd6104d2fb27`.
Model revisions remain `UNKNOWN`; exact files are hash-bound.

| Binary | Fair SHA-256 | FIFO-control SHA-256 |
| --- | --- | --- |
| `umm-infer` | `faf3314b02bb31b4098a9f8e651be79afdc2d2017fadf22022f45d0db305a532` | same |
| `umm-native-stage-asr` | `5428f0213f87b206cbfbed9b982c92deb00b22d29f5e4dd5886895805bfe025e` | same |
| `umm-native-stage-qwen38` | `ba84c91dbf9fd933ed33f563219b626de2cf92d1a8178e93e7a545274c20b328` | same |
| `umm-native-stage-tts` | `f0ff5a45b90ba155d4f27e474e53471cf0342ffc7fe3402341eded834e9678f4` | `de1f4e8f4a6e7de1c366b422215f497106f2a8daed56e77ac629af49b398a213` |

Analyzer and live-driver SHA-256 values are
`6e5249a93212703844680156fd2c8d3c26617b4022533f89cd3ac6191e0c2d73` and
`e489ae111db38e639c101d42b0c90d0a066b21d50b81339c82bb247ae9557ade`.

## Primary paired latency result

All values are milliseconds. Quantiles use linear interpolation at `(n-1)*p`;
CV is population standard deviation / mean. Complete samples and Tukey outliers
remain in each linked full report.

| Run | Policy | n | Median | p95 | IQR | MAD | CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | FIFO | 18 | 18,134.312 | 29,965.327 | 14,194.105 | 7,099.705 | 0.56407 |
| R1 | Fair | 18 | 9,763.313 | 11,907.207 | 3,828.604 | 1,844.217 | 0.46139 |
| R2 | FIFO | 18 | 18,097.438 | 29,868.541 | 14,134.970 | 7,070.171 | 0.56356 |
| R2 | Fair | 18 | 9,786.961 | 11,933.151 | 3,859.843 | 1,853.788 | 0.46129 |

| Pair | Median reduction | p95 reduction | Serial median change | Improved / worsened / equal primary pairs | Exact sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIFO R1 → fair R1 | 46.1611% | 60.2634% | +0.0457% | 17 / 1 / 0 | 0.0001449585 |
| FIFO R2 → fair R2 | 45.9207% | 60.0478% | −0.0110% | 16 / 2 / 0 | 0.0013122559 |

Median paired improvement is 7,987.227 ms (R1) and 8,315.106 ms (R2).
The R1 worsened pair is only −0.126 ms; R2 retains worsenings up to −9.683 ms.
These misses remain visible in the machine evidence rather than being rewritten
as uniform per-pair improvement.

The preregistered gates require at least 20% median improvement, no more than
5% primary p95 or serial median regression, and serial CV ≤ 0.10. Both pairs
pass. The largest serial-cold-short CV is 0.02115. All cold requests have
`prefix_reused=0`; warm measured reuse reaches 244 tokens in every run.

### Stage attribution and retained tradeoffs

| Primary metric | FIFO R1 median | Fair R1 median | FIFO R2 median | Fair R2 median |
| --- | ---: | ---: | ---: | ---: |
| TTS submit → first native PCM | 13,655.099 | 4,756.661 | 13,612.448 | 4,763.260 |
| Native compute → first PCM | 72.632 | 127.113 | 72.657 | 128.109 |
| First native PCM → public WS audio | 3.911 | 6.719 | 7.238 | 7.422 |

Fair admission starts first-audio work earlier, while VLM work remains active on
the shared GPU. Native first-PCM compute therefore rises rather than being
presented as free. The large queue-admission reduction still dominates TTFA.
No transport or native-kernel optimization is claimed.

### Every arm, both repetitions

Each cell is `FIFO median → fair median` in milliseconds.

| Arm | R1 | R2 |
| --- | ---: | ---: |
| Serial cold short | 381.499 → 381.674 | 381.650 → 381.608 |
| Serial warm short | 542.484 → 542.455 | 542.568 → 542.533 |
| Concurrent cold short | 18,134.312 → 9,763.313 | 18,097.438 → 9,786.961 |
| Concurrent warm short | 23,800.548 → 10,082.761 | 23,763.638 → 10,108.736 |
| Serial cold alternate | 361.610 → 351.562 | 361.517 → 361.536 |
| Concurrent cold alternate | 3,637.003 → 3,632.737 | 3,659.215 → 3,663.106 |
| Concurrent cold short + 2 s reader pause | 18,107.777 → 9,756.872 | 18,129.820 → 9,818.999 |

The alternate arms show no material generalized improvement. The warm and
backpressure gains apply only to this fixed topology and plan.

## Exact correctness and lossless playback evidence

Every run independently proves:

- 81 measured responses and 96 total server responses, with 96 completed and four
  deliberate cancellation callbacks across the regression block;
- 380 accepted playback ACKs, 853 ACK requests, and 473 intentionally ignored
  invalid/decreasing/foreign/unissued ACKs;
- exact native/public PCM for every measured per-TTS-leg chunk, exact stream /
  terminal identity, complete text and prompt history, UTF-8 cursors, and final
  ACKs;
- one terminal callback kind per response, no stale old-response audio/text/ACK,
  protected echo cut-ins `0`, and successful double-talk cancellation in both
  modes;
- cold prefix reuse `0`, warm reuse observed, `in_flight=0`, all owned processes
  exited, and both assigned GPUs at 3 MiB / 0% with no compute process.

Both cross-policy comparisons require all 81 transcript/text/prompt/cursor/PCM
pairs to match. Both same-policy comparisons independently match all 81 pairs.
The existing comparator's same-policy status remains `FAIL` because its unrelated
“at least 20% self-improvement” rule is not met; that status is retained verbatim
as `comparison_status_retained`, **not** relabeled as an A/B PASS. Only
`exact_quality_equivalence=true` and zero quality mismatches are consumed from
those A/A controls. The two actual A/B reports themselves are PASS.

### ACK-anchored barge-in timing

The delayed regression consumes an exact 61,440-byte PCM prefix at its 24 kHz
sample clock before the first valid partial ACK, then retains a further
1,186,560 emitted-but-unplayed bytes. On client receipt of `response.cancelled`,
the harness immediately clears that pending virtual-playback queue. Independent
verification re-decodes raw frames, recomputes the played PCM SHA-256
`af9c189bed0cb3721fb66fba8917cdf6f0ad3d8eed221e4a9c4c17542959935f`, and checks
the accepted cursor `10`, cancellation cursor, pending suffix, and all clocks.

| Run | First new-speech append → cancel, including VAD (ms) | Signed speech-started notification → cancel (ms) | Client cancel receipt → virtual silence (ms) |
| --- | ---: | ---: | ---: |
| Fair R1 | 42.903429 | −0.049876 | 0.091307 |
| FIFO R1 | 3.043577 | −0.033585 | 0.070320 |
| Fair R2 | 3.199809 | −0.056623 | 0.045882 |
| FIFO R2 | 42.547344 | −0.075291 | 0.061991 |

The production server emits `response.cancelled` before the public
`speech_started` notification, so the signed notification interval is negative
and is never clamped. First-append timing includes VAD detection and chunk
transport. Cancel-to-silence starts at the client's raw cancellation receipt,
not server emission time. These are digital timed virtual-playback results, not
physical DAC/speaker latency. No barge-in improvement is claimed from these four
diagnostic samples.

## Raw evidence and independent verification

Each raw tree has 235 files and a complete self-excluded `SHA256SUMS`. After the
initial report generation, two remote and two local fresh processes independently
reconstruct the identical report and validate the sealed manifest. Complete raw
PCM/frame/observer trees and full reports remain unmodified outside Git. The
tracked JSON is only a compact derivative, never a substitute for raw evidence.

| Run | External full-report SHA-256 | Self-excluded manifest SHA-256 |
| --- | --- | --- |
| Fair R1 | `5149c7e43ecc8aa02067d171a8266bd959a2f44a3aeb543485df3259585d93c9` | `f1ec699117170945f2037ae6535259e5f9fa27dd05bc9ac58f502fe784819e2b` |
| FIFO R1 | `a00efabd288cb1ded3e6560af0c7a7e0fbc2f03ab2aa1a340cdcd3ea98f6a17f` | `c238cf4b230642ca6500461ffa52c2af164a3c4c1486f27af3cb4eca490c48c8` |
| Fair R2 | `4a485bbd2a7c3b45374bde0c6a413a48db4ff6257910a276afb0ef3aa196e42b` | `35a7da113e7310caa3939f032ce99da1e3df087e3d543f9a325890ff745953eb` |
| FIFO R2 | `c1887b2de81336fbdc5cd5bd0f28e58ecd1aa0c1e37eaba35f9c76ecdc83530e` | `f144015691b4d79084a86c7a7c59b94d9b05790d98e1e58e6c83e3db568ee70b` |

Paired deltas and sign tests are included in the compact projection. The original
external A/B derivative hashes are
`8c938314b8fc7fce48ead8a9400a42a55ebc896cf65e62c5979839fdb4ae38f2` (R1) and
`c87d03f42074cbf679412442851b6195c798e7f38fa1c4ce1446313971a4dc14` (R2).
The [compact machine summary](evidence/realtime-r4-post-determinism-summary.json)
has SHA-256 `5be2d38c0ececf83e77aa46e34ca92a95b668d8899ab4f2be601f6782920c7c1`.

## Candidate network, reconnect, and public 29-session gates

The final fair binary also passed the existing real six-session/three-round
network harness: 15 connections, 30 actual ASR → VLM → native TTS responses, 30
exact native VLM prompts, 21 history turns, nine reconnects, and nine barge-in
cancellations. Independent raw verification reports 27 terminal callbacks, zero
stale events after cancellation, zero cross-session/connection leaks, zero
duplicate terminal callback kinds, zero invalid accepted cursors, and final
`in_flight=0`. The 41 connection-scoped response identities include 11 ACK-only
entries and are not misreported as 41 server responses.

The 44-file raw network manifest SHA-256 is
`f593a02150fc194b494bc99b1e906fd3fc2c354ac93f007fc402a92794fa20cd`; its
external independent report SHA-256 is
`ac971ecfb82ff81b1c85bfafd502da0c9f21406bab8d25d2d514cc0a420e7873`.
Across nine cancellations, first new-speech append → cancel (including VAD) has
median/p95 **2.996942 / 27.168510 ms**. Cancel → public speech-started
notification has median/p95 **0.060340 / 0.109708 ms**. Playback-ACK round trip
has median/p95 **0.041956 / 28.701745 ms**, with every outlier retained.
Final-old-audio → cancellation is separately retained and is not mislabeled as
playback silence. Physical playout remains NOT_COVERED.

The first follow-up audit incorrectly selected the pinned `task41-r3` profile and
rejected the new run's 1,019 audio frames against that historical run's 1,024.
The real network verifier had already passed. No analyzer assertion was changed
to bypass this; the documented `generic` profile was used for the new source.
An initial cross-host comparison then exposed three last-bit CPU-mean differences
from Python-version-dependent float summation. Commit
`e5bb1fff87ad4038dee0590c47742fcb3325417f` adds a focused red/green
`stable_mean` fix using `math.fsum`; all four R4 report hashes remain unchanged.
The final external stable network audit is byte-identical across both hosts at
`cdb94f01bf729477427231fde3f0fa43847f39062a03090b8b9b9ba6100ad01c`.

The final public HTTP and delayed-realtime capture independently passes:

- HTTP: 25 responses (24 completed + one deliberate cancellation), 44 PCM legs,
  184 chunks, 10,698,240 bytes including the valid one-chunk cancel prefix, and
  exact retry/repeat identity;
- direct native continuation control: four responses, each 737,280-byte PCM
  SHA-256 `3eaa2791bba77428558a314086e2264e3279123b03d89a4998f24a3cce11773d`;
- realtime: **29/29** completed responses with one terminal and final ACK each,
  493 exact native/public PCM legs, 4,176 chunks, and 115,034,880 bytes;
- delayed release: all 29 readers paused, 4,096 native audio events during pause,
  zero public reads, and exactly **4,125−29=4,096** pending audio events;
- cleanup: `in_flight=0`, all owned processes exited, GPU 0 at 3 MiB / 0% with no
  compute process.

Three remote and two isolated-local fresh public verifier processes produce the
same report SHA-256
`fa24b43d78d351bd209566b60a37ba2cba0b7504296d0dfe2f6c8b02a045023a`.
The 1,251-file self-excluded manifest SHA-256 is
`b1f2f34fd7cd0e0d865ddcd65b16677d143d773bf21bc9472946315d5beafff5`.
The local verifier used an isolated temporary `msgpack 1.2.4` environment after
the initial missing-dependency attempts produced no report. The full public
report contains task-local capture paths and remains in the sealed raw bundle;
its exact hash is retained without committing those paths.

[Compact follow-up machine summary](evidence/realtime-followup-post-determinism-summary.json)
SHA-256: `ffe8f93888eccf9df457825c0bdb19eef582a710a7cd571c27ed7110e14e5acf`.
HTTP PASS, R4/realtime parity, digital playback ACK timing, physical playback,
and perceptual quality remain separate claims.

## Deterministic checks and negative attempts

- Complete `umm-test-local`, ReleaseFast unified tests, 6 first-audio tests, 21
  public-PCM tests, 12 network tests, and 10 independent latency-audit fixtures:
  PASS.
- Native stage host CTest: 11/11 PASS; remote native TTS: 3/3 PASS, including the
  real CUDA duplicate-history determinism fixture.
- Realtime stress: 3,360 actions PASS, trace `0x4ce990317a0823eb`; dedicated
  realtime TSan and voice ASan/UBSan PASS. The actual scheduler regression also
  passes a focused ASan/UBSan+leak build.
- Full-source unified ASan/UBSan exceeded 600 seconds and was stopped:
  **NOT_RUN**, not a passing or failing product result.
- The short timing probe R1 refused launch when GPU 0 transiently reported 2%
  utilization after CUDA CTest, despite 3 MiB and no compute process. R2 launched
  only from explicit idleness and passed. Neither probe is counted as an R4
  measured sample.
- An initial remote build launcher used `sh` with unsupported `pipefail` and
  exited before compilation or model launch. The corrected `bash` build passed.
  No failed launcher or probe is counted as PCM evidence.
