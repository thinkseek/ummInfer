# NS-05 duration-aware video and voice co-creation — Task #88

## Decision

**NOT_PROMOTED; final default runtime candidate REMOVED.** The single bounded
candidate was an engine-observed duration rule: only when the client omitted
`frame_selection.max_frames`, a uniform source of at least 60 seconds and at
least 16 declared frames resolved to the existing exact-source uniform-16 path.
Explicit 4/8/16 budgets, temporal selection, motion thresholds, prompts,
sampling, public schemas, and model weights were unchanged.

Candidate execution was `feb8e9721d35ee9b44eb4d768e6fb2898a6d5988`.
Commit `038842b443cf5fdb3014d7a29a8478456ea84a82` reverts it from final
runtime. The accepted default-eight decoder is retained. No second candidate,
prompt correction, scoring alias, training, optimizer, or selector-threshold
change followed the observations.

The paired shot-change family improved reproducibly from **0/16 to 16/16** and
the phone family did not regress (**0/16 to 0/16**), but candidate first-voice
p95 exceeded 3 seconds in both families. This is a hard NOT_PROMOTED result,
not a promoted quality improvement and not an NS-05 score increase.

The separate real voice co-creation gate is **NOT_MET in both arms**. Before
and After each autonomously emitted video `slot-0` with `source=slot-0`; the
existing fail-closed reference validator rejected that self-reference. Neither
arm produced a FLUX image or Wan I2V. No direct-render fallback, manual
directive, synthetic artifact, or post-observation prompt edit was used.

## Frozen protocol and source

[Preregistration](../benchmarks/ns05-video-voice-coc/preregistration.json) fixes
the source, candidate, populations, scoring, voice bytes, journey, and gates.
The media is the Task #75 continuous 30–150 second CC-BY-4.0 derivative:

- 120.000 seconds, 3,600 frames, 768×432, 19,821,335 bytes;
- derivative SHA-256 `ffef335edc161a2f6dead512e0fde96cf48fb997ae6b6491067e36c76eec58d0`;
- upstream SHA-256 `fd24e652c92a5759127aa9c7f23fea63d73aebb4bcea72763abe72ef06a20cf0`;
- license SHA-256 `e7edaaaeb0cd6da6a3dd97c54f3e8443ef027be685253fab64bf6eed2419ee7e`.

Four fresh stacks ran baseline A → candidate B → candidate C → baseline D on
physical GPU 0. Each ran two frozen cases with one warmup plus eight measured
repetitions: 72/72 requests and 64 measured answers. Every request omitted
`frame_selection`, used greedy seed 510075, and logged `prefix_reused=0`.
Baseline execution was official main
`e084d8404b1d96bee5cd6f83d8299c2498a3af65`; all four used the same gateway,
ASR, TTS, model files, and sampling. All 36 same-implementation fresh-stack
pairs had exact answer, native-semantic, and PCM identity.

The candidate reused the existing uniform-16 decoder: exact source RGB and
PTS, first source frame 0 and last source frame 3599, temporal-2 pair-first
timestamps, and the existing mRoPE path. It did not touch the temporal selector
or synthesize geometry. Final runtime has no native/src/include diff from the
accepted baseline.

## Paired latency and comprehension

Each cell has 16 measured requests. Values are median / nearest-rank p95 in
milliseconds; no outliers were removed.

| Implementation | Case | Metric | Median | p95 |
| --- | --- | --- | ---: | ---: |
| baseline | phone-action | decode+prefill | 1250.960 | 1369.893 |
| baseline | phone-action | first public text | 1897.311 | 1975.907 |
| baseline | phone-action | first public voice | 2117.408 | 2214.344 |
| baseline | phone-action | public total | 2875.767 | 3143.631 |
| candidate | phone-action | decode+prefill | 2230.809 | 2314.994 |
| candidate | phone-action | first public text | 2839.082 | 2973.610 |
| candidate | phone-action | first public voice | **3073.484** | **3214.377** |
| candidate | phone-action | public total | 4169.186 | 4366.522 |
| baseline | driver-scene-change | decode+prefill | 1239.935 | 1368.859 |
| baseline | driver-scene-change | first public text | 1890.659 | 1969.050 |
| baseline | driver-scene-change | first public voice | 2109.798 | 2210.662 |
| baseline | driver-scene-change | public total | 3518.868 | 3981.240 |
| candidate | driver-scene-change | decode+prefill | 2236.098 | 2277.002 |
| candidate | driver-scene-change | first public text | 2820.283 | 2941.039 |
| candidate | driver-scene-change | first public voice | **3063.103** | **3178.574** |
| candidate | driver-scene-change | public total | 3774.433 | 3999.306 |

First-text p95 passes in every cell. Candidate first-voice p95 fails at
**3214.377 ms** and **3178.574 ms**. Decode/prefill rose about 78–80% and first
voice about 45%; this explains the latency rejection and was not hidden by the
quality gain.

| Family | Baseline | Candidate | Frozen raw answers and scoring |
| --- | ---: | ---: | --- |
| phone-action | 0/16 | 0/16 | Baseline `He holds it to his ear to talk.`; candidate `He holds it up to his ear to make a call.` Both omit the required literal `phone` token. |
| driver-scene-change | 0/16 | **16/16** | Baseline `The driver stops talking on the phone and looks at the camera.`; candidate `The driver changes from a man to a woman.` |

The phone outcome is a frozen-scoring miss even though the action is
recognizable. No alias was added. The scene result is a repeatable temporal-
coverage improvement, but cannot override the first-response hard gate.

Raw answer/PCM identities, all metrics, and threshold outcomes are in the
compact [scorecard](evidence/ns05-duration-budget-voice-cocreation-sm90.json).

## Real voice co-creation: fail-closed evidence

Three unmodified native-TTS recordings were resampled once to PCM16 mono 16 kHz
and then actually transcribed by native Qwen3-ASR. Their 16 kHz SHA-256 values
are phone `ade900f915247d1bf6334f8dd72b021b7470b872c07f10c4b6907ae41b91ee5d`,
scene `3ce35b3522c1c965bb45db743a99da6ee4957ceddcfb4b264309617ac611b0df`, and
cyberpunk `f42b91a373bda81965c4759b63947ac6240de75ee02ae36af2f92ef28327c20d`.
No transcript was injected; the same source video bytes accompanied every turn.

The scene answer was interrupted, the initial cyberpunk request was cancelled
at its autonomous visual slot, and the identical body was immediately retried
in the same session. All four public response identities were distinct, old and
initial responses were cancelled, the retry had one complete terminal, no
stale/cross-session events were observed, and final `in_flight=0`. Exact native
TTS → public PCM → terminal-WAV joins passed independently in both arms.

| Measure | Before | After |
| --- | ---: | ---: |
| Phone first public voice (ms) | 2647.391 | 3496.251 |
| Retry first public voice from retry start (ms) | 4967.338 | 5816.669 |
| Full barge → retry first public audio (ms) | 9972.694 | 11698.035 |
| FLUX image artifacts | 0 | 0 |
| Native Wan I2V artifacts / submits | 0 / 0 | 0 / 0 |

Both retries failed at the model's self-reference, reported exactly as
`source slot slot-0 is not an earlier slot in this request`. The co-creation
loop is not claimed complete. Full-path barge timings are diagnostic and do
not change the accepted NS-04 result. Human perceptual, physical acoustic, and
full production gates remain NOT_RUN.

## Verification and retained corrections

- Native red/green: the unmodified decoder failed the new 64-second default-16
  test in three checks; the candidate passed it, including explicit-8
  preservation and byte equality with explicit-16. The final revert restores
  the accepted baseline and its tests.
- `umm-test-local`: PASS on the candidate and again on final runtime; final log
  SHA-256 `6cbde95cb564a2ba1915ac9da2abb57d329150ba9ca7770e4682a42c198808fb`.
- CUDA 12.4.131 / SM90 native Qwen build: PASS; Qwen host and video-input CTest
  **2/2 PASS**.
- Task #72 final-runtime controls: **320/320 requests, 304/304 parity**, all
  165 source frames independently redecoded, default/uniform/temporal-8 42/48,
  both four-frame arms 48/48, all 18 historical misses and zero outliers.
  Against accepted baseline records there are zero answer mismatches and zero
  mismatches across the 24 semantic native parity fields. Report SHA-256
  `cd74523c59d9c1d17f1fa10e34e27830a2d007bc57fba7ded095d811cb685d0e`;
  manifest `caf96b1afb79e2dd56c28d166775636e1cc58059722cb6057e6eaf5bca492d04`.
- Frozen Task #81 Chinese lifecycle on final runtime: **43/43 natural EOS**, 43
  native terminals, 537 exact native/public/terminal PCM chunks and
  14,987,520 bytes. Clean text SHA-256 is
  `883802d889885bd032b18a3dc21c8f7e1ce77773f020ab0ee3e69a816ac8b572`;
  aggregate PCM is `2ce2c53454349fd70edceab97e6afe6d93cf8ffc22e7bcdf497c303222d94f78`.
  Two fresh Task #81-specific raw verifications passed; receipt
  `fe8f1c03a5aa614bf68a5152ee99d8201d8d940b44b0228c1be51b906fb9de4b`.
  The generic narrative postprocessor again reached its documented no-video
  `StopIteration`; it is not presented as a serving-model failure or used for
  this control.
- Public narrative FLUX → explicit native I2V: **CONTROLLED_LIVE_PASS** with
  independent verification. Source PNG is 512×512 / 301,844 bytes /
  `01af1c29f95aa2362b57067bd6592a032617b8c46d91509e7b946d7b0d4e150b`;
  H.264 I2V is 704×480, 29 frames at 16 fps / 140,397 bytes /
  `49c6ae4e5460a17a387fb5e31f821a0739cf5e789a21419fc125a63194110b60`.
  First-frame/motion MAE is 2.259115/5.124479; report
  `9bfd7032a740c108df31d6a6c93b1a285b755a0365a2c76f0006a8fa2dd9f319`,
  verifier `c5326b329108e2995a67bbc1c35edb4e550a1dfa56dd20e7fcdf486c24b1024a`.
- Public HTTP + delayed realtime PCM: **25 HTTP responses (24 complete plus one
  exact cancel), 29/29 realtime responses and final ACKs**, 493 exact native/
  public legs, 4,176 chunks and 115,034,880 bytes. The release was exactly
  4,125 native audio events minus 29 public events = 4,096 pending, with all
  29 readers paused and zero public reads during the pause. Verification
  `389104a350b56757ef0dec2d66c172d2474bd8021d3a506f3e59b23120fdc4e8`;
  1,251-file evidence digest
  `7810763fecddac3aaef2fc5313a25ef7f82cb5a2a620a58967c63273b13a6f57`.
  The earlier wrong-fixture attempt produced only 406 native events and timed
  out; it remains negative evidence and is not counted as a passing control.
- Realtime network/reconnect: six sessions, three rounds, 15 connections,
  41 connection-scoped identities, 30 exact next-turn prompts and 32 terminal
  callbacks; zero stale/cross-session/decreasing-cursor events and final
  `in_flight=0`. Independent receipt
  `2becfebbe0416382438d050af349107bcad0becc0b9d546132ec2e07e99ce4b8`;
  manifest `e06748b5cab490b9badc9d8d14cd2be09bd0a5929b27b2dc6e9911aac77439bc`.
- The co-creation harness initially consumed SSE in an HTTP-200 assertion
  message and later rejected both real self-reference outputs. The harness
  transport correction did not alter any model result. An independent verifier
  correction uses numeric tap ordinals rather than lexicographic TTS filenames.
- Task #72 verification added only semantic normalization for FFprobe `K_`
  versus `K__` flag padding and an explicit GPU-0 execution override. Frozen
  media, packet PTS/DTS, RGB, scores, questions, answers, and native evidence
  were not changed.

## External evidence and identities

Two fresh independent paired verifications were byte-identical and the complete
self-excluded manifest passed. Analysis SHA-256 is
`f3d0d3d02d44c40d2ea425928578aeb20ce61fcd4eb6f3d5f9bf6e718c4cffab`,
independent receipt `e6d2d14894cfc1d83d6c2fe95718eaec3bf3b3da287561a878e36d057b6d2b08`,
and paired manifest `9fa3ce05c9bda987929cb9a0d3eea71be8608df3b390a6cb56192bca47e7fb66`.

The fail-closed Before/After co-creation receipt is
`f440870444fd5712f153cca8a654e9477af6cca20bb6dc54c5f1b955c3e2e5bd`;
its complete manifest is
`4e1ccbe5c9485e5219400c4c8c7544d1b91df8747664b1f2d250dd19e2b2ec48`.
Raw media, SSE, MessagePack, PCM, logs, negative attempts, and complete
self-excluded manifests remain external. Tracked evidence contains compact
hashes/results only. The compact scorecard SHA-256 is
`fd24bd1d6449dd32e87268e8d458d0e139f8c3f586b0a64b505b30953eed4164`.
The complete external root has 6,828 files / 1,790,619,793 bytes and passed two
fresh self-excluded manifest checks; root `SHA256SUMS` is
`8d8d3d687e44a0ab99d89615f1cf55353e37fafc771d6e01b3b88f95543fbac6`
and its verification receipt is
`2ca5b9f0dc132eaca76698fcdfba5f3261014eb7ff960bfee346e455a0c16e2b`.

| Object | SHA-256 |
| --- | --- |
| Shared gateway | `275fc42ef78e0dde9ce928d2092c76d2f28c62ed1d566a8b883425e04ac3a698` |
| Baseline VLM | `686ff04d28c41eb7f34ffe3b80bffa73941f9776a9412df002a5b6a6bcef7500` |
| Candidate VLM | `31fb30f216e3c900b49d1ca54121600688aa1404e0d17fba45b6121592e5612c` |
| Shared ASR | `e936045361d1f37ffe22717974dab9705afbb7c1e11023c44b003267afa42f86` |
| Shared TTS | `92fd8e24ddfb3cf7a21748ade17db719c33b2d4633c4e1b580a94b7f5b500c91` |

Only explicitly idle physical GPU 0 was used for CUDA work. Task-owned
processes were stopped, final `in_flight=0`, and the resident service and GPUs
1–7 were not used or restarted.
