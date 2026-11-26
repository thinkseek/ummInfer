# Qwen3.8 native video temporal-selection live evidence

## Scope and evidence labels

This is the deferred real-model gate for the accepted native temporal video
implementation. The requested object `8f078d96c0b404bc2132b8165ff7862f9e38d603`
was not present in the source mirror, so integration used the reviewed final
branch content at `99d0fd425904e9bb16aa59196facf013a165afbb` on base
`585dc83d1ee0d7c347218b2a614bebbeb1bd828d`. Only native/server video selection,
its tests, request-keyed native diagnostics, and this evidence are in scope.
GRPO/training and Web/macOS client files were not changed.

`R1` below is an integration dry run on the pre-final working tree. It is real
Qwen3.8-27B output on NVIDIA Hopper SM90, not a contract mock. The serial R1
client was run before the harness added per-clip warmups; the batched R1 client
used one warmup per clip. R1 therefore supports functional and serial/batched
parity claims, not a clean cross-mode performance comparison. Final
committed-tree warm serial/batch measurements are recorded separately below
when available.

**Four clips are a limited fixture set. 4/4 expected-answer coverage in any arm
is not evidence of generalized video-quality improvement.** No such claim is
made. The single-word answers are also intentionally coarse and are retained
verbatim for review.

## Controlled request and native-input contract

Every case runs three arms:

1. `default`: `video_url.frame_selection` is omitted, preserving the historical
   request shape and deterministic uniform decoder;
2. `uniform`: `frame_selection={"policy":"uniform"}` is explicit;
3. `temporal`: `frame_selection={"policy":"temporal",
   "motion_threshold":0.02,"scene_threshold":0.25}`.

All arms otherwise use the same video bytes, prompt, `qwen3-native` model,
`temperature=0`, `top_p=1`, `seed=42`, `max_tokens=128`, SSE streaming, native
8-frame/64-candidate ceilings, 3584-premerge-patch budget, and thresholds.
`tests/video_temporal_live.py --self-check` checks that removing the explicit
selection object makes the three requests identical. Live mode requires real
opt-in, a loopback URL other than the resident port, a provenance file pinned
to physical GPU 0, native logs, source revision, and a new evidence directory.
Expected-answer misses affect only the recorded coverage, never the transport
gate or a generalized quality claim.

The native stage emits one `qwen38_video_evidence` JSON record per video
request. It carries the request id, exact temporal source frame indices and
PTS, honest nominal uniform positions/timestamps, selected/candidate counts,
temporal-2 pair-first timestamps, grid and token counts, media fingerprint,
MRoPE vector digest/next position, and serial/batched engine path. The harness
validates `grid_t=ceil(selected/2)`, pair-first PTS, premerge and merged token
counts, frame budgets, default/uniform native parity, and serial/batched native
parity. The MRoPE digest is non-cryptographic FNV-1a diagnostic evidence, not a
security hash. The long-video extension adds `selected_frame_content_fnv1a64`
for independent scaled-RGB source binding without changing default uniform
pixels or historical nominal timestamps. Its 449-frame fixtures, budgets,
seeded repetitions, and two-stack controlled contract are documented in
[video-temporal-long.md](video-temporal-long.md); no long live run has executed.

## Clip provenance and explicit expectations

Sources are the tracked real MP4 assets below. Derived fixtures use lossless
H.264 (`libx264 crf=0`), 768x432, 8 fps, 48 frames, no audio. The rise/drain
pair has byte-identical decoded first-frame RGB SHA-256
`9921cdc727d1b046e8e9878c9418fe72ef56a0746f46d0909678c4c3fb024859`.
The drain clip reverses only subsequent real source motion and retains that
exact first frame. The static control repeats frame zero. The scene-cut control
joins glass frames 0..23 to building frames 24..47 at 3.0 seconds.

| Fixture | SHA-256 | Derivation | Question expectation |
| --- | --- | --- | --- |
| `liquid-rise.mp4` | `673182f9f614af9e1fe3378632ffbd9bee07e832a3ae1a0c83f59769ca97bc2b` | storyboard seconds 3..9, forward | `rise` or `fill` |
| `liquid-drain.mp4` | `934a105a875a583e69ea6f42b6a0d6835d76f3cf295beae894f928e3eac437bd` | subsequent frames reversed; same first frame | `drain`, `fall`, or `lower` |
| `static.mp4` | `c8258255a6eff968c4ef6cce031c79a149e32490a226c3a9193df686cf2ead69` | first frame repeated 48 times | `static`, `unchanged`, `unchanging`, or `remain` |
| `scene-cut.mp4` | `4cc0e8c398b37789ce596457ea8e655932630f7b42789acf5be15e523b5a6dce` | glass then building at frame 24 | `yes` |

Source SHA-256:

- `docs/assets/umminfer-showcase-storyboard.mp4`:
  `16219101a974935db472572620222258d4f2115790ee7eff06207b0f00bc3ae1`;
- `docs/assets/umminfer-showcase-duplex.mp4`:
  `66e30e4a132e516086cb69890520afbc008656330c6662bdb1833a983857b5a8`.

The R1 fixture manifest SHA-256 was
`c0aad68a3e81668265eff5851c10ea6d2ef49cc64e97aeb0f9244572f41f31cf`.
The manifest records these derivations, source hashes, per-case expectations,
and expected first-frame RGB hashes. Raw reports also embed FFprobe metadata,
per-video/config/source/model hashes, complete SSE frames, and all native
records. Evidence stays outside tracked files and contains no host identity.

## Model and available-weight provenance

The same local `qwen3-native` Qwen3.8-27B checkpoint was used for every arm.
`config.json` SHA-256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`;
`model.safetensors.index.json` SHA-256 is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.
The complete 29-file model `SHA256SUMS` manifest has SHA-256
`4a18d624b21af2bf0c5912914b97cc9a167845049ed9951632bcfc4748ed7ad5`.
All 18 available weight shards were hashed (not just the index):

| Shard | SHA-256 |
| --- | --- |
| 01 | `ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c` |
| 02 | `06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb` |
| 03 | `2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968` |
| 04 | `511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a` |
| 05 | `635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701` |
| 06 | `0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0` |
| 07 | `80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892` |
| 08 | `7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79` |
| 09 | `af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8` |
| 10 | `163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9` |
| 11 | `5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8` |
| 12 | `a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55` |
| 13 | `06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a` |
| 14 | `4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b` |
| 15 | `69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22` |
| 16 | `73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe` |
| 17 | `beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8` |
| 18 | `1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06` |

## R1 selected frames, packing, and MRoPE

Uniform positions below are decoder-output indices with historical nominal
timestamps, not claimed source PTS. Temporal indices and PTS are exact source
mappings. Every case selected 8 frames into 4 temporal-2 slices, grid
`4x16x30`, 1920 premerge rows, 120 merged tokens per slice, and 480 video
tokens. Default and explicit uniform had identical media and MRoPE digests in
every case.

| Case | Uniform decoded/selected; nominal indices | Temporal decoded/selected; exact source indices | Temporal selected PTS seconds |
| --- | --- | --- | --- |
| liquid-rise | 8/8; `0,1,2,3,4,5,6,7` | 48/8; `0,7,13,20,27,34,40,47` | `0,0.875,1.625,2.5,3.375,4.25,5,5.875` |
| liquid-drain | 8/8; `0,1,2,3,4,5,6,7` | 48/8; `0,1,7,13,20,27,34,47` | `0,0.125,0.875,1.625,2.5,3.375,4.25,5.875` |
| static | 8/8; `0,1,2,3,4,5,6,7` | 48/8; `0,7,13,20,27,34,40,47` | `0,0.875,1.625,2.5,3.375,4.25,5,5.875` |
| scene-cut | 8/8; `0,1,2,3,4,5,6,7` | 48/8; `0,7,13,20,24,27,34,47` | `0,0.875,1.625,2.5,3,3.375,4.25,5.875` |

The scene boundary at source frame 24 is retained. Static input uses bounded
uniform fallback. Rise/drain and temporal variants keep the same native
packing path rather than becoming independent still images. Prompt counts
were 557/557/544/546 tokens respectively; MRoPE next positions were
137/137/124/126. Serial and batch-four native records matched across all 12
arms, including media hashes, grids, prompt/video token counts, pair-first
timestamps, and MRoPE digests.

## R1 answers, correctness, and exploratory latency

Answers are verbatim except that `Yes.` is shown without code formatting
changes. Every default/uniform answer pair was byte-identical. Every serial
and corresponding batch-four answer was also byte-identical. Expected-answer
coverage was 4/4 for each of default, explicit uniform, and temporal in both
modes. This is correctness on the four explicit expectations only.

Times are TTFT/total seconds. R1 serial had no per-clip warmup; R1 batch-four
had one warmup per clip and concurrency 4. They must not be used as a clean
serial-versus-batched speed ratio.

| Case / arm | Answer | R1 serial TTFT / total s | R1 batch-four TTFT / total s |
| --- | --- | ---: | ---: |
| rise / default | `rise` | 1.290 / 1.314 | 1.283 / 1.308 |
| rise / uniform | `rise` | 0.572 / 0.596 | 1.448 / 3.079 |
| rise / temporal | `rise` | 1.249 / 1.274 | 2.214 / 3.041 |
| drain / default | `drain` | 1.224 / 1.267 | 2.994 / 3.192 |
| drain / uniform | `drain` | 0.579 / 0.621 | 1.911 / 1.941 |
| drain / temporal | `drain` | 1.245 / 1.290 | 2.806 / 3.650 |
| static / default | `static` | 1.016 / 1.036 | 1.042 / 1.062 |
| static / uniform | `static` | 0.370 / 0.390 | 1.059 / 2.679 |
| static / temporal | `static` | 1.040 / 1.060 | 1.833 / 2.658 |
| scene-cut / default | `Yes.` | 1.162 / 1.203 | 2.549 / 3.541 |
| scene-cut / uniform | `Yes.` | 0.521 / 0.563 | 0.950 / 1.795 |
| scene-cut / temporal | `Yes.` | 1.189 / 1.230 | 1.736 / 1.784 |

R1 artifacts included gateway SHA-256
`e602bb32742d4ea2805a738fe61c42c4b150cd8d17af989e996a3d2a998418f7`,
native stage SHA-256
`81deb57ccb370ea3bf25218c4e10fb05f6e8437200917e514baab51dde00028e`,
serial harness SHA-256
`4106d5aa5f79a204c129abf57c38d81c0fecaf4fbf2880b3a622c0130b4f95ed`,
and warmup-enabled batch harness SHA-256
`ca4f9119a90c84462dd8a81539c8157b122063c25671dcad43abf50240567e3c`.
Both dry-run gates returned `gate_pass=true`; all isolated processes were
stopped and physical GPU 0 returned to 3 MiB. GPUs 1 and 7 were not used, and
GPUs 2..6 and the resident service were not contacted.

## Committed-tree runs and prefix-cache control

The first committed-tree candidate `f5f789f5f348684e8240cea5c62d6d190aed9e4b`
(tree `6a79aef2aac99a038be9f3ff11eb3ea7a5e76306`) passed the serial functional
gate, but its latency was rejected as a clean policy comparison. Native request
logs showed that explicit-uniform requests following an identical default
request reused 550/550/537/539 prefix tokens and skipped ViT, while default and
temporal performed full prefill. A per-clip warmup did not prevent this because
the measured default request itself populated the reusable snapshot for the
following uniform request. Those numbers are not presented as a policy speedup
or as clean serial/batch latency.

The follow-up harness commit `89299e806bdf2585f2203cc60caf41f8dc08d1cb` makes
this condition fail closed. Clean runs require both
`UMM_QWEN38_PREFIX_CACHE=0` in the native stage and
`qwen38_prefix_cache=disabled` in provenance, parse the existing native request
summary for every measured request, and require `prefix_reused=0`. The harness
also records native prompt/prefill/reuse/token counts, vision/prefill/decode and
elapsed times. Cache-enabled measurements require an explicit
`--allow-prefix-reuse` and are labelled not clean.

The clean run uses fresh IPC/evidence paths, one unmeasured default warmup per
clip in both modes, serial `--batch-size 1`, and batch-four `--batch-size 4`
with client concurrency 4. Its source/artifact hashes and measured results are
recorded below. No quality or performance result is extrapolated beyond these
four clips and this exact checkpoint/configuration.

### Clean R3 results

Clean serial used source `89299e806bdf2585f2203cc60caf41f8dc08d1cb` (tree
`8b2010a2132a584209a88a43481fa226a8f4f5d2`), gateway SHA-256
`fb7fb9522eeee461499fca8df7c8472f001e0fcfdc67cb3ed49966de828d7f58`, and
stage SHA-256 `9da101b4405474b072d92ccf085080d6129df9d3cdc6fa52f7cbf596058cc70c`.
The first batch attempt rejected interleaved native summaries as malformed
evidence. Atomic request logging fix
`bca1fb21fa5ced6f8a10c7cd8b1549c967aa0cbe` (tree
`1d79b9b4c17f013f78480a8ac0ad5d645c9881c3`) made each summary a single stderr
insertion; the fresh batch retry used stage SHA-256
`c8f053f7f0be34677e475a4fce8f2438c425782d361ba8e362932b7c53456cc1`.
Both used harness SHA-256
`e005ae55d9f218181532895dd8ae3ef3cedce036d0f22a245fec09595969c46e` and the
same fixture/model/config hashes above.

All 24 measured requests (12 serial + 12 batch-four) logged
`prefix_reused=0`. Both modes returned `clean_policy_latency=true`,
`native_evidence_complete=true`, and `gate_pass=true`; batch additionally
returned `serial_batched_comparison=true`. Default/uniform requests and answers
remained byte-compatible, all 12 serial/batch answer pairs and native
selection/packing/MRoPE records matched, and each arm covered 4/4 explicit
expectations. The selected frames and grids remained exactly those in the R1
table. These are single-run measurements, not a statistical performance study.

| Case / arm | Answer | Clean serial TTFT / total s | Clean batch-four TTFT / total s |
| --- | --- | ---: | ---: |
| rise / default | `rise` | 1.074 / 1.096 | 1.126 / 1.149 |
| rise / uniform | `rise` | 1.074 / 1.096 | 1.781 / 3.134 |
| rise / temporal | `rise` | 1.106 / 1.129 | 3.069 / 3.097 |
| drain / default | `drain` | 1.066 / 1.108 | 2.398 / 3.754 |
| drain / uniform | `drain` | 1.067 / 1.109 | 2.616 / 3.971 |
| drain / temporal | `drain` | 1.096 / 1.140 | 1.959 / 2.670 |
| static / default | `static` | 0.871 / 0.891 | 1.303 / 1.982 |
| static / uniform | `static` | 0.867 / 0.887 | 1.974 / 1.997 |
| static / temporal | `static` | 0.899 / 0.919 | 1.322 / 1.995 |
| scene-cut / default | `Yes.` | 1.013 / 1.054 | 1.961 / 3.309 |
| scene-cut / uniform | `Yes.` | 1.022 / 1.063 | 1.961 / 2.658 |
| scene-cut / temporal | `Yes.` | 1.046 / 1.087 | 2.599 / 2.648 |

Batch times include real concurrency-4 queueing/batching and are not compared
as speed ratios to serial. Temporal's small serial TTFT deltas (about 24–33 ms
on these runs) are descriptive only. All isolated Qwen/gateway/entry processes
were stopped after the run; physical GPU 0 returned to 3 MiB. GPUs 1 and 7 were
not used, and GPUs 2..6 and the resident service were not contacted.
