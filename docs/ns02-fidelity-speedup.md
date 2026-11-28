# NS-02 fidelity-preserving Wan2.2 interactive-video speedup (Task #86, 2026-11-28)

## Result and promotion boundary

**CONTROLLED_LIVE / NOT_PROMOTED.** On one explicitly idle NVIDIA H20 SM90,
physical GPU 7, the frozen automatic boundary2 TeaCache candidate reduced every
measured wide/tall/square public complete-artifact latency below 5 seconds and
passed cancellation, malformed-input, recovery, terminal, and reproducibility
controls. It nevertheless failed both required fidelity controls: none of the
15 pairs had exact decoded first-frame RGB parity, and tall SSIM was
`0.896092 < 0.90`.

The candidate is therefore absent from the final default runtime. Final runtime
commit `c804a1d0d5dc7f11e6d3bd7faf9f216cfd513380` has zero `native/` or `src/`
diff from official main `b23f6fd1f2b3c50165ddcc5cac86dc5aa7723dc4`.
Human/perceptual approval is **NOT_RUN**; numeric SSIM and RGB controls are not a
full NS-02 perceptual PASS. No deployment, training, optimizer, or resident
service validation is claimed.

The compact independently verified scorecard is
[ns02-fidelity-speedup-sm90.json](evidence/ns02-fidelity-speedup-sm90.json).
Raw media, RGB, MessagePack frames, process logs, manifests, and verification
outputs remain external only.

## Frozen candidate and protocol

Candidate commit `b4c92ac1e967aa3b661d42059cff281060c73398` selected one bounded
policy: when both the request and native Stage CLI omit `teacache_thresh`,
native I2V uses TeaCache `0.25` with full 30-layer DiT computation for both CFG
branches at UniPC steps **0, 1, 18, and 19**. Only interior steps 2–17 may reuse
a residual. Explicit request/Stage values, including `0.0` and explicit `0.25`,
retain their historical behavior; T2V and the legacy direct-engine constructor
remain uncached.

The candidate did not change CUDA arithmetic, solver/update math, precision,
resolution, frame count, sampling, codec settings, public schema, or lifecycle
behavior, and added no Fast/Quality control. It is distinct from the rejected
Task #79 automatic TeaCache 0.25 candidate, whose documentation, failed evidence,
and preregistration remain unchanged as diagnostic context.

The frozen protocol is
[preregistration.json](../benchmarks/ns02-fidelity-speedup/preregistration.json).
It binds baseline profile report
`73600dc655ecdd938ba61a20177c5db3cd58d3b557eaaa74fa22f0657b803e1e`, the exact
real FLUX.2 sources, seed 42, 29 frames, 20 UniPC steps, 16 fps, guidance 5.0,
five process-separated warm pairs per geometry, one excluded warmup per shape in
each fresh arm block, and seeded arm order `79025`. Both public arms omit
`teacache_thresh`. Every observation and outlier remains in all statistics.
Unlike the historical pooled-only diagnostic, this task additionally requires
**every individual measured artifact in every geometry** to be below 5,000 ms.

### Retained R1 tooling correction

R1 stopped on its first candidate warmup before any accepted measured candidate
pair or report. The tooling incorrectly required exactly 28 cache hits; four
protected steps permit at most 32, and a legitimate threshold-driven interior
recompute reduces the count. The observed candidate log reported 30/40 and
`teacache_boundary2=1`. R2 changed only this diagnostic expectation to
`0 < hits <= 32`; the candidate, threshold, protected steps, inputs, population,
and every promotion gate stayed unchanged. R2 subsequently observed 30/40 in
all measured candidate runs.

The excluded R1 bundle contains 456 regular files plus a self-excluded
`SHA256SUMS`; its manifest SHA-256 is
`053a26027b01c23f28e582c45c00866bb9a0cdc0547e523bb2d37e4dd90d1c42` and its
retained failure-log SHA-256 is
`50d3b536e693f97b48ef1bb669d4ec2b7043b2ebb812495f0ff2e0a54dbf65c8`.

## NS-02 paired public latency

The primary metric is median paired candidate-minus-baseline public
complete-artifact latency. Public playable and complete-body timings are both
retained in the raw report; the table uses complete artifacts. All 15 paired
deltas were negative. The median paired change was **−6,587.153505 ms
(−58.882592%)**.

| Geometry | Before median / p95 / max (ms) | After median / p95 / max (ms) | Median paired delta (ms) | Every After <5 s |
| --- | ---: | ---: | ---: | --- |
| Wide 704×480 | 11,161.239554 / 11,280.574050 / 11,295.174771 | 4,583.901644 / 4,624.625932 / 4,629.326626 | −6,571.688505 | PASS |
| Tall 480×704 | 11,207.533852 / 11,217.499764 / 11,218.466746 | 4,606.393572 / 4,626.542222 / 4,627.650586 | −6,590.816160 | PASS |
| Square 512×512 → 704×480 center-cover | 11,158.020200 / 11,284.674788 / 11,309.111324 | 4,581.172696 / 4,610.788386 / 4,615.222749 | −6,583.775628 | PASS |
| **Pooled** | **11,186.928644 / 11,299.355737 / 11,309.111324** | **4,583.901644 / 4,628.153398 / 4,629.326626** | **−6,587.153505** | **PASS** |

No After IQR outlier occurred. Retained Before outliers include tall
`11,136.117632 ms` and square `11,309.111324 ms`; retained paired-delta outliers
include `−6,711.273127` and `−6,716.060388 ms`. The scorecard contains every raw
value and all per-shape outlier sets without discarding them.

Native DiT medians fell from 8,837.89 / 8,835.37 / 8,848.55 ms to
2,262.98 / 2,266.54 / 2,265.68 ms for wide/tall/square. VAE decode remained
about 2.0 s and was not a claimed optimization. The separate fresh default-path
profile had nine warm public measurements at 11,163.921352 ms median and
11,184.832192 ms p95.

## Fidelity rejection

Each geometry produced the same fixed-seed MP4 within an arm, so all five pairs
per shape have the same numeric fidelity values.

| Geometry | Exact paired first-frame RGB | SSIM | All-frame PSNR (dB) | Temporal-motion ratio | Quality gate |
| --- | --- | ---: | ---: | ---: | --- |
| Wide | **FAIL 5/5 pairs** | 0.925993 | 28.908439 | 0.908676 | **FAIL: exact RGB** |
| Tall | **FAIL 5/5 pairs** | **0.896092** | 27.003652 | 0.532097 | **FAIL: exact RGB and SSIM <0.90** |
| Square | **FAIL 5/5 pairs** | 0.960630 | 28.808751 | 1.143593 | **FAIL: exact RGB** |

Source-first-frame continuity proxies passed (candidate MAE 1.956124 /
1.959975 / 1.893939), as did nonzero motion (MAE 7.729167 / 7.231882 /
6.209596), PSNR ≥24 dB, and motion ratios within [0.5, 1.5]. These controls do
not repair the 15 exact-parity failures and five tall SSIM failures. No gate was
relaxed and no successive threshold was tested.

All successful videos independently decoded as H.264/yuv420p with exact target
width, height, 29 frames, and 16 fps. Exact source bytes reached native Wan;
wide/tall/square source SHA-256 values remained respectively
`477cfafd…3451`, `390465a6…ce27`, and `c46c343f…d1da`.

## Cancellation, malformed input, and recovery

One real non-streaming `/v1/visual/render` disconnect per arm occurred one second
after observed native submit.

| Arm | Disconnect → native abort terminal | Disconnect → client/health settled | Late native records / public media | Final `in_flight` |
| --- | ---: | ---: | ---: | ---: |
| Before | 0.571871 ms | 1.539011 ms | 0 / 0 | 0 |
| After | **0.544237 ms** | **1.494911 ms** | **0 / 0** | **0** |

Both arms pass <500 ms. This route exposes no request id, so `/abort` ACK remains
`NOT_APPLICABLE`, not a fabricated measurement. Native terminal dedup passed.
Malformed 500×480 input failed with HTTP 500 and no media in each arm. Immediate
post-cancel retry and post-malformed recovery returned each arm's exact wide MP4
hash: Before `6e487e7f…e934`, After `2f86c991…d014`.

## Independent verification and raw hashes

The accepted R2 bundle contains **1,566 regular files** and a recursive
self-excluded top-level `SHA256SUMS`; nested baseline/evidence manifests remain
covered. A fresh stdlib-first process rehashed every entry and all supplied
binaries/model files, independently decoded MessagePack and media, reconstructed
all raw pairs/statistics/outliers, recomputed SSIM/PSNR/motion, checked every
journal and source/artifact binding, and recursively recomputed the promotion
decision. A second fresh-process verification was byte-identical to the first.
This is **evidence-integrity PASS**, not candidate promotion.

| Record | SHA-256 |
| --- | --- |
| Compact scorecard | `a299516c6380a9e549a1d6f3b1e330c002f11abcadb6e1fd487571ebb2c9b6b4` |
| R2 report | `dcde6aa167d21bc2bceb74a66e936b366aac177db2cfaecf90c603e36e4e9bb9` |
| R2 manifest, 1,566 files | `e49aa5dfc502f67439d0f511786b2f250e4e6d299499ed86b615d2546d83819a` |
| R2 first and second independent verification | `39eddeef69c8e33af39290da316d77cb41fcdbd0c762b1761f4e4f992584a444` |
| Frozen R2 preregistration | `e57c55a0981b0a4d46c83a15e6268dc59bdd6240db4544259eda84be278eeefd` |
| Baseline profile report | `73600dc655ecdd938ba61a20177c5db3cd58d3b557eaaa74fa22f0657b803e1e` |
| Baseline profile manifest | `9db5b663a6f0b9655467aeaa53f976b388baf9b4e6c192e0d2109376d213c89a` |

Execution identities include server
`c43934ce6f9ffc4c4218f0930c2038a551737f04f643b2de2c4d107abd98460f`, baseline
Wan `15ed9fd41de4ec635fe0a87b4d3ecd17d2cbd5330934e939d1479b2d28b8adb3`, and
candidate Wan `b8c17a8293f0ec65781580d94be16d820d298254616b567847b166ad47ed3ab1`.
The Wan model tree remains
`b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616`; model
revision is `UNKNOWN`, with exact file/tree hashes providing identity.

## Tests and cleanup

- Candidate and final-runtime `umm-test-local`: PASS.
- Deterministic NS-02 protocol/evidence controls: 13/13 PASS; compact scorecard
  controls: 3/3 PASS. The boundary schedule's pre-change red control failed on
  the absent policy; the candidate host implementation passed green.
- Native Wan host CTest: 1/1 PASS. Public contract-stub completed, cancelled,
  failed, and recovery controls passed with final `in_flight=0`.
- Narrative source-link self-check: PASS; raw-SSE replay: 11/11 PASS; storyboard
  replay verification: 10/10 PASS.
- Final SM90 Stage CTest under `CUDA_VISIBLE_DEVICES=7`: 12/13 applicable PASS.
  `stage_common_test` fails `an unclosed marker remains exact clean prose` on
  both the untouched base and candidate/final trees, so it is retained as an
  unrelated pre-existing failure. The image2 allocator test requires two visible
  devices and is NOT_RUN under the GPU-7-only constraint.

Only task-owned processes were stopped. GPU 7 returned to 3 MiB with zero compute
processes; residual utilization was rechecked to 0%. GPUs 0/1 were not used, and
the resident service/GPUs 2–6 were never contacted, restarted, or deployed to.

## Reproduction

Use exact local assets, an explicitly idle permitted GPU, and fresh
caller-selected evidence directories. Raw outputs have no tracked absolute path
or default output directory. Build baseline from `b23f6fd1` and the historical
candidate from `b4c92ac1`; run the retained harness from `c21aaa4` or newer with
`tests/i2v_geometry_latency_live.py --mode ns02`, the frozen v2 preregistration,
and the hash-bound baseline profile. Independent verification is
`tests/i2v_geometry_latency_verify.py`; compact scorecard generation is:

```sh
python3 -I tests/ns02_fidelity_speedup_scorecard.py \
  --evidence "<sealed-r2-evidence>" \
  --verification "<fresh-independent-verification.json>" \
  --out "<new-scorecard.json>"
```

The final default runtime intentionally contains no boundary2 candidate. No
additional candidate or GPU measurement was launched after non-promotion.
