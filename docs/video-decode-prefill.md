# Native Qwen3.8 natural-video decode/prefill — Task #75

## Revision 1 final disposition — NOT_PROMOTED

The direct NS-05 golden-case gate completed on a reviewed CC-BY-4.0 continuous
120-second multi-shot source. Seventy-two default-eight requests passed the <3 s
text/voice thresholds and exact answer/native/PCM parity, but both first-voice
bootstrap intervals crossed zero and neither sign test was significant. The
candidate is therefore **NOT_PROMOTED** and was removed from the final runtime by
`44199d9fa323d3708074922ff63dd7ffc30ae69b`. Merged-main Task #72 non-regression
passed with all 320 answers/native records equal to baseline. See
[NS-05 Before/After](ns05-video-before-after.md) and its
[compact scorecard](evidence/ns05-video-before-after-sm90.json).

Everything below is the unchanged revision-0 diagnostic record. Its frozen 5%
public-total gate remains **FAIL** and its candidate history/raw evidence is not
rewritten by the final removal.

## Outcome and claim boundary

A focused, semantics-preserving runtime candidate was implemented and measured on
the existing 330-frame Apache-2.0 natural-footage source. It reduces serial
16-frame video-input latency by **61.220 ms (9.73%)**, with exact selected RGB,
source PTS, temporal-2/mRoPE, generated-output, serial/batch-four, and repetition
parity. The frozen Task #72 natural control also passes with byte-identical
answers to the accepted baseline.

The preregistered primary acceptance is nevertheless **FAIL**: public serial
temporal-16 total latency improves by a paired median **53.740 ms**, but that is
**2.33%**, below the required 5%. The absolute 25 ms, bootstrap, sign-test, CV,
and secondary-regression conditions pass; the relative condition does not. This
is a measured improvement in one video-input stage on one real source, not a
generalized throughput or quality claim.

- `OUTPUT_AND_NATIVE_SEMANTICS_PARITY: PASS`
- `NATURAL_VIDEO_FIXTURE_ACCURACY: UNCHANGED_MEASURED_TASK72_CONTROL`
- `FIVE_PERCENT_PUBLIC_TOTAL_ACCEPTANCE: FAIL`
- `GENERALIZED_QUALITY: NOT_COVERED`
- `PERCEPTUAL_QUALITY: NOT_RUN`
- `ISOLATED_H2D_TRANSFER: NOT_COVERED`
- `REAL_CANCELLATION: NOT_RUN` (unchanged offline cancellation contracts pass)

## Profile-selected candidate

`native/qwen38/image_preprocess.cpp` historically lets FFmpeg scale to the fixed
four-slice target and then, for a 16-frame/eight-slice request, applies the tighter
per-slice patch cap with a second host bicubic resize. On the 1920×1080 source
these dimensions are 256×480 followed by 224×448. Changing the FFmpeg target would
change selected RGB and was therefore rejected.

The candidate runs the **existing scalar bicubic computation independently for
each frame with at most four workers**, only for non-exact 16-frame inputs. Every
frame writes its own output vector. Worker errors are retained and rethrown after
joining. Exact-size inputs, the default/eight-frame path, four-frame paths, frame
selection, ordering, patch packing, and model execution remain serial and
unchanged. The change adds no public field or configuration knob.

Remote host profile, two warmups plus nine interleaved measurements per arm:

| temporal-16 stage | Baseline median / p95 ms | Candidate median / p95 ms |
| --- | ---: | ---: |
| `decode_video_bytes` total | 511.496 / 538.805 | 467.512 / 490.758 |
| second resize + temporal-2 packing | 68.209 / 72.210 | 25.668 / 28.864 |
| FFmpeg demux/decode/RGB conversion | 339.263 | 338.039 (median) |
| demux/probe | 76.594 | 77.266 (median) |
| temporal analysis | 7.760 | 7.631 (median) |
| frame selection | 0.006 | 0.006 (median) |

All seven arms (default-8, uniform/temporal 4/8/16) on the long source and three
retained natural excerpts produced 28 byte-identical baseline/candidate semantic
binaries. One baseline temporal-4 host repetition changed only
`has_nominal_timestamps` and its encompassing diagnostic FNV because an
**unselected** candidate lacked finite showinfo PTS. Selected RGB, source indices
and PTS, pair-first timestamps, geometry, and pixel count stayed exact. This
variation is retained, not hidden.

Rejected experiments remain external with raw profiles and patches:

| Rejected variant | Local temporal-16 packing median | Decision |
| --- | ---: | --- |
| direct pointers + 256-entry normalization LUT | 79.161 ms | no improvement; reverted |
| direct pointers + scalar normalization | 79.360 ms | no improvement; reverted |
| four-way exact-size slice packing | 74.019 ms | insufficient and not the non-exact hotspot; reverted |
| AVX2 exact-size packing | 77.599 ms | unreachable for 16-frame non-exact input; reverted |

## Preregistered real-model protocol

Official main is `a4320e7678748a2336f74e3c2477c21b30805fb8`. The instrumented
baseline is `03b4ab65ea861491374c00954b06269f9baadc4c`; candidate execution is
`1ff89b764962d270613d42344b9879eab122c006`. The runtime candidate itself is
`4a5287a0874fa43517f964a39993b01ad34abf99`.

The source is historical Git blob `fa078dc543b273922dfc8c07e74f63306778a3ed`,
SHA-256 `81178f8af6c859d908cd6fa4a5348ed9bbe271c071a94a196ff2488c63f0db2e`:
330 H.264 frames, 30 fps, 11 seconds, two B-frames, under the repository's
Apache-2.0 license. No media was downloaded or procedurally generated.

Serial and batch-four each ran fresh **baseline → candidate → candidate →
baseline** stacks on physical GPU 1. Every stack had four warmup requests
(including one explicitly marked default-8 padding request) and 24 seeded,
interleaved measured requests across default-8, uniform-16, and temporal-16.
Batch groups were complete groups of four. Prefix caching was disabled and every
request logged `prefix_reused=0`. Sampling was temperature 0, top-p 1, seed
510075, and at most 32 tokens; both `stop` and `length` are valid real terminals.
The 224/224 requests completed, with 192 measured and 112 exact A/B pairs.

Binary/model identities:

| Object | SHA-256 / identity |
| --- | --- |
| Gateway | `d805a0cab3a21b699405c7db231660f800a9cb639df8590d4e8648cb90d052e2` |
| Baseline native stage | `2ba39a26067419a7ec49864b00db23275c054a18d7769f7386b03846c7eabc42` |
| Candidate native stage | `df46ab96ec6de742b3549d262f62ce5dbcc7d08c3e6b7313bd758622b1b769cf` |
| Baseline / candidate profiler | `d7b27c14262ee4188228f1a82f50ebdb0f5a6f3ed0a68dae7e0af612d2d63fe9` / `3846b4da4dba6e90a8b8e0000c2d32c640a754d04cf923b0d2201b5065448e06` |
| Qwen3.8 config | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Complete 29-file model manifest | `198a6d8552768d55e79537bc500656f488a381f4dd07035b6c862013c0cc8278` |

Model revision is `UNKNOWN_LOCAL_NO_GIT`; CUDA is 12.4.131 and FFmpeg/ffprobe is
6.1.1-3ubuntu5. The resident service and all other physical GPUs were not used.

## Real public latency

Nearest-rank p95, inclusive-quartile IQR, and median absolute deviation are in
milliseconds. Every cell has 16 measured requests. No timing value was removed.

| Mode | Arm | Implementation | Median | p95 | IQR | MAD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| serial | default-8 | baseline | 1903.337 | 1975.996 | 17.778 | 10.278 |
| serial | default-8 | candidate | 1889.295 | 1929.449 | 8.498 | 6.073 |
| serial | uniform-16 | baseline | 2275.779 | 2342.736 | 23.878 | 10.588 |
| serial | uniform-16 | candidate | 2222.108 | 2250.408 | 23.556 | 9.133 |
| serial | temporal-16 | baseline | 2309.817 | 2338.963 | 29.457 | 16.820 |
| serial | temporal-16 | candidate | 2244.868 | 2277.491 | 9.915 | 4.290 |
| batch4 | default-8 | baseline | 5043.342 | 5238.118 | 265.906 | 134.165 |
| batch4 | default-8 | candidate | 4998.375 | 5241.477 | 267.443 | 134.611 |
| batch4 | uniform-16 | baseline | 5178.413 | 5289.211 | 113.006 | 46.268 |
| batch4 | uniform-16 | candidate | 5161.969 | 5301.956 | 157.399 | 60.638 |
| batch4 | temporal-16 | baseline | 5192.226 | 5267.008 | 120.076 | 41.940 |
| batch4 | temporal-16 | candidate | 5166.046 | 5273.636 | 136.789 | 52.783 |

Primary serial temporal-16 decomposition:

| Metric | Baseline median / p95 / IQR / MAD ms | Candidate median / p95 / IQR / MAD ms |
| --- | --- | --- |
| video input | 629.530 / 656.745 / 22.163 / 12.610 | 568.310 / 601.082 / 11.747 / 5.301 |
| TTFT | 1697.654 / 1726.617 / 27.883 / 15.936 | 1635.189 / 1667.053 / 9.568 / 4.145 |
| native elapsed | 2169.500 / 2197.000 / 22.750 / 13.000 | 2107.000 / 2140.000 / 12.000 / 5.500 |
| vision encoding | 140.300 / 140.500 / 0.200 / 0.100 | 140.300 / 140.500 / 0.100 / 0.000 |
| text prefill | 781.500 / 781.700 / 0.125 / 0.100 | 781.500 / 781.600 / 0.100 / 0.050 |
| decode | 603.000 / 603.500 / 0.225 / 0.100 | 603.100 / 603.600 / 0.425 / 0.200 |
| prompt preparation | 5.303 / 5.402 / 0.087 / 0.043 | 5.267 / 5.339 / 0.010 / 0.005 |

The vision measurement includes host conversion, transfer enqueue, and encoding;
an isolated H2D interval was not measured. Batch-four `unattributed` includes
queueing and is not interpreted as model execution.

Paired candidate-minus-baseline public-total comparisons (16 pairs each):

| Mode / arm | Median / mean delta ms | MAD ms | Negative / positive | Exact sign-test p | Bootstrap 95% median delta ms |
| --- | ---: | ---: | ---: | ---: | --- |
| serial / temporal-16 | -53.740 / -58.654 | 15.280 | 16 / 0 | 0.0000305 | [-75.114, -48.511] |
| serial / uniform-16 | -52.869 / -57.365 | 16.750 | 16 / 0 | 0.0000305 | [-65.130, -40.186] |
| serial / default-8 | -9.066 / -14.973 | 7.714 | 15 / 1 | 0.0005188 | [-17.931, -6.516] |
| batch4 / temporal-16 | -13.381 / -23.313 | 41.751 | 11 / 5 | 0.2101 | [-81.275, +0.011] |
| batch4 / uniform-16 | -1.879 / -15.822 | 24.775 | 8 / 8 | 1.0 | [-25.430, +20.436] |
| batch4 / default-8 | -3.870 / -15.327 | 26.700 | 10 / 6 | 0.4545 | [-39.900, +18.199] |

The candidate does not execute for default-8; that small serial difference is
reported as an ABBA control observation, not attributed to the code change.
Batch-four temporal improvement is not significant. No secondary arm/mode had a
median or p95 regression over 5%.

Nineteen throughput timing outliers are retained: 15 baseline and four candidate.
By cell, serial default-8 total/native has 1/1 per implementation; serial
uniform-16 baseline 1/1; serial temporal-16 candidate 1/1; batch-four baseline
uniform-16 3/3 and temporal-16 3/2 (total/native). All per-request values and
fences are in the external raw report.

## Frozen natural and Task #66 controls

The unchanged Task #72 candidate run passed 320/320 requests and 304/304 parity
checks. All **320 answers and 320 native semantic records are exactly equal** to
the accepted baseline. Default-8, uniform-8, and temporal-8 each score 42/48;
uniform-4 and temporal-4 each score 48/48. The primary temporal-8 versus uniform-8
comparison remains 0 improved / 0 worsened pairs, `p=1.0`.

All 18 misses are retained: six `Pappina` answers for `natural-07` in each of
default-8, uniform-8, and temporal-8. No aliases were added. Six natural-control
timing outliers are retained. Local and remote fresh-process natural verification
receipts are byte-identical; each independently decoded all 165 source frames and
checked the 34 frozen fixture files plus 138 unchanged Task #66 control files.
Task #66 deterministic tests pass 14/14 and natural tests 17/17. A new real-model
Task #66 replay was **NOT_RUN**.

## Verification, negative setup attempts, and cleanup

- `umm-test-local test`: PASS.
- Local native host CTest: 3/3 PASS; focused varied 16-frame non-exact parity and
  repeated-output test PASS. The pre-change benchmark was red for material
  improvement; the selected host candidate is green. The public 5% gate stays red.
- Remote native Qwen3.8/ASR CUDA build and CTest: 3/3 PASS on physical GPU 1.
- Full native stage CUDA build: PASS; 12/13 tests passed with device remapping.
  `image2_request_allocator_test` explicitly selects physical device 1 and failed
  under remapping; rerun without remapping passed 1/1. No test was skipped or
  weakened.
- Independent Task #75 fresh-process verifier: PASS on 224 requests, 112 pairs,
  28 semantic binaries, and 1,537 manifest files.
- Excluded preflights retained: PID-namespace assumption, length-terminal helper,
  malformed preregistration JSON (no process/model traffic), temporal fields on a
  uniform request, and a post-request report variable collision. None contributes
  to accepted measurements.

The authoritative external evidence contains 1,537 files / 970,526,413 bytes.
`artifact-manifest.json` and `SHA256SUMS` explicitly self-exclude their own
recursive entries; `SHA256SUMS` covers the manifest. Exact hashes:

| Artifact | SHA-256 |
| --- | --- |
| External `artifact-manifest.json` | `8d65ffbdcc9c4a7e7913a86b2afaeb4b21f244f3f629a23efa9d4fc195c26501` |
| External `SHA256SUMS` | `35b8c665d25e1fbf6062e5d5af6cb6b94718c6430401cdd2f34a1fd775634117` |
| Raw throughput report | `8cf3923500bae33436057a31aac0b1198b086c5681ef4a0322818881e9bc698e` |
| Independent Task #75 verification | `089b1244f6a9911da1e4e4e90ae1d21d7c9f0cf5f03291e4ee5fe349c5163f0d` |
| Natural candidate report | `b468475ffc2b17307a368654a3d2ed412e99c9fddd0ec556c1cf5c5f7c67ffa8` |
| Natural candidate `SHA256SUMS` | `80e2c245551e8771e9d5cdb6f78a5c2afc7023a45cbfa553e6aac476e227303c` |
| Natural local/remote verification | `d5a3f0516bede6f90acef65474321cc5f0c44825d0244856f286440f5b78206b` |

The compact machine-readable aggregate is
[video-decode-prefill-sm90.json](evidence/video-decode-prefill-sm90.json). The
preregistration and all amendments are in
[benchmarks/video-decode-prefill/preregistration.json](../benchmarks/video-decode-prefill/preregistration.json).

Fresh verification (both paths are caller-supplied and outside the repository):

```sh
python3 -I tests/video_decode_prefill_verify.py "$EVIDENCE_ROOT" \
  --output "$VERIFY_OUTPUT"
```

Every recorded task-owned process received SIGTERM and stopped. No SIGKILL was
needed for accepted runs. Final physical GPU 1 state is **3 MiB, zero compute
processes**. No resident process was stopped or restarted.
