# NS-01/NS-02 native interactive-preview diagnostic (Task #79, 2026-11-27)

## Result and promotion boundary

**CONTROLLED_LIVE / NOT_PROMOTED.** On one explicitly idle NVIDIA H20 SM90,
physical GPU 7, the frozen automatic TeaCache 0.25 candidate produced a real,
repeatable multi-second Wan2.2-TI2V-5B speedup, but failed preregistered visual
quality controls and the combined NS-01 hard thresholds. The candidate is absent
from the final default runtime. Final runtime commit
`4b727875798b6682cacee823a24d1e4a484aea04` has zero `native/` or `src/` diff
from official main `91310b36184c86a1c147debb27db8643cc0cb3e4`.

No full NS-01 or NS-02 PASS, deployment, perceptual judgment, training, or
optimizer result is claimed. Human/perceptual review is **NOT_RUN**.

The compact independently verified scorecard is
[ns-interactive-preview-sm90.json](evidence/ns-interactive-preview-sm90.json).
Raw responses, media, MessagePack frames, process logs, RGB, and review media are
external only. The complete external bundle contains 29,845 regular files and a
self-excluded top-level `SHA256SUMS`; nested evidence manifests remain covered.

## Frozen candidate and protocol

Candidate commit `3f98fe1645d4d82877372056ba8ad2544aa399c5` selected only the
existing native TeaCache residual policy:

- with both the request and Stage CLI option absent, native I2V used threshold
  0.25 automatically;
- an explicit request/Stage value, including 0.0, retained precedence;
- T2V and the legacy direct-engine constructor stayed uncached;
- no CUDA kernel, solver arithmetic, precision, frame count, resolution,
  sampling, public schema, or client Fast/Quality control changed.

The frozen protocol is
[preregistration.json](../benchmarks/ns-interactive-preview/preregistration.json).
It fixes five process-separated warm pairs per geometry, seeded arm order
`79025`, one excluded warmup per shape/block, exact frozen real FLUX.2 sources,
seed 42, 29 frames, 20 UniPC steps, 16 fps, and guidance 5.0. Both public arms
omit `teacache_thresh`. The three NS-01 pairs use seeded order
Before→After, Before→After, After→Before. Every observation and IQR outlier is
retained.

The accepted NS-01 semantic control uses a short plain-prose English request;
this is **not** an exact Chinese rubric PASS. All eight earlier Chinese/long or
truncated setup attempts and the NS-02 R1 tooling failure are separately retained
with raw trajectory/log hashes.

## Final default-path profile

The new official-main NS-02 profile independently passed 291 files. Its nine warm
public I2V measurements were **11,076.665 ms median / 11,187.189 ms p95**, with
DiT dominating at about 8.8 s and VAE decode about 2.0 s. The three frozen public
FLUX sources exactly matched Task #74:

| Geometry | Source SHA-256 |
| --- | --- |
| Wide 704×480 | `477cfafdcbd663b3d573d70140c5da355a4874fea920bc721dedc329a8ed3451` |
| Tall 480×704 | `390465a6b7c01e25faaed49e73bb9cf6a49d7e607636988e491805176605ce27` |
| Square 512×512 → 704×480 center-cover | `c46c343fdae91d00a561759c697c53ac7693e39d43e47429a00436ad0067d1da` |

Profile report `2851b23433b4de53a1b880f02ef0d4f2ce5d4ba4154854ead899bcd2f0b83e6d`;
manifest `2b78a1272a9698ab27cbb391e332f6dfab5c8276e0e549e8a0f833a353ccb53a`;
fresh verification `565e8ef8543d37e478b0529651b4cc116b1e9c863b3b16e79c516d647454a053`.

## NS-02 paired click-to-animate result

The corrected R2 execution used commit
`2f0a241290deaed19b99776e84eccf389960fa79`, the same frozen candidate and
protocol as R1, plus crash-safe per-record journals and a fixed FFmpeg SSIM
parser. R1 is excluded because its parser failure occurred before the report and
before per-record timing journals existed. R2 did not change the candidate,
settings, thresholds, sources, or schedule.

| Geometry | Before median / p95 (ms) | After median / p95 (ms) | Median paired delta (ms) | <5 s |
| --- | ---: | ---: | ---: | --- |
| Wide | 11,028.298 / 11,103.809 | 4,112.208 / 4,133.796 | −6,926.658 | PASS |
| Tall | 11,040.370 / 11,118.497 | 4,093.395 / 4,150.852 | −6,945.184 | PASS |
| Square center-cover | 11,032.266 / 11,065.124 | 4,088.612 / 4,203.988 | −6,934.487 | PASS |
| **Pooled** | **11,038.579 / 11,125.341** | **4,101.640 / 4,178.276** | **−6,928.849 (−62.769%)** | **PASS** |

All 15 paired deltas were negative. After IQR/MAD were 38.749/13.028 ms. The
4220.477543 ms square observation and every Before outlier remain in all
statistics. Every successful video decoded as H.264/yuv420p with the requested
width, height, 29 frames, and 16 fps. Exact source bytes reached native Wan;
within-arm fixed-seed MP4s were byte-reproducible.

### Quality rejection

The speedup is not semantics-preserving and did not meet the frozen approximate
quality gate:

- all 15 pairs failed exact paired first-frame RGB equality;
- wide/tall/square SSIM were 0.912704 / **0.894088** / 0.93706, with tall below
  the 0.90 minimum;
- all-frame PSNR was 26.506 / 25.285 / 26.191 dB, above 24 dB;
- temporal-motion ratios were 0.9743 / 0.5787 / 1.2131, inside [0.5, 1.5];
- nonzero motion and source-first-frame continuity proxies passed, but do not
  repair the failed exact/fidelity controls.

Therefore `quality_gate_pass=false` and `promotion_decision=NOT_PROMOTED`.
Numeric proxies are not human/perceptual approval.

### Cancellation, malformed input, and recovery

One real non-streaming `/v1/visual/render` disconnect per arm occurred one second
after observed native submit. After native abort terminal was **0.588345 ms**
(client/health settled **1.434646 ms**), versus 0.562761/1.503389 ms Before;
median=p95 because there is one preregistered observation per arm. Both are below
500 ms. There were zero late native records or public cancelled media across the
complete retained history, `in_flight=0`, and immediate retry plus malformed-size
recovery reproduced each arm's exact MP4 hash.

This route exposes no request id, so `/abort` ACK is **NOT_APPLICABLE**, not a
fabricated measurement. Client disconnect, native cooperative exit, and health
settlement are separately recorded.

## NS-01 paired autonomous narrative result

Three complete process-separated pairs used identical clean text and exact
source bytes, one autonomous key image, and an explicit I2V link to that image.
Raw streams had zero tag leakage, correct visual/artifact/terminal ordering, one
terminal, completed slots, exact PCM joins, and final `in_flight=0`.

| Metric | Before median / p95 (ms) | After median / p95 (ms) | Hard result |
| --- | ---: | ---: | --- |
| TTFT | 556.799 / 558.078 | 550.893 / 560.254 | **PASS <800 ms, all 3 After** |
| TTFA | 17,443.210 / 17,452.287 | 17,388.069 / 17,400.633 | **FAIL <800 ms** |
| First image | 7,105.569 / 7,142.737 | 7,088.715 / 7,097.805 | **FAIL <2,000 ms** |
| First public/playable I2V artifact | 58,467.882 / 59,353.945 | 51,159.547 / 51,258.215 | **FAIL <5,000 ms** |

Paired public-video changes were −8,183.217, −7,308.335, and −7,128.249 ms;
median −7,308.335 ms. Native DiT changed from 8,911.830→1,886.940 ms median;
native denoise 12,005.800→4,959.630 ms. Cold native I2V elapsed was
12,140→5,119 ms median, so even native cold execution missed 5 s. VAE decode was
2,217.300→2,230.170 ms, not a claimed improvement.

All three After MP4s were byte-identical
(`6505a68137b6b5b7de03381332ba05a38921c8451198f5a248fd50b6616a47bc`);
Before MP4 was `0ddcde8cbf8c59edc6dadb531a8435d4a4f6c08d78e96cd2f14e295a26868f28`.
Exact shared source was
`b68b27ade8e68d3c64f312b76e8422168b9a42d48f6917130c97d6e73f0b18ff`.
Nevertheless every NS-01 pair failed exact paired first-frame RGB, with SSIM
0.887097, all-frame PSNR 24.875 dB, and motion ratio 0.5226. This independently
fails quality promotion even apart from public latency.

The exact Chinese rubric attempts exposed existing native TTS failures rather
than passing: short token ceilings omitted or cut directives; longer responses
hit non-terminating Markdown-only spans (one reached 8,192 frames / 655.36 s),
the streamed-artifact cap, or aborted video. The successful plain-prose English
semantic pairs do not erase those failures.

## Failed attempts and verifier corrections

Nine failed setup/tooling attempts are sealed separately from accepted pairs:

1. Chinese 224-token truncation, no autonomous visual slot.
2. Chinese 640-token run reached both slots but exceeded streamed-audio cap and
   aborted video.
3. Chinese 384-token run stopped before video.
4. Chinese 448-token run hit the native TTS maximum/audio cap.
5. Long English 384-token run stopped before video.
6. English 448-token run cut inside `[[video:...]]`, leaked it to TTS, and was
   explicitly aborted.
7. English 640-token full response stalled in a later native TTS span and was
   explicitly aborted.
8. Short English response with Markdown headings stalled in native TTS and was
   explicitly aborted.
9. NS-02 R1 generated the frozen pairs but its parser expected `SSIM All:` where
   the installed FFmpeg emits `SSIM ... All:`. R1 has no accepted statistics.

For R2, an independent review also required unique NS-01 pair IDs, exhaustive
late-native-frame scanning, and full NS-02 decision recomputation. A verifier-only
correction excluded process-volatile FFmpeg log digests while requiring exact
numeric SSIM/PSNR/MAE/motion parity. Fresh verification then passed without media
regeneration. Earlier missing-path and isolated-import invocation errors remain
outside the accepted PASS chronology.

## Exact identities

All model revisions are `UNKNOWN`; exact local tree/config hashes are the binding
identity. Raw evidence includes every individual file hash.

| Object | SHA-256 |
| --- | --- |
| Official server binary | `723902d868208668fb1396f2c3433fc5fd28d64c1638a58b2c01458994d7ca5c` |
| Before Wan binary | `ab90b11772e5cd44dc8e590f6c6fe7e9fea39ecf06ae85cd0715bda66760d017` |
| Candidate Wan binary | `8546f01326e42b96b844f647ac7e1745389b5c53174ed3a9e016c111a40cb355` |
| Qwen3.8 binary | `10107ed609952d86288c2074c876969624939d58cbf38e471aa74f8684719474` |
| Qwen3-TTS binary | `d5ad3e825f30f77f918949b253b685d6070171d946a2fe4065547430a928afd3` |
| Wan tree / model_index | `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616` / `6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7` |
| FLUX.2 tree / model_index | `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997` / `51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0` |
| Qwen3.8 tree / config | `22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304` / `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Qwen3-TTS tree / config | `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d` / `17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9` |
| FFmpeg / ffprobe | `ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41` / `272f6ebc634a63d9c8b4ca68e964119d980f25154e5aa2c35e5487da48e9a58f` |

The runtime used CUDA 12.4.131 for SM90 and the existing isolated Torch
2.6.0+cu124 / Diffusers 0.43.0 environment.

## Independent verification and hashes

| Record | SHA-256 |
| --- | --- |
| Compact scorecard | `fe5c0cedbd29e02de3fe453adcceddf28942cecffbc34b0441194e5a21e73931` |
| Complete external manifest, 29,845 files | `667f22b17ff9fcd879c80900b000c569e03e63520117523c5c28b1a1f072b701` |
| Final fresh whole-bundle verification | `65d7b78db7c0b3b742d6b43db98a906b256cf24ca002764d2316f8ff7c584338` |
| Closeout hash record | `611875273bb8cb99ef2cb335fef757eb41e19735e52ee25ca3d307988d9b1fbe` |
| NS-02 R2 report | `bbfdb7995a55cb6c3a403f371f7d7bf7a0fd26df8649ea41e2e5c2c757259ed2` |
| NS-02 R2 manifest, 1,529 files | `a779ca9296d0f05ccebcd4fec420447b41b359ffddcb72ab914218e7ff691844` |
| NS-02 corrected fresh verification | `fdb2956c7f2190784e5b897c09a9a462914d34ac518d28e93291887c136e6d07` |

The final verifier rehashed every manifest entry, reconstructed all summaries and
thresholds, checked unique pairs and all media hashes, recomputed NS-01 quality,
and recursively recomputed the complete NS-02 decision. Its result is PASS for
**evidence integrity**, not candidate promotion.

## Tests and cleanup

- `umm-test-local`: PASS on the candidate with evidence fixes and PASS again on
  the final unpromoted runtime; final log SHA-256
  `e5e70a810de81d5c18b0e86137b622c13c15510aab41511a9c0e2b6b5149fd5a`.
- Deterministic evidence/SSIM/manifest controls: 12/12 PASS.
- Native video host CTest: 1/1 PASS; native Stage host CTest: 11/11 PASS.
- Candidate SM90 Stage CTest: 13/13 applicable PASS, log
  `798162157c9679e1d9971b278b07ff002275b0b72e6c23528f1f819c85d382f2`.
  The unrelated image2 allocator test requires two visible devices and is
  NOT_RUN under the GPU-7-only constraint.
- Narrative source-link self-check and raw-SSE replay: PASS (10/10 replay tests).
- Real candidate anchors, malformed rejection, cancellation, terminal dedup,
  `in_flight=0`, and immediate/fault recovery passed as recorded above.

Only task-owned processes were stopped. GPU 7 returned to 3 MiB with zero compute
processes; residual utilization samples were rechecked to 0%. GPUs 0/1 were not
used, and the resident service/GPUs 2–6 were never contacted, restarted, or
deployed to.

## Reproduction

Use exact local assets, an explicitly idle permitted GPU, and fresh
caller-selected evidence directories. The protocol and tools have no absolute
output default or tracked host/address. Start with the profile, then freeze the
preregistered baseline report hash before any candidate run:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/i2v_geometry_latency_live.py --mode profile \
  --python /path/to/venv/bin/python \
  --server-binary /path/to/umm-infer \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py \
  --flux-model /path/to/FLUX.2-klein-4B \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir "<new-profile-evidence>" --gpu 7 \
  --base-commit 91310b36184c86a1c147debb27db8643cc0cb3e4 \
  --execution-commit 91310b36184c86a1c147debb27db8643cc0cb3e4
```

The historical candidate requires commit `3f98fe16`; the final default runtime
intentionally does not contain it. NS-01 uses
`public_narrative_i2v_live.py --scenario ns01-black-hole`; NS-02 uses
`i2v_geometry_latency_live.py --mode ns02` with explicit baseline/candidate
binaries, frozen profile, preregistration, and fresh output. Compact scorecard
and independent verification entry points are
`tests/ns_interactive_preview_scorecard.py` and
`tests/ns_interactive_preview_verify.py`. All require caller-selected paths.
