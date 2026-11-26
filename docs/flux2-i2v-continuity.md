# Direct FLUX.2 to native Wan2.2 I2V continuity gate (Task #62, 2026-11-27)

## Result and scope

**CONTROLLED_LIVE / PASS** on one explicitly idle NVIDIA Hopper SM90 H20,
physical GPU 1, using official Wave 86 base
`27c584e8da571e16434969e9c1a11f46aabc6e76` and execution commit
`e5a1ed322391898442463d2981334b4a7012da64`.

The gate ran the real `black-forest-labs/FLUX.2-klein-4B` Diffusers Stage once,
joined and unloaded it, verified GPU 1 was idle, then loaded the pure native
C++/CUDA `Wan2.2-TI2V-5B-Diffusers` Stage. It sent the exact retained PNG bytes
as MessagePack binary to three explicit I2V requests and ran three interleaved,
same-settings T2V controls without an image. No VLM, Qwen-Image, mock artifact,
TTS, realtime, online-training, or Qwen video-selection path was used or changed.

Seven measured generation requests succeeded: **1 source PNG + 3 I2V MP4 + 3
T2V MP4**. Two additional anchor-negative requests were rejected before model
execution. A separate pre-fix diagnostic that generated one MP4 is retained and
described below; it is not counted in the accepted seven-request gate.

The machine-readable record, including every model file hash, request hash,
monotonic timestamp, RGB hash, native-path log hash, and negative result, is
[flux2-i2v-continuity-sm90.json](evidence/flux2-i2v-continuity-sm90.json).
Raw PNG/MP4/RGB/media logs remain outside the tracked tree at
`/tmp/umminfer-flux2-i2v-62-e5a1ed3/final-evidence/`.

## Frozen requests and exact content binding

| Setting | FLUX.2 source | Wan2.2 I2V and T2V control |
| --- | --- | --- |
| Prompt | `A single vivid red paper lantern above a misty blue mountain lake at dawn, cinematic composition, clear subject, gentle ripples` | `The red lantern sways gently while the camera slowly orbits above the misty lake` |
| Seed | 42 | 42 |
| Size | 512x512 | 704x480 |
| Steps | 4 | 20 |
| Guidance | 4.0 | 5.0 |
| Frames / FPS | N/A | 29 / 16 |
| Output | PNG | H.264 yuv420p MP4 |

The source PNG is **300,373 bytes**, SHA-256
`c46c343fdae91d00a561759c697c53ac7693e39d43e47429a00436ad0067d1da`.
Every I2V input carries that exact byte string as `image`, plus
`anchor_explicit=true`, `anchor_slot_index=0`, `anchor_crop=center_cover`, and
`anchor_sha256=c46c343f…d1da`. T2V inputs omit every image/anchor field. All
other request fields are exactly equal between the two arms.

Native execution logs contain one `backend=native-cuda-wan` line per measured
request, with `i2v=1` for all anchored requests and `i2v=0` for all controls.
The fresh verifier independently reads those lines rather than trusting the
report's labels. All three I2V outputs are byte-identical to each other, and all
three T2V outputs are byte-identical to each other:

| Arm | MP4 bytes | SHA-256 |
| --- | ---: | --- |
| I2V (each of 3) | 117,921 | `9c01d51996aa1fa6c74657f1177ce84e85165bd52957d42bdcf96bb558a6bfdc` |
| T2V control (each of 3) | 267,836 | `cc5395a420c77d941089551934891f9677d89f5087fb99394f36559c40972ef8` |

The distinct native flags and media hashes prove different execution paths and
outputs; they do not establish perceptual quality.

## Frozen numeric continuity and motion proxies

Metric version `center-cover-64x40-rgb8-frame0-frame28-v1` is shared with the
accepted narrative gate. FFmpeg applies aspect-preserving scale followed by
center crop to 64x40 RGB8. First-frame MAE compares source RGB with video frame
0; the motion proxy is RGB MAE between frames 0 and 28. The I2V gates remain
**first-frame MAE <= 48.0** and **motion MAE >= 1.0**. T2V metrics are descriptive
controls only and do not relax or participate in those I2V thresholds.

| Arm | First-frame MAE, runs 0/1/2 | Motion MAE, runs 0/1/2 | Result |
| --- | --- | --- | --- |
| I2V | 2.286328 / 2.286328 / 2.286328 | 3.466276 / 3.466276 / 3.466276 | PASS on frozen numeric gates |
| T2V | 85.225521 / 85.225521 / 85.225521 | 15.288021 / 15.288021 / 15.288021 | Descriptive control |

Full-resolution decoded first/last RGB (704x480x3, 1,013,760 bytes each) and the
64x40 evaluation RGB are retained for every video. For the identical I2V runs,
the full first/last RGB SHA-256 values are
`9a0a018918f3c0fd6bc658ec067b25faa98ad002cca920ce799a42f375ff0c85` and
`63f5623a5b5d2b37c55d43391974558fed613db723103ff9a606910ad9795ad3`.
For the identical T2V controls they are
`a94ae42e04a304b9a604544b151cdbda609087dfbb7dbd8ec1813208a90e1a16` and
`9b8b32e7c75d9a64a2e2af91f63abd27cd878890a44a7c6d0bb0e0ca95929318`.
The source full RGB is 786,432 bytes, SHA-256
`fcbdd19599ea1718180c8f6f255e97facdde1eb599824d93565cb6b782544879`.

These are byte/continuity/motion proxies only. **PERCEPTUAL_QUALITY and human
visual review remain NOT_RUN.**

## Latency and monotonic lifecycle evidence

No sample was dropped. Fixed execution order was
`i2v-r0,t2v-r0,i2v-r1,t2v-r1,i2v-r2,t2v-r2`.

| Measurement | All observed request latencies (ms) | Median (ms) |
| --- | --- | ---: |
| FLUX source | 1,449.525705 | 1,449.525705 |
| I2V | 11,781.219021 / 11,106.948048 / 11,110.524069 | **11,110.524069** |
| T2V | 11,062.335302 / 11,119.048482 / 11,080.626647 | **11,080.626647** |

Native denoise medians were 10,985.380656 ms (I2V) and 10,954.953972 ms (T2V).
FLUX reported 1,229.392 ms denoise. These three-pair observations are not a
generalized performance or service-latency benchmark.

Every request has integer `time.monotonic_ns()` submit/complete timestamps and a
latency derived from their difference. Both stages have start/ready/stop ticks,
return code 0, and no stop error. The FLUX stop tick precedes the Wan start tick;
ready precedes every submit, serial completions precede the next submit, and all
completions precede their stage stop. The independent verifier recomputes every
latency and checks this entire chain.

GPU 1 used 3 MiB before FLUX, 3 MiB after FLUX exited and before Wan loaded, and
3 MiB after Wan exited. A final query found zero GPU 1 compute processes and zero
task-owned Stage/Python processes. GPUs 0 and 7 remained at 3 MiB. The resident
service and its GPUs were not contacted, restarted, or used.

## Wrong-hash/tamper rejection and the pre-fix defect

A raw pre-fix request with exact source bytes and 64-zero `anchor_sha256` was
incorrectly accepted by native Wan and generated a 29,983-byte MP4, SHA-256
`f850698ff433cbb9d4f655d8356adcca1bc95076aa6f2fb1601c15f0738a2a0c`, with
`i2v=1`. This raw record motivated the only runtime fix: `parse_video_work` now
validates optional `anchor_explicit`, requires image bytes for an explicit link,
and recomputes SHA-256 over the exact supplied bytes when `anchor_sha256` is
present. Requests without these fields retain the previous implicit T2V/I2V
behavior. No public request schema changed.

The final fixed-binary controls both returned
`video.image does not match anchor_sha256`, produced no media, and emitted no
native generation line:

| Control | Submitted bytes | Declared anchor | Latency (ms) |
| --- | --- | --- | ---: |
| Wrong hash | Original source SHA `c46c343f…d1da` | 64 zeroes | 47.566361 |
| Tampered source | Valid modified PNG, 300,378 B, SHA `424b12886f5cd0faf604d5e6e2314f4c0f17f8194355a995eae8b10123e24fa4` | Original source SHA `c46c343f…d1da` | 2.087339 |

The deterministic native regression failed on the wrong-hash case before the
fix and passes after it. It covers matching raw bytes including NUL/non-ASCII
bytes, wrong hash, content tampering, hash without image, explicit anchor without
image, and malformed `anchor_explicit`. The narrative/replay tests separately
cover immediate and delayed source settlement, stale-anchor source tampering,
wrong hashes, cancellation, failed sources, and retained-valid-record recovery.

## Provenance and independent verification

| Object | SHA-256 |
| --- | --- |
| FLUX Stage source | `4efe0f79ec227540382ddb64486a28dd1675159d2c5263964bdc9091cfd307d0` |
| Native Wan executable (3,173,280 B) | `e089d86f844b922021a5cdc5e9e92e8cdcc1461b595dc2bfd8c701213cf5b36e` |
| Live harness | `0c819a52a300fbd33b11878e83e036b2d519af4f55d18c5a86f74005fd921c79` |
| Independent verifier | `6c8aa76aa43e12dcce9f92e554204e96f1bfb9644c5374924bd1ec8d2d9e3af4` |
| FFmpeg / ffprobe | `ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41` / `272f6ebc634a63d9c8b4ca68e964119d980f25154e5aa2c35e5487da48e9a58f` |
| FLUX model tree (25 files) | `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997` |
| Wan model tree (30 files) | `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616` |
| FLUX `model_index.json` | `51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0` |
| Wan `model_index.json` | `6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7` |
| Report / SHA256SUMS | `c5c588180f259725569f2623369cc776f8a68689bc8a2ff384ec977254e52529` / `1b3123b93bcc84544a235df2ab25f9446a51d3e945f8bc95d826e621a6f30fea` |
| Independent verification JSON | `2d34e969ec9f08e341dc54239b019a66441dc01c430c267e55ab02685432073f` |

Both model revisions are reported as `UNKNOWN`; file/tree hashes are therefore
the exact local identity, not an upstream revision claim. The isolated runtime
used Torch 2.6.0+cu124, Diffusers 0.43.0, Transformers 5.6.0, Hugging Face Hub
1.33.0, pyzmq 27.3.0, and msgpack 1.2.4. The complete runtime and all individual
weight/config hashes are in the evidence JSON.

A fresh verifier process passed remotely, then the sealed directory was copied
to a fresh local directory and a second fresh process independently passed all
39 manifest-bound files. The verifier imports neither the live driver nor the
narrative fixture. It rechecks media bytes, hashes, H.264 geometry/frame rate,
full/evaluation RGB, metrics, identical fixed settings, native logs, negative
controls, request/stage timing, generation counts, and latency medians.

## Reproduction and tests

Use an isolated Python environment and exact local assets; the command launches
only task-owned stages and refuses a nonempty evidence directory, nonzero GPU 1
memory above 16 MiB, any GPU other than physical 1, or missing opt-in:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/flux2_i2v_continuity_live.py \
  --python /path/to/venv/bin/python \
  --flux-stage python/umminfer_flux2_stage.py \
  --wan-stage /path/to/umm-native-stage-video \
  --flux-model /path/to/FLUX.2-klein-4B \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir /tmp/umm-flux-i2v-62-reproduction \
  --base-commit 27c584e8da571e16434969e9c1a11f46aabc6e76 \
  --execution-commit e5a1ed322391898442463d2981334b4a7012da64
```

Offline checks:

```sh
python3 -I tests/flux2_i2v_continuity_live.py --self-check
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/sse_replay_test.py
python3 -I tests/flux2_stage_contract_test.py
umm-test-local
```

Results: full `umm-test-local` PASS; native Stage host CTest 9/9; standalone
native video host CTest 1/1; focused remote video CTest 4/4; replay tests 10/10;
FLUX contract tests 14/14 with zero skips in the isolated remote environment
(two ZMQ cases skip on the dependency-light local host). The native
contract-stub lifecycle gate passed completed/cancelled/failed transitions,
wrong-session cancellation 403, matching-session cancellation 200, and final
`in_flight=0`. A direct contract-only smoke proved wrong-hash rejection followed
by valid recovery; mock/stub outputs are not real-generation evidence.

## Explicit limits

- Full public VLM narrative orchestration for this task: **NOT_RUN**.
- Perceptual quality, human visual review, and generalized quality: **NOT_RUN /
  NOT_COVERED** as applicable.
- Native real-model cancellation/fault injection beyond the two parser failures:
  **NOT_RUN** in this gate; cancellation/failure coverage above is deterministic
  contract-stub/replay coverage.
- Generalized throughput, cross-host reproducibility, production deployment, and
  resident-service validation: **NOT_RUN**.
- Public schema and implicit default behavior: unchanged.
