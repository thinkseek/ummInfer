# Real native Wan2.2 cancellation and fault-recovery gate (Task #67, 2026-11-27)

## Result and scope

**CONTROLLED_LIVE / PASS** on one isolated NVIDIA H20 SM90, physical GPU 7,
using official Wave 91 main
`ced4b145857e6d8d25e1fa56b2dd538ad2aa0e17`. The runtime fix and binary build
are commit `c88e71c4d07343e2a77026139b59104ea5274bde`; the accepted producer
execution is `8efa3ea30aa341129441e2e5a91e129e5237406e`. The final
quantization-correct independent verifier is commit
`e64f05ba3649db6b9a04446a7a168cca3ed5a52a`.

The exercised path was public HTTP/SSE → `UnifiedEngine` → C++ `Stage` → real
native C++/CUDA `Wan2.2-TI2V-5B-Diffusers`. Two existing CPU echo stages supplied
only the required text/readiness path. No contract-stub, fabricated video, real
VLM narrative, TTS, realtime, online optimizer, client, or Qwen video-selection
path was used as Wan evidence or modified.

The accepted run contains **1 valid control before injection, 6 valid real Wan
generations, 4 cancellations (3 active + 1 after terminal), 1 native
invalid-input fault, and 4 following exact-hash recovery requests**. All ten
public/native request cases have one terminal outcome. Final `/health` reported
`in_flight=0`; all four task-owned processes returned 0, and GPU 7 returned to
3 MiB with zero compute processes. The resident service was not contacted.

The compact record is
[native-wan-cancel-fault-sm90.json](evidence/native-wan-cancel-fault-sm90.json).
Raw SSE transport chunks, all raw MessagePack submit/abort/complete messages,
valid/partial MP4 bytes, native/process logs, and GPU samples remain outside the
tracked tree at `/tmp/umminfer-wan-cancel-67-8efa3ea-evidence/`.

## Frozen generation settings and exact recovery proof

| Setting | Initial control, terminal control, all recoveries | Denoise cancellation | Artifact-emission cancellation |
| --- | --- | --- | --- |
| Size | 352x256 | 704x480 | 1280x704 |
| Frames | 9 | 29 | 77 |
| Steps | 4 | 20 | 1 |
| FPS / guidance / seed | 16 / 5.0 / 42 | 16 / 5.0 / 42 | 16 / 5.0 / 42 |
| Observed native boundary | complete | `aborted at step 4/20` | `aborted during MP4 encoding` |

Every valid control/recovery produced a 38,895-byte H.264/yuv420p MP4, 352x256,
9 frames at 16 fps, with exactly the same SHA-256:

`b1a4d7d174a6aa8e621b34de57052a78cdec402668b1d6c11f607b96670d7c21`

This identity holds for the initial control, the valid request used for the
post-terminal cancellation, and all four recoveries. Raw native terminal media
also matches the public reconstruction byte for byte. A real generation control
completed and decoded before any abort or invalid input was submitted.

The invalid-input request used `size=500x480`. Native Wan rejected it before
model generation with `video.size 500x480 must be divisible by 32`. Its raw
native `complete` has `success=false`, no media, and one terminal. Under the
existing optional-output contract, the public request has one `complete` with
`legs[video].state=failed`; it is not mislabeled as a public cancellation. The
next valid request reproduced the control hash exactly.

## Cancellation windows and measured timing

All aborts returned HTTP 200 `{"success":true}`. Monotonic measurements below
start at the `/abort` send. Native wall-clock log fields have 1 ms truncation,
so their send-to-detection/exit results are reported as `[raw, raw + 1 ms)`
intervals. `boundary observed` is the independent monotonic observation upper
bound, not an exact device timestamp.

| Case | Abort ACK (ms) | Public terminal (ms) | Owning native terminal (ms) | Native detection interval (ms) | Native exit interval (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Queued behind serial running Wan | 0.433925 | 1.050791 | 0.962994 | N/A: never dispatched | N/A: never dispatched |
| Running denoise, step 4/20 | 0.612149 | 1.170762 | 2.394733 | [432.577881, 433.577881) | [432.577881, 433.577881) |
| MP4 artifact emission | 0.603444 | 1.091464 | 2.480554 | [-0.308594, 0.691406) | [50.691406, 51.691406) |
| After terminal completion | 0.503078 | Existing sole `complete` retained | No new native message | N/A | N/A |

The negative lower endpoint in the MP4 detection interval is only the declared
integer-millisecond truncation uncertainty; it is not a negative physical
latency. The monotonic boundary observation was 53.647217 ms after abort send.
Denoise boundary observation was 434.684811 ms after abort send. These values
show that public/native terminal acknowledgment does **not** mean the CUDA call
was immediately preempted.

The queued victim's raw native submit arrived while the serial denoise request
was still running. It settled before that blocker exited, emitted no native
generation/abort-boundary log, and produced no media. The running request, in
contrast, reached a measured cooperative denoise boundary. This explicitly
distinguishes queue removal from cancellation of a running CUDA call.

## Partial artifact and stale-event gates

Cancellation during MP4 emission retained a bounded native partial artifact:

- bytes: **262,192** (below the existing 48 MiB cap);
- SHA-256:
  `02a1bff927606b200d93e3f8483d7ada1eb98675da10102a382b71de9f3a615a`;
- no public artifact or successful native terminal escaped;
- the native temporary MP4 was removed, no task-owned ffmpeg child remained,
  and the following valid request exactly reproduced the control hash.

A cancelled partial MP4 is retained as bytes for audit only; it is not required
or claimed to decode as a complete video. Across all cancelled requests there
are zero public events after terminal, zero late cancelled artifacts, and
exactly one owning `from_stage=video` native terminal per request. Late abort of
an already terminal public request produced no second public outcome and no
native abort/message.

## Raw-evidenced defect and minimal fix

Pre-fix execution `6e69fbf9fcc92d6ac6ce37eb91ec4f220ea8614c` retained two
byte-identical owning video `complete{success=false,error="request aborted"}`
messages for one request. The reliable PUSH and low-latency PUB abort paths both
called `Stage::fail_request`; the Coordinator hid the duplicate from public
callers, but the raw native protocol did not have one terminal outcome.

A deterministic real-socket regression sent three PUSH plus three PUB aborts to
a running cancellable Stage. It failed before the fix with **6 native abort
terminals**, then passed after the fix with **1 terminal and a successful valid
recovery**. The minimal change in `Stage::abort_request` shares bounded
10-minute per-Stage/request abort-terminal tombstones across both channels.
Normal failure completions, abort-before-submit handling, public schema,
scheduler defaults, scoring, anchor validation, and implicit fallbacks are
unchanged.

The producer audit separately caught a convenience field naming the first
broadcast recipient instead of the owning video Stage. Execution `8efa3ea`
binds that field to `from_stage=video`, and the independent verifier rejects the
older attribution. This reporting-only correction did not alter the runtime or
regenerate the defect record.

## Exact model, config, binary, and source identities

| Object | Identity |
| --- | --- |
| Wan model | `Wan2.2-TI2V-5B-Diffusers`, revision `UNKNOWN` |
| Wan model tree, all 30 files rehashed | `990af3f115910af2d659a39c73bd9ea0fb727c6ad4ae6120181ddab9c2314f3a` |
| `model_index.json` | `6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7` |
| `scheduler/scheduler_config.json` | `571a3eed68bcd943a61b9fea8efa9e141472216ada59f928c8b5128ce24b32e0` |
| `text_encoder/config.json` | `a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07` |
| `transformer/config.json` | `dc00d9866e72cf77db6b531aaa33be4dc7148fef9338442bd0ae9181f7075e9b` |
| `vae/config.json` | `d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360` |
| `umm-infer`, 54,818,680 B | `603a87d1b23578fb481e0864a83b612dd80ba86804d293583fcc894d87899046` |
| `umm-native-stage-video`, 3,175,488 B | `754075285c639b2e2e1dbbee1529039208d27ec43bd4ce32348193450fdf530d` |
| Fixed `src/pipeline/pipeline.cpp` | `a8e185031f0c239b3c781a623bd41200f9ff42c7e19b87ac77570756b49a29b2` |
| Executed live harness | `6bf180fa437a8fcea65708024bdfbd126e728236b1bcf0083e5501620de8f725` |
| Final independent verifier | `5b0e039bd3b4f7561ebce625ab3130bd11c16c2835efc64f4f3c95a162a3d2e1` |
| FFmpeg / ffprobe | `ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41` / `272f6ebc634a63d9c8b4ca68e964119d980f25154e5aa2c35e5487da48e9a58f` |
| Tracked compact evidence JSON | `78bbf023c581bc3a262a397a2fc23fe31ad94c49a8b932d4094cd4800437e1ee` |
| Independent verification JSON | `ba5f9cd3fd52f267010b9e65cfc644fe28d14ed1e25a073d97ef2eb319aaf61d` |
| Raw `SHA256SUMS`, 268 files | `6de1446031f2e07bc272fb9a3fb332eb5a11d33db07e3acefee0515c5403df1f` |

The evidence JSON includes every individual model file hash and the execution's
recorded native/control-plane/harness source hashes. The runtime used Python 3.12.3, msgpack 1.2.4, pyzmq
27.1.0, CUDA 12.4.131, and an SM90 Release build. NVML reports driver/host PIDs
whereas the managed process records use container PIDs; ownership is established
by exactly one NVML compute owner on isolated GPU 7 plus the task-owned process
start/stop records, without claiming the PID numbers are in the same namespace.

## Independent verification and tests

A fresh remote verifier rehashed all model files, decoded every retained raw
MessagePack message, reconstructed public events/media from exact transport
chunks, checked monotonic/native timing, probed all valid MP4s, and sealed 268
files. A second fresh local process passed the same raw-message, timing, media,
process/GPU, and manifest checks without importing the live driver. The verifier
correction for integer-millisecond wall-log quantization was rerun on the same
retained raw evidence; no new GPU result was substituted.

Final checks:

```sh
python3 -I tests/native_wan_cancel_fault_live.py --self-check
python3 -I tests/native_wan_cancel_fault_verify.py --self-check
umm-test-local
```

- Live self-checks: **5/5 PASS**, including early transport-only request IDs.
- Independent verifier self-checks: **4/4 PASS**, including wall-time
  quantization and stale-event rejection.
- Full `umm-test-local`: **PASS**, with both Python self-checks registered in
  `build.zig` and the dual-channel C++ regression.
- Native video host CTest: **1/1 PASS**.
- Native Stage host CTest: **9/9 PASS**.
- CUDA native `stage_video_common_test`: **4/4 PASS**.

Reproduction requires exact local model assets, explicit opt-in, an empty output
directory, and idle physical GPU 7:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/python -I \
  tests/native_wan_cancel_fault_live.py \
  --server-binary /path/to/umm-infer \
  --wan-binary /path/to/umm-native-stage-video \
  --model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --source-root /path/to/ummInfer \
  --out-dir /tmp/umm-native-wan-67-reproduction \
  --base-commit ced4b145857e6d8d25e1fa56b2dd538ad2aa0e17 \
  --execution-commit 8efa3ea30aa341129441e2e5a91e129e5237406e
```

Earlier failed attempts remain separate and uncounted: one pre-generation PID
namespace assertion, two missed queued windows caused by ignored transport-only
IDs, the raw duplicate-terminal defect run, and a superseded fixed-runtime run
with incorrect producer broadcast attribution. None is presented as an accepted
cancellation window.

## Explicit limits

- Cancellation is cooperative at native call/step/encoder boundaries; immediate
  CUDA device preemption is **not claimed**.
- VAE-interior cancellation is **NOT_RUN**; Wan VAE decode has no internal abort
  callback. The real covered artifact window is MP4 encoding/emission.
- Forced process crash, direct unknown-ID native abort fuzzing, and a real VLM
  public narrative are **NOT_RUN**. The fault case is native invalid input.
- Perceptual quality, generalized performance, cross-host reproducibility, and
  production deployment are **NOT_RUN**/**NOT_COVERED** as applicable.
- No resident-service validation or deployment claim is made; GPUs 0/1 and the
  resident service were not used.
