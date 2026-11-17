# Soak and fault recovery on the native H20B service

Repeatable long-run test of the native deployment: mixed request load, a
memory and queue time series, and three fault injections with measured
recovery. Every number below was produced by `tests/soak_native.py` on the
deployment described in [Environment](#environment). No number is estimated.

## Environment

| Item | Value |
| --- | --- |
| Commit under test | `889b727c504db49c6acd64ac01be0b9fee487b7c` |
| Host | H20B, container `linyu-dev` |
| GPU | 4 x NVIDIA H20, driver `535.161.08` |
| GPU assignment | ASR 2, Qwen3.8-27B 3, Qwen-Image-2512 4, TTS 5 |
| CPU / RAM | Xeon Platinum 8457C, 180 cores / 1928 GiB |
| Kernel | `5.4.250-9-velinux1u2-amd64` |
| CUDA | 12.4.131 (nvcc) |
| Python | 3.12.3 |
| Zig | 0.16.0 |
| Server | `172.17.0.2:43301`, supervisor `service/h20b-service.json` |

Installed binary SHA256, built from `889b727` on the host:

| Binary | SHA256 |
| --- | --- |
| `umm-infer` | `64581317b2764f2c87228e61b63ec72e5e7f6075cb44a14d035dbaeeb2eb6dd1` |
| `umm-native-stage-asr` | `ee601434f2ff5a72e315d04e2574021241b8ab154c8be57c92f3bbda1385b651` |
| `umm-native-stage-qwen38` | `b43ebb6318821bac56799bcb33f8d5eeac5d5d375f0c95ad1ec9c7dd1b9bc9a5` |
| `umm-native-stage-image` | `97e9406ee2b1d82bb1883cfe9362a9cb767b1d050428c82c6c0eb5c91d3ccfbc` |
| `umm-native-stage-tts` | `9424cb6b5f37c9c5cc2b2498df918133b09e7d5ff238c820e443abe004e88fc0` |

The binaries were built from `git archive HEAD` on the H20B host
(`zig build -Doptimize=ReleaseFast`; CMake `-DUMM_STAGE_CUDA=ON`), installed
over the previous deployment, and the supervisor was restarted through its own
`stop` / `start` commands. The previous binaries are kept in
`/root/umm-native-service/bin/pre-soak-20261117/`.

## Load model

`tests/soak_native.py --duration-s 1800 --concurrency 4` drives five request
types over the public HTTP surface:

| Type | Route | Shape |
| --- | --- | --- |
| text | `POST /v1/chat/completions` | text only, `max_tokens=48` |
| asr | `POST /v1/audio/transcriptions` | real 15 s WAV (481718 bytes), multipart |
| vlm_image | `POST /v1/chat/completions` | one image part, streamed |
| tts | `POST /v1/chat/completions` | `modalities=[text,audio]`, streamed |
| image | `POST /v1/chat/completions` | `modalities=[text,image]`, 512x512, streamed |

Every stage is serial, so the two expensive types are rate-gated instead of
weight-sampled: at most one image request per 240 s and one TTS request per
12 s. A gated type that is not due falls through to text, ASR or image-VLM.
Without the gates the admission queue fills with requests that cannot run.

Sampling every 20 s records `VmRSS` for the server and each stage, per-GPU
memory, temperature and utilisation from `nvidia-smi`, and the server's
`umm_queue_depth`, `umm_stage_in_flight` and `/health.in_flight`. 92 samples
cover the 30 minute run.

## Results

Reported run: 1800 s, no process restart during the window
(`restart_delta` all zero).

| Metric | Value |
| --- | --- |
| Requests | 1324 |
| Success | 1324 |
| Success rate | 1.0000 |
| End-to-end p50 | 1841 ms |
| End-to-end p95 | 13541 ms |
| End-to-end p99 | 88537 ms |
| Maximum | 227822 ms |

Per type:

| Type | Count | Success | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| text | 527 | 100% | 1813 ms | 4943 ms | 6147 ms |
| asr | 350 | 100% | 1015 ms | 1844 ms | 1977 ms |
| vlm_image | 350 | 100% | 2058 ms | 3951 ms | 4758 ms |
| tts | 91 | 100% | 28352 ms | 89663 ms | 104942 ms |
| image | 6 | 100% | 227480 ms | 227822 ms | 227822 ms |

The overall tail is the image leg. That leg runs one request at a time, loads
the 53 GB model per request, and denoises 50 steps: 10 s load + 213 s denoise
+ VAE. The TTS tail is queueing behind one serial TTS stage; its own compute is
about 11 s.

Queue depth over the run, from `/metrics` and the container's own sampler:

| Gauge | Max |
| --- | --- |
| `umm_queue_depth{queue="unified_admission"}` | 4 |
| `umm_queue_depth{queue="coordinator_pending"}` | 0 |
| `umm_stage_in_flight{stage="asr"}` | 2 |
| `umm_stage_in_flight{stage="qwen38"}` | 4 |
| `umm_stage_in_flight{stage="image"}` | 1 |
| `umm_stage_in_flight{stage="tts"}` | 4 |

Admission never exceeded the four concurrent clients. The per-stage numbers
are in flight, not queue depth: the server exposes no separate per-stage queue
metric. `coordinator_pending` stayed at zero, so no request waited for a stage
to become ready.

## Memory

Start, middle and end medians (first, middle and last thirds of the run):

| Process | Start | Mid | End | Delta |
| --- | --- | --- | --- | --- |
| server | 550.4 MiB | 650.6 MiB | 679.8 MiB | +129.4 MiB |
| stage asr | 155.8 MiB | 155.8 MiB | 155.8 MiB | 0.0 MiB |
| stage qwen38 | 52378.2 MiB | 52378.2 MiB | 52378.2 MiB | 0.0 MiB |
| stage tts | 195.3 MiB | 195.3 MiB | 195.3 MiB | 0.0 MiB |
| stage image | 315.8 MiB | 52770.5 MiB | 316.0 MiB | oscillates |

GPU memory follows the same shape:

| GPU | Start | Mid | End |
| --- | --- | --- | --- |
| 2 (asr) | 65301 MiB | 62922 MiB | 62920 MiB |
| 3 (qwen38) | 55773 MiB | 55773 MiB | 55773 MiB |
| 4 (image) | 329 MiB | 53039 MiB | 329 MiB |
| 5 (tts) | 30100 MiB | 20846 MiB | 15385 MiB |

The image stage is not leaking. It loads the model at the start of every
image request and releases it at the end, so its host RSS and GPU memory step
between 316 MiB / 329 MiB and 52.8 GB / 53.0 GB once per 240 s gate. It
returned to the same baseline after all six requests. The other three stages
are flat to 0.1 MiB.

The server grew by 129 MiB, so the growth was attributed by request type:

| Sequence | Server RSS |
| --- | --- |
| before | 679.8 MiB |
| +400 text | 679.8 MiB |
| +400 text | 679.8 MiB |
| +100 asr | 705.0 MiB |
| +100 text | 705.0 MiB |

Text requests do not move RSS. ASR requests add about 0.25 MiB each. The
allocation is not retained: `malloc_trim(0)` on the live server returned
20.6 MiB, and the process then stayed flat at 711.6 MiB for the idle
observation. The per-request growth is glibc arena retention of the 481 KB
multipart/PCM buffers, freed but not returned to the kernel. Conclusion: no
heap growth attributable to a retained object in any stage or in the server.
Heap attribution stopped at RSS plus `malloc_trim` because valgrind and
heaptrack are not installed in the container.

## Fault injection

Each fault ran alone after the soak, with the service otherwise idle.

### (a) Kill a stage during a request

Target: `umm-native-stage-qwen38`, SIGKILL after 2 s of decode. The script
verifies the PID against the supervisor's recorded start time and `argv[0]`
before signalling, so it cannot signal a reused PID or a foreign process.

| Measurement | Value |
| --- | --- |
| Supervisor detected the death and started a new PID | 2.503 s |
| First successful chat request after the kill | 12.264 s |
| `/ready` returned 200 after the kill | 0.002 s |
| Terminal event for the in-flight request | none |
| Client outcome | read timeout at 90 s |
| `in_flight` after the client disconnected | 0 |

The supervisor recovers the stage quickly: a new PID in 2.5 s and a real
answer after 12.3 s, which is the 51 GB model load. Subsequent requests
recover with no user action.

The in-flight request is the problem. It never received `error` or
`cancelled`. The Coordinator has no notice that a stage died, so it kept the
record until the client closed the socket; the client saw a socket timeout,
not an explicit error. The server-side deadline is 600 s, so without a client
timeout the same request would have hung for ten minutes. Requirement (a) is
therefore not met. See [Problems found](#problems-found) for the proposed fix,
which is a design change and was not made here.

`/ready` also stayed 200 for the whole outage: the readiness flag is a
one-shot `stage_ready` registration that is never cleared when the stage dies.
A load balancer would keep sending traffic to a server whose VLM stage is gone.

### (b) Request timeout

The client stopped reading an image request after 20 s (`socket` read
timeout) and closed the connection.

| Measurement | Value |
| --- | --- |
| Client read timeout | 20 s |
| `in_flight` back to 0 | 20.426 s |
| `umm_aborts_total{source="client"}` delta | +1 |
| Next successful request | 20.618 s |

The server detected the closed peer through the HTTP sink's liveness check,
cancelled the image leg, and reclaimed the record within one sample of the
client giving up. The next request succeeded immediately. No resource stayed
attached to the abandoned request.

This is a client-side timeout. The server's own 600 s request deadline was not
exercised; the deadline has no command-line override on `serve`, so it would
cost ten minutes per request to test and was left out. It is listed as not
covered.

### (c) Fill the admission queue

48 streaming chat requests were released simultaneously against
`max_in_flight = 32`.

| Measurement | Value |
| --- | --- |
| HTTP 200 | 32 |
| HTTP 503 | 16 |
| Rejection latency (median) | 0.011 s |
| Rejection body | `{"error":{"message":"unified engine queue is full","type":"pipeline_error"}}` |
| `in_flight` after the burst | 0 |
| Next successful request | 0.181 s |

The rejection is explicit and immediate: the excess requests fail at submit
time instead of being buffered, and the server is healthy again as soon as the
accepted requests drain. The status is 503 for every rejected request.

On `origin/main` (`c7ac443`) the same path returns a dedicated
`QueueFullError` response with `Retry-After: 1` and the body `server is at
capacity`. That commit was merged after this build; the response body above is
what `889b727` returns.

## Problems found

1. **An in-flight request does not fail when its stage dies.** It hangs until
   the client disconnects or the 600 s deadline expires. Measured: no terminal
   in 90 s; recovery of the stage at 12.3 s did not settle the request. This
   needs a design decision, not a local patch. Two options:
   (a) treat a repeated `stage_ready` from an already-ready stage as a restart
   and fail every pending request whose `current_stage` is that stage, or
   (b) add a stage-death detector in the Coordinator. Option (a) reuses the
   existing registration message and is the smaller change, but it changes
   protocol semantics, so it is left to the owner.

2. **`/ready` does not become 503 when a required stage dies.** The ready set
   is written once by `stage_ready` and never cleared. Same root cause as
   problem 1 and the same decision applies.

3. **The previously deployed binaries were older than `HEAD`.** The resident
   `umm-infer` had SHA256 `6e7fbbce…` and its source tree had no `src/obs/`,
   so it predated the metrics commit. With that binary, two of four image
   probes returned `leg_error: "request aborted"` at 150 s and 183 s with no
   client action and no stage death. After installing the `889b727` binaries,
   ten image requests outside the disturbance window below produced no leg
   error. The abort could not be root-caused on the old binary because it was
   replaced; the numbers in this page are from the `889b727` binaries.

4. **The first 30 minute run was disturbed by an outside SIGKILL.** At
   12:24:11 the supervisor logged `server exited code=-9`; it restarted the
   server and all four stages by 12:24:32. No OOM was recorded and the
   container has no memory limit; the script sent no signal. A second soak
   stack owned by another session on port 43311 / GPUs 6 and 7 restarted in the
   same minute. That run produced 13869 connection failures in a 20 s window
   and is excluded from the results above. The clean rerun (`restart_delta`
   all zero) is the one reported. The failure count is a property of the test
   harness as much as the outage: the workers retried without backoff. The
   script now backs off 0.5 s after a connection error, so a future outage
   produces a bounded number of failures.

5. **Per-stage queue depth is not exposed.** `/metrics` has
   `umm_stage_in_flight` and `coordinator_pending`, but no per-stage queue
   gauge, so the per-stage "queue depth" in this report is in-flight count. A
   real queue exists only in the admission gate, and it was measured.

6. **The image stage loads and unloads its 53 GB model per request.** This is
   correct but costs 10 s on every image request and leaves GPU 4 alternating
   between 329 MiB and 53 GB. It is a throughput limitation, not a leak.

## Not covered

- **Server-side request timeout (600 s).** Only the client-side timeout was
  exercised. `serve` has no flag to shorten the deadline, and one request at
  600 s per fault would dominate the run.
- **Multi-hour duration and repeated fault cycles.** The measured window is
  30 minutes with one cycle of each fault, not a 24 h run.
- **Heap-level proof.** Attribution used RSS and `malloc_trim`; no ASAN,
  valgrind or heaptrack run was possible in the container.
- **GPU 0, 1, 6, 7 and port 43300.** Not touched, as required. GPU 6 and 7
  carried another session's stack.
- **Commits merged to `origin/main` after `889b727`.** `c7ac443` (dedicated
  overload response) and `b917eb0` (TTS per-sentence streaming, VLM prefill
  reuse) landed while this test ran. The numbers here are for `889b727`; the
  merged changes were not re-measured. The TTS latency and the overload body
  are the values most likely to move.
- **The realtime WebSocket path under soak.** The load model covers HTTP and
  SSE only; full-duplex L1 is covered by `docs/full-duplex.md`.

## Reproduce

Inside the container, next to the service:

```sh
python3 -I tests/soak_native.py --out /root/soak-out --duration-s 1800
python3 -I tests/soak_native.py --mode faults --out /root/soak-faults
python3 -I tests/soak_native.py --self-check
```

Raw output of the reported run: `/root/soak-out-run2/` on the H20B host
(`soak-summary.json`, `soak-samples.csv`, `soak-requests.jsonl`,
`console.log`), fault rerun `/root/soak-faults-run3/`. Neither directory is
committed.
