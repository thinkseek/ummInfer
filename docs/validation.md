# User end-to-end validation

Use these journeys to validate the current source. The end-to-end checks below
start their own server, external Python stage, and real model; argument-parsing
checks exit before any process starts. No test below accepts a fake server or
handwritten model response as evidence.

## 0. Fast local black-box gate

```sh
python3 -I tests/blackbox_integration.py
```

This runs the local suites one after another and reuses them unchanged. It is a
gate, not new evidence: it adds no assertions and each suite keeps its own exit
code, printed as `REAL_EXIT <suite> <code>`. The gate exits non-zero if any
suite fails.

Included suites: `build`, `zig-test`, `smoke`, `unified-test`, and the two
realtime contract E2Es. All of them run without H20B, CUDA, SGLang, or RL
training.

Behavior: suites run serially in their own session; combined output per suite
is capped at 64 KiB (tail kept); a suite that exceeds `--timeout` (default
1800 s) has its process group killed. Use `--deps-prefix`, `--curl-prefix`,
and `--binary` when the defaults under `/opt/homebrew` and `zig-out/bin` do not
match this host.

Boundary. A pass proves the local contracts only. It does not replace the H20B
runs in this document: real Qwen3 backends, the realtime live run, or the real
RL optimizer-step validation. Those still need their dedicated evidence
directories.

## 1. Prepare the environment

Use an Apple Silicon Mac, Zig 0.16.0, the Homebrew dependencies from README,
and a Python environment with `mlx-lm`, `pyzmq`, `msgpack`, and `websockets`.

Set these paths for the automated checks:

```sh
export UMMINFER_BINARY="$PWD/zig-out/bin/umm-infer"
export MLX_MODEL=/absolute/path/to/local/mlx-model
```

Use a local MLX-LM model with a text chat template. The ASR checks generate a
new deterministic TinyCTC model in a temporary directory. TinyCTC checks do not
measure recognition quality.

Build the runtime:

```sh
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew \
  -Dcurl-prefix=/opt/homebrew/opt/curl
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

Expected result: the build exits 0, the contract suite passes, and the binary
exists. The suite drives the real Coordinator over IPC sockets with bound
PUSH/PULL stage sockets and no model worker. It covers the ASR -> VLM event
contract: ordering, aggregation, cancellation, stage failures, backpressure,
and duplicate or out-of-order handling. It does not load a model.

## 2. Complete the primary user journey

`tests/client_smoke.py` was removed in `ab51cf5`. The real end-to-end journey is
the committed case reproduction:

```sh
SSH_OPTS="-o ControlPath=$HOME/.ssh/cm/tn-h20b" sh docs/cases/0001_repro.sh
```

Case record, per-leg timings and hashes:
[docs/cases/0001-end-to-end-native.md](cases/0001-end-to-end-native.md).

## 3. Exercise input and client boundaries

```sh
python3 -I tests/client_boundary_e2e.py "$UMMINFER_BINARY" --model "$MLX_MODEL"
python3 -I tests/realtime_client_contract_e2e.py "$UMMINFER_BINARY"
```

Expected results:

- Invalid CLI options return usage exit 2. Invalid realtime `--chunk-ms` values
  are rejected before any network connection.
- Invalid chat and transcription URLs fail before inference.
- Invalid WAV structures and odd PCM16 byte counts fail as bad requests.
- Unsupported server fields retain the server's error reason.
- A request whose generation metadata the real TinyCTC stage omits after a real
  forward returns HTTP 500 with the stage's reason. The transcribe CLI exits 1
  with empty stdout, and the real stage hook emits its omission marker once.
- A real HTTP 400 that echoes the request model under the bearer secret is
  redacted to `unsupported model: [key]`, and the secret does not appear.
- The public `ASRClient` maps a real TinyCTC short-frame failure to
  `kServer`/500/retryable with the stage reason, and a real invalid entry
  endpoint to `kServer`/503/retryable with the `zmq_connect` reason. After the
  500 case, valid audio still transcribes on the same server.
- Final realtime segments do not invoke the public `on_partial` callback.
- Missing optional generation metadata leaves token count null, length-limited
  state null, and finish reason empty in the public result.
- A large append is split into real messages of 1,048,575 and 151,519 bytes;
  the reconstructed PCM hash matches the original input.
- Active close and SIGKILL produce abort events, return `in_flight` to zero, and
  allow a following real request to succeed.
- Real continuation fragments are reassembled.
- A 20-round normal-close regression opens a real WebSocket session through
  the wire proxy and closes it. Every round must record both Close frames and
  no forwarded-error entry. ENOTCONN/EPIPE count as teardown; ECONNRESET counts
  only after an observed Close frame or EOF. Other transport errors and
  protocol assertions remain test failures. The local classifier test
  `tests/wire_proxy_teardown_test.py` was removed in `ab51cf5`; the real
  20-round regression above is the remaining evidence.
- `--ping-interval-ms` makes the real CLI send PING control frames while it
  streams audio. The wire proxy observes those frames and injects none. The
  stream keeps its order, ends with exactly one terminal event, and later
  requests succeed.
- The real-backend realtime live harness
  `tests/realtime_streaming_live.py` was removed in `7542ba0` with the retired
  SGLang backends. Partials, PING/PONG interleaving, terminal count and
  cancellation now have no live-backend test; the local realtime contract E2Es
  above remain.

Check each script's argument help if its local stage defaults differ from your
checkout. Do not use `--skip-real` when validating the end-to-end journey.

### 3a. H20B realtime live run

The H20B container ships libcurl 8.5.0 without the `ws` protocol, so the
realtime client refused to start. Build libcurl 8.5.0 from source with
`--enable-websockets` into a dedicated prefix, then pass that prefix to the
build. Do not replace the system library.

```sh
PREFIX=/path/to/curl-prefix
curl -fsSLO http://archive.ubuntu.com/ubuntu/pool/main/c/curl/curl_8.5.0.orig.tar.gz
tar xzf curl_8.5.0.orig.tar.gz && cd curl-8.5.0
./configure --prefix="$PREFIX" --enable-websockets \
  --with-openssl --with-zlib --disable-ldap --disable-ldaps --disable-manual
make -j8 && make install
cd /path/to/umminfer
zig build -Doptimize=ReleaseFast -Dcurl-prefix="$PREFIX" \
  -Ddeps-prefix=/path/to/deps
```

`curl-config --protocols` must list `ws` and `wss`. The harness
`tests/realtime_streaming_live.py` was removed in `7542ba0`; the build recipe
below remains valid and the results are a historical record. On H20B 2026-11-14
a dedicated `qwen3-asr` backend ran on GPU 5, port 31005. It needed
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` because the installed driver
535.161.08 is older than the container CUDA 13 build.

The harness that produced these results, `tests/realtime_streaming_live.py`,
was removed in `7542ba0` with the retired SGLang backends. The table below is a
historical record; none of it is reproducible on the current source:

| Check | Result |
| --- | --- |
| Startup | `PASS real_startup` with dedicated free ports |
| Incremental partials | `PASS real_incremental_partials partials=7`, final text `甚至出现交易几乎停滞的情况。`, one terminal |
| Control frames | `PASS real_control_frame_interleaving pings=419 pongs=188`, PING/PONG interleaved with data |
| Cancel | `PASS real_cancel_disconnect in_flight=0` |
| Backend abort | `PASS real_gpu_abort_confirmed` with `SGLANG_ABORT_REQUEST`, `SGLANG_ABORT_CONFIRMED`, `finish=abort` |
| Recovery | `PASS real_request_after_cancel`; a later stream completed |

Evidence is in `/tmp/umminfer-phase6-realtime-h20b-evidence/`: build command and
`curl-config` output, libcurl and binary SHA256, raw frame JSON with payload
hashes, stage abort lines, capped server, stage and backend logs, and
`SHA256SUMS`.

## 4. Exercise network retry and stall behavior

```sh
python3 -I tests/client_retry_e2e.py "$UMMINFER_BINARY" --model "$MLX_MODEL"
```

The test inserts a TCP byte relay between the real client and real server. The
relay only forwards, delays, holds, truncates, resets, or observes actual bytes.
It never writes HTTP, SSE, WebSocket, or control-frame payloads of its own.

Both relays ignore connections that are not part of the protocol under test. A
localhost port scanner or health probe sends `GET /` and must not consume a
relay plan slot or bind the WebSocket session. The local pinning test
`tests/proxy_isolation_test.py` was removed in `ab51cf5`; the relay isolation
behaviour is now covered only by the real byte-tap runs above.

Expected result: retries occur only where the client contract allows them.
Emitted stream content is not silently restarted. Stall and total timeouts
produce client errors. Subsequent requests and process cleanup pass.

These network cases complement the primary smoke. They do not prove immediate
interruption of an MLX forward call.

## 5. Check the MLX backend separately

`tests/mlx_smoke.py` was removed in `ab51cf5`. The deterministic MLX stage
contract that remains is `tests/mlx_stage_result_test.py`; run it after
`zig build test`:

```sh
python3 -I tests/mlx_stage_result_test.py
```

It loads no model and starts no server. It checks that the stage forwards the
prompt and completion counts MLX-LM reported and never invents a count it did
not observe. Real-model MLX latency measurement remains in `tests/mlx_perf.py`
(see `docs/mlx.md`).

## 5b. Mac unified low-resource profile

Run the unified ASR -> text path with the local TinyCTC test ASR stage and a
real MLX text stage:

```sh
python3 -I tests/mac_unified_validation.py \
  --model "$MLX_MODEL" --evidence-dir /tmp/umminfer-phase5-mac-evidence
```

See `docs/mac-unified.md` for scope, recorded results, evidence layout, and
limits. TinyCTC is an inference test model; it is not recognition-quality
evidence.

## 5c. Unified ASR partials and VLM GPU abort

RETIRED. `tests/unified_partial_abort_live.py` was removed in `ab51cf5`, and the
SGLang backends it drove were retired in `7542ba0`. The paragraph below is a
historical record of the 2026-11-13 H20B run; it cannot be reproduced on the
current source. The deterministic partial and abort contracts remain in
`zig build unified-test` (section 5d).

`tests/unified_partial_abort_live.py` drove the real Qwen3-ASR and Qwen3-Omni
workers through public HTTP/SSE. It required `UMMINFER_LIVE_OPT_IN=1`, the
ReleaseFast binary, both SGLang backends, and real audio.

Expected result, recorded on H20B 2026-11-13: a default request carried one
`asr_final` and no `asr_partial`; `stream_options.include_asr_partials` added
ordered `asr_partial` deltas before one `asr_final`, then `vlm_delta`, with
gap-free `seq` and one terminal; a cancel during the VLM stream yielded exactly
one `cancelled`, no later delta, `/abort` 200, `in_flight=0`, and the Omni
worker logged `SGLANG_ABORT_CONFIRMED ... finish=abort`; the next request
completed. `tests/sglang_abort_probe.py` (also removed) recorded the
backend-only confirmation: `/abort_request` 200 and a raw SSE
`finish_reason="abort"` frame.

## 5d. Optional multi-turn session history

The deterministic local contract is the C++ unified test binary, which uses
bound PUSH/PULL sockets and no model worker:

```sh
zig build unified-test -Ddeps-prefix=/opt/homebrew \
  -Dcurl-prefix=/opt/homebrew/opt/curl
```

It covers history ordering, session isolation, independent request id/seq/
terminal, concurrency, cancel/error exclusion, per-session turn and character
eviction, the global session cap, default-off behavior at both gates, the
non-unified and disabled-engine HTTP 400 responses, and history loss across an
engine restart. These are contract tests, not real-model validation.

Real H20B multi-turn validation was recorded on 2026-11-14 with
`tests/unified_remote_image_history_live.py`. That harness was removed in
`ab51cf5` with the SGLang backends in `7542ba0`, so the description and command
below are a historical record, not a runnable test. It started the real server
twice: once without `--session-history` for the default gate, then with the
switch and a fresh worker pair. It served `man_ironing.png` from a local HTTP
server and drove the default gate, two successful turns, a cancel, a failed VLM
leg, a follow-up turn, and a request with no opt-in. One extra request without
the image checked that the image answer was not generic.

`asr_en_16k.wav` is the repository recording converted to the contract format
with `ffmpeg -i asr_en.wav -ar 16000 -ac 1 -sample_fmt s16 -c:a pcm_s16le`;
the original is 48 kHz 24-bit and the server correctly rejects it.

Expected result, recorded three times on H20B:

| Check | Expected |
| --- | --- |
| Remote image URL | Default server: complete; the Omni log holds the `http://` URL; the local server records one GET with the input SHA256; the answer names the man and the ironing/garment; the same request without the image answers that no image was present |
| Default gate | Server without `--session-history` returns HTTP 400 `session_history is not enabled on this server` for the opt-in |
| Turn 1 | Complete, distinct request id, gap-free `seq`, one terminal, prompt has no prior turn |
| Turn 2 | Prompt is `[user: turn-1 transcript, assistant: turn-1 answer, user: current]`; distinct request id; gap-free `seq`; one terminal |
| Cancel | Exactly one `cancelled` terminal; `/abort` 200; the worker logs `SGLANG_ABORT_CONFIRMED rid=<id>:vlm@... finish=abort`; the prompt had the committed turn |
| Error | Exactly one `error` terminal for the failed VLM leg after a valid ASR transcript |
| Turn 3 | Complete; contains the committed turns and neither the cancelled nor the failed transcript |
| History off | Request without `session_history` on the history-enabled server has one prompt message and no prior turn |

The worker prompt log, raw SSE frames, health responses, image access log, input
SHA256 values, commands, and `SHA256SUMS` are in
`/tmp/umminfer-phase5-final-h20b-evidence/live/`.

## 5e. Backend decode stop: scheduler, SSE, and GPU hardware evidence

RETIRED. Every harness named in this section (`tests/sglang_abort_probe.py`,
`tests/gpu_abort_evidence.py`, `tests/gpu_decode_stop_evidence.sh`) was removed
in `ab51cf5`, and the SGLang backends were retired in `7542ba0`. The section is
kept as a historical record of the 2026-11-14 measurement. None of its commands
run on the current source.

Three distinct evidence levels were recorded for one abort. They answered
different questions and are not interchangeable. Before 2026-11-14 only levels
1 and 2 existed; level 3 was the open `NOT_COVERED` gap closed by the run below.

1. Scheduler abort. `tests/sglang_abort_probe.py` sampled the backend
   scheduler counter `/v1/loads` before, during and after the request, then
   posted `/abort_request`. Recorded on H20B 2026-11-14 in 10 consecutive runs:
   `/abort_request` 200; `num_running_reqs` went 0 -> 1 -> 0;
   `counter_decode_stopped=true`. This was the scheduler's view of the running
   batch, not a GPU hardware counter.

2. SSE abort. The generate stream ended with `finish_reason="abort"`. This is a
   protocol frame. It proves the backend told the client the request ended, not
   that the GPU stopped executing kernels.

3. GPU hardware stop. `tests/gpu_abort_evidence.py` plus
   `tests/gpu_decode_stop_evidence.sh` ran on a dedicated GPU 2 against an
   independent SGLang instance (the shared GPU 0/1 backends were never
   restarted). Two independent hardware instruments recorded the stop, each with
   request id and wall-clock markers:

   - CUPTI kernel timeline. SGLang's own `torch.profiler` (Kineto) subscribes
     to the CUPTI Activity API inside the scheduler process while the probe
     sends one streaming request and aborts it. A monotonic-clock calibration
     maps the probe markers onto the trace. The 2026-11-14 trace records 26
     `step[DECODE bs=1]` spans and 11,573 CUDA kernels before the abort
     (59.85 ms GPU-active), then a bounded scheduler teardown of 1,754 kernels
     (7.97 ms) between the abort POST and the abort frame, and then zero CUDA
     kernels for a 2.83 s idle window. The teardown is bounded and its timing
     is read directly from the trace: two `step[DECODE bs=1]` spans start
     about 64.333 ms and 64.392 ms after the abort POST and both end before
     the abort frame (which arrives 129.382 ms after the POST). These are
     steps the scheduler still launched after the abort arrived, so the trace
     must not be read as "no new decode step starts after the abort POST".
     What stops is the sustained decode stream: every teardown span finishes
     before the abort frame, and no CUDA kernel runs in the window that starts
     0.2 s after the abort frame.
   - DCGM hardware counters. `dcgmi dmon -i 2 -e 1001,1002,1004,1005`
     (graphics/SM/tensor/DRAM activity) samples at 100 ms during a second
     probe. Decode-phase `sm_active` reaches 0.033; after a 0.5 s teardown
     grace window, 55 post-abort samples have `sm_active` max 0.0 and mean
     0.0.

   Run:

   ```sh
   tests/gpu_decode_stop_evidence.sh <repo-checkout-dir>   # removed in ab51cf5
   ```

   The verdict gates are in the analyzer and covered by its `selfcheck`
   subcommand. Raw traces, probes, calibration, DCGM rows, commands,
   environment, the failed nsys attempts with an explanation, and
   `SHA256SUMS` are under `/tmp/umminfer-phase6-gpu-evidence/` (not a
   committed artifact). Text logs are capped at 64 KiB.

### Why not Nsight Systems

External `nsys profile -t cuda,nvtx` was tried first with both the image
default Nsight Systems 2023.4.4 and a user-space 2026.4.1 (the 2023.4 build
predates CUDA 13). Both capture model-load kernels and then stop delivering
every CUPTI event, kernel and runtime API alike, once SGLang's
multiprocessing-spawn scheduler enters its steady-state event loop at about
36.6 s. The abort happens later, so those traces contain no decode kernel. A
minimal spawn + sleep(45) + kernel reproducer traces fully under the same
nsys, which localizes the failure to the SGLang scheduler runtime under
external CUPTI injection on the CUDA 13 forward-compat driver (580 user-state
libcuda on a 535 kernel driver), not to delayed CUDA use or tool age. The two
nsys reports are retained under `nsys-failed/`. Kineto runs inside the
scheduler process and is not affected.

Failure injection inside real model compute is measured in section 5f with an
independent SGLang copy on GPU 3. SGLang 0.5.15's only relevant switch is
`enable_nan_detection`, off by default; its sampler replaces NaN logits with
-1e5 or raises when `crash_on_warnings()` is set
(`python/sglang/srt/layers/sampler.py`). It is a detector, not an injector, so
the copy adds a decode-time activation corruption and lets the real sampler
detect it. The shared backends on GPUs 0 and 1 stay unmodified.

## 5f. Real model-compute NaN failure, recorded on 2026-11-14

RETIRED. `tests/unified_compute_failure_live.py` and
`tests/sglang_nan_injection.py` were removed in `ab51cf5` with the SGLang
backends in `7542ba0`. This section is a historical record; its command does not
run on the current source.

`tests/unified_compute_failure_live.py` drove one fault leg through the public
unified HTTP/SSE path. `tests/sglang_nan_injection.py` patched a copy of
SGLang 0.5.15 (`/sgl-workspace/sglang`) at
`srt/layers/logits_processor.py`. On the first decode forward after the trigger
file exists, the copy multiplies the final hidden states by NaN on the GPU, so
the real LM-head matmul produces NaN logits. The copy runs on GPU 3 with
`--enable-nan-detection`, `SGLANG_IS_IN_CI=1` (which makes `crash_on_warnings()`
raise), and `--disable-cuda-graph`: CUDA graph replay skips the Python logits
path.

The server is the ReleaseFast binary; both workers are the committed Python
scripts. All three come from c8ce8a2. The ASR backend is port 43100 for the ASR
leg; the Omni backend is port 43101 for the VLM leg. The other leg uses the unmodified shared backend on
31000/31001.

The recorded command was:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/unified_compute_failure_live.py \
  --leg asr --backend-port 43100 --other-backend http://127.0.0.1:31001 \
  --server ./zig-out/bin/umm-infer --repo . --audio asr_zh.wav \
  --image man_ironing.png --evidence-dir /tmp/phase6-asr
```

| Check | ASR leg | VLM leg |
| --- | --- | --- |
| Injection point | `(1, 1024) bfloat16 cuda:0` | `(1, 2048) bfloat16 cuda:0` |
| SGLang detector | `Detected errors during sampling! NaN in the logits.` then `ValueError` | same |
| Scheduler | `Scheduler hit an exception` traceback at `sampler.py:89` | same |
| Backend exit | exited on its own after the traceback | same |
| Public terminal | exactly one `error`, `stage=asr`, seq 0 | `asr_final` seq 0, then one `error`, `stage=vlm`, seq 1 |
| Success events | no `asr_final`, `asr_partial`, `vlm_delta`, `complete` | no `vlm_delta`, no `complete` |
| `in_flight` after fault | 0 | 0 |
| Recovery | restart the same copy without a trigger; one `complete`; `in_flight=0` | same |

The error frame is `{"error":{"code":500,"message":"Remote end closed connection
without response","type":"generation_error"}}`. The public server stayed up
across both legs. Only the fault backend restarted for recovery.

All checks pass: 16 in the ASR leg and 17 in the VLM leg (the VLM leg also
asserts that the healthy ASR final is present). Each `result.json` records
`passed=true`.

Evidence: `/tmp/umminfer-phase6-compute-failure-evidence/` with per-leg
`result.json`, raw SSE for baseline/fault/recovery, capped backend and worker
logs (each at most 64 KiB), `environment.txt`, and `SHA256SUMS`. The copy's
patch is `logits_processor.patch`; the injection tool is committed as
`tests/sglang_nan_injection.py`. The evidence directory is not a committed
artifact.

This covers a NaN produced inside the real decode path and detected by SGLang's
own sampler. It does not cover a CUDA hardware fault, a NaN in a tensor that
sampling never reads, or a fault that leaves the backend alive.

## 6. Manual smoke during development

Start the text or audio processes as shown in README. Then check:

```sh
curl --fail-with-body http://127.0.0.1:30000/health
curl --fail-with-body http://127.0.0.1:30000/ready

curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to three."}],"max_tokens":32,"stream":true}'

./zig-out/bin/umm-infer transcribe --url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --max-new-tokens 64

./zig-out/bin/umm-infer realtime --url http://127.0.0.1:31001 --final-url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --live-result --max-new-tokens 64
```

Expected result: readiness is 200 only after the model stage is ready. SSE
contains incremental deltas and a terminating done event. Both audio commands
exit 0 and print the real TinyCTC output.

For authentication, configure the server key through its `serve --api-key`
option in a protected local test environment. Pass the client key only through
`UMMINFER_API_KEY` or `OPENAI_API_KEY`. Repeat transcription and realtime with
no key, a wrong key, and the correct key. Check the server and client logs for
accidental key disclosure.

Stop each owned server and stage with SIGTERM. Confirm that later launches can
bind fresh IPC endpoints and that no test-owned process remains.

## 7. Future model acceptance gates

These gates apply to the model support plan in `roadmap.md`. They are not
current capabilities. A passing HTTP status alone never accepts a model.

Every accepted case must use this real path:

```text
real CLI or public API -> ummInfer server
  -> real backend with a local real model, or authorized official API with a real model
  -> decoded user artifact
```

A direct SGLang run that bypasses ummInfer is supporting backend evidence, not
final acceptance. An unimplemented integration remains `PLANNED` or
`IN_PROGRESS — integration planning`.

Do not use a fake server, fake runtime, fake backend, handwritten model
response, or generated artifact in place of model output. Record the exact
model id and revision, local weights or API endpoint type, backend version,
precision, quantization method, input and output parameters, random seed where
applicable, and artifact SHA256. Record wall time, output duration or count,
peak accelerator memory, host RAM, and resource limits. Mark an unexecuted
case `NOT_COVERED` with the missing hardware, weights, backend, or API access.

### Qwen3.5-Omni

**PLANNED.** Only the authorized official API is planned. The verified model
IDs are `qwen3.5-omni-plus` and `qwen3.5-omni-flash`. They accept text, image,
audio, and video input and return text or streaming speech. Do not claim image
or video generation. Authorized account/API access is unverified and no hosted
execution has been accepted. Until access exists, every API execution case is
`NOT_COVERED`.

After access is verified, exercise each declared input modality through the
complete CLI/public API-to-ummInfer-to-official-API path. Record the provider
model identifier without exposing credentials. Check the returned text. Decode
the streaming speech and check its format, duration, channel and sample layout,
and SHA256.

### MiniMaxAI/MiniMax-H3

**IN_PROGRESS — integration planning.** Establish the full-stack BF16 baseline
on H20B for high-precision acceptance and quality/resource comparisons. Include
DiT, complete text encoders, VAE, activations, offload, and host RAM. Official
Hugging Face metadata for the original H3 FL2VA partition is approximately
144.05 GB. This disk size is not a VRAM requirement.

Cover text, image, video, and audio inputs separately and in supported
combinations. Record the actual input files, hashes, model precision, and all
generation parameters. Decode the real 768p video output and native 32 kHz
stereo audio output. Check video frame decoding, dimensions, frame count and
timing; check audio decoding, sample rate, channel count, sample format and
duration. Save both artifact hashes and the parameters needed to reproduce the
run.

The local open H3-Base acceptance plan covers only the 768p path. 2K requires
hosted Context-IR/Regenerate-2K modules that are not fully open. Do not claim
fully local 2K.

There is no official quantized checkpoint. Official weights with runtime online
quantization are distinct from third-party converted or pruned quantized
packages. Verify a third-party package's revision, license, tensor completeness,
hashes, and real outputs. Do not call it official MiniMax quantization. A trusted
quantized run may execute before the H20B baseline, but it is a separate
precision configuration. Compare its real outputs and resource use with the
BF16 baseline when H20B is available. Without a trusted artifact or compatible
hardware, mark the case `BLOCKED` or `NOT_COVERED`.

There is no H3-specific Mac validation, memory guarantee, or performance
commitment. Generic MPS or offload examples do not establish model acceptance.
Limit Mac execution to low-resource real-model tests. Unsupported or unrun
modalities remain `NOT_COVERED`.

### MiniMaxAI/MiniMax-H3-Image

**IN_PROGRESS — integration planning.** Use the same full-stack BF16 baseline
scope as H3: DiT, complete text encoders, VAE, activations, offload, and host
RAM. The H3-Image selective layout for SGLang/vLLM is approximately 135.38 GB.
This disk size is not a VRAM requirement.

Cover text-to-image generation, editing with one reference image, and editing
with 16 reference images. Record each input image hash and the applicable size,
count, seed, and generation parameters. Decode every returned image with an
actual image decoder. Check its format, pixel dimensions, color mode, and full
artifact SHA256. Do not accept a JSON success field or HTTP 200 as an image
result.

There is no official quantized checkpoint. Apply H3's distinction between
runtime online quantization and third-party packages, including all verification
and `BLOCKED`/`NOT_COVERED` rules. The H3 Mac limits apply unchanged: there is
no H3-Image-specific Mac validation, memory guarantee, or performance
commitment, and execution stays limited to low-resource real-model tests. Any
editing, reference-count, or output-format case that was not executed remains
`NOT_COVERED`.

## 8. CUDA build, IPC, sanitizer and stress on H20B

RETIRED. `tests/cuda_ipc_test.cpp`, `tests/cuda_pipeline_test.cpp`,
`tests/cuda_sanitize.py`, and `tests/cuda_stress.py` were removed in `ab51cf5`,
and the `cuda-stress` build step no longer exists. The measurement below is a
historical record of the 2026-11-14 H20B run.

The only CUDA checks that remain on the current source are the build and the
relay benchmark:

```sh
zig build -Doptimize=ReleaseFast -Dcuda -Dcuda-prefix=/usr/local/cuda \
  -Ddeps-prefix=/path/to/deps
zig build cuda-bench -Dcuda -Dcuda-prefix=/usr/local/cuda \
  -Ddeps-prefix=/path/to/deps -Doptimize=ReleaseFast
```

`-Ddeps-prefix` must contain the pinned `httplib.h`, `nlohmann/`, and the
`libzmq`, `libmsgpackc`, `libcurl`, and `libwebsockets` libraries. The build
puts that prefix before distro headers.

Historical results, recorded on H20B 2026-11-14 on GPUs 2 and 3:

| Check | Expected |
| --- | --- |
| Build | Exit 0; `cuda_ipc_test`, `cuda_pipeline_test`, `cuda_bench`, and `umm-infer` installed |
| CUDA IPC | Both test binaries exit 0 and print `all checks passed` |
| Sanitizer | Exit 0. The clean `ipc` and `pipeline` runs report `ERROR SUMMARY: 0 error`. The default-API `ipc` run exits 99 and reports `cudaIpcOpenMemHandle`, because test [5] opens a corrupted handle in a child process on purpose |
| Stress | `cuda_stress: 10/10 rounds passed`; the stress environment also adds the 1 MiB and slot-limit-minus-one cases |

Evidence is in `/tmp/umminfer-phase5-cuda-stress-evidence/`; it is not a
committed artifact. This run does not cover CUDA relay under real model load,
zero-copy device-tensor transfer, or multi-host IPC.

## NOT_COVERED

- A first-party HTTP 429 with `Retry-After`: this server does not emit one.
  Synthetic protocol responses are not validation evidence.
- `other_501_502_public_client`: this binary snapshot is not evaluated for a
  real public-client 501 or 502 response, so both remain unverified.
- Immediate cancellation inside MLX prefill or a model call.
- CUDA execution and CUDA IPC on this Apple Silicon environment.
- Production ASR accuracy and generalization: TinyCTC is an inference test model.
- Full OpenAI Realtime compatibility: that API is not implemented.
