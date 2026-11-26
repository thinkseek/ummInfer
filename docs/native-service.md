# SM90 resident service

This page is the runbook for the resident ummInfer service on SM90: one C++
server and one native C++/CUDA stage per model, all inside the SM90 GPU
container. No Python model runtime, PyTorch or SGLang is on the serving path.
`python/` appears only as the optional process supervisor and as legacy code.

## Topology

| Piece | Process | GPU |
| --- | --- | --- |
| ummInfer server | `bin/umm-infer serve` on `<container-ip>:43301` (`$UMM_NATIVE_HOST:43301`, with `--session-history --web-root <deploy-root>/web`) | none |
| ASR stage | `bin/umm-native-stage-asr` (`Qwen3-ASR-1.7B`) | 2 |
| VLM stage | `bin/umm-native-stage-qwen38` (default, `--batch-size 8`) | 3 |
| VLM stage, optional | `bin/umm-native-stage-omni` (rollback option) | 3 |
| Image stage | `python/umminfer_flux2_stage.py` (`FLUX.2-klein-4B`, Apache-2.0, t2i + i2i edit) | 4 |
| Image stage, optional | `bin/umm-native-stage-image` (`Qwen-Image-2512`, rollback option) | 4 |
| TTS stage | `bin/umm-native-stage-tts` (`Qwen3-TTS-12Hz-1.7B-CustomVoice`; `1.7B-Base` drop-in optional) | 5 |
| Video stage | `bin/umm-native-stage-video` (`Wan2.2-TI2V-5B`, pure native C++/CUDA UniPC-BH2 + streaming 3D-VAE) | 6 |
| Supervisor | `python3 -I python/umminfer_service.py --config <deploy-root>/config.json` | none |

Exactly one VLM stage runs at a time (`vlm_stage` selects `qwen38` or `omni` on
GPU 3), and one image stage holds GPU 4 (`flux2` or native `image`). GPUs 0, 1,
and 7 are not part of this resident service.

Stages talk to the server over ZeroMQ and MessagePack, with payloads in POSIX
shared memory. Endpoints live under `ipc_dir` (`/tmp/umminfer-native-ipc` by
default): `asr`, `qwen38` (or `omni`), `image`, `tts`, `video`, plus the shared
`done`/`completion` and `abort` edges.

## Address and exposure

The server binds `127.0.0.1` by default, or the container's private bridge
address (`<container-ip>`, configured via `UMM_NATIVE_HOST` or `"host"` in
`service/native-service.json`). It does not bind `0.0.0.0`, it does not bind the
host public interface, and no Docker port is published. The supervisor refuses
a configuration whose `host` is `0.0.0.0` or `::`.

A Mac cannot reach `<container-ip>` directly. SSH `-L` on the Linux SM90 GPU host
(substituting your real `<container-ip>` and `<gpu-host>`) reaches the
container bridge address:

```sh
/usr/bin/ssh -N -L 43301:<container-ip>:43301 <gpu-host>
```

`127.0.0.1:43301` on the Mac then reaches the container service through the SM90
host.

## Install

Two build systems produce the artifact set. `zig build` builds only the server;
`native/stage_cli` builds the five stage binaries. Neither builds the other.
Set `CURL_PREFIX`, `DEPS_PREFIX`, and `UMM_NATIVE_ROOT` to your local prefixes:

```sh
zig build -Doptimize=ReleaseFast \
  -Dcurl-prefix="${CURL_PREFIX:-/opt/umminfer/curl-prefix}" \
  -Ddeps-prefix="${DEPS_PREFIX:-/opt/umminfer/deps}"

cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 16

install -m755 zig-out/bin/umm-infer build/umm-native-stage-* "${UMM_NATIVE_ROOT:-/opt/umminfer/native-service}/bin/"
```

`service/native-service.sh` expects those binaries in `$UMM_NATIVE_ROOT/bin`
(`/opt/umminfer/native-service/bin` by default). The supervisor reads
`service/native-service.json`, which names `binary`, `stage_binary_dir`, `host`,
`env.LD_LIBRARY_PATH`, and the four `native_stages` (edit those paths and
`host` to match your container before starting).

Install the config next to the supervisor:

```sh
install -m644 service/native-service.json "${UMM_NATIVE_ROOT:-/opt/umminfer/native-service}/config.json"
```

Record the identity of the deployed bytes:

```sh
sha256sum "${UMM_NATIVE_ROOT:-/opt/umminfer/native-service}"/bin/umm-infer \
  "${UMM_NATIVE_ROOT:-/opt/umminfer/native-service}"/bin/umm-native-stage-*
git rev-parse HEAD
```

## Opt-in request-native allocator telemetry

A chat request may explicitly ask for native allocator peaks without enabling a
process-wide profiler:

```json
{
  "model": "qwen3-native",
  "stream": true,
  "messages": [{"role": "user", "content": "hello"}],
  "telemetry": {"native_allocator": true}
}
```

The option is propagated as the Stage request parameter
`native_allocator_telemetry=true`. The terminal unified event (or the non-SSE
JSON body) then carries one bounded `telemetry` envelope. A covered stage entry
has this shape:

```json
{
  "coverage": "COVERED",
  "scope": "request-native-cuda-runtime",
  "request_id": "http-1",
  "stage": "vlm",
  "native_stage": "qwen38",
  "backend": "qwen38",
  "device_index": 0,
  "lifecycle_status": "completed",
  "execution": {"mode": "serial", "batch_capacity": 1},
  "epoch": {
    "id": 17,
    "reset": true,
    "semantics": "new_allocation_ownership_epoch"
  },
  "allocated_bytes_peak": 123456,
  "reserved_bytes_peak": 123456,
  "allocated_bytes_final": 4096,
  "reserved_bytes_final": 4096,
  "sample_coverage": {
    "allocation_events": 4,
    "free_events": 3,
    "terminal_snapshots": 1
  }
}
```

`allocated` and `reserved` are equal because these are direct CUDA runtime
blocks, not a caching-allocator reservation estimate. Peaks include only blocks
**first allocated and owned by that request epoch**. Model weights and other
resident allocations made before the request are excluded, never subtracted from
a device or process baseline. CUDA context/driver allocations and allocations
inside external libraries are outside this scope. A nonzero final value can be a
retained cache or snapshot owned by the completed request; it is not, by itself,
a leak verdict.

An epoch id is monotonically increasing within a native stage process. `reset`
means a new allocation-ownership log, **not** `cudaDeviceReset`, a global
high-water reset, or a reset of another request. A late free updates its original
owner, including after that owner's terminal snapshot, and cannot contaminate a
later epoch. Overlapping supposedly serial scopes poison both records to
`NOT_COVERED` rather than divide or reset shared counters.

Current coverage is deliberately narrow:

| Path | Request allocator result |
| --- | --- |
| Qwen3.8 native stage, `--batch-size 1` | Instrumented explicit native `cudaMalloc`/`cudaFree`; `COVERED` when attribution invariants hold |
| Wan2.2 native stage, `cuda` or `cuda-scaffold`, serial | Instrumented explicit native runtime allocations in UMT5, DiT, VAE, and scaffold paths |
| Qwen3.8 `--batch-size > 1` | `NOT_COVERED`, reason `batched_request_attribution_unavailable`; configured capacity and request identity remain attached |
| Concurrent/shared execution or overlapping serial epochs | `NOT_COVERED`; no peer subtraction or shared-counter split |
| Native ASR, Omni, Image, and TTS stages | `NOT_COVERED`, `backend_not_instrumented` |
| Video `contract-stub` | `NOT_COVERED`, `contract_stub_has_no_cuda_allocator` |
| Python/Diffusers/FLUX stages and fixed external proxies | No truthful native epoch is inferred; absent/missing terminal data is normalized to `NOT_COVERED` |

Success and failure completions carry their native terminal snapshot. The
scheduler also retains a covered cancelled snapshot when compute observes the
abort. The Coordinator's public cancellation is intentionally immediate: if it
wins before the native terminal arrives, the public envelope instead records
`cancelled` plus `stage_terminal_snapshot_missing`; late data is not used to
rewrite an already sent terminal event.

No request content, model path, IPC path, or free-form backend diagnostic is
accepted into the public projection. Identifiers and reasons are allow-listed,
unknown fields are dropped, malformed byte/counter/epoch values become fixed
`NOT_COVERED` records, and the envelope is capped at 64 stage entries. The native
pointer table contains only live blocks owned by active or retained request
epochs; frees remove entries. There are no per-frame samples or telemetry log
files. Allocation/free locking exists only for opted-in covered serial epochs.
Without the opt-in, request parameters, result payloads, SSE, and JSON remain
byte-identical and no telemetry field is emitted.

These peaks are **not** whole-device VRAM. `/metrics` NVML gauges and controlled
`measurement.resources` remain separate `whole-device-used-memory` observations,
including resident weights and other processes. The request envelope's
`whole_device_nvml` value is always `separate_measurement_resources`.

### Required SM90 validation gate

The implementation in this revision has CPU-only host/fake-allocator and
protocol evidence. Real CUDA allocation measurements are **NOT_RUN**. A later
exclusive SM90 window must:

1. build the gateway and Qwen3.8/Wan2.2 stages with CUDA for SM90 and run the
   native CTests; use only the agreed idle GPUs and leave the resident service
   untouched;
2. issue serial Qwen3.8 and Wan2.2 requests with telemetry off/on, verify output
   bytes and default wire parity, then verify nonzero peaks, monotonic distinct
   epoch ids, event counts, and eventual tracked-block cleanup;
3. repeat serial requests with prefix/VAE caches retained, and verify a later
   request neither inherits nor subtracts the earlier resident blocks;
4. verify batch/concurrent and every uninstrumented backend returns its declared
   `NOT_COVERED` reason with null, not zero, peaks;
5. exercise completed, failed, and cancelled requests, checking public lifecycle,
   the documented immediate-cancellation limitation, `in_flight=0`, and device
   cleanup; and
6. run a controlled ablation with `request_memory.enabled=true` and
   `required=true`, requiring `request_scoped_telemetry.coverage=COVERED` while
   independently retaining the 50 ms whole-device NVML samples and hashes.

No real CUDA peak or production-deployment claim is made until that gate passes.

### SM90 validation gate — PASS (2026-11-26)

The real serial gate ran on one isolated loopback stack with ASR, Qwen3.8
`--batch-size 1`, and Wan2.2 CUDA all confined to explicitly idle physical
GPU 0. The resident five-GPU service was not contacted or reconfigured.
`tests/request_telemetry_live.py` was run with `UMMINFER_LIVE_OPT_IN=1`; it
never starts or stops a process. Final NVML showed GPU 0 returned to `3 MiB`
with no compute process.

- Qwen3.8's first covered request reported allocated/reserved peaks of
  **742,129,664 bytes** (covered epochs `1..8, 10`). Warm prefix/workspace
  repeats truthfully reported zero new request-owned allocations rather than
  inherited resident bytes (prefix reuse observed at 12 and 36 tokens).
- Wan2.2 reported **156,241,920 bytes** on both measured serial video requests,
  including the VAE-cache repeat, with 28 allocations and 28 frees per request
  (epochs `1, 2`).
- Telemetry-off/on JSON and SSE protocol projections matched; default responses
  emitted no telemetry fields; Wan off/on MP4 bytes and SHA-256 were identical.
  Nonstream artifact descriptors strip the internal `request_telemetry` key.
- Real completed and failed Qwen requests retained covered terminal records.
  Immediate public cancellation returned `cancelled` with null peaks and
  `stage_terminal_snapshot_missing`. ASR returned `backend_not_instrumented`,
  Qwen `--batch-size 2` returned `batched_request_attribution_unavailable`, and
  Wan `contract-stub` returned `contract_stub_has_no_cuda_allocator`, all with
  null peaks; `in_flight` drained to zero.
- `benchmarks/ablations/request-memory-controlled-sm90.json`
  (`request_memory.enabled=true`, `required=true`, two-arm
  `request_native_backend=(qwen38, wan22)`) passed `--require-controlled` as
  **CONTROLLED_LIVE / COMPLETED** (8/8 measured requests and 12/12 stage entries
  `COVERED`, independent 50 ms whole-device NVML `88,656 MiB` baseline/final and
  `88,806 MiB` Wan peak, **52/52 `SHA256SUMS` PASS**).

Batched/concurrent/shared attribution, native Image/Omni/TTS, Python/external
backends, and production deployment remain `NOT_COVERED` or `NOT_RUN` as
declared above.

## Operate

Without supervision:

```sh
service/native-service.sh start
service/native-service.sh status
service/native-service.sh stop
```

With the supervisor, which restarts unhealthy children:

```sh
CFG=/opt/umminfer/umm-native-service/config.json
python3 -I python/umminfer_service.py --config $CFG start
python3 -I python/umminfer_service.py --config $CFG status
python3 -I python/umminfer_service.py --config $CFG logs --name asr
python3 -I python/umminfer_service.py --config $CFG stop
```

The supervisor `start` forks twice, calls `setsid`, and redirects its own stdio
to `/dev/null`; the command returns after the pid file appears. `status` prints
JSON from `run/state.json`. `logs` prints at most 64 KiB from the tail of one
log.

## Supervisor behavior

- Ownership. Each child is spawned with its PID and kernel start time
  (`/proc/<pid>/stat` field 22 on Linux, `ps -o lstart=` elsewhere). `stop` and
  every signal path compare the current start time with the recorded one before
  signalling. A reused PID, or a hand-edited state file, receives no signal.
- Liveness. The server is probed on `GET /health` every 2 s after an 8 s
  startup grace. Three consecutive failures restart the server.
- Readiness. `/ready` is an admission gate, not a heartbeat: it returns 503
  until the entry and VLM stages have sent `stage_ready`. It never waits for the
  image or TTS legs. A stack that stays unready past `ready_grace_s` (180 s) is
  restarted.
- Restart. A dead child is restarted with backoff `1, 2, 4, 8, 15, 30` s,
  capped at 30 s, reset after 60 s of stable uptime.
- Dependency. Stages are started only after the server answers `/health`, but a
  server restart no longer restarts them. Each stage re-sends an idempotent
  `stage_ready` heartbeat (C++ `Stage::Config::ready_heartbeat_ms`, default
  1000; Python video stage: 1.0 s), so a new Coordinator on the same IPC
  endpoints re-registers the already-running stage and its GPU weights stay
  resident. The supervisor keeps the stage children alive through a server
  exit or failed health check. If `/ready` is still failing `ready_grace_s`
  after the new server is up, the supervisor falls back to restarting the
  stages once. Set `"restart_workers_on_server_restart": true` in the config to
  restore the old restart-everything behaviour.
- Logs. Child stdout and stderr are merged and capped at 65536 bytes each
  (`server.log`, `asr.log`, `qwen38.log`, `image.log`, `tts.log`); the
  supervisor keeps its own `supervisor.log` under the same cap. The cap is a
  rolling tail, not a truncation at process exit.

## Limits

- The supervisor is not registered with systemd or any init. A container
  restart loses the service; run `start` again.
- The container bridge address can change if the container network is
  recreated. The config must then be updated.
- Model revisions stay `UNKNOWN`; the model directories carry no `.git`
  metadata. See
  [performance baseline](performance-baseline.md#model-snapshot-and-provenance).
- `service/native-service.sh` does no supervision: it writes pid files,
  and `stop` signals those pids. Use the supervisor when restart matters.
- Neither launcher cleans up other users' processes. Each signals only the PIDs
  it recorded.
- GPUs 2-5 are hardcoded defaults in the config and in the script. A second
  stack on the same host must override `UMM_NATIVE_GPU_*` and
  `UMM_NATIVE_PORT`; the config file needs the same edit.

## Validation, 2026-11-25 (Wave 7–8 five-stage rollout)

The resident SM90 service on `<container-ip>:43301` was upgraded rollback-safely
(timestamped backups of prior `bin/`, `python/`, and `config.json`) to commit
`c73e3e68` running the full five-stage stack across GPUs 2–6 (`asr` 1.7B,
`qwen38` `--batch-size 8`, `flux2` `FLUX.2-klein-4B`, `tts` 1.7B-CustomVoice,
`video` native CUDA `Wan2.2-TI2V-5B` with streaming 3D-VAE) plus `--session-history`
and `--web-root`:

- Supervisor reached `/ready` 200 with all 5 stages ready in `28 s`, 0 restarts
  across all 6 processes (steady VRAM: GPU 2 `5,105 MiB`, GPU 3 `61,441 MiB`,
  GPU 4 `15,617 MiB`, GPU 5 `6,417 MiB`, GPU 6 `20,861 MiB`; GPUs 0/1/7 `3 MiB`).
- `tests/native_service_live.py` (`UMMINFER_LIVE_OPT_IN=1`): **15/15 PASS**,
  `failures=0` (ASR->VLM, gap-free seq, live `/abort`, recovery, `in_flight=0`).
- Live 5-modality online checks: TTS 1.7B WAV/PCM (`1.73 s`, RMS `2613.4`) ->
  ASR 1.7B (`0.42 s`, exact transcript match); Qwen3.8-27B (`0.48 s`, `seed=42`
  exact parity + 2-turn `session_history` codeword recall); FLUX.2 1024x1024
  t2i (`3.21 s`, `1,523,599 B`) + 512x512 i2i edit (`1.15 s`, `l1_diff=92.06`);
  Wan2.2 native CUDA video 480x320 13f 15 steps with streaming 3D-VAE (`7.74 s`,
  H.264 MP4 `177,034 B`, `ffprobe` verified); Web UI `GET /` `30,476 B`;
  `/health` `in_flight=0`.
- Wave 9 live CLI E2E, gateway hot-upgrade, and 1280x704 77-frame E2E (`6144a621`):
  - Native `umm-infer` CLI verified against `43301` over a loopback tunnel:
    `speech` WAV (`153,644 B`) & PCM (`88,320 B`) -> `transcribe` exact verbatim
    roundtrip; `chat --seed --image-steps` t2i (`363,169 B`) -> `--image-ref`
    FLUX.2 edit (`417,646 B`, `l1_diff=45.47`); `chat --video-out` (`70,144 B`,
    480x320 13f H.264 MP4).
  - Hot-upgraded `/bin/umm-infer` in place (`umm-infer.wave8` backup preserved)
    and sent `SIGTERM` only to the server PID: supervisor respawned the server
    and reached `/ready` 200 in **`3 s`** with **`0` stage restarts** across all
    5 GPU stages (GPUs 2–6 kept resident via `stage_ready` heartbeat), followed
    by live `chat` 200 and `/health` `in_flight=0`.
  - Full end-to-end 1280x704 77-frame native CUDA Wan2.2 (UMT5-XXL + DiT 2-step
    + streaming 3D-VAE + H.264 MP4) via `umm-infer serve` + `umm-native-stage-video`
    + `umm-infer chat --video-out` on free GPU 7: `92.66 s` (`270,505 B` H.264
    MP4, `ffprobe` verified), GPU 7 returned to `3 MiB`.
- Wave 10 live interleaved narrative, global streaming voice, and visual swap (`f0519cc3`):
  - Hot-upgraded `/bin/umm-infer` in place (`umm-infer.wave9` backup preserved)
    and updated `web/index.html`: `/ready` returned 200 in **`3 s`** with **`0`
    stage restarts** across all 5 GPU stages (GPUs 2–6 stayed hot across back-to-back
    Wave 9 and Wave 10 gateway upgrades).
  - `tests/narrative_live_e2e.py --url http://127.0.0.1:43301 --model qwen3-native`:
    TTFT `0.58 s`, global streaming PCM16 TTFA **`0.78 s`** (159 chunks, `4.10 MB`),
    `slot-0` FLUX.2 PNG (`287,400 B`) dispatched mid-stream at `4.05 s` (`2.80 s`
    before VLM finished) and delivered in `1.46 s`, `slot-1` Wan2.2 H.264 MP4
    (`100,541 B`) in `7.96 s`, `clean_prose=true`, live `POST /v1/visual/render`
    PNG -> H.264 MP4 swap (`51,368 B`, `ffprobe` PASS), `/health` `in_flight=0`.
- Wave 11 live Wan2.2-TI2V-5B native CUDA I2V animation (`dba0538b`):
  - Hot-upgraded `/bin/umm-infer`, GPU 6 `/bin/umm-native-stage-video` (preserving
    `.wave10` backups), and `web/index.html`: `/ready` 200 and GPU 6 `ready` in
    `26 s` with **`0` restarts** on GPUs 2, 3, 4, and 5.
  - GPU 7 Diffusers parity: `encode_wan_vae_image_fp32` `cosine=1.000000`,
    `PSNR=134.07 dB` (`217.3 ms`); scalar T2V bit-identical MD5 to prior baseline.
  - Live `tests/narrative_live_e2e.py` on `43301`: animated `slot-0` FLUX.2 PNG
    (`298,032 B`, `0.90 s`) in place via `POST /v1/visual/render` (`i2v=true`,
    `704x480`, `29` frames, `20` steps) -> H.264 MP4 (`83,359 B`, `ffprobe` PASS,
    frame 0/28 visually verified crisp & faithful), `/health` `in_flight=0`.
- Wave 12 live Wan2.2 aspect-preserving center-crop I2V (`b80805c`):
  - Hot-upgraded GPU 6 `/bin/umm-native-stage-video` (preserving `.wave11`
    backup) in `2 s` with `0` restarts on GPUs 2–5; verified `704x480` 29-frame
    I2V from a square FLUX.2 PNG (`84,974 B` H.264 MP4) with zero aspect
    stretching on frames 0 and 28.
- Wave 13 live native video input understanding (`video_url` + Web UI `+ Vid`):
  - Hot-upgraded `/bin/umm-infer`, GPU 3 `/bin/umm-native-stage-qwen38`
    (preserving `.wave12` backups), and `web/index.html`; `/ready` 200 in `36 s`
    with **`0` restarts** on GPUs 2, 4, 5, and 6 (GPU 3 steady `60,733 MiB`,
    GPUs 0/1/7 `3 MiB`).
  - Live `POST /v1/chat/completions` (`model=qwen3-native`) with real `704x480`
    29-frame H.264 MP4 (`84,974 B`) as `video_url` (`data:video/mp4;base64,...`):
    FFmpeg extracts 8 RGB keyframes into 4 temporal-2 slices (`grid_t=4`,
    `384` visual tokens, `533` prompt tokens, `prefill=814 ms`, `22.1 tok/s`,
    `elapsed=6.32 s` SSE / `8.53 s` non-stream), accurately describing the
    glowing red Chinese lantern, misty karst mountains, water reflection, and
    gentle swaying motion.
  - Zero-regression single-image VQA (`2.58 s`) and live `video_url` ->
    interleaved narrative + FLUX.2 image slot + streaming PCM16 voice
    (`ttft=1.78 s`, `ttfa=2.01 s`, `54` PCM chunks, `1` visual slot) both
    passed; `/health` `in_flight=0`.
- Wave 14 live gateway hot-upgrade (`2625e47`):
  - Hot-upgraded `/bin/umm-infer` in place (preserving `.wave13` backup) in
    **`2 s`** with **`0` restarts** across all 5 GPU stages (GPUs 2–6 kept hot),
    carrying Stage B full-duplex `playback_ack` barge-in history continuity
    (real 3-GPU SM90 validation on idle GPUs 0/1/7 recorded in
    `docs/full-duplex.md`), `/ready` 200 and `/health` `in_flight=0`.
- Wave 15 live multi-stage hot-upgrades (`68cf88b`, `ea72f96`, `2dabb35`, `7fe0227`, `a6f88ea`):
  - **GPU 6 `umm-native-stage-video`** (`68cf88b` Tier-1 lossless DiT `kQueryTile=512` +
    persistent GPU workspace/RoPE + 2-entry cross-attn KV cache, then `2dabb35`
    opt-in `--teacache-thresh`, preserving `.wave14` backup): hot-swapped in
    `2–3 s` with `0` restarts on GPUs 2–5; live `POST /v1/visual/render`
    (`704x480` 29f 15 steps `seed=42`) sped up **`24.07 s -> 22.13 s`** steady
    state with **bit-exact identical MP4** (`81,076 B`, `sha256=b371ed5a78a92873`,
    `teacache_thresh=0 teacache_hits=0/30`).
  - **GPU 3 `umm-native-stage-qwen38`** (`ea72f96` cross-turn multimodal prefix
    KV-cache reuse with `media_hash` + `cached_pos_next` + unified `grid_t` pixel
    H2D, preserving `.wave14` backup): hot-swapped in `13 s` with `0` restarts on
    GPUs 2/4/5/6 (steady `60,733 MiB`).
  - **Gateway `/bin/umm-infer` & `web/index.html`** (`a6f88ea` OpenAI multi-turn
    multimodal `messages`, `7fe0227` Web full-duplex Realtime WS + Stage B
    `playback_ack`, preserving `.wave14` backups): hot-swapped in `3 s` with `0`
    restarts across all 5 GPU stages; live 3-turn OpenAI video follow-up
    (`POST /v1/chat/completions` `[user(video_url+text), assistant, user, assistant, user]`)
    verified chained multimodal prefix KV-cache reuse:
    - Turn 1 (cold video): `prompt=522`, `image_tokens=468`, `vision_ms=542.2 ms`,
      `prefill_ms=966.7 ms` (TTFT `1,508.9 ms`).
    - Turn 2 (prefix hit): `prompt=574`, **`prefix_reused=515`**, **`vision_ms=0.0 ms`**,
      `prefill_ms=367.1 ms` (**4.11× faster TTFT**; `8.95×` `169.1 ms` on exact-prefix repeat).
    - Turn 3 (chained hit): `prompt=659`, **`prefix_reused=567`**, **`vision_ms=0.0 ms`**,
      `prefill_ms=412.7 ms`, accurate visual detail recall across all 3 turns.
- Wave 16 live multi-stage performance & North-Star showcase upgrades (`46fc52c`, `b88c58e`, `96f5964`, `ecea4a0`, `3acadce`, `7b87980`):
  - **GPU 6 `umm-native-stage-video`** (`96f5964` SM90 TF32 Tensor Cores in 3D-VAE +
    `ecea4a0` bounded GPU scratch `BufferPool`, `1x1x1` pointwise fast path,
    2D-grid `im2col`, fused `rms_norm+silu`, in-place causal cache, and
    `umt5_ms/dit_ms/vae_ms` telemetry, preserving `.wave15` backup): hot-swapped
    in `3 s` with `0` restarts on GPUs 2–5. Measured live on SM90 (`704x480` 29f
    15 steps `seed=42`):
    - Default (`teacache_thresh=0.0`): **`24.07 s -> 15.60 s`** (`1.54x`,
      `umt5=220 ms`, `dit=8,537 ms`, `vae=6,667 ms`, `peak_vram=21,609 MiB` zero
      extra VRAM, `PSNR=47.09 dB` vs pedantic FP32, `PSNR=inf dB` bit-identical
      to prior TF32).
    - Opt-in `--teacache-thresh 0.05` (`10/30` hits, `+32 MiB`): **`12.88 s`**
      (`1.87x` faster than Wave 14, `PSNR=30.30 dB`).
  - **GPU 3 `umm-native-stage-qwen38`** (`b88c58e` batched multi-slice video ViT
    Linear/MLP/Merger + 3D-grid per-slice attention, `0` extra VRAM): hot-swapped
    in `13 s` with `0` restarts on GPUs 2/4/5/6 (`vision_ms: 542.2 -> 512.4 ms`,
    Turn 2 `prefix_reused=515`, `vision_ms=0.0 ms`).
  - **Gateway `/bin/umm-infer`** (`7b87980` opt-in `narrative.allow_text_only` +
    sharpened `auto` director prompt): hot-swapped in `3 s` with `0` restarts;
    live 4-journey North-Star Showcase (`tools/umminfer_showcase.py --url http://127.0.0.1:43301`)
    passed **4/4 journeys** (`TTFT=0.91 s`, `TTFA=1.12 s`, FLUX.2 `0.66 s`,
    704x480 I2V `17.12 s`, 3-turn video QA `4.89 -> 3.00 -> 2.94 s`, and
    `pure_text_slots=0` / `visual_scene_slots=1` / `zero_redundant=True`).
  - **Stage C SM90 3-GPU routing ablation** (`46fc52c`, isolated GPUs 0/1/7
    returned cleanly to `3 MiB`): `8/8` cases completed (`SHA256SUMS` verified),
    `auto_tag` achieved `F1=1.000` / `redundant=0.000` vs `forced_image`
    (`0.800`) and `forced_video` (`0.400`, `0.500` redundant).
- Wave 17 live multi-stage performance, full-duplex VAD, UI & Stage C upgrades (`aeaf7e4`, `256c5a5`, `90e59c3`, `9525a7b`, `982e950`, `9db9a63`):
  - **GPU 6 `umm-native-stage-video`** (`982e950` + `9db9a63` selectable
    `UMM_VAE_GEMM_PREC=bf16|tf32|fp16|fp32` defaulting to BF16 Tensor Core GEMM
    with lazy cached low-precision weights, fused `im2col`->BF16/FP16,
    specialized `1x3x3`/`3x3x3` kernels, fused ResBlock/upsample kernels, and
    pinned host D2H staging, preserving `.wave16` backup): hot-swapped in `3 s`
    with `0` restarts on GPUs 2–5. Measured on isolated SM90 GPU 7 (`704x480`
    29f 15 steps `seed=42`, GPU 7 returned to `3 MiB`):
    - Default **BF16 VAE** (`teacache_thresh=0.0`): **`12.61 s`** (`umt5=221 ms`,
      `dit=8,548 ms`, **`vae=3,689 ms`** — **-44.7% vs Wave 16**, **1.91x faster
      than Wave 14 `24.07 s`**, **`PSNR=52.77 dB`** avg / `51.39 dB` min vs TF32,
      **`+0 MiB` peak VRAM** at `21,605 MiB`).
    - Opt-in **BF16 VAE + `--teacache-thresh 0.05`** (`10/30` hits): **`9.85 s`**
      (sub-10s 704x480 29f, **2.44x faster than Wave 14**, `PSNR=38.05 dB`).
    - Live `43301` Journey 2 end-to-end 704x480 29f I2V HTTP render dropped from
      `17.12 s` -> **`14.22 s`** (`SHOWCASE PASS [LIVE_SERVER]: 2/2`, `in_flight=0`).
  - **Gateway `/bin/umm-infer`** (`9525a7b` bounded per-session server-side
    echo-robust & double-talk VAD in `src/serve/echo_robust_vad.hpp`, preserving
    `.wave16` backup): hot-swapped in `1 s` with `0` GPU stage restarts
    (`echo_only_false_barge_ins=0/50`, `echo_only_false_starts=0/100`,
    `delay_jitter_false_starts=0/1000`, `double_talk_cancel=PASS`).
  - **Web UI `/web/index.html` & macOS `ummmlx`** (`aeaf7e4`): exposed
    `narrative.allow_text_only` (`Smart text-only`) default-on toggle;
    zero-restart hot-deployed to `43301` Web UI and rebuilt Release `ummmlx.app`
    (`107/107` Swift tests PASS, `0` `_AVKit_SwiftUI`).
  - **Stage C Live-to-Training E2E & SM90 Visual-Form Ablation** (`256c5a5`,
    `90e59c3`): `tools/umminfer_stage_c_pipeline.py --capture-live http://127.0.0.1:43301`
    verified `8/8` live SM90 trajectories -> CAS SHA-256 -> SFT (`6/2`) +
    DPO/GRPO (`4` pairs / `4` groups) -> `0` leakage -> `8/8` ablation-matrix
    CPU reference micro-steps (`REAL_TRAINING: NOT_RUN`); isolated SM90 3-GPU
    Visual Form ablation (`image_only` `12.80 s`, `video_only` `9.02 s`,
    `image_then_i2v_animate` `12.78 s` + I2V `5.89 s`, 4 PNG + 4 MP4 decoded,
    `SHA256SUMS` PASS, GPUs 0/1/7 returned to `3 MiB`).
- Wave 18 live multi-stage CUDA FlashAttention-2, Qwen3.8 52 tok/s decode, CLI voice & real GPU Joint-SFT upgrades (`9e11649`, `476ac07`, `39fd3da`, `6167954`, `a627fa8`):
  - **GPU 6 `umm-native-stage-video`** (`9e11649` hand-crafted SM90
    `mma.sync.aligned.m16n8k16` BF16 Tensor-Core FlashAttention-2 + fused
    `bias+rms_norm+rope` / `bias+gelu` / `residual+bias` / single-pass
    `layer_norm`, cutting DiT launches `77 -> 22`/block, preserving `.wave17`
    backup): hot-swapped in `3 s` with `0` restarts on GPUs 2–5 (`-194 MiB` live
    VRAM on GPU 6: `21,611 -> 21,417 MiB`). Measured on isolated SM90 GPU 1
    (`704x480` 29f 15 steps `seed=42`, GPU 1 returned to `3 MiB`):
    - Default **FA2 + Fused DiT + BF16 VAE**: **`11.14 s`** (`umt5=215 ms`,
      `dit=7,072 ms`, `vae=3,699 ms` — **2.16x faster than Wave 14 `24.07 s`**,
      **`PSNR=47.79 dB`** avg / `45.77 dB` min vs cuBLAS ref, **`-192 MiB` peak
      VRAM** at `21,413 MiB`).
    - Opt-in **FA2 + TeaCache `0.05`**: **`8.86 s`** (**2.72x faster than Wave
      14**, `PSNR=36.47 dB`).
    - Live `43301` Showcase Journey 1+2 (`SHOWCASE PASS [LIVE_SERVER]: 2/2`):
      Journey 1 `TTFT=0.593 s` / `TTFA=0.803 s`; Journey 2 704x480 29f I2V
      dropped **`17.12 s -> 14.22 s -> 12.83 s`**.
  - **GPU 3 `umm-native-stage-qwen38`** (`a627fa8` 4x K-unrolled `__ldcs` decode
    GEMV, fused SwiGLU dual-output GEMV, fused 4-way GDN & 3-way GQA projections,
    fused `residual+rms_norm`, 4x parallel Gated DeltaNet, 1-pass multi-warp
    online softmax, pre-resolved weights & pinned host staging, cutting `m=1`
    launches `~1,235 -> ~499`, preserving `.wave16` backup): hot-swapped in
    `13 s` with `0` restarts on GPUs 2/4/5/6. Live `43301` `temperature=0.0`
    A/B (identical `157` chunks / `975` chars greedy parity):
    - Decode throughput: **`38.95 -> 52.18 tok/s` (+34.0% / 1.34x)**;
    - Turn latency: **`4,795 ms -> 3,574 ms`** (`TTFT: 763.8 -> 564.7 ms` warm);
    - GPU 3 VRAM: **`61,441 -> 60,573 MiB` (-868 MiB)**;
    - Live Journey 3 (3-turn video QA + multimodal prefix cache): **`4.89 -> 3.00 -> 2.94 s`** (Wave 16) improved to **`2.86 -> 2.01 -> 1.85 s`** (`SHOWCASE PASS [LIVE_SERVER]: 2/2`, Journey 4 `zero_redundant=True`).
  - **Full-Duplex CLI `umm-infer voice`, Real SM90 Joint-SFT Gate & Controlled
    Ablation Layer** (`476ac07`, `39fd3da`, `6167954`): landed ALSA/macOS
    AudioQueue `umm-infer voice` + `realtime_api.cpp` handshake-close fix
    (`12/12` `voice_client_test` + `zig build test` PASS), bounded real SM90 GPU
    7 Joint-SFT AdamW acceptance (`Qwen2.5-0.5B` BF16 2 real steps + fresh-process
    checkpoint resume equivalence, GPU 7 returned to `3 MiB`), and Stage C
    controlled-measurement contract (`59/59` tests PASS).
- Wave 19 live native TTS 7.00x realtime CUDA Graphs, per-request Fast Video TeaCache & real SM90 Joint-DPO upgrades (`77da8ac`, `08b1fdf`, `c7d42f6`):
  - **GPU 5 `umm-native-stage-tts`** (`08b1fdf` CUDA Graph capture for 28-layer
    Talker + 15-step Code Predictor collapsing `~2,086 -> ~8` launches/frame,
    O(T) stateful causal Conv1d/ConvTranspose1d + rolling KV codec replacing
    O(T^2) streaming & 256x redundant `snake`/`transpose` threads, 1-warp/row
    4x `__ldcs` GEMV, fused QKV/SwiGLU/QK-norm+RoPE+KV-append, and 1-pass Online
    Softmax GQA, preserving `.wave18` backup): hot-swapped in `3 s` with `0`
    restarts on GPUs 2/3/4/6. Live `43301` `/v1/audio/speech` & Journey 1 A/B:
    - `/v1/audio/speech` (15.7 s 24 kHz WAV): **`7.63 s (1.86x RT, RTF 0.538) ->
      2.24 s (7.00x RT, RTF 0.143)`** (**3.76x faster**, `rms=2717.5`);
    - GPU 5 steady VRAM: **`10,997 MiB -> 5,835 MiB`** (**-5,162 MiB / -5.04 GiB
      saved, -47% VRAM**);
    - Live Journey 1 (interleaved prose + FLUX.2 image + 85.8 s streaming PCM16
      audio `4,116,480 B`): **`62.72 s -> 16.86 s`** (**3.72x faster**),
      `vlm_finish_s: 11.53 -> 7.06 s`, `TTFT=0.875 s`, `TTFA=0.996 s` (`121 ms`
      VLM-to-first-audio gap, `SHOWCASE PASS [LIVE_SERVER]: 1/1`).
  - **Gateway `/bin/umm-infer`, GPU 6 `umm-native-stage-video`, Web UI & macOS
    `ummmlx`** (`77da8ac` opt-in per-request `video.teacache_thresh` in
    `[0.0, 0.25]` with RAII default restore, `Fast video` UI toggle, plus
    `realtime_api.cpp` rejected-handshake close fix, preserving `.wave18`
    backups): hot-swapped live with `0` restarts on GPUs 2/3/4/5 and rebuilt
    Release `ummmlx.app` (`108/108` Swift tests, `0 _AVKit_SwiftUI`). Live
    `43301` `/v1/visual/render` verified `teacache_thresh: 0.05` (`5.78 s`,
    `ftyp=True`) vs omitted lossless default (`6.94 s`) and HTTP `400` on
    out-of-range `0.50`.
  - **Stage C Real SM90 Joint-DPO Gate** (`c7d42f6`, isolated GPU 7 returned to
    `3 MiB`): `Qwen2.5-0.5B` BF16 DPO (`beta=0.1`) 2 real AdamW steps (`loss
    0.6931 -> 0.6131`, 290/290 non-zero grad tensors, exact fresh-process
    checkpoint resume equivalence, `25/25` Stage C tests PASS; GRPO & quality
    remain `NOT_RUN`).
- Wave 20 live native ASR 99.3x realtime CUDA Graph & temporal video keyframe selection upgrades (`47db3b2`, `038d544`):
  - **GPU 2 `umm-native-stage-asr`** (`038d544` hand-crafted SM90 64x64x64 BF16
    Tensor-Core `mma.sync` GEMM + Implicit-GEMM Conv2d 3x3s2p1+GELU, 4x `__ldcs`
    GEMV, fused 3-way QKV / dual-output SwiGLU / `qk_norm+rope+kv_append`,
    1-pass Online Softmax GQA, full 28-layer decode CUDA Graph, pre-resolved
    weights, pinned host staging, and bit-exact trailing-zero STFT skip,
    preserving `.wave19` backup): hot-swapped in `9 s` with `0` restarts on
    GPUs 3–6. Live `43301` `/v1/audio/transcriptions` on `15.69 s` 16 kHz speech
    (identical 37-token / 167-char transcript):
    - Tunnel E2E latency dropped **`2,918 ms -> 1,682 ms`** (`~1,521 ms` upload
      RTT);
    - Server-side GPU 2 stage log (`asr.log`): **`elapsed_ms = 158 ms`**
      (**`99.3x` realtime, `RTF = 0.010`** — ~`8.8x` server speedup, completing
      our 5-for-5 SM90 native GPU optimization sweep).
  - **Gateway `/bin/umm-infer` & GPU 3 `umm-native-stage-qwen38`** (`47db3b2`
    opt-in `video_url.frame_selection` bounded temporal motion/scene-boundary
    keyframe selection, preserving `.wave19` backups): hot-swapped live with
    `0` restarts on GPUs 2/4/5/6 and verified live on `43301` with
    `{"policy": "temporal", "motion_threshold": 0.02, "scene_threshold": 0.25}`
    returning accurate motion understanding (`in_flight=0`).

- Wave 45 Real `Qwen/Qwen3.8-27B` Fast-Tokenizer Gate & `token_export` Format v2 (`4685125`):
  - Harvested accepted Task #37 (`revisions=0`, CPU-only real fast-tokenizer
    gate): preserved v1 byte-for-byte (golden canonical record SHA-256
    `ea53770a...` unchanged) after isolating real NFC (`e` + U+0301 -> `é`) and
    emoji ByteLevel byte-fallback offset behavior; added backward-compatible
    `token_export` format v2 in `python/umminfer_token_export.py` +
    `tests/token_export_test.py` + `docs/token-export-real-qwen.md` (and updated
    `docs/rl.md`, `docs/joint-training.md`, `docs/roadmap.md`). Real
    `Qwen/Qwen3.8-27B` 6-file tokenizer gate (`transformers 5.6.0` /
    `tokenizers 0.22.2`) passed **10/10 OK (0 skips)** (`REAL_TOKENIZER_FIXTURE:
    PASS`) with fresh-process export + independent re-encode strong validation;
    local 8-suite Stage C regression passed (`64/64` tests; real tokenizer
    honestly reports `NOT_RUN` when not supplied locally).

- Wave 44 SM90 Real Request-Native CUDA Allocator Telemetry Gate & Nonstream Leak Fix (`4ff5133`):
  - Harvested accepted Task #35 (`revisions=0`, isolated SM90 GPU 0 real CUDA
    validation gate): landed `tests/request_telemetry_live.py`,
    `benchmarks/ablations/request-memory-controlled-sm90.json`
    (`variable=request_native_backend`, `request_memory.enabled=true`,
    `required=true`), nonstream artifact descriptor strip fix in
    `src/serve/openai_chat.cpp` + regression in `tests/unified_contract_test.cpp`,
    `python/umminfer_ablation_run.py` two-arm support, `docs/observability.md`,
    and `docs/narrative-routing-ablation.md`. Real SM90 GPU 0 peaks: Qwen3.8
    `742,129,664 B`, Wan2.2 `156,241,920 B`, `CONTROLLED_LIVE / COMPLETED`
    (8/8 measured requests, 12/12 stage entries `COVERED`, 52/52 `SHA256SUMS`,
    GPU 0 returned to `3 MiB`). Local verification: `zig build test` PASS,
    `controlled_measurement_test.py` 38/38 OK, `ablation_runner_test.py` 17/17 OK,
    `stage_c_pipeline_test.py` 6/6 OK.

- Wave 43 Stage C Decoupled Online GRPO Rollout Generation, Validation & Replay Contract (`db43e7e`):
  - Harvested accepted Task #36 (`revisions=0`, CPU-only contract): added
    stdlib trainer-side HTTP orchestrator `python/umminfer_online_rollout.py`,
    fixture `configs/training/online_grpo_rollout_fixture.json`,
    `tests/online_rollout_test.py` (10/10 PASS), backward-compatible
    `python/umminfer_rl_trajectory.py` hooks, pre-ML online-group rejection in
    `python/umminfer_joint_grpo_live.py`, and updated `docs/joint-training.md`,
    `docs/rl.md`, and `docs/roadmap.md`. Verified all 10 Stage C Python suites
    PASS (`online_rollout_test.py` 10/10, `joint_grpo_live_test.py` 4/4,
    `token_export_test.py` 5/5 + 1 `REAL_TOKENIZER_FIXTURE: NOT_RUN`,
    `rl_architecture_test.py` 16/16, `rl_phase2_test.py` 6/6,
    `sft_dataset_test.py` 8/8, `joint_trainer_test.py` 6/6,
    `stage_c_pipeline_test.py` 6/6, `ablation_runner_test.py` 17/17,
    `sse_replay_test.py` 9/9). Real online generation, online optimizer
    execution, and model-quality evaluation remain explicitly `NOT_RUN`, and
    behavior log-probabilities/masks remain `NOT_COVERED` unless exposed by
    the endpoint.

- Wave 42 Native C++ CLI `umm-infer voice` Per-Turn Barge-In Cut-Point, Real PCM16 RMS Mic Level & Interactive Echo Toggle (`58a58c0`):
  - Completed full 3-client parity (Web UI + macOS `ummmlx` + native C++ CLI
    `umm-infer`) for Wave 34's full-duplex barge-in & mic observability:
    added `compute_pcm16_rms_level`, `VoiceBargeInCutPoint` /
    `format_barge_in_cut_point` (`Interrupted · <N> B spoken` /
    `Interrupted before audio`), monotonic `response.playback_ack` + optional
    `response.cancelled.text_cursor` resolution, `on_response_cancelled` /
    `on_mic_level` callbacks, `last_barge_in_cut_point()` /
    `last_mic_rms_level()` accessors with clean reconnect/stop reset, and
    interactive stdin `echo on` / `echo off` runtime toggle in
    `src/client/voice.{hpp,cpp}`, `src/client/cli.cpp`,
    `tests/voice_client_test.cpp`, and `docs/client.md`
    (`zig build test` PASS, `voice_client_test` **20/20 PASS**,
    `client_media_cli_e2e.py` **9/9 PASS**; physical mic/speaker `NOT_RUN`).

- Wave 41 Tokenizer-Backed Trajectory Token-ID & Span Export (`python/umminfer_token_export.py`):
  - Harvested accepted Task #34 (`f098749` + `f847a5a`, `revisions=0`): added
    optional additive `token_export` format v1 with explicit tokenizer file
    SHA-256 binding, exact half-open UTF-8 byte offsets, source event/raw-SSE
    frame attribution, and SFT `joint_full` mask equivalence across
    `python/umminfer_token_export.py`, `python/umminfer_rl_worker.py`,
    `python/umminfer_sft_dataset.py`, `tests/token_export_test.py`, `docs/rl.md`,
    and `docs/joint-training.md` (6-suite Python regression PASS:
    `token_export_test.py` 5/5 + explicit `REAL_TOKENIZER_FIXTURE: NOT_RUN` skip,
    `sft_dataset_test.py` 8/8, `rl_architecture_test.py` 16/16,
    `rl_phase2_test.py` 6/6, `sse_replay_test.py` 9/9,
    `stage_c_pipeline_test.py` 6/6).

- Wave 40 Native C++ CLI `umm-infer chat --native-allocator-telemetry` (3-Client Parity):
  - Completed 3-client parity (Web UI + macOS `ummmlx` + native C++ CLI
    `umm-infer`) for Wave 37's `telemetry.native_allocator=true`: added
    `ChatRequest::native_allocator_telemetry`, `ChatResponse::telemetry`,
    default-omission request encoding, SSE/non-stream telemetry extraction,
    and concise stderr `[telemetry] VRAM: <peak> peak` / `VRAM: N/C (<reasons>)`
    reporting without altering stdout prose (`zig build test` PASS,
    `tests/client_media_cli_e2e.py` **9/9 PASS**).

- Wave 39 Web UI & macOS `ummmlx` Opt-In Native VRAM Allocator Telemetry Pills (`8319791`):
  - Exposed Wave 37's `telemetry.native_allocator=true` across Web UI
    (`web/index.html`) and macOS `ummmlx` (`UnifiedStream.swift`,
    `ChatClient.swift`, `ChatModel.swift`, `App.swift`): default OFF omits
    `telemetry` completely (byte-identical default requests); opted-in ON
    parses both `unified.telemetry` and top-level `telemetry` envelopes and
    renders `VRAM: <peak> peak` (when `>=1` stage is `COVERED`) or honest
    `VRAM: N/C` with real server reason tooltips (never mapping `null` to `0`):
    - `web_ui_contract_test.py`: **17/17 PASS** (`2,914` script lines);
      macOS `swift test`: **136/136 PASS**; Release `ummmlx.app` rebuilt
      (zero `_AVKit_SwiftUI`).
    - Zero-restart hot-deployed `web/index.html` (`.wave36` backup) live onto
      SM90 `43301` (`id="nativeAllocatorTelemetry"` &
      `summarizeNativeAllocatorTelemetry` verified live, `/health in_flight=0`).

- Wave 38 Isolated SM90 Public-SSE Live Capture/Replay Gate (`869854f`):
  - Harvested accepted Task #33 (`81c593c`, 7 files): landed focused live
    harness `tests/sse_replay_live.py`, `docs/sse-replay-live.md`, hash-only
    SM90 evidence `docs/evidence/sse-replay-live-sm90.json`, and 9-test
    `tests/sse_replay_test.py` (`9/9 OK`, `rl_architecture_test.py 16/16 OK`,
    `stage_c_pipeline_test.py 6/6 OK`).
  - Isolated SM90 GPU 0 gate (resident `43301` & GPUs 2–6 untouched; GPU 0
    returned to `3 MiB`, `/health in_flight=0`, `33/33 SHA256SUMS OK`) captured
    3 public streams (text-only `1,654 B`, interleaved visual contract
    `235,713 B`, real in-flight `/abort` `725 B` -> terminal `cancelled`) with
    observed `perf_counter_ns` ticks and verified fresh-process CLI replay
    produced byte-identical `26,040 B` JSONL (`b63b60dc...`), 5/5 CAS match,
    split/coalesced framing invariance, and fail-closed rejection on byte and
    re-hashed tick tampering (visual bytes explicitly marked contract fixtures).

- Wave 37 Opt-In Request-Native CUDA Allocator Telemetry (`6ea24d5`):
  - Harvested accepted Task #31 (`d0c7822`, 31 files): added
    `telemetry.native_allocator=true` opt-in request-native CUDA allocator epoch
    tracking (`include/umminfer/obs/request_telemetry.hpp`,
    `src/obs/request_telemetry.cpp`, `native/common/cuda_request_allocator.*`,
    `native/stage_cli/request_telemetry_stage.*`, Qwen3.8 & Wan2.2 explicit
    allocation instrumentation, public bounded 64-stage projection, and
    controlled-ablation `request_memory` integration).
  - Verified locally: `zig build test` PASS (`unified_contract_test` +
    `voice_client_test` 16/16), `controlled_measurement_test.py` **37/37 OK**,
    `ablation_runner_test.py` **17/17 OK**, `stage_c_pipeline_test.py` **6/6 OK**,
    `sse_replay_test.py` **8/8 OK**. Real CUDA allocation measurements remain
    `NOT_RUN` pending the exclusive SM90 gate.

- Wave 36 Web UI & macOS `ummmlx` 4th Showcase Preset (`Temporal Video QA`) & Explicit `source=slot-0` Storyboard Preset (`3a53f89`):
  - Added `temporal_video_qa` (`Temporal Video QA`: configures `temporal` frame
    selection `0.02/0.25` + `allowTextOnly` + non-blocking attach-clip hint,
    never auto-sends, resets to `uniform` when switching presets) and upgraded
    `storyboard_i2v` to explicitly request `source=slot-0` anchor binding across
    both Web UI (`web/index.html`) and macOS `ummmlx`:
    - `web_ui_contract_test.py`: **17/17 PASS**; macOS `swift test`: **134/134 PASS**;
      Release `ummmlx.app` rebuilt (`9.12 s`, zero `_AVKit_SwiftUI`).
    - Zero-restart hot-deployed `web/index.html` (`.wave34` backup) live onto
      SM90 `43301` (`data-preset="temporal_video_qa"` & `source=slot-0` verified
      live, `/health in_flight=0`).

- Wave 34 Web UI & macOS `ummmlx` Per-Turn Barge-In Cut-Point Badges & Real PCM16 Mic Level Meter (`e6d7291`):
  - Exposed real server-acknowledged spoken prefix cut-point badges
    (`Interrupted · <N> B spoken` / `Interrupted before audio` from observed
    `response.playback_ack` / `response.cancelled` `text_cursor`, never shown on
    completed turns) and real captured 16 kHz PCM16 RMS mic level meters (zero
    synthetic animation, reset strictly to `0` on disconnect/stop) across both
    Web UI (`web/index.html`) and macOS `ummmlx`:
    - `web_ui_contract_test.py`: **17/17 PASS** (including Node behavioral check
      `realtime_audio_meter_and_barge_in_behavior`).
    - macOS `ummmlx`: `swift test` **134/134 PASS**; Release `ummmlx.app` rebuilt
      (`9.20 s`, zero `_AVKit_SwiftUI`).
    - Zero-restart hot-deployed `web/index.html` (`.wave32` backup) live onto
      SM90 `43301` (`id="realtimeMicMeter"` & `computePcm16RmsLevel` verified
      live, `/health in_flight=0`).

- Wave 33 Native C++ CLI `umm-infer voice` Echo-Suppression & Barge-In Telemetry (`9147b84`):
  - Completed 3-client parity (Web UI + macOS `ummmlx` + C++ CLI `umm-infer voice`)
    for Wave 31's server-side `turn_detection.echo_suppression`: added
    `VoiceConfig::echo_suppression` (`--no-echo-suppression`, default ON omitted
    on wire), narrow mid-response `VoiceClient::set_echo_suppression(bool)`,
    strict opt-out confirmation check, and real observed `barge_in_count()`
    (`zig build test` PASS: `voice_client_test` **16/16 PASS** +
    `unified_contract_test` PASS).

- Wave 32 Web UI & macOS `ummmlx` Server-Side `echo_suppression` Toggle & Live Barge-In Status (`f18834d`):
  - **Web UI `/web/index.html` & macOS `ummmlx`** (`f18834d`): exposed Wave 31's
    `turn_detection.echo_suppression` control (default ON omitted on wire to
    preserve backward compatibility, opt-out `false` at connect, and narrow
    `turn_detection`-only active mid-session/mid-response toggle) alongside
    server-confirmed status (`Echo suppression: On` / `Off (A/B)`) and real
    observed `Barge-ins: N` (`response.cancelled` count, zero fabricated values):
    - `web_ui_contract_test.py`: **16/16 PASS** (including Node behavioral check
      `realtime_echo_suppression_behavior`).
    - macOS `ummmlx`: `swift test` **130/130 PASS**; Release `ummmlx.app` rebuilt
      (`9.00 s`, zero `_AVKit_SwiftUI`).
    - Zero-restart hot-deployed `web/index.html` (`.wave31` backup) live onto
      SM90 `43301` (`id="realtimeEchoSuppression"` &
      `buildRealtimeEchoSuppressionUpdate` verified live, `/health in_flight=0`).

- Wave 31 Live Gateway Echo-Suppression Rollout (`8ac1777`) & Exact-Byte Raw SSE Replay (`207ddb5`):
  - **Gateway `/bin/umm-infer` hot-swap (`EchoRobustVad` + active `echo_suppression` A/B toggle)**
    (`8ac1777`, `src/serve/realtime_api.cpp`, `tests/full_duplex_echo_live.py`,
    `docs/full-duplex.md`): harvested accepted Task #28 (`zig build test` PASS
    including `REALTIME_ECHO_ACTIVE_AB same_pcm=1 protected_cutins=0 baseline_cutins=1`,
    `full_duplex_echo_live.py --self-check` PASS), rebuilt Linux `ReleaseFast`
    `umm-infer` (`830ec93d...`), and hot-swapped ONLY the supervised gateway
    (`.wave30` backup; `/ready` 200 with `0` stage restarts across GPUs 2–6,
    `/health in_flight=0`). Task #28's isolated SM90 GPUs 0/1/7 live gate
    (real Qwen3-ASR-1.7B + Qwen3.8-27B + Qwen3-TTS-12Hz-1.7B, `63/63 SHA256SUMS`)
    verified `0/6` echo false cut-ins vs `6/6` disabled same-PCM A/B cut-ins,
    near-end onset `3.355 ms`, double-talk cancellation `3.375 / 3.256 ms`,
    and stale-reference pre-audio cancellation `0.323 ms` (physical speaker/mic
    AEC remains `NOT_RUN`).
  - **Exact-Byte Raw SSE Trajectory Replay (`python/umminfer_sse_replay.py`)**
    (`207ddb5`): harvested accepted Task #30 (`53/53` RL/SFT/SSE-replay/Stage-C
    offline tests PASS while preserving Wave 29's 0-based `slot-N` prompt fix)
    and verified live against resident SM90 `43301` (`record_chat_completion(..., capture_replay=True)`
    on 5-modality `i2v_chain` + streaming audio): captured `3,933,658 B` raw
    `.sse` sidecar (`637f7447...`, `304` transport chunks, `198` SSE frames,
    `196` normalized events, real `monotonic_ns` ticks up to `9,936.66 ms`,
    PNG/MP4/PCM CAS artifacts), and confirmed fresh-process CLI replay
    (`python/umminfer_sse_replay.py`) produced **byte-identical** canonical JSONL
    (`9d52bbdd...`) and matching CAS/raw SHA-256 digests (`/health in_flight=0`).

- Wave 30 Explicit Anchor UI (`5493d85`) & Frozen 3-Round Multimodal Stability Gate Orchestrator (`a30e32a`):
  - **Frozen 3-Round Multimodal Stability Orchestrator & SM90 Config**
    (`a30e32a`, `tools/multimodal_stability.py`,
    `benchmarks/multimodal-stability-sm90.json`,
    `tests/multimodal_stability_test.py`): harvested accepted Task #29 adding
    1-warmup + 3-fresh-process measured round orchestration, full source/binary/
    weight/data/topology hash pinning, raw-SSE metric re-derivation, blocking
    10% per-task median & p95 gates, and recursive `SHA256SUMS` sealing
    (`18/18` offline contract tests + `multimodal_benchmark_test.py` +
    `multimodal_coverage_audit_test.py` PASS; live 3-round same-binary run
    remains `NOT_RUN` pending same-binary baseline).
  - **Web UI `/web/index.html` & macOS `ummmlx` Explicit `source=slot-N` UI**
    (`5493d85`): exposed Wave 29 `anchor_explicit` & `anchor_sha256` across
    `visual_slot`, `artifact`, `leg_error`, and `complete.legs` with
    `Explicit I2V ← Slot # · <short8>` badges, full SHA-256 tooltips, and
    fail-closed slot error cards; zero-restart hot-deployed `web/index.html`
    (`.wave29` backup) onto SM90 `43301` (`normalizeAnchorSha256` &
    `updateSlotContinuity` verified live, `/health in_flight=0`), rebuilt
    Release `ummmlx.app` (`8.93 s`, zero `_AVKit_SwiftUI`),
    `web_ui_contract_test.py 15/15 PASS`, `swift test 126/126 PASS`.

- Wave 29 Explicit `source=slot-N` Narrative Anchor Binding (`e56d11e`, `efabf39`):
  - **Gateway hot-swap & Live SM90 5-Modality Explicit `source=slot-0` I2V Verification**:
    harvested accepted Task #24 (`e56d11e`, 56/56 Python tests + full `zig build test`
    PASS), fixed a live-discovered 0-based `slot-N` prompt numbering omission in
    `src/pipeline/unified_schema.cpp` (`efabf39`), rebuilt Linux `ReleaseFast`
    `umm-infer` (`a4c9750a...`), and hot-swapped ONLY the supervised gateway
    (`.wave28` backup; ready in `1.5 s`, all 5 GPU stages `restarts=0`):
    - Live SM90 `43301` Journey 5 (`storyboard_i2v_chain`): **`PASS`** in
      **`11.86 s`** (`TTFT 0.967 s`, `TTFA 1.105 s`, 62 PCM16 chunks / 32.8 s
      audio, Slot 0 FLUX.2 PNG `0.669 s` `440,745 B` SHA-256 `3a9d5bbe...` ->
      Slot 1 Wan2.2 704×480 29f I2V MP4 `7.225 s` `170,245 B` with native
      `anchor_explicit=true`, `anchor_slot_index=0`, and exact matching
      `anchor_sha256=3a9d5bbe...`; `5/5 SHA256SUMS`, `/health in_flight=0`).

- Wave 28 Controlled SM90 Visual-Form Gate (`5890a4c`), 70/70 Real SM90 Multimodal Coverage (`13d9274`), and Live Web/macOS Showcase Presets & Telemetry HUD (`81227f1`):
  - **Web UI `/web/index.html` & macOS `ummmlx` North-Star Showcase Presets + Live Telemetry Strip**
    (`81227f1`): zero-restart hot-deployed `web/index.html` (`.wave27` backup)
    onto SM90 `43301` (`id="showcasePresets"` & `formatTurnTelemetry` verified
    live, `/health in_flight=0`) and rebuilt Release `ummmlx.app` (`8.68 s`,
    zero `_AVKit_SwiftUI`); `web_ui_contract_test.py` `15/15 PASS`,
    `swift test` `122/122 PASS`.
  - **Controlled Single-GPU SM90 Visual-Form Ablation (`CONTROLLED_LIVE / COMPLETED`)**
    (`5890a4c`, `benchmarks/ablations/visual-form-controlled-sm90.json`,
    `docs/narrative-routing-ablation.md`): isolated GPU 0 (`1` warmup + `3`
    interleaved repetitions = `18` measured chat + `6` measured I2V renders,
    `0` failures, `26/26 SHA256SUMS OK`, `78/78` independent assertions PASS,
    true 50 ms polled VRAM peaks, GPU 0 returned to `3 MiB`, resident GPUs 2–6
    untouched): `image_only` TTFT `313.6 ms` (`IQR 2.4`), `video_only` TTFT
    `314.3 ms` (`IQR 2.1`), `image_then_i2v_animate` TTFT `313.5 ms` + I2V
    render `3,426.6 ms` (`IQR 60.6`), first-frame PSNR **`46.08 / 44.44 dB`**
    (center-crop MAE `1.02 / 1.25` vs stretch `11.36 / 3.88`).
  - **70/70 Real SM90 Multimodal Coverage & Independent Auditor**
    (`13d9274`, `tools/multimodal_coverage_audit.py`,
    `docs/multimodal-benchmark-results.md`): added remaining 48 real SM90 cases
    on isolated GPU 1 (`Qwen3-ASR-1.7B`) + GPU 7 (`Qwen3.8-27B`, GPUs returned
    to `3 MiB`): English ASR `16/16` (**WER `1.18%`**, `5/425`), Mandarin ASR
    `16/16`, GSM8K `14/16` (**`87.5%`**), combining with Wave 26 16-VQA
    (`95.83%`) + 6-SpeechCOCO Joint (`0.0%` WER / `97.22%` macro hit) for
    **70/70** verified real coverage (`287/287 SHA256SUMS OK`, byte-identical
    independent re-audit, `BASELINE_REQUIRED` retained).

- Wave 27 Stage C Controlled Voice-Strategy (`f31e2ed`), Row-5 Objective-Mix Pareto SM90 Gate (`281f3b4`), and Upgraded 3-Way CUDA-Graph Full-Duplex Barge-in Verification:
  - **Stage C Ablation #5 Post-Training Objective-Mix & Quality-vs-Latency Pareto SM90 Gate**
    (`281f3b4`, `python/umminfer_objective_mix_ablation.py`,
    `configs/training/objective_mix_ablation.json`, `docs/objective-mix-ablation.md`):
    executed all 4 single-variable objective-mix arms from bit-identical initial
    `Qwen2.5-0.5B-Instruct` BF16 weights (`4699d9ff...`) on isolated SM90 GPU 7
    (`26.9 s`, `torch_memory_allocated=0`, `torch_memory_reserved=0`, post-exit
    GPU 7 `3 MiB`, `5/5 SHA256SUMS OK`, `all_arms_distinct_final_parameters=true`):
    - `sft_plus_consistency` (`q=0.9365`, `p95=27,915.0 ms`) and
      `sft_plus_full_latency_pareto` (`q=0.9362`, `p95=27,813.0 ms`, `-425.4 ms`
      vs baseline, `+0.4150` optimal-fast `Δp_policy`) form a 2-arm non-dominated
      Pareto frontier strictly dominating `sft_only` (`0.6768 / 28,240.1 ms`) and
      `sft_plus_routing` (`0.8754 / 28,361.0 ms`).
  - **Upgraded 3-Way CUDA-Graph Full-Duplex Stage B Barge-in Regression**
    (`tests/full_duplex_barge_in_live.py`): executed on isolated GPUs 0/1/7
    (`99.3×` Qwen3-ASR-1.7B + `53.6 tok/s` Qwen3.8-27B + `7.00×` Qwen3-TTS-12Hz-1.7B;
    resident GPUs 2–6 untouched): `PASS` (`first=400.7 ms`, `second=400.8 ms`
    including 400 ms leading fixture silence -> `0.7–0.8 ms` server interrupt
    latency, exact 12 B acknowledged prefix, `0` stale events, `in_flight=0`,
    GPUs 0/1/7 returned to `3 MiB`).
  - **Controlled SM90 Voice-Strategy Ablation (`CONTROLLED_LIVE / COMPLETED`)**
    (`f31e2ed`, `docs/voice-strategy-ablation.md`,
    `benchmarks/ablations/voice-strategy-controlled-sm90.json`): executed on
    isolated SM90 GPU 1 (`Qwen3.8-27B` CUDA Graph, `UMM_QWEN38_PREFIX_CACHE=0`,
    `--max-new-tokens 256`, `--batch-size 1`) + GPU 7 (`Qwen3-TTS-12Hz-1.7B`
    dual CUDA Graph + idle required entry `Qwen3-ASR-1.7B`) over ephemeral
    loopback (`127.0.0.1:43911`; resident GPUs 2–6 untouched; pre/post GPU 1/7
    `3 MiB / 0` compute processes; post `/health in_flight=0`; `3/3 SHA256SUMS OK`):
    - Schedule: `1` warmup (`6` requests excluded) + `3` interleaved alternating
      repetitions (`18` measured = `9/9` per arm, `0` failures, `172.2 s`).
    - `sentence_streaming_tts` (`audio.stream=true`) vs `full_response_tts`
      (`audio.stream=false`): median TTFT `153.3 vs 153.4 ms`; median TTFA
      **`265.8 ms` (`p95 266.6 ms`, `IQR 1.9 ms`, `MAD 0.7 ms`)** vs
      `5,285.4 ms` (**`19.88×` TTFA speedup**, `-5,019.5 ms`); post-TTFT onset
      **`112.5 ms` (`IQR 0.2 ms`)** vs `5,132.0 ms` (**`45.62×`**); stalls
      **`0 / 687` (`0.0%`)**, pooled inter-chunk gap p95 `210.6 ms`.

- Wave 26 real SM90 16-VQA + 6-SpeechCOCO Joint Multimodal Benchmark Gate (`a2f9eec`):
  - **Real SM90 Qwen3.8-27B + Qwen3-ASR-1.7B multimodal benchmark** (`a2f9eec`,
    `docs/multimodal-benchmark-results.md`): executed 22/22 usable cases over
    public HTTP/SSE with strict terminal-leg validation (isolated GPUs returned
    to `3 MiB`, resident `43301` untouched):
    - VQA (`16/16`): **95.83%** accuracy, TTFT p50/p95 `657.3 / 817.4 ms`,
      terminal p50/p95 `684.4 / 836.7 ms`.
    - SpeechCOCO Joint (`6/6`): **ASR WER 0.0%** (`0/58` words), macro keyword
      hit **97.22%**, TTFT p50/p95 `666.5 / 719.5 ms`.

- Wave 25 live Web/macOS Narrative Continuity UI & Real SM90 Uniform vs Temporal Video A/B Gate (`15eee86`, `34b3bae`):
  - **Web UI `/web/index.html` & macOS `ummmlx`** (`15eee86`): exposed
    `narrative.continuity_mode` (`off`/`prompt_anchor`/`i2v_chain`),
    `narrative.style_anchor` (UTF-8 `<=256B` pre-flight check), and inline
    `I2V <- Slot #` / `Anchor` slot attribution badges
    (`web_ui_contract_test.py` 13/13 PASS, macOS `swift test` 116/116 PASS);
    zero-restart hot-deployed to SM90 Web UI (`.wave22` backup) and rebuilt
    Release `ummmlx.app` (`8.18 s`, zero `_AVKit_SwiftUI`).
  - **Real SM90 Uniform vs Temporal Video Keyframe A/B Gate** (`34b3bae`):
    isolated GPU 0 cold-prefix (`UMM_QWEN38_PREFIX_CACHE=0`, `prefix_reused=0`)
    24/24 measured requests (12 serial + 12 batch-4) passed with 4/4 expected
    answers per arm across rise/drain/static/scene-cut clips, retaining exact
    scene-cut frame 24 (`3.0 s`) under temporal mode (`docs/video-temporal-live.md`);
    added atomic batched request logging and `qwen38_video_evidence` diagnostics
    in `native/stage_cli/qwen38_stage.cpp` (hot-swapped onto GPU 3).

- Wave 24 live Wan2.2 DiT Fused-QKV (`+0 B` VRAM) & 30-Layer CUDA Graph (`7588455`):
  - **GPU 6 `umm-native-stage-video`** (`7588455` contiguous `attn1_qkv_w`
    packing in `bf16_weights_`, `ws_.ff` output reuse, fused
    `qkv_split_bias_norm_rope_kernel`, and 2-slot LRU 30-layer DiT CUDA Graph
    capture with preallocated 32 MiB `cublasSetWorkspace` and
    `UMM_DIT_NO_GRAPH=1` opt-out, preserving `.wave23` backup): hot-swapped live
    via supervisor respawn (`load_ms=11,771`, `0` restarts on GPUs 2–5).
    Measured live on SM90 GPU 6:
    - Bit-exact parity vs Wave 23 (`PSNR=inf`, `sha=e09287236a2867c1`).
    - `704x480` 29f 15 steps: `8.78 s` lossless / `6.66 s` TeaCache `0.05`
      (confirming BF16 Tensor-Core compute saturation at `M=5,280`).
    - `480x320` 13f 10 steps lossless: **`1.72 s`** (`dit_ms=1,081`,
      `vae_ms=421`).

- Wave 23 live Persistent GPU-Resident Wan2.2 3D-VAE + Fused GPU RGB8 Decode (`776d153`):
  - **GPU 6 `umm-native-stage-video`** (`776d153` persistent `WanVae` preloaded
    decoder + pre-converted BF16/FP16 weights + persistent `cublasHandle_t` /
    `BufferPool` / pinned staging + lazy cached I2V encoder + fused
    `unpatchify_clamp_rgb8_kernel` 4x smaller D2H, with `UMM_VAE_NO_PERSIST=1`
    fallback, preserving `.wave21` backup): hot-swapped live via supervisor
    respawn (`load_ms=12,190`, `28,323 MiB` steady after lazy I2V encoder cache,
    `0` restarts on GPUs 2–5). Measured live on SM90 GPU 6 (`704x480` 29f 15
    steps `seed=42`):
    - T2V lossless (`teacache_thresh=0.0`): `vae_ms` **`3,792 -> 1,993 ms`**
      (**`-47.4%`**), total **`10.66 -> 8.79 s`** (**`2.74×` faster than Wave 14
      `24.07 s`**), `PSNR=55.58 dB` avg (`max=inf`) vs Wave 21, bit-exact repeat
      outputs (`sha=e09287236a2867c1`, `310,732 B`).
    - T2V Fast (`teacache_thresh=0.05`, `10/30` hits): `vae_ms=1,983 ms`, total
      **`6.62 s`** (**`3.64×` faster than Wave 14**; `7.05 s` end-to-end HTTP).
    - I2V Fast (`i2v=1, teacache_thresh=0.05`, Journey 5 warm): `vae_ms`
      **`4,486 -> 2,024 ms`** (**`-54.9%`**), total **`9.21 -> 6.69 s`**
      (**`3.60×` faster than Wave 14**), bit-exact repeat outputs (`162,532 B`).
  - **Full 5/5 North-Star Showcase live on Wave 23 SM90 stack (`tools/umminfer_showcase.py --url http://127.0.0.1:43301 --journey all`)**:
    passed **5/5 journeys** (`9/9 SHA256SUMS`): Journey 1 `TTFT=0.877 s` /
    **`TTFA=0.996 s`** (FLUX.2 `0.652 s`), Journey 2 lossless `704x480` 29f I2V
    HTTP **`17.12 -> 9.713 s`** (sub-10s!), Journey 3 3-turn video QA
    **`2.38 -> 1.61 -> 1.56 s`**, Journey 4 `zero_redundant=True`, Journey 5
    `storyboard_i2v_chain` (`TTFT=0.873 s`, `TTFA=1.024 s`, `anchor_slot_index=0`).

- Wave 22 live Web/macOS Temporal Video UI, Journey 5 `storyboard_i2v_chain` Showcase & Stage C Voice Collaboration Ablation (`ea59116`, `b7c30ee`):
  - **Web UI `/web/index.html` & macOS `ummmlx`** (`ea59116`): exposed
    Uniform/Temporal video frame selection and motion/scene thresholds
    (`web_ui_contract_test.py` 11/11 PASS, macOS `swift test` 112/112 PASS);
    zero-restart hot-deployed to live SM90 Web UI (preserving `.wave21` backup)
    and rebuilt Release `ummmlx.app` (`8.92 s`, zero `_AVKit_SwiftUI`).
  - **North-Star Showcase Journey 5 (`storyboard_i2v_chain`)** (`b7c30ee`):
    live SM90 `43301` run passed (`1/1`, `5/5 SHA256SUMS`): `TTFT=0.874 s`,
    `TTFA=1.026 s`, `20.4 s` streaming WAV (`979,244 B`), Slot 0 FLUX.2 PNG
    (`426,002 B`) automatically anchoring Slot 1 Wan2.2 `704x480` 29f I2V MP4
    (`anchor_slot_index=0`, `162,460 B`).
  - **Stage C Ablation #3 (`voice_strategy`, `UNCONTROLLED_LIVE`)** (`b7c30ee`):
    live SM90 `43301` run (`3/3` cases/arm, `SHA256SUMS` 3/3 OK) measured
    `sentence_streaming_tts` median TTFA **`825.0 ms`** (**`112.8 ms`**
    post-TTFT, `p95` inter-chunk gap `212.3 ms`, **`0/229` stalls**) vs
    `full_response_tts` `6,261.0 ms` (**`7.59×` TTFA speedup**, **`49.15×`**
    post-TTFT audio onset speedup; isolated `CONTROLLED_LIVE` and perceptual
    quality remain `NOT_RUN` / `NOT_COVERED`).

- Wave 21 live Qwen3.8 CUDA Graph, Wan2.2 Batched-CFG, native `i2v_chain` continuity & Stage C controlled/GRPO gates (`2f331bd`, `bb4c471`, `f2fa842`, `eb30eba`, `39412ff`):
  - **GPU 3 `umm-native-stage-qwen38`** (`bb4c471` lazy per-slot `m=1` 64-layer
    decode CUDA Graph with device-side mRoPE and separate `kv_pos`/`rope_pos`,
    preserving `.wave20` backup): hot-swapped live (`load_ms=9,608`, `61,325 MiB`
    steady, `+44 MiB` graph overhead, `0` restarts on GPUs 2/4/5/6). Measured
    live on `43301` (`--batch-size 8`, `temperature=0.0`): server
    **`decode_tok_per_s = 53.6`** (`18.94 ms/tok`), client SSE **`51.74 ->
    53.26 tok/s`** (`+37.7%` vs Wave 15 `38.95 tok/s`, bit-identical 165 chunks
    / 749 chars / `sha=0a1bda565320e74b`, multimodal image VQA verified).
  - **GPU 6 `umm-native-stage-video`** (`39412ff` Batched-CFG `B=2`
    `WanDit::forward_cfg()`, `grid.z=2` FlashAttention-2, GPU patchify/unpatchify
    + fused CFG combine, dedicated non-blocking stream + pinned host buffers,
    preserving `.wave20` backup): measured on isolated SM90 GPU 1 (`704x480` 29f
    15 steps `seed=42`, GPU 1 returned to `3 MiB`): lossless `dit_ms`
    **`6,967 -> 6,655 ms`**, total **`11.16 -> 10.73 s`** (`2.24x` vs Wave 14
    `24.07 s`), TeaCache `0.05` **`8.86 -> 8.59 s`** (`2.80x` vs Wave 14),
    1-step Batched vs Sequential `PSNR=48.71 dB`. Hot-swapped live on GPU 6
    (`20,859 MiB` steady).
  - **Gateway `/bin/umm-infer`** (`eb30eba` opt-in `narrative.continuity_mode`
    `off|prompt_anchor|i2v_chain` + `narrative.style_anchor`, preserving `.wave20`
    backup): hot-swapped live; end-to-end 5-modality `POST /v1/chat/completions`
    with `continuity_mode="i2v_chain"` and `video.teacache_thresh=0.05` completed
    in **`12.53 s`** (`TTFT=0.91 s`), automatically feeding Slot 0 FLUX.2 PNG
    into Slot 1 Wan2.2 I2V (`request_id=http-1:slot:1 i2v=1 teacache_hits=10/30
    mp4_bytes=185,890 dit_ms=4,447 elapsed_ms=9,257`).
  - **Stage C Controlled Routing & Real SM90 Joint-GRPO Gates** (`2f331bd`,
    `f2fa842`): recorded the accepted **`CONTROLLED_LIVE / COMPLETED`** SM90
    narrative-routing gate on isolated GPUs 0/1 (24 measured + 8 warmup requests,
    25/25 `SHA256SUMS`, 36 Safetensors manifests) and the bounded real SM90 GPU 7
    Joint-GRPO optimizer gate (`Qwen2.5-0.5B` BF16, `clip_eps=0.2`, Schulman KL
    `0.04`, 290/290 nonzero grad tensors, exact fresh-process resume, completing
    the SFT + DPO + GRPO triad; online rollout & quality claims remain `NOT_RUN`).

- Wave 18 CPU-only native video temporal keyframe selection:
  - Added opt-in `video_url.frame_selection.policy=temporal` with bounded
    scene/motion selection (64 candidates -> at most 8 frames), chronological
    source timestamps, deterministic tie-breaking/fallback, and unchanged
    default uniform decode bytes. Existing 32 MiB input and 3584-patch budgets
    remain in force; native temporal-2 packing, video prompt, 3D MRoPE, batched
    multi-slice ViT, and serial/batched engine calls are shared.
  - Synthetic FFmpeg/host, Stage schema, public HTTP, C++ client/CLI, narrative,
    and deterministic benchmark checks cover policy propagation and edge cases.
    `tests/video_temporal_live.py` is the later SM90 uniform-vs-temporal
    motion/question A/B harness. This task accessed no GPU or resident service:
    **REAL_MODEL_QUALITY: NOT_RUN; LIVE_LATENCY: NOT_RUN; deployed: NOT_RUN**.

## Validation, 2026-11-19 (historical)

Offline supervisor coverage now lives in `tests/umminfer_service_test.py`
(10 tests). On 2026-11-19 both launch paths were also exercised live on GPU 6 (ASR,
Qwen3.8-27B) and GPU 7 (image, TTS), because GPUs 2-5 are held by the resident
43301 service.

### Unsupervised path

```sh
UMM_NATIVE_ROOT=/opt/umminfer/umm-native-verify UMM_NATIVE_PORT=43302 \
UMM_NATIVE_IPC=/tmp/umminfer-native-verify-ipc \
UMM_NATIVE_GPU_ASR=6 UMM_NATIVE_GPU_VLM=6 \
UMM_NATIVE_GPU_IMAGE=7 UMM_NATIVE_GPU_TTS=7 \
  /opt/umminfer/native-service.sh start
```

| Check | Result |
| --- | --- |
| Stage binaries | installed from `build/umm-native-stage-*`; asr SHA-256 `1bf9485f...cf072` matches the CMake output byte for byte |
| Server binary | `zig-out/bin/umm-infer`, SHA-256 `5953963c...30864` |
| Stage start | asr pid 1065000 gpu 6, qwen38 pid 1065003 gpu 6, image pid 1065006 gpu 7, tts pid 1065009 gpu 7 |
| `/health` | 200 |
| `/ready` | 200 at 11 s (entry and VLM stages registered; the stages log `ready` at 4.1 s and 10.0 s) |
| All four stages | `native stage <kind> ready` in every log |
| One chat request | HTTP 200, `{"content": "ok", "role": "assistant"}` |
| `stop` | all pids gone, port 43302 closed |
| GPU after stop | GPU 6 3 MiB, GPU 7 3 MiB (before start: 3 MiB each) |

The stage binaries were not rebuilt during this check; they were installed from
an existing `native/stage_cli` build directory whose asr output is byte
identical to the binary used in the accepted benchmark baseline.

### Supervised path

The same root and GPUs, with a copy of `service/native-service.json` edited for
port 43302, its own `run_dir`, `log_dir` and `ipc_dir`, and GPUs 6/6/7/7:

```sh
python3 -I /opt/umminfer/umm-native-verify/umminfer_service.py \
  --config /opt/umminfer/umm-native-verify/config.json start
python3 -I /opt/umminfer/umm-native-verify/umminfer_service.py \
  --config /opt/umminfer/umm-native-verify/config.json status
python3 -I /opt/umminfer/umm-native-verify/umminfer_service.py \
  --config /opt/umminfer/umm-native-verify/config.json stop
```

| Check | Result |
| --- | --- |
| `start` | `{"started": true, "supervisor": {"pid": 1066623}}` |
| `status` | server 1066624 and the asr, qwen38, image and tts children `alive`, `restarts: 0` |
| `/health`, `/ready` | 200 / 200 |
| `stop` | `{"stopped": [{"pid": 1066623, "signal": "SIGTERM"}]}`; every child then `alive: false` |
| GPU after stop | GPU 6 3 MiB, GPU 7 3 MiB |

## History

Through 2026-11-15 the service ran the phase-7 topology: a Python supervisor
driving `umminfer_qwen_asr_stage.py` and `umminfer_qwen_omni_worker.py`, which
called two pre-existing SGLang backends on GPUs 0 and 1 over loopback. That
topology is retired. Its deployment record, the SGLang abort evidence and the
chat-contract fix history stay in git history at commit `04bf1b0` and earlier.
