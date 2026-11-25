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
