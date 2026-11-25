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
