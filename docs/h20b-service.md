# H20B resident service

This page is the runbook for the resident ummInfer service on H20B: one C++
server and one native C++/CUDA stage per model, all inside the container
`linyu-dev`. No Python model runtime, PyTorch or SGLang is on the serving path.
`python/` appears only as the optional process supervisor and as legacy code.

## Topology

| Piece | Process | GPU |
| --- | --- | --- |
| ummInfer server | `bin/umm-infer serve` on `172.17.0.2:43301` | none |
| ASR stage | `bin/umm-native-stage-asr` | 2 |
| VLM stage | `bin/umm-native-stage-qwen38` (default) | 3 |
| VLM stage, optional | `bin/umm-native-stage-omni` | 3 |
| Image stage | `bin/umm-native-stage-image` | 4 |
| TTS stage | `bin/umm-native-stage-tts` | 5 |
| Supervisor, optional | `python3 -I python/umminfer_service.py --config service/h20b-service.json` | none |

Exactly one VLM stage runs at a time: `vlm_stage` in `service/h20b-service.json`
selects `qwen38` or `omni`, and only one of them may hold GPU 3. GPUs 0 and 1
are not part of this service.

Stages talk to the server over ZeroMQ and MessagePack, with payloads in POSIX
shared memory. Endpoints live under `ipc_dir` (`/tmp/umminfer-native-ipc` by
default): `asr`, `qwen38` or `omni`, `image`, `tts`, plus the shared
`completion` and `abort` edges.

## Address and exposure

The server binds `172.17.0.2`, the container's private bridge address. It does
not bind `0.0.0.0`, it does not bind the host public interface, and no Docker
port is published. The supervisor refuses a configuration whose `host` is
`0.0.0.0` or `::`.

A Mac cannot reach `172.17.0.2` directly. SSH `-L` on the H20B host reaches the
container bridge address:

```sh
/usr/bin/ssh -N -L 43301:172.17.0.2:43301 h20b
```

`127.0.0.1:43301` on the Mac then reaches the container service through the H20B
host.

## Install

Two build systems produce the artifact set. `zig build` builds only the server;
`native/stage_cli` builds the five stage binaries. Neither builds the other.

```sh
zig build -Doptimize=ReleaseFast \
  -Dcurl-prefix=/root/umm-c6rt-ckl/curl-prefix \
  -Ddeps-prefix=/root/umm-c6rt-ckl/deps

cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 16

install -m755 zig-out/bin/umm-infer build/umm-native-stage-* /root/umm-native-service/bin/
```

`service/h20b-native-service.sh` expects those binaries in `$UMM_NATIVE_ROOT/bin`
(`/root/umm-native-service/bin` by default). The supervisor reads
`service/h20b-service.json`, which names `binary`, `stage_binary_dir` and the
four `native_stages`.

Install the config next to the supervisor:

```sh
install -m644 service/h20b-service.json /root/umm-native-service/config.json
```

Record the identity of the deployed bytes:

```sh
sha256sum /root/umm-native-service/bin/umm-infer /root/umm-native-service/bin/umm-native-stage-*
git rev-parse HEAD
```

## Operate

Without supervision:

```sh
service/h20b-native-service.sh start
service/h20b-native-service.sh status
service/h20b-native-service.sh stop
```

With the supervisor, which restarts unhealthy children:

```sh
CFG=/root/umm-native-service/config.json
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
- Dependency. A server restart also restarts every stage, because the
  Coordinator loses the one-shot `stage_ready` registration the stages send at
  startup. Stages are started only after the server answers `/health`.
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
- `service/h20b-native-service.sh` does no supervision: it writes pid files,
  and `stop` signals those pids. Use the supervisor when restart matters.
- Neither launcher cleans up other users' processes. Each signals only the PIDs
  it recorded.
- GPUs 2-5 are hardcoded defaults in the config and in the script. A second
  stack on the same host must override `UMM_NATIVE_GPU_*` and
  `UMM_NATIVE_PORT`; the config file needs the same edit.

## Validation, 2026-11-19

There is no offline test for the supervisor; `tests/umminfer_service_test.py`
does not exist. Both launch paths were exercised live on GPU 6 (ASR,
Qwen3.8-27B) and GPU 7 (image, TTS), because GPUs 2-5 are held by the resident
43301 service.

### Unsupervised path

```sh
UMM_NATIVE_ROOT=/root/umm-native-verify UMM_NATIVE_PORT=43302 \
UMM_NATIVE_IPC=/tmp/umminfer-native-verify-ipc \
UMM_NATIVE_GPU_ASR=6 UMM_NATIVE_GPU_VLM=6 \
UMM_NATIVE_GPU_IMAGE=7 UMM_NATIVE_GPU_TTS=7 \
  /root/h20b-native-service.sh start
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

The same root and GPUs, with a copy of `service/h20b-service.json` edited for
port 43302, its own `run_dir`, `log_dir` and `ipc_dir`, and GPUs 6/6/7/7:

```sh
python3 -I /root/umm-native-verify/umminfer_service.py \
  --config /root/umm-native-verify/config.json start
python3 -I /root/umm-native-verify/umminfer_service.py \
  --config /root/umm-native-verify/config.json status
python3 -I /root/umm-native-verify/umminfer_service.py \
  --config /root/umm-native-verify/config.json stop
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
