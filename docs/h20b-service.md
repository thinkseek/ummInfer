# H20B resident service

This page is the runbook for the resident ummInfer service on H20B: one
supervisor, one C++ server and two Qwen workers, all inside the container
`linyu-dev`. The service reuses the two SGLang backends that were already
running. It never starts, stops, or reconfigures them.

## Topology

| Piece | Process | Address |
| --- | --- | --- |
| Supervisor | `python3 -I python/umminfer_service.py` | `/root/umm-phase7-service/run/` |
| ummInfer server | `bin/umm-infer serve` | `172.17.0.2:43300` |
| Qwen ASR worker | `python3 -I python/umminfer_qwen_asr_stage.py serve` | ZeroMQ `ipc:///tmp/umminfer-phase7-ipc/asr` |
| Qwen Omni worker | `python3 -I python/umminfer_qwen_omni_worker.py` | ZeroMQ `ipc:///tmp/umminfer-phase7-ipc/vlm` |
| ASR backend (existing) | SGLang `qwen3-asr` | loopback `127.0.0.1:31000`, GPU 1 |
| Omni backend (existing) | SGLang `qwen3-omni` | loopback `127.0.0.1:31001`, GPU 0 |

All four owned processes run inside the container. The Mac is only a client.

## Address and exposure

The server binds `172.17.0.2`, the container's private bridge address. It does
not bind `0.0.0.0`, it does not bind the host public interface, and no Docker
port is published. The supervisor refuses a configuration whose `host` is
`0.0.0.0` or `::`.

The binder is the container loopback in the sense that no host interface is
exposed; a Mac cannot reach `172.17.0.2` directly. SSH `-L` on the H20B host can
reach the container bridge address, so the client tunnel is:

```sh
/usr/bin/ssh -N -L 43300:172.17.0.2:43300 h20b
```

`127.0.0.1:43300` on the Mac then reaches the container service through the H20B
host. The tunnel listens on Mac loopback only.

## Install

Build the binary from the committed source on the H20B host, inside the
container, with the toolchain already present at `/root/umm-c6rt-ckl`:

```sh
git archive --format=tar.gz HEAD > /tmp/umm-phase7-src.tar.gz
scp /tmp/umm-phase7-src.tar.gz h20b:/root/
ssh h20b docker cp /root/umm-phase7-src.tar.gz linyu-dev:/root/
ssh h20b docker exec linyu-dev bash -lc '
  rm -rf /root/umm-phase7-service/src &&
  mkdir -p /root/umm-phase7-service/src &&
  tar xzf /root/umm-phase7-src.tar.gz -C /root/umm-phase7-service/src &&
  cd /root/umm-phase7-service/src &&
  /root/umm-c6rt-ckl/zig/zig build -Doptimize=ReleaseFast \
    -Dcurl-prefix=/root/umm-c6rt-ckl/curl-prefix \
    -Ddeps-prefix=/root/umm-c6rt-ckl/deps'
```

Install the scoped copy:

```sh
ssh h20b docker exec linyu-dev bash -lc '
  mkdir -p /root/umm-phase7-service/{bin,python,run,logs}
  install -m755 /root/umm-phase7-service/src/zig-out/bin/umm-infer \
    /root/umm-phase7-service/bin/umm-infer
  cp /root/umm-phase7-service/src/python/*.py /root/umm-phase7-service/python/
  cp /root/umm-phase7-service/src/service/h20b-service.json \
    /root/umm-phase7-service/config.json'
```

Record the identity of the deployed bytes:

```sh
ssh h20b docker exec linyu-dev sha256sum /root/umm-phase7-service/bin/umm-infer
sha256sum /tmp/umm-phase7-src.tar.gz
git rev-parse HEAD
```

## Operate

```sh
CFG=/root/umm-phase7-service/config.json
ssh h20b docker exec linyu-dev python3 -I \
  /root/umm-phase7-service/python/umminfer_service.py --config $CFG start
ssh h20b docker exec linyu-dev python3 -I \
  /root/umm-phase7-service/python/umminfer_service.py --config $CFG status
ssh h20b docker exec linyu-dev python3 -I \
  /root/umm-phase7-service/python/umminfer_service.py --config $CFG logs --name asr
ssh h20b docker exec linyu-dev python3 -I \
  /root/umm-phase7-service/python/umminfer_service.py --config $CFG stop
```

`start` forks twice, calls `setsid`, and redirects its own stdio to
`/dev/null`; the SSH command returns after the pid file appears. The supervisor
therefore does not depend on the interactive session that started it.
`status` prints JSON from `run/state.json`. `logs` prints at most 64 KiB from
the tail of one log.

## Supervisor behavior

- Ownership. Each child is spawned with its PID and kernel start time
  (`/proc/<pid>/stat` field 22 on Linux, `ps -o lstart=` elsewhere). `stop`
  and every signal path compare the current start time with the recorded one
  before signalling. A reused PID, or a hand-edited state file, receives no
  signal.
- Liveness. The server is probed on `GET /health` every 2 s after an 8 s
  startup grace. Three consecutive failures restart the server.
- Restart. A dead child is restarted with backoff `1, 2, 4, 8, 15, 30` s,
  capped at 30 s, reset after 60 s of stable uptime.
- Dependency. A server restart also restarts both workers, because the
  Coordinator loses the one-shot `stage_ready` registration that the workers
  send at startup. Workers are started only after the server answers
  `/health`.
- Logs. Child stdout and stderr are merged and capped at exactly 65536 bytes
  each (`server.log`, `asr.log`, `omni.log`); the supervisor keeps its own
  `supervisor.log` under the same cap. The cap is a rolling tail, not a
  truncation at process exit.
- Shutdown. `stop` sends SIGTERM to the supervisor, which terminates its
  children, writes `state.json` with `stopped=true`, and removes the pid file.
  If the supervisor is already gone, `stop` signals only child PIDs whose
  recorded start time still matches.

`/ready` is an admission gate, not a heartbeat: it returns 503 until both
workers have sent `stage_ready`. Use `/health` for liveness.

## Limits

- The supervisor is not registered with systemd or any init. A container
  restart loses the service; run `start` again.
- The container bridge address can change if the container network is
  recreated. The config must then be updated.
- Model revisions stay `UNKNOWN`; the backends were launched from local model
  directories without `.git` metadata. See
  [performance baseline](performance-baseline.md#model-snapshot-and-provenance).
- The supervisor does not clean up other users' processes. It signals only the
  PIDs in its own state file.

## Validation

Local contract test, no H20B needed:

```sh
python3 -I tests/umminfer_service_test.py
```

Live check through the SSH loopback forward, with the real backends:

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/h20b_service_live.py \
  --base-url http://127.0.0.1:43300 \
  --audio /tmp/umminfer-phase4-bench-evidence/inputs/asr_en_pcm16.wav \
  --audio-second /tmp/umminfer-phase4-bench-evidence/inputs/asr_zh.wav \
  --image /tmp/umminfer-phase4-bench-evidence/inputs/man_ironing.png \
  --evidence-dir /tmp/umminfer-phase7-service-evidence/live
```

The live check covers `/health`, `/ready`, a real ASR -> VLM stream, a joint
image + audio stream, a cancel during the VLM stream with `in_flight` returning
to zero, and a later request that completes.

The cancel assertion uses the terminal event as the boundary, not the `seq`
where `/abort` was posted. Tokens already queued when the abort arrives are
delivered before the `cancelled` terminal; the final run delivered 7 such
deltas after the abort sequence number. The worker log separately records the
backend confirmation
`SGLANG_ABORT_CONFIRMED rid=http-3:vlm@227690 finish=abort`.

## Recorded deployment, 2026-11-15

| Item | Value |
| --- | --- |
| Local commit | the commit that adds this page |
| Build source | `git archive HEAD` of that commit |
| Deployed binary SHA256 | `d00b97a8b5847c22afe5f3b0d5a1937de01fa9bf952442244b10cd1d1c9d3bfe` |
| Installed supervisor SHA256 | `b2dd0450a246c6e3e7f37593b353f022f950e2445d829322fb0c1d9c4ec2240f` |
| Installed config SHA256 | `2f73ac41ac7745bc6dfd387b40a6444546e745d6f4ae28eb198726657355e12a` |
| Container / GPUs | `linyu-dev`, ASR backend GPU 1, Omni backend GPU 0 |
| Server address | `172.17.0.2:43300`, state `LISTEN` on `020011AC:A924` in `/proc/net/tcp` |
| Docker published ports | none (`docker port linyu-dev` is empty) |
| Backends | reused, PID 34134 (ASR) and 35212 (Omni) before and after every step |
| Resident supervisor | PID 227436, parent PID 1 |
| Resident children | server 227437, ASR 227688, Omni 227690, parent 227436 |
| Child environment | `LD_LIBRARY_PATH=/root/umm-c6rt-ckl/deps/lib:/root/umm-c6rt-ckl/curl-prefix/lib` in all three `/proc/<pid>/environ` |

The first installation copied the supervisor before the `env` merge was added,
so the running process had source SHA `daabec11…` while the commit had
`b2dd0450…`. The rework stopped that supervisor through its own pid file,
copied the committed files again, and restarted. The table above is the state
after the restart; `rework1/rework.json` records the before and after hashes.
The three running children now carry the configured `LD_LIBRARY_PATH`.

Measured results:

| Check | Result |
| --- | --- |
| `/health`, `/ready` | 200 / 200; `in_flight=0` before and after every case |
| ASR -> VLM stream | `asr_final`, 30 `vlm_delta`, one `complete`; gap-free `seq`; real transcript and answer |
| Joint image + audio | one `asr_final`, 128 deltas, one `complete`; answer describes the real screenshot content (man ironing a blue shirt on a yellow SUV) |
| Cancel during VLM stream | `/abort` 200, one `cancelled` terminal, no event after it, `in_flight` back to 0, worker logged `SGLANG_ABORT_CONFIRMED rid=http-3:vlm@227690 finish=abort` |
| Recovery after cancel | one `complete`, non-empty answer |
| ASR worker SIGKILL | new worker PID within 1 s, `restarts=1`, Omni untouched |
| Server SIGKILL | new server PID within 1 s; both workers restarted; `/ready` 200 at 5 s |
| `stop` | supervisor SIGTERM; server, ASR and Omni PIDs gone; SGLang PIDs 34134/35212 alive |
| SSH session exit | supervisor PPID 1 and alive after the starting command returned |
| Log cap | local contract test writes 300000 bytes into each child log; the files stay at 65536 bytes and keep the newest tail |

Evidence is under `/tmp/umminfer-phase7-service-evidence/`. The final run is
`rework1/`: `rework1/rework.json` (stop, install hashes, start, parent PIDs,
listener, child environment), `rework1/live/live-run.log` and
`rework1/live/*-frames.json`, `rework1/live/omni-abort.txt`,
`rework1/final-state.txt`, `rework1/logs/service-state.tgz`. The earlier run is
in `live/`, `resilience/`, and `access/`. `SHA256SUMS` covers the directory. It
is not a committed artifact.

A stopped supervisor remains as a zombie (`state Z`) reparented to container
PID 1, because this container's PID 1 never reaps orphans; `ps` already lists
hundreds of such entries from earlier phases, including `[umm-infer] <defunct>`
PID 179057 from 2026-11-14. The stopped supervisors 225970 and 226747 and the
stopped server/worker PIDs likewise show no live process, socket, or child.
Removing those entries needs the container to run with an init that reaps, which
is outside this service's scope.
