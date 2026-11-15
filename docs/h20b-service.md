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

## Public chat contract fix, 2026-11-15

The real multimodal baseline exposed two public chat contract defects. A GSM8K
request with a plain string `messages[].content` was rejected with 400
`unified chat content must be a content part array`, and a VQA image+text
request without an `input_audio` part was rejected with 400
`unified chat requires exactly one nonempty input_audio part`. The fix is one
scoped commit on top of `8ccae24`, titled "Accept string content and
audio-free image+text in unified chat". The exact commit SHA and every hash
below are also recorded in the deployment evidence JSON
(`/root/umm-phase7-service/contract-fix-20261115.json` in the container and the
Mac evidence directory); the code content is pinned directly by the source-file
SHA256 values in that file, so the record does not depend on a self-referential
commit hash.

Behavior after the fix:

- A plain string content and a text content-part array are both accepted.
- An image+text request with no audio skips the ASR leg and runs the VLM leg
  directly. The VLM-only leg keeps the pre-allocated HTTP request id; a
  generated id would orphan the SSE sink and hang the stream on the role
  frame. A streaming regression in `unified_contract_test.cpp` fails on the
  orphaned id.
- A request with an `input_audio` part keeps the existing ASR -> VLM path. At
  most one audio part, one image part, and one text unit are accepted.

Build and install used the same `git archive HEAD` procedure as the section
above. The Zig build is reproducible: rebuilding the same tree yields the same
binary SHA256 byte for byte, verified in the container. Only the `umm-infer`
binary and its source tree were replaced; the config and supervisor were not
overwritten (their SHA256 values are unchanged from the table above).

| Item | Value |
| --- | --- |
| Base commit | `8ccae24` |
| Source archive SHA256 | see deployment evidence JSON |
| Deployed binary SHA256 | `de94ed67195244ed14f0d5aa309b8fa4c1bd4655f9b6639b26c8f590f6c25e57` |
| Installed config SHA256 (unchanged) | `2f73ac41ac7745bc6dfd387b40a6444546e745d6f4ae28eb198726657355e12a` |
| Installed supervisor SHA256 (unchanged) | `b2dd0450a246c6e3e7f37593b353f022f950e2445d829322fb0c1d9c4ec2240f` |
| Container / address | `linyu-dev`, `172.17.0.2:43300`, sole `LISTEN` on `020011AC:A924` |
| Backends | reused, PID 34134 (ASR) and 35212 (Omni), unchanged across both restarts |
| Final supervisor | PID 228823, parent PID 1 |
| Final children | server 228824, ASR 229009, Omni 229011, all parent 228823, restarts 0 |
| Final health / ready | 200 / 200, `in_flight=0` |

The first build of this fix (`67d27af3…`) passed the local non-streaming tests
but its streaming canary hung: the VLM-only leg used a Coordinator-generated id
and the SSE sink never received its events. That pre-fix evidence is retained
under the Mac directory `umminfer-phase7-canary-before-fix/`
(4 `public_contract`, 2 `performance`, 0 `infrastructure`). The id fix is in
the final binary above; the post-fix canary is clean.

Post-fix canary, four real cases through the Mac SSH loopback tunnel
(`tests/multimodal_benchmark.py --cases`, subset verdict
`BASELINE_REQUIRED`/`INELIGIBLE_SUBSET` gate semantics, judged per case, not
claimed as a stable PASS):

| Case | Route | HTTP | `[DONE]` | Result |
| --- | --- | --- | --- | --- |
| `gsm8k-test-0000` | VLM-only string | 200 | yes | 68 frames, metric 1.0, gold answer 18 |
| `vqa2-val-262148000` | VLM-only image+text | 200 | yes | 4 frames, metric 1.0 |
| `asr-en-ls-1089-134686-0000` | ASR transcriptions | 200 | n/a | WER 0.0357 |
| `joint-speechcoco-val-0000` | ASR -> VLM joint | 200 | yes | 22 frames, WER 0.0, keyword hit 1.0 |

Post-fix failure counts: `infrastructure=0`, `public_contract=0`,
`quality=0`, `performance=0`. Raw SSE frames are in the Mac evidence directory
`umminfer-phase7-canary-contract-fix/round-1/events/`, with `cases.jsonl`,
`failures.json`, answers and transcripts alongside. The full 54-case
multi-round benchmark was not run; it stays with the controlling review.

## Round-1 rework: VLM-only boundary rules, 2026-11-15

Review found two boundary gaps in the change above. Both are fixed in the same
scoped commit (amended, still one commit on top of `8ccae24`, not pushed):

1. An audio-free request with no text and no image — for example an empty
   `content: []` array — previously passed the HTTP boundary and would reach
   the VLM as an empty prompt. `unified_input_parts` now rejects it with 400
   `unified chat without input_audio requires text or image content`. An empty
   string content is already rejected by the nonempty-string check.
2. Session history on a VLM-only turn recorded the empty ASR transcript as the
   user message, so the real text was lost. The engine now extracts the current
   turn's real user text from the string content or text parts
   (`user_text_from_inputs`, same shape handling as `build_vlm_inputs`) and
   stores that on a VLM-only turn; an audio turn still stores the ASR
   transcript.

New regression coverage: an HTTP 400 for the empty array and empty string, an
engine two-turn VLM-only history test (string turn then image+text turn, same
session) asserting the first turn's real user text and answer are carried into
the second VLM prompt, plus the existing string/array, image+text, streaming-id
and joint tests. Both new checks fail when the corresponding fix is reverted
(mutation-verified). The local gate stayed green: black-box 13/13 exit 0,
benchmark 44 tests OK, service contract 21 checks PASS, unified C++ all pass.

| Item | Value |
| --- | --- |
| Build commit (source of the deployed binary) | `04bf1b0ede8dae95329605d308f39573378d7ab9` |
| Build source archive SHA256 | `9ba5be1d7d6db38bd45f05d6df0002ab2922507ce4d8b8aa4068180ee46f1cc4` |
| Deployed binary SHA256 | `980ee8720e7039808ef4c02590a54a4fb968a2d49ca4a78575ddff629a9d0fb9` |
| Config / supervisor SHA256 | unchanged (`2f73ac41…` / `b2dd0450…`) |
| Final supervisor | PID 229938, parent PID 1 |
| Final children | server 229939, ASR 230124, Omni 230126, parent 229938, restarts 0 |
| Backends | reused, PID 34134 (ASR) and 35212 (Omni), unchanged |
| Listener | sole `LISTEN` on `172.17.0.2:43300` (`020011AC:A924`) |
| Health / ready | 200 / 200, `in_flight=0` |

Live boundary verification through the Mac tunnel against the resident service
(Mac evidence `umminfer-phase7-canary-r1/raw-http/empty-content-400.txt`):

- `content: []` (audio-free) -> HTTP 400
  `unified chat without input_audio requires text or image content`.
- `content: ""` -> HTTP 400 `message content string must be nonempty`.
- A valid string request in the same run -> HTTP 200.

A two-turn VLM-only history canary could not use the resident server because it
is started without `--session-history` and the config must not change. A
temporary, independent stack was started inside the container on port 43301 with
its own IPC directory and the same SGLang backends, then stopped; the resident
43300 service was never touched. Turn 1 sent plain string content
("Remember this exact phrase: purple kumquat ledger.") and turn 2 sent
image+text with the same `session_id` and `session_history=true`. The Omni
worker request log for turn 2 shows turn 1's real user string, the assistant
reply, and the current image+text turn — proving the history kept the real
user input. Both turns returned 200 with `vlm_delta` -> `complete` and
`[DONE]`. After stop the three canary processes are state `Z` (the container
PID 1 never reaps, same as the documented existing zombies), hold no listener
(port 43301 absent from `/proc/net/tcp`), and the resident 43300 listener and
SGLang PIDs are unchanged. Evidence:
`umminfer-phase7-canary-r1/history/{turn1-string.sse,turn2-image-text.sse,omni-worker-requests.log}`.

The four-case canary was rerun on the new resident binary
(`umminfer-phase7-canary-r1/canary4`): `infrastructure=0`, `public_contract=0`,
`quality=0`, `performance=0`, with GSM8K and VQA metric 1.0, ASR WER 0.0357,
joint WER 0.0 and keyword hit 1.0. The full 54-case multi-round benchmark was
again not run and stays with the controlling review.
