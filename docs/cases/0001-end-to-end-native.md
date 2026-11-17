# Case 0001: one native end-to-end request on H20B

Status: the four native stages served one real request end to end. One request
with one `request_id` returned a text answer, a PNG and a WAV. The same request
was repeated four times. Abort was verified during the VLM decode. Resources
returned to baseline after `stop`.

One acceptance item does not hold: the SGLang service on port 43300 was already
down before this run (section 7).

## 1. Deployment

Source: commit `1f851a08ff685ad059b8a5496add4f6da35cc4f7`, `git archive HEAD`
SHA256 `aae4e74c10e4176ce4b0d7adca2985474f9b8329e39c104ebedc803e51c0170f`.

Clean rebuild inside container `linyu-dev`:

```sh
rm -rf .zig-cache zig-out build
/root/umm-c6rt-ckl/zig/zig build -Doptimize=ReleaseFast \
  -Dcurl-prefix=/root/umm-c6rt-ckl/curl-prefix \
  -Ddeps-prefix=/root/umm-c6rt-ckl/deps
cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

The Zig step took 27 s and the CMake step 19 s. The removed `build/` directory
was recreated with 52 object files, all with the rebuild timestamp; no old
`.o` was reused. The four stage binaries have different SHA256 values from the
previous deployment, so the new C++ stage code is in the served bytes. The
`umm-infer` build is byte-reproducible; the C++/CUDA stage builds are not. A
second clean build of the same commit produced the same `umm-infer` bytes and
different stage bytes (`asr eae959ff…`, `image 3a19836c…`, `omni 24d2ff20…`,
`tts 0b5973f4…`). Binary identity for a run was therefore checked with
`/proc/<pid>/exe` against the installed file, not by hash equality across
builds.

| Binary | SHA256 (installed) | SHA256 (previous deployment) |
| --- | --- | --- |
| `umm-infer` | `7a547c0c1e7da76c6adb243b887fdf46b386ccd2611b4d50c77a27ea813e3226` | same (same source, reproducible Zig build) |
| `umm-native-stage-asr` | `a06ea00e099fbb274a56566c392593ed2ce3c1c2cab4cfc52e8fa1d5b7c87cda` | `52e760c5a94675ce95d5e6fa8caef42a3d56d6e226f537e37bbaece1d57ff9ba` |
| `umm-native-stage-omni` | `0c72d7585ce34a71f97e7c0017c1df3d4f3bf93b270697bdf17e4eb9fa996024` | `a1a693bcc478054c8b14426095b460cf56e896d02bdb3cea19449f61ebb9171c` |
| `umm-native-stage-image` | `e0411a63767820e6209c023b609b42cfeb7dd8f449b8554ef144c116c3eca2e1` | `92b1836eb6f2287c3297449383558bf44ad3273a48166ff99cac0f652c6d619e` |
| `umm-native-stage-tts` | `5fd2b2df300509466e6acb85d1f40077d1ed0e6eac3e6e13f6adce179bf76469` | `0b7fb87df20197391be24bcce60352495edd7221ed97083c89b81920adf12211` |

The table is the first clean build, which served runs 1-3. The reproduction
run rebuilt from the same commit and installed the second-build stage hashes
listed above.

`/proc/<pid>/exe` SHA256 for the five running processes matched the installed
files above at the start of run 1. Strings markers present in the served binaries: `image-entry`
(3), `tts-entry` (3), `output_modalities` (1), `image_options` (2) in
`umm-infer`; `image request size=`, `omni request prompt_tokens=`,
`tts request voice=`, `asr request audio_samples=` in the matching stage.

Resident service (`python3 -I python/umminfer_service.py --config config.json`):

| Piece | PID | GPU | Endpoint |
| --- | --- | --- | --- |
| supervisor | 529298 | - | `/root/umm-native-service/run` |
| server | 529299 | - | `172.17.0.2:43301` |
| ASR stage | 529532 | 2 | `ipc:///tmp/umminfer-native-ipc/asr` |
| Omni stage | 529534 | 3 | `ipc:///tmp/umminfer-native-ipc/omni` |
| Image stage | 529536 | 4 | `ipc:///tmp/umminfer-native-ipc/image` |
| TTS stage | 529538 | 5 | `ipc:///tmp/umminfer-native-ipc/tts` |

`/health` returned `{"in_flight":0,"status":"ok"}` and `/ready` returned 200
after the stages loaded. A later restart for the third run produced supervisor
538151, server 538152, ASR 538384, Omni 538386, Image 538388, TTS 538390; the
same addresses and GPUs.

## 2. Request

Input audio: `/root/umm-e2e/asr_en_pcm16.wav`, 16 kHz mono PCM16, 240,820
samples (15.051 s), 481,718 bytes, SHA256
`64fbd9f7ecafc160ba457595570f4a8b338f10e16acc0933be30aedc7eedf86d`.

Request body (`input_audio.data` omitted; the redacted JSON SHA256 is
`ac87fea046cb6a3c0d4a96b74f97d4dca5f234d1b9a236667bf58f0cfd83251f`):

```json
{
  "model": "qwen3-native",
  "stream": true,
  "temperature": 0,
  "max_tokens": 64,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Listen to the audio. In one sentence, state the speaker's main point, then name one object that would make a good illustration."},
    {"type": "input_audio", "input_audio": {"data": "<omitted>", "format": "wav"}}
  ]}],
  "modalities": ["text", "image", "audio"],
  "image": {"prompt": "A songwriter at a grand piano in a warm studio, soft window light, photographic, 512x512",
            "size": "512x512", "seed": 42, "output_format": "png"},
  "audio": {"voice": "vivian", "format": "wav", "sample_rate": 24000}
}
```

HTTP status 200. First SSE frame after 15 ms. Event order for run 1
(`request_id` `http-1`):

| t (s) | kind | stage | note |
| --- | --- | --- | --- |
| 0.015 | role frame | - | no `unified` object |
| 1.033 | `asr_final` | asr | transcript below |
| 112.520 | `artifact` | tts | `art-0`, WAV |
| 230.878 | `artifact` | image | `art-1`, PNG |
| 230.878 | answer delta | - | full VLM answer in one chunk |
| 230.878 | `complete` | vlm | `legs`: image completed, tts completed |

The mixed-output stream delivers the answer after the output legs finish. The
native Omni stage returns one aggregated result, so there are no per-token
`vlm_delta` events in this path.

## 3. Outputs

Transcript (verbatim):

> Oh yeah, yeah. He wasn't even that big when I started listening to him. But in his solo music, didn't do overly well, but he did very well when he started writing for other people.

Answer (verbatim, `\n` shown as a line break):

> The speaker's main point is that an artist had more success writing songs for others than he did with his own solo music.
> An object that would make a good illustration is a songwriting notebook.

Artifacts (paths inside the container; the same bytes were copied to the Mac
evidence directory):

| Artifact | Path | Bytes | SHA256 | Geometry / format |
| --- | --- | --- | --- | --- |
| PNG | `/root/umm-e2e/out/run1/art-1.png` | 486,463 | `65b506b879c695ec4ab1f198f201ffa3b405c2311fe0ac5eb949c060e6517a34` | 512x512, 8-bit PNG, seed 42, 50 steps |
| WAV | `/root/umm-e2e/out/run1/art-0.wav` | 883,244 | `54a57092c21036a7c930b1919314cab7f49d1d74c9745acc018e00507e489ede` | 24 kHz, mono, PCM16, 441,600 samples, 18.4 s |

WAV level from the TTS stage log: RMS 0.0489132, peak 0.42575, non-silence
fraction 0.601404. The artifact descriptor's `sha256` matched the bytes
decoded from the stream for both artifacts.

Per-leg timing. Stage values are the `elapsed_ms` the stage itself wrote;
stream values are the event arrival times above.

| Leg | Run 1 | Run 2 | Run 3 | Source |
| --- | --- | --- | --- | --- |
| ASR | 1,014 ms | 1,010 ms | 1,023 ms | stage log; `asr_final` at 1.033 / 1.028 / 1.043 s |
| VLM total | 989 ms | 986 ms | 1,015 ms | stage log (78 prompt tokens, 40 generated tokens) |
| VLM prefill | ~111 ms | ~111 ms | ~114 ms | derived, see below |
| VLM decode | ~878 ms | ~875 ms | ~901 ms | derived, see below |
| Image leg | 228,837 ms | 227,487 ms | 228,646 ms | stage log (`load 9,995` + `encode 43` + `denoise 216,386` + `vae 764`) |
| TTS leg | 110,468 ms | 110,450 ms | 110,476 ms | stage log |
| Total to terminal | 230,878 ms | 229,518 ms | 230,723 ms | stream |

The VLM split is derived, not printed by the stage. The stage logs only the
total. Five calibration requests on the same Omni stage gave
`elapsed = a + 1.56 ms x prompt_tokens + 22.7 ms x generated_tokens`
(`a = -7 ms`, R2 = 0.9996) over these measured points: (78, 40, 990 ms),
(62, 2, 148 ms), (63, 96, 2,275 ms), (65, 96, 2,279 ms), (115, 32, 908 ms).
Applied to the request's 78 prompt and 40 generated tokens, the split is 11.2%
prefill and 88.8% decode of the measured total. The total itself is measured.

GPU memory peak during each request, sampled every 200 ms with `nvidia-smi`
(`memory.used` MiB):

| Run | GPU2 (ASR) | GPU3 (Omni) | GPU4 (Image) | GPU5 (TTS) | GPU6 (unused) |
| --- | --- | --- | --- | --- | --- |
| run1 | 13,508 | 89,876 | 64,160 | 18,322 | 11,125 |
| run2 | 2,387 | 78,755 | 53,039 | 7,201 | 3 |
| run3 | 2,387 | 78,755 | 53,113 | 7,201 | 3 |

Run 1 overlapped other tenants' containers, which held 11,118 MiB on each of
GPU2, 3, 4, 5 and 11,125 MiB on GPU6 (`nvidia-smi --query-compute-apps`
reported host PIDs that are not in `linyu-dev`). Run 2 and run 3 had no foreign
allocation. The image stage's first request after start loads the model, which
adds about 11 GiB and 9.5 s; the steady-state request peak on GPU4 is 53.0 GiB.

Server log tail (each file is far below the 64 KiB cap):

```
== asr.log (1017 bytes)
native stage model loaded kind=asr model=/root/omniinfer-model-validation-20261111-k3/models/asr gpu=2 load_ms=4685
native stage asr ready kind=asr
asr request audio_samples=240820 generated_tokens=46 chars=180 elapsed_ms=1014
asr request audio_samples=240820 generated_tokens=46 chars=180 elapsed_ms=1010
...
== omni.log (1225 bytes)
loaded 19218 thinker tensors, 57.87 GiB weights
native stage model loaded kind=omni model=/root/omniinfer-model-validation-20261111-k3/models/omni gpu=3 load_ms=16318
native stage omni ready kind=omni
omni request prompt_tokens=78 image_tokens=0 generated_tokens=40 chars=196 elapsed_ms=989
omni request prompt_tokens=78 image_tokens=0 generated_tokens=40 chars=196 elapsed_ms=986
...
== image.log (539 bytes)
native stage model loaded kind=image model=/root/umm-mixed-output-20261116/models/Qwen-Image-2512 gpu=4 load_ms=0
native stage image ready kind=image
image request size=512x512 steps=50 seed=42 png_bytes=486463 load_ms=9995 encode_ms=43 denoise_ms=216386 vae_ms=764 elapsed_ms=228837
image request size=512x512 steps=50 seed=42 png_bytes=486463 load_ms=8818 encode_ms=31 denoise_ms=216379 vae_ms=781 elapsed_ms=227487
...
== tts.log (609 bytes)
native stage tts ready kind=tts
tts request voice=vivian language=english samples=441600 duration_s=18.4 rms=0.0489132 peak=0.42575 non_silence=0.601404 wav_bytes=883244 elapsed_ms=110468
...
== server.log (38 bytes)
umm-infer serving on 172.17.0.2:43301
```

Full logs are in the evidence archive; the supervisor caps each child log at
exactly 65,536 bytes.

## 4. Repeatability

The redacted request body SHA256 was identical for all runs. Run 1 and run 2
used the same server process; run 3 used a fresh server process.

| Run | request_id | PNG SHA256 | WAV SHA256 |
| --- | --- | --- | --- |
| run1 | `http-1` | `65b506b879c695ec4ab1f198f201ffa3b405c2311fe0ac5eb949c060e6517a34` | `54a57092c21036a7c930b1919314cab7f49d1d74c9745acc018e00507e489ede` |
| run2 | `http-2` | same | `cd15bab461f92ee2fa22e114fe834374cdc20ca1a65da82774a93e6c76ed16e6` |
| run3 | `http-1` (new server) | same | `328af938e9d1898ca90075c51595a3673ad888c95082ae3e9d2e3c0aa24a954e` |
| repro (second clean build) | `http-1` (new server) | same | `8418def54403123fa0c0b40cfcff2a20db6b3eaebcad795bbcf4a12bc5a90eed` |

The PNG is bit-identical because the image seed is fixed at 42. The WAV is
not: all four files are 883,244 bytes, 441,600 samples, 18.4 s, and the
answers are identical, but the bytes differ. The cause was not isolated here.
The TTS leg takes no seed parameter, and the same text produced different
bytes in every run. BF16 reduction order in the codec is a candidate; that was
not verified. The text answer was identical in all runs.

## 5. Abort

Run `http-11` sent `POST /abort {"request_id":"http-11"}` 1.2 s after the first
frame, during VLM decode. The client sends the abort from a timer thread because
the SSE read blocks while the output legs run.

| Item | Value |
| --- | --- |
| Abort HTTP status | 200, body `{"success":true}` |
| Request events | role frame, `asr_final` (before the abort), `cancelled`; no artifact |
| Abort send to `cancelled` terminal | 0 ms (same millisecond) |
| Abort send to socket EOF | 1 ms |
| Omni stage line | `omni request aborted boundary="aborted at token 4" abort_epoch_ms=1794893297330 exit_epoch_ms=1794893297330 compute_ms=199` |
| `in_flight` after | 0 |

An earlier attempt sent the abort at 112.5 s, after the TTS artifact had been
produced. The cancel then reached the still-running image leg:
`image request aborted boundary="denoise step" abort_epoch_ms=1794893294200
exit_epoch_ms=1794893295381 compute_ms=114439`, which is 1,181 ms from abort
signal to compute exit. The image engine synchronizes once per denoise step.

No GPU process from the aborted request remained; the health check immediately
after returned `in_flight=0`.

## 6. Resource cleanup

After `stop`, the supervisor answered
`{"stopped": [{"pid": 529298, "signal": "SIGTERM"}], "supervisor_was_running": true}`.
A second stop after run 3 produced the same shape for PID 538151, and the
reproduction run stopped PID 543292 the same way. The final check after the
reproduction run: no live `umm-infer` or stage process, GPU0-GPU7 all 0 MiB,
ports 43300 and 43301 both `000`.

| Check | Result |
| --- | --- |
| Live `umm-infer` / stage / supervisor processes | none |
| GPU2, GPU3, GPU4, GPU5, GPU6 | 0 MiB each; GPU0/1 and GPU7 also 0 MiB |
| Port 43301 | `000` (closed) |
| Port 43300 | `000` (see section 7) |
| Orphans | the stopped supervisors (PIDs 529298, 538151, 543292) remain as zombie `python3` entries; container PID 1 does not reap, so they stay in `ps` with state `Z`. They hold no GPU, socket or child. |

No process outside this task was signalled at any point.

## 7. Failed item and limits

- Port 43300 is not 200. It was `000` at 05:13:52 UTC, before this task
  stopped or started anything, and `000` after cleanup. `nvidia-smi` shows no
  live SGLang process, and `ps` lists only `[sglang::...] <defunct>` entries
  from earlier phases. Commit `1f851a0` records that the phase-7 service and
  its SGLang backends were stopped when the native resident service was
  accepted. Restarting them is outside this task and would need the owner of
  GPU0/1 to decide.
- The VLM prefill/decode split is derived from a linear fit over five measured
  requests; the stage reports only the total. The total is measured.
- The native stage ignores the request `max_tokens`. Two calibration requests
  with `max_tokens` 1 and 41 both generated 40 tokens and stopped at EOS. The
  launcher cap is `--max-new-tokens 96`.
- The mixed-output path emits no incremental VLM deltas. The answer arrives in
  one chunk after both output legs complete, because the native Omni stage is
  non-streaming.
- The image stage loads its weights on the first request after start: about
  9.5 s and 11 GiB extra on GPU4. Later requests use 53.0 GiB.
- Stage logs carry no request id; lines are matched to runs by order and by the
  client's event timestamps.
- During run 1, other tenants allocated about 11 GiB on GPU2, 3, 4, 5 and
  GPU6. Those allocations are not part of this task and are the reason the run-1
  peak table is higher than run 2/3.

## 8. Reproduction

From a checkout of the recorded commit, with the WAV fixture present:

```sh
SSH_OPTS="-o ControlPath=$HOME/.ssh/cm/tn-h20b" sh docs/cases/0001_repro.sh
```

The script archives `HEAD`, copies the source and the client into the
container, does the clean rebuild, installs the binaries, starts the resident
supervisor, sends the same request, prints the answer and artifact hashes,
stops the service, and copies the evidence archive to `evidence/`.

Executed on 2026-11-17 against commit `1f851a0` with a second clean build:
exit 0, request `http-1` returned HTTP 200 in 231,107 ms. PNG SHA256
`65b506b8…` (same as the earlier runs), WAV SHA256 `8418def5…`, answer
identical, `asr_final` at 1.033 s, TTS artifact at 112.520 s, image artifact
and terminal at 231.107 s. The trailing `stop` returned
`{"stopped": [{"pid": 543292, "signal": "SIGTERM"}], "supervisor_was_running": true}`.
After it, `ps` showed no live umm process, GPU0-7 were 0 MiB, and ports 43300
and 43301 both answered `000`.

The client is `docs/cases/0001_e2e_client.py`; without the script it is:

```sh
python3 -I 0001_e2e_client.py run --url http://172.17.0.2:43301 \
  --audio input.wav --out out/run1 --timeout 1200
```

## 9. Evidence

Container: `/root/umm-e2e/out/{run1,run2,run3,abort1,abort2}/` with
`request.json`, `response.sse`, `events.json`, `art-1.png` / `art-0.wav`,
`gpu.csv`, `result.json`, `text.txt`, `transcript.txt`.

Mac copy: `/tmp/umm-e2e/umm-e2e-evidence.tgz` and
`/tmp/umm-e2e/umm-native-logs.tgz`. The reproduction script writes its own
archive to `evidence/case0001-out.tgz` in the working directory (2,213,516
bytes for the run recorded above). Binary artifacts are not committed.
