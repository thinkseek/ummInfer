# Multimodal benchmark results — H20B, 2026-11-19

First recorded real-model run of `tests/multimodal_benchmark.py` against the
native service. The accepted baseline is run D below: 54 cases, 0
infrastructure failures, 0 public-contract failures, 1 quality failure.

## 1. Result

| Task | Cases | Successes | Metric | Value | p50 ms | p95 ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| ASR (LibriSpeech test-clean) | 16 | 16 | corpus WER / CER | 0.0329 / 0.0274 | 2341 | 5329 |
| GSM8K exact | 16 | 16 | exact-match accuracy | 1.0000 | 2899 | 3768 |
| Image VQA (VQAv2 val) | 16 | 16 | VQA accuracy | 0.8958 | 2435 | 3790 |
| Joint image + real speech | 6 | 6 | ASR WER / keyword hit | 0.0000 / 1.0000 | 2810 | 4242 |
| **Total** | **54** | **54** | | | 2679 | 3790 |

Failure counts: infrastructure 0, public_contract 0, quality 1, performance 0.
`successes` follows the harness definition: cases with no infrastructure or
public-contract failure. A quality miss does not reduce it.

Verdict: `BASELINE_ESTABLISHED`. Wall time for the 54 cases: 146.9 s.

The one quality failure is `vqa2-val-262197000` ("How many skyscrapers are
there?"): the model answered `1`; the ten reference answers are six `2` and
four `0`. Genuine model miss, not a harness defect.

## 2. Runs

Four full 54-case runs were made. Only run D is the accepted baseline.

| Run | Evidence | Verdict | infra | contract | quality | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | `/tmp/mmbench/evidence-real` | `BASELINE_ESTABLISHED` | 0 | 0 | 4 | Harness still had the two bugs fixed below; not used as the baseline |
| B | `/tmp/mmbench/evidence-v3` | `BASELINE_REFUSED` | 1 | 0 | 3 | One ASR `RemoteDisconnected` |
| C | `/tmp/mmbench/evidence-v4` | `BASELINE_REFUSED` | 0 | 1 | 3 | In-band `image too large for slice` error event |
| D | `/tmp/mmbench/evidence-v5` | `BASELINE_ESTABLISHED` | 0 | 0 | 1 | Accepted baseline |

Run B was started immediately after a previous run was killed mid-request.
Both failures are intermittent; section 7 records the reproduction.

## 3. Environment

| Item | Value |
| --- | --- |
| Host | H20B, 8 × H20 97871 MiB, driver 535.161.08 |
| Container | `linyu-dev` |
| GPUs used | 6 (ASR), 7 (Omni). GPU 0/1 untouched |
| Service port | `172.17.0.2:43301` (43300 untouched) |
| Client | Mac, harness connects through `ssh -L 43301:172.17.0.2:43301 h20b` |
| Service commit | `a83ae44` (`umm-infer` built from it) |
| Harness commit | `1a0e321` (adds the three fixes in section 6) |
| ASR model | `/root/omniinfer-model-validation-20261111-k3/models/asr`, 1.8 G |
| Omni model | `/root/omniinfer-model-validation-20261111-k3/models/omni`, 66 G, 15 shards |
| ASR `config.json` SHA-256 | `76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c` |
| Omni `config.json` SHA-256 | `eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d` |
| ASR weights SHA-256 | `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea` |
| `umm-infer` SHA-256 | `5953963c1030433b54af62a4aef74e718b8866337f2367926f09410a58430864` |
| `umm-native-stage-asr` SHA-256 | `1bf9485f9a103b0b14e946e0235b2a7397271251184c4f6f47ecadad106cf072` |
| `umm-native-stage-omni` SHA-256 | `8ca0225d082075f57030fc4a62f3ef793645bc8721bea74f8307fa458f4dadd8` |
| Process PIDs (run D) | server 1054714, ASR 1054719, Omni 1054720 |

The service for run D was started fresh: server first, then the ASR stage on
GPU 6 and the Omni stage on GPU 7. `/health` and `/ready` were both 200 before
the run. Model load times: ASR 4145 ms, Omni 15824 ms. GPU memory at run
time: GPU 6 2389 MiB, GPU 7 78757 MiB.

The deployed binary is built from `a83ae44`. The three harness fixes in
section 6 are Python-only and do not change the C++ binary.

### Build and start notes

- The Torch CUDA 13 runtime needs a newer driver than 535. The container ships
  the forward-compatibility library at `/usr/local/cuda-13/compat`. Add it to
  `LD_LIBRARY_PATH` for any SGLang or Torch process; without it, Torch reports
  `driver too old (found version 12020)` and sees no device.
- `umm-infer` comes from `zig build`. The five `umm-native-stage-*` binaries
  come from the separate CMake project `native/stage_cli`
  (`-DUMM_STAGE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90`). `zig build` does not
  build them.

## 4. Data

All 54 inputs were downloaded and SHA-256 verified on the Mac:
`PASS data: 54 cases verified (manifest 2026-11-15.1)`.

| Source | Cases | Verified object | SHA-256 |
| --- | ---: | --- | --- |
| LibriSpeech test-clean | 16 | `test-clean.tar.gz`, 346663984 B | `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23` |
| GSM8K test | 16 | `test.jsonl` | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| VQAv2 val | 16 | questions zip | `e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a` |
| VQAv2 val | 16 | annotations zip | `0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e` |
| VQAv2 val | 16 | 16 images | per-image SHA-256 from the manifest |
| SpeechCoco | 6 | revision `ba337542de8f4ee66cfc285b5eeb69c326dc1576` | per-audio and per-image SHA-256 from the manifest |

The LibriSpeech archive was fetched with eight parallel range requests from
`openslr.trmal.net`; a single stream ran at about 129 KB/s. The archive size
and SHA-256 match the manifest exactly.

Container egress for this run: `openslr.trmal.net`, S3 and
`images.cocodataset.org` reachable; `raw.githubusercontent.com` and
`huggingface.co` timed out. Data was therefore fetched on the Mac. The
container network was also slower (openslr about 20 KB/s), so a container-side
download was not used.

## 5. Harness self-test

```sh
python3 -I tests/multimodal_benchmark_test.py
```

Result: `PASS contract test: 54 cases, full evidence layout, metrics match the
replay`. The test starts a local mock server, runs the real runner over the
real manifest and the cached inputs with `--evidence-class contract-test`, and
checks the evidence layout, the failure counts and the scores against the
replayed reference. The mock repeats each case's reference answer, so a correct
harness reports WER 0, GSM8K 1.0, VQA 0.979 and joint WER 0 / keyword hit 1.0.

`tests/multimodal_benchmark_test.py` did not exist before this task, although
`docs/benchmark-system.md` section 7 and the runner docstring both referenced
it. It was added in `8e6152a`.

## 6. Harness defects found and fixed

1. **Token rate divided by a near-zero interval** (`6776c09`). A one-token
   answer makes `first_delta` and `last_delta` the same clock reading, so
   `stream_ms` is near zero. VQA reported `tokens_per_s_p50 = 3.4e6`. The
   basis now falls back to `total_ms` when the streaming span is under 1 ms.
2. **VQA punctuation replaced with a space** (`61f9c69`). `vqa_normalize`
   used `re.sub(r"[^\w\s]", " ", ...)`, so `FOODIEBAKER.COM` became
   `foodiebaker com` and never matched the manifest answer `foodiebakercom`.
   The documented behaviour is punctuation removal; the substitution is now
   deletion. This raised `vqa2-val-393225000` from 0.0 to 1.0.
3. **In-band error event reported as a contract violation** (`1a0e321`). When
   the server ends a stream with an `error` event and no `finish_reason`, the
   harness reported `missing_finish_reason` and hid the real message. It now
   reports `stream_error_event` as an infrastructure failure and skips the
   missing-marker checks for that stream.

## 7. Intermittent failures observed

These are real and reproduced or measured; they are not counted as passes.

1. **Omni image preprocessing rejects a valid image, intermittently.**
   `vqa2-val-262189000` ended with
   `{"error":{"code":500,"message":"image too large for slice",
   "type":"generation_error"}}`. The limit is `kMaxImgTokens = 1296` in
   `native/omni/vision_forward.cu:11`. The same image bytes (SHA-256
   `5b6076cc09e50c0b0603510010a9c40c4891caefdd73b4232d4c43b624ae14f4`)
   succeeded in runs A, B and D and failed in run C. A sequential reproduction
   of three identical requests failed once (1/3). The input is identical, so
   this is state in the engine or its scratch buffers, not a fixed size limit.
2. **One ASR request was closed without a response.** Run B,
   `asr-en-ls-1089-134691-0000`: `RemoteDisconnected: Remote end closed
   connection without response`. A 40-request sequential stress test of
   `/v1/audio/transcriptions` afterwards returned 40/40 HTTP 200. Not
   reproduced.
3. **Greedy decoding is not run-to-run deterministic.** `gsm8k-test-0008`
   ended at `#### 75` in run A and `#### 45` (the gold answer) in run D;
   `gsm8k-test-0012` ended at `#### 12` in run A and `#### 13` (gold) in run D.
   Both runs used `temperature=0`. The Omni log shows `prefix_reused=64` on
   some requests, so KV or prefix reuse is a likely source. This matters for
   the stability gate: a future run can move GSM8K accuracy between 0.875 and
   1.0, which is outside the 0.05 quality tolerance.

## 8. Coverage and gaps

| Capability | Status |
| --- | --- |
| English ASR (16 cases) | Covered, real model, real audio |
| Mandarin ASR | `NOT_COVERED` — no licensed test set in the manifest |
| GSM8K exact answer (16) | Covered |
| Image VQA (16) | Covered |
| Joint image + real speech (6) | Covered; objective sub-metrics only, free-form answer needs human review |
| Multi-round stability | `NOT_RUN` — one round only; three consecutive rounds are required for a stability verdict |
| Token counts | `word_estimate` for chat (no `usage.completion_tokens` in the stream) |
| AOCI cognition maintenance | `NOT_COVERED` — no AOCI MCP tools in this session |

The joint task works through the unified path: the ASR stage transcribes the
spoken caption, then the Omni stage answers with the image and the transcript.
The native Omni engine does not accept audio directly
(`native/OMNI_NATIVE.md`), and the benchmark does not send audio to it.

Repository documentation drift found during this run:

- `docs/h20b-service.md` describes the retired phase-7 topology (Python
  workers plus SGLang backends). The current head is native-only.
- `service/h20b-native-service.sh` and `README.md` invoke
  `umm-native-stage-*` from `zig build`; those binaries come from the separate
  CMake project `native/stage_cli`.

## 9. Per-case results

All values come from `/tmp/mmbench/evidence-v5/round-1/cases.jsonl`.

| case | task | metric | value | total ms | failures |
| --- | --- | --- | ---: | ---: | --- |
| asr-en-ls-1089-134686-0000 | asr | wer | 0.0714 | 3288 | - |
| asr-en-ls-1089-134691-0000 | asr | wer | 0.0000 | 3701 | - |
| asr-en-ls-1188-133604-0000 | asr | wer | 0.1053 | 2460 | - |
| asr-en-ls-121-121726-0000 | asr | wer | 0.0000 | 1893 | - |
| asr-en-ls-121-123852-0000 | asr | wer | 0.1034 | 3278 | - |
| asr-en-ls-121-123859-0000 | asr | wer | 0.0270 | 3472 | - |
| asr-en-ls-121-127105-0000 | asr | wer | 0.0000 | 2341 | - |
| asr-en-ls-1221-135766-0000 | asr | wer | 0.0000 | 2820 | - |
| asr-en-ls-1221-135767-0000 | asr | wer | 0.0556 | 5329 | - |
| asr-en-ls-1284-1180-0000 | asr | wer | 0.0000 | 2015 | - |
| asr-en-ls-1284-1181-0000 | asr | wer | 0.0000 | 1344 | - |
| asr-en-ls-1284-134647-0000 | asr | wer | 0.0476 | 2307 | - |
| asr-en-ls-1320-122612-0000 | asr | wer | 0.0000 | 2990 | - |
| asr-en-ls-1320-122617-0000 | asr | wer | 0.0000 | 1868 | - |
| asr-en-ls-1580-141083-0000 | asr | wer | 0.0333 | 2066 | - |
| asr-en-ls-1580-141084-0000 | asr | wer | 0.0000 | 1326 | - |
| gsm8k-test-0000 | gsm8k | gsm8k_exact | 1.0000 | 2899 | - |
| gsm8k-test-0001 | gsm8k | gsm8k_exact | 1.0000 | 1454 | - |
| gsm8k-test-0002 | gsm8k | gsm8k_exact | 1.0000 | 3057 | - |
| gsm8k-test-0003 | gsm8k | gsm8k_exact | 1.0000 | 2386 | - |
| gsm8k-test-0004 | gsm8k | gsm8k_exact | 1.0000 | 3768 | - |
| gsm8k-test-0005 | gsm8k | gsm8k_exact | 1.0000 | 2679 | - |
| gsm8k-test-0006 | gsm8k | gsm8k_exact | 1.0000 | 1994 | - |
| gsm8k-test-0007 | gsm8k | gsm8k_exact | 1.0000 | 3067 | - |
| gsm8k-test-0008 | gsm8k | gsm8k_exact | 1.0000 | 3606 | - |
| gsm8k-test-0009 | gsm8k | gsm8k_exact | 1.0000 | 2829 | - |
| gsm8k-test-0010 | gsm8k | gsm8k_exact | 1.0000 | 3015 | - |
| gsm8k-test-0011 | gsm8k | gsm8k_exact | 1.0000 | 3492 | - |
| gsm8k-test-0012 | gsm8k | gsm8k_exact | 1.0000 | 3108 | - |
| gsm8k-test-0013 | gsm8k | gsm8k_exact | 1.0000 | 2855 | - |
| gsm8k-test-0014 | gsm8k | gsm8k_exact | 1.0000 | 2626 | - |
| gsm8k-test-0015 | gsm8k | gsm8k_exact | 1.0000 | 3291 | - |
| vqa2-val-262148000 | vqa | vqa_accuracy | 1.0000 | 2215 | - |
| vqa2-val-393225000 | vqa | vqa_accuracy | 1.0000 | 2560 | - |
| vqa2-val-393226000 | vqa | vqa_accuracy | 1.0000 | 2781 | - |
| vqa2-val-240301000 | vqa | vqa_accuracy | 0.3333 | 2985 | - |
| vqa2-val-131089000 | vqa | vqa_accuracy | 1.0000 | 1922 | - |
| vqa2-val-262162000 | vqa | vqa_accuracy | 1.0000 | 2083 | - |
| vqa2-val-393243000 | vqa | vqa_accuracy | 1.0000 | 2233 | - |
| vqa2-val-262175000 | vqa | vqa_accuracy | 1.0000 | 3278 | - |
| vqa2-val-131108000 | vqa | vqa_accuracy | 1.0000 | 2435 | - |
| vqa2-val-393254000 | vqa | vqa_accuracy | 1.0000 | 2921 | - |
| vqa2-val-42000 | vqa | vqa_accuracy | 1.0000 | 2671 | - |
| vqa2-val-131115000 | vqa | vqa_accuracy | 1.0000 | 2641 | - |
| vqa2-val-262189000 | vqa | vqa_accuracy | 1.0000 | 3790 | - |
| vqa2-val-393266000 | vqa | vqa_accuracy | 1.0000 | 2304 | - |
| vqa2-val-393267000 | vqa | vqa_accuracy | 1.0000 | 1770 | - |
| vqa2-val-262197000 | vqa | vqa_accuracy | 0.0000 | 1975 | vqa_wrong |
| joint-speechcoco-val-0000 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 2930 | - |
| joint-speechcoco-val-0001 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 2185 | - |
| joint-speechcoco-val-0002 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 4242 | - |
| joint-speechcoco-val-0003 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 2810 | - |
| joint-speechcoco-val-0004 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 2678 | - |
| joint-speechcoco-val-0005 | joint | joint_asr_wer_keyword_hit | asr_wer=0.000, keyword_hit=1.000 | 2835 | - |

## 10. Reproduce

```sh
# 1. Inputs (idempotent, SHA-256 checked)
python3 -I tests/multimodal_benchmark.py --verify-data

# 2. Offline harness check
python3 -I tests/multimodal_benchmark_test.py

# 3. Service: SGLang is not on the path. Build the server and the stages:
zig build -Doptimize=ReleaseFast \
  -Dcurl-prefix=/root/umm-c6rt-ckl/curl-prefix \
  -Ddeps-prefix=/root/umm-c6rt-ckl/deps
cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 16
# Then start umm-infer serve with --entry asr --vlm-entry omni, the ASR stage on
# GPU 6 and the Omni stage on GPU 7 (see section 3 for addresses and flags).

# 4. Benchmark (first full run establishes the baseline)
python3 -I tests/multimodal_benchmark.py \
  --base-url http://127.0.0.1:43301 --model qwen3-native \
  --evidence-dir /tmp/mm-evidence --evidence-class real --accept-baseline
```

## 11. Cleanup

| Item | Before stop | After stop |
| --- | ---: | ---: |
| GPU 6 used | 2389 MiB | 3 MiB |
| GPU 7 used | 78757 MiB | 3 MiB |

The server, ASR stage and Omni stage were stopped by PID. No `/root/mmbench/bin/`
process remains. The SSH tunnel on port 43301 was stopped and the port is free.
GPU 0/1 were never touched. The 43300 service was never started or stopped.
