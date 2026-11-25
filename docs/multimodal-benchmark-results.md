# Multimodal benchmark results — SM90, 2026-11-19

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
| Host | SM90, 8 × H20 97871 MiB, driver 535.161.08 |
| Container | SM90 GPU container (`$CONTAINER`) |
| GPUs used | 6 (ASR), 7 (Omni). GPU 0/1 untouched |
| Service port | `<container-ip>:43301` (43300 untouched) |
| Client | Mac, harness connects through `ssh -L 43301:<container-ip>:43301 <gpu-host>` |
| Service commit | `a83ae44` (`umm-infer` built from it) |
| Harness commit | `1a0e321` (adds the three fixes in section 6) |
| ASR model | `/opt/umminfer/omniinfer-model-validation-20261111-k3/models/asr`, 1.8 G |
| Omni model | `/opt/umminfer/omniinfer-model-validation-20261111-k3/models/omni`, 66 G, 15 shards |
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
| Mandarin ASR (16 cases) | Covered and measured from 2026-11-19: FLEURS `zh_cn` test split, CC BY 4.0, scored by CER and by WER over jieba words. Corpus CER 0.0916 (58/633) and WER 0.0503 (17/338) over the 16 cases, section 17.3 |
| GSM8K exact answer (16) | Covered |
| Image VQA (16) | Covered |
| Joint image + real speech (6) | Covered; objective sub-metrics only, free-form answer needs human review |
| Multi-round stability | `PARTIAL` — five full 54-case rounds on 2026-11-19, section 12. Quality metrics reproduce exactly when no request drops (CV 0 over the clean rounds); latency CV exceeds 0.10 on three of the four tasks, so the latency figures are not a stable baseline |
| Token counts | `word_estimate` for chat (no `usage.completion_tokens` in the stream) |
| AOCI cognition maintenance | `NOT_COVERED` — no AOCI MCP tools in this session |

The joint task works through the unified path: the ASR stage transcribes the
spoken caption, then the Omni stage answers with the image and the transcript.
The native Omni engine does not accept audio directly
(`native/OMNI_NATIVE.md`), and the benchmark does not send audio to it.

Repository documentation drift found during this run, and its state on
2026-11-19 (section 13):

- `docs/native-service.md` described the retired phase-7 topology (Python
  workers plus SGLang backends). Fixed: rewritten for the native topology.
- `service/native-service.sh` did not state where its binaries come from.
  Fixed: the header now names both build systems.
- `README.md` was reported as invoking `umm-native-stage-*` from `zig build`.
  It already separated the two build systems at commit `44de086`; the only
  wrong sentence was "configured by `service/native-service.json`" for the
  shell-only path. Fixed.

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
  -Dcurl-prefix="${CURL_PREFIX:-/opt/umminfer/curl-prefix}" \
  -Ddeps-prefix="${DEPS_PREFIX:-/opt/umminfer/deps}"
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

The server, ASR stage and Omni stage were stopped by PID. No `/opt/umminfer/mmbench/bin/`
process remains. The SSH tunnel on port 43301 was stopped and the port is free.
GPU 0/1 were never touched. The 43300 service was never started or stopped.

## 12. Multi-round stability, 2026-11-19

Five consecutive full 54-case rounds were run against one service instance.
Three rounds were required; five were run because two rounds contained dropped
requests and, under the documented `cv > 0.10` rule, a round like that does not
count.

Setup, unchanged from section 3 except port and GPU:

| Item | Value |
| --- | --- |
| Service | one instance, `<container-ip>:43302`, started fresh, never restarted during the 5 rounds |
| GPUs | ASR stage 6, Omni stage 7. GPU 0/1 untouched, port 43300 untouched |
| Resident 43301 service | running throughout on GPUs 2-5; it is the same service that ran during the baseline window and was not touched |
| Binaries | `umm-infer` `5953963c…30864`, `umm-native-stage-asr` `1bf9485f…cf072`, `umm-native-stage-omni` `8ca0225d…dadd8` (identical to section 3) |
| Harness | `3d747bc`, same manifest `2026-11-15.1`, same cached inputs, `--baseline /tmp/mmbench/evidence-v5/baseline.json` |
| Client | Mac through `ssh -L 43302:<container-ip>:43302 <gpu-host>` |
| Evidence | `/tmp/mmbench/stability/round-{1..5}/round-1/` (`cases.jsonl`, `events/`, `answers/`, `transcripts/`, `failures.json`, `report.json`) |

### Per-round results

| Round | Wall s | Cases | Successes | infra | contract | quality | WER | CER | VQA acc | GSM8K | joint WER | keyword hit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 204.5 | 54 | 53 | 1 | 0 | 1 | 0.0329 | 0.0274 | 0.8333 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 217.6 | 54 | 53 | 1 | 0 | 1 | 0.0329 | 0.0274 | 0.8333 | 1.0000 | 0.0000 | 1.0000 |
| 3 | 201.2 | 54 | 52 | 2 | 0 | 1 | 0.1200 | 0.1144 | 0.8333 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 203.0 | 54 | 53 | 1 | 0 | 1 | 0.0329 | 0.0274 | 0.8333 | 1.0000 | 0.0000 | 1.0000 |
| 5 | 280.1 | 54 | 50 | 4 | 0 | 1 | 0.1529 | 0.1410 | 0.8333 | 1.0000 | 0.0345 | 1.0000 |

Per-task latency, median / p95 in ms:

| Round | ASR | GSM8K | VQA | joint |
| ---: | --- | --- | --- | --- |
| 1 | 2320 / 5512 | 2894 / 4763 | 3344 / 6698 | 2815 / 3763 |
| 2 | 2252 / 5527 | 2838 / 6004 | 2635 / 8172 | 2921 / 5121 |
| 3 | 2484 / 8362 | 2791 / 4866 | 2522 / 6599 | 3061 / 4932 |
| 4 | 3210 / 8729 | 3487 / 5880 | 2341 / 5299 | 3096 / 8853 |
| 5 | 7357 / 23357 | 3004 / 6086 | 2569 / 6800 | 2924 / 7333 |

### Coefficient of variation

CV is `stdev / median` over the rounds, the same definition as
`docs/mlx.md` and `tests/mlx_perf.py`. The gate: `cv > 0.10` means the window
was not quiet, and that round does not count.

| Metric | CV, all 5 rounds | CV, rounds 1/2/4 | Gate |
| --- | ---: | ---: | --- |
| WER | 1.757 | 0.000 | fail over 5, pass over clean rounds |
| CER | 2.030 | 0.000 | fail over 5, pass over clean rounds |
| VQA accuracy | 0.000 | 0.000 | pass |
| GSM8K accuracy | 0.000 | 0.000 | pass |
| joint keyword hit | 0.000 | 0.000 | pass |
| joint ASR WER | undefined (four rounds 0, one 0.0345) | 0 (all three rounds 0, no variation) | pass over clean rounds |
| ASR median | 0.876 | 0.230 | fail |
| ASR p95 | 0.892 | 0.335 | fail |
| GSM8K median | 0.097 | 0.124 | borderline, fail over clean rounds |
| GSM8K p95 | 0.110 | 0.116 | fail |
| VQA median | 0.150 | 0.196 | fail |
| VQA p95 | 0.152 | 0.215 | fail |
| joint median | 0.039 | 0.049 | pass |
| joint p95 | 0.401 | 0.515 | fail |
| wall time | 0.164 | — | fail |

Per-case latency spread is wider than the per-task medians suggest. Over the
three clean rounds 1/2/4, 54 of 54 cases have three samples of `total_ms`, and
the median per-case CV is 0.2568. The widest case is
`asr-en-ls-1089-134691-0000` at 946.81 / 829.13 / 4552.72 ms, CV 2.2356. The
next widest are `asr-en-ls-1284-1181-0000` (CV 1.0593) and `vqa2-val-393266000`
(CV 1.0433).

### What the variance is made of

Three separate effects, not one noise source.

1. **Dropped requests.** Rounds 3 and 5 each contained ASR requests that ended
   with `RemoteDisconnected: Remote end closed connection without response`
   (round 3: `asr-en-ls-1221-135766-0000`; round 5:
   `asr-en-ls-1089-134691-0000`, `asr-en-ls-1188-133604-0000`,
   `asr-en-ls-1580-141083-0000`). The failing case is scored with an empty
   transcript, so one drop adds about 28 word errors and moves corpus WER from
   0.0329 to 0.1200 or 0.1529. This is the failure documented in section 7
   item 2, now at 4 occurrences in 80 ASR requests.
2. **A deterministic engine failure, every round.** `vqa2-val-262189000` ended
   with `{"error":{"code":500,"message":"image too large for slice",...}}` in
   5 of 5 rounds. In the baseline run D the same image passed. The limit is
   `kMaxImgTokens = 1296` in `native/omni/vision_forward.cu:11` and the input
   bytes are unchanged, so the failure is engine state, not image size. It
   costs 1/16 of the VQA score: 0.8958 in the baseline, 0.8333 in all five
   rounds.
3. **One transcription miss in round 5.** `joint-speechcoco-val-0000` scored
   `asr_wer = 0.25` there and 0.0 in rounds 1-4. The transcript is real and
   shorter than the caption; it is not a transport error.

The two known deterministic quality outcomes are unchanged in every round: the
VQA miss on `vqa2-val-262197000` (`1` against six `2` and four `0`) and
GSM8K 16/16.

### Conclusion

The quality numbers are a credible baseline; the latency numbers are not.

- **Quality: credible.** On rounds with no dropped request (1, 2, 4) every
  quality metric is identical to five decimal places, CV = 0. The model is
  deterministic on this input set. Two caveats: WER/CER can only be quoted
  per-round with the drop rate stated, because 2 of 5 rounds lost a request;
  and VQA accuracy is 0.8333, not the 0.8958 recorded in section 1, while
  `vqa2-val-262189000` keeps failing.
- **Latency: not credible as a baseline.** Even over the three clean rounds,
  ASR median CV is 0.230, VQA median CV is 0.196 and GSM8K median CV is 0.124,
  all above the 0.10 gate. The host is shared with the resident 43301 service
  on GPUs 2-5 and the per-case spread (median CV 0.2568) shows scheduling
  stalls, not model variance. The medians in section 1 are one sample of a
  wide distribution; treat them as an order of magnitude, not a measurement.
- **The stability gate itself would not pass.** `stability_verdict` requires
  three consecutive rounds with zero infrastructure failures. Five rounds were
  run and none had zero, so the harness verdict is `FAIL` for reasons 1 and 2
  above.

No claim of a stable PASS is made. The two defects behind it are recorded
above and are engine work, not harness work.

## 13. Documentation and script drift, fixed 2026-11-19

Three items were checked; one needed a rewrite, one needed a header, one was
already correct.

| Item | State at `3d747bc` | Action |
| --- | --- | --- |
| `docs/native-service.md` | described the retired phase-7 topology: Python workers `umminfer_qwen_asr_stage.py` / `umminfer_qwen_omni_worker.py` and SGLang backends on GPU 0/1 | rewritten for the native topology, with the phase-7 record compressed to one history paragraph |
| `service/native-service.sh` | never stated where `$BIN/umm-infer` and `$BIN/umm-native-stage-*` come from | header now names both build systems and the install line |
| `README.md` | already separated `zig build` from `native/stage_cli` at commit `44de086`; it wrongly said the shell script is "configured by `service/native-service.json`" | sentence corrected: the script reads `UMM_NATIVE_*`; the JSON is the supervisor's config |

The claim that both build systems are separate is verified, not asserted: the
CMake output `/opt/umminfer/mmbench/stage-build/umm-native-stage-asr` and the deployed
`/opt/umminfer/mmbench/bin/umm-native-stage-asr` (the binary used in the baseline run)
have the same SHA-256 `1bf9485f9a103b0b14e946e0235b2a7397271251184c4f6f47ecadad106cf072`.

### Startup verification

Run on the container, from a fresh root, using the binaries the runbook names:

```sh
ROOT=/opt/umminfer/umm-native-verify
mkdir -p $ROOT/bin $ROOT/run $ROOT/logs
install -m755 /opt/umminfer/mmbench/src/zig-out/bin/umm-infer $ROOT/bin/
install -m755 /opt/umminfer/mmbench/stage-build/umm-native-stage-* $ROOT/bin/

UMM_NATIVE_ROOT=$ROOT UMM_NATIVE_PORT=43302 \
UMM_NATIVE_IPC=/tmp/umminfer-native-verify-ipc \
UMM_NATIVE_GPU_ASR=6 UMM_NATIVE_GPU_VLM=6 \
UMM_NATIVE_GPU_IMAGE=7 UMM_NATIVE_GPU_TTS=7 \
  /opt/umminfer/native-service.sh start
```

The GPU overrides exist because GPUs 2-5 and port 43301 are held by the live
resident service; the defaults in the script and in `service/native-service.json`
are unchanged.

Observed output:

```text
started asr pid 1065000 gpu 6
started qwen38 pid 1065003 gpu 6
started image pid 1065006 gpu 7
started tts pid 1065009 gpu 7
ready_after_s=11
health=200 ready=200
server running pid 1064808
asr running pid 1065000
qwen38 running pid 1065003
image running pid 1065006
tts running pid 1065009
```

All four stages logged `native stage <kind> ready` (load times 4149, 9961,
0 and 5702 ms). One chat request returned HTTP 200,
`{"content": "ok", "role": "assistant"}`. `stop` removed every pid, closed port
43302, and left GPU 6 at 3 MiB and GPU 7 at 3 MiB.

The supervised path was checked the same way, with a config copy pointed at the
same root, port and GPUs: `start` returned a supervisor pid, `status` reported
all five children alive with `restarts: 0`, `/health` and `/ready` answered
200 / 200, and `stop` signalled the supervisor and left the children dead.

One correction to the old runbook: it cited `python3 -I
tests/umminfer_service_test.py` as a local contract test for the supervisor.
That file does not exist and never did; the new page states that no offline
test exists and records the live checks instead.

## 14. Cleanup, 2026-11-19

| Item | Before start | During run | After stop |
| --- | ---: | ---: | ---: |
| GPU 6 used | 3 MiB | 2387 MiB (2389 during the startup check) | 3 MiB |
| GPU 7 used | 3 MiB | 78755 MiB | 3 MiB |
| GPU 0/1 | 3 MiB | 3 MiB, never touched | 3 MiB |
| Port 43302 | free | listening | free |
| Port 43300 | — | never started or stopped | — |

The server, ASR stage and Omni stage were stopped by pid. Their entries remain
as `Z` zombies reparented to container PID 1, which does not reap orphans; this
is the same state documented for earlier phases. No process holds GPU 6 or 7
and no listener claims 43302. The Mac tunnel was closed.

## 15. Engine defects, reproduced and fixed, 2026-11-19

Section 12 left two failures that blocked the stability gate. Both were
investigated on the 54 cases of the accepted baseline, selected by id from the
current manifest (`2026-11-19.1`, which adds Mandarin cases), with a fresh
service on port 43310, ASR stage on GPU 6 and Omni stage on GPU 7. GPUs 0/1 and
port 43300 were never touched. The case ids and the baseline comparison come
from `/tmp/mmbench/evidence-v5` (manifest `2026-11-15.1`).

Deployed bytes for every run in this section:

| Artifact | SHA-256 |
| --- | --- |
| `umm-infer` | `fb092678c6328e1552fe7357c1e1293f6d131e71e2fd8f302b3f9411c00d254f` |
| `umm-native-stage-asr` | `81ac3287be1fb48d186859c65c25f7ffa138389a4a6050ca23e9f23cacdd9221` |
| `umm-native-stage-omni` | `6fb3bd93413c987f7a9caf230234bedf164b46461d66c7d99530569d3604075c` |
| LibriSpeech archive | `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23` |

### 15.1 `image too large for slice`: a fixed size bound, not engine state

Reproduction with the real request bytes, on a freshly started service:

| Probe | Order | Result |
| --- | --- | ---: |
| `vqa2-val-262189000`, five consecutive requests | first requests after start | 5/5 failed |
| target after a passing image (`262148`), twice | interleaved | 4/4 failed |
| small/target interleaved, three rounds | 3 rounds | 3/3 failed |

Every failure arrived in 28–77 ms, before any vision work, and the same bytes
failed every time. Order and history never changed the result, so the "stale
slicing cache or sequence length" hypothesis is excluded.

The arithmetic is the whole cause. `smart_resize` keeps the resized area within
`max_pixels`, and the preprocessor then cuts the result into 16×16 patches:

| Quantity | Value for COCO 262189 |
| --- | --- |
| source bytes | 279986, SHA-256 `5b6076cc…4ae14f4` |
| source size | 587×640 |
| `smart_resize(factor 32, max_pixels 12845056)` | 640×576 (368640 px) |
| pre-merge patch rows | 40×36 = **1440** |
| previous bound `kMaxImgTokens` | **1296** (36×36, that is 576×576) |

`native/omni/vision_forward.cu` rejected any image above 36×36 patch rows,
while `native/stage_cli/omni_stage.cpp` accepted up to 12845056 px. The two
constants were written independently and never agreed. A 640×480 image resizes
to 30×40 = 1200 rows and passed; a 587×640 image needs 1440 and failed. That is
the whole difference between the passing and the failing VQA cases.

The run-D stage log no longer exists (later runs overwrote it), so its pass on
this case cannot be re-derived from logs. The check is a pure function of the
resized size, so a pass in run D requires different deployed bytes or a
different input size, not engine memory.

Fix, in one place (`native/omni/image_preprocess.hpp`):

```cpp
constexpr int kMaxImagePixels = 1024 * 1024;              // 1024x1024
constexpr int kMaxImagePatchRows = kMaxImagePixels / 256; // 4096 patch rows
constexpr int kMaxImageTokens = kMaxImagePixels / 1024;   // 1024 decoder pads
static_assert(kMaxImageTokens * 4 == kMaxImagePatchRows, "...");
```

The preprocessor default, the vision scratch (`kVT` in `omni_engine.cu`) and
the guard in `vision_forward.cu` now read the same constants, so `smart_resize`
output can never exceed the allocation. A 1024×1024 image leaves 1024 of the
default 2048-token context for text and generated tokens. The static assertion
makes the pixel/row/pad relation a compile-time check.

After the fix every one of the 11 target requests sent after the change answered
("four distinct street signs", against the ten reference answers `4`), across
two builds and both probe orders. The stage log reports `image_tokens=360` for
the case, which is 1440/4.

### 15.2 ASR requests closed without a response: transport, not engine

Section 12 counted four drops in 80 ASR requests. The ASR stage log shows where
the requests stopped:

| Round | ASR cases | `asr request` lines in the stage log |
| ---: | ---: | ---: |
| 1 | 16 | 16 |
| 2 | 16 | 16 |
| 3 | 16 | 15 |
| 4 | 16 | 16 |
| 5 | 16 | 13 |

The shortfalls match the dropped cases exactly: one in round 3, three in round
5. Every request that reached the stage completed, with `elapsed_ms` from 191 to
2288 ms over all 112 logged requests. The dropped requests never reached the
engine.

Controlled probes against the same binary on the same network path:

| Probe | Result |
| --- | --- |
| 40 sequential transcriptions, 1 MB body | 40/40 HTTP 200, p50 696 ms |
| headers and half body, 6 s stall | HTTP 400 `Connection: close` at 6.0 s |
| connect, send nothing for 6 s, then send | EOF with no status line |
| connect, send nothing for 12 s, then send | EOF with no status line |

The last two rows reproduce the client-visible signature:
`RemoteDisconnected: Remote end closed connection without response`. The server
closes a connection whose request head does not arrive within httplib's default
5 s socket read timeout. That close carries no status line and no log line. The
body-read timeout is a different path and answers with 400.

The observed durations (9.9, 12.3 and 11.8 s) and the fact that only the ASR
route failed, and only in the slowest round, put the break in the connection
phase before the handler. The client-to-server path stalled, the head arrived
late, and the server closed the connection. No ASR engine, Omni engine or
pipeline code is involved, and the stage log shows the engine never saw the
traffic.

Code changes in the server, and what was not changed:

- `set_error_logger` now logs httplib transport errors, and every transcription
  logs one line with `body_bytes`, `handler_ms` and `state`. Before this, a
  route could fail with no trace in the server log, which is why section 12
  could not attribute the drops.
- The abandoned-request branch now answers 503. It previously returned with no
  status, and httplib turned that into an empty 200.
- No retry, no timeout change, no engine change. There is no engine defect at
  this layer, and a longer timeout only moves the threshold.

### 15.3 Verification: three rounds of the same 54 cases

The client ran inside the container against `<container-ip>:43310`, the container's
bridge address. That is a real veth path, not loopback. Each round is a separate
process; the harness stops after one round when the case set is a subset, so
`--rounds 3` produced one round and the three rounds were started separately.

| Round | Cases | Successes | infra | contract | quality | WER | CER | VQA acc | GSM8K | joint WER | hit | p50 ms | p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54 | 54 | 0 | 0 | 1 | 0.0329 | 0.0274 | 0.8958 | 1.0000 | 0.0000 | 1.0000 | 1091 | 2686 |
| 2 | 54 | 54 | 0 | 0 | 1 | 0.0329 | 0.0274 | 0.8958 | 1.0000 | 0.0000 | 1.0000 | 1091 | 2699 |
| 3 | 54 | 54 | 0 | 0 | 1 | 0.0329 | 0.0274 | 0.8958 | 1.0000 | 0.0000 | 1.0000 | 1091 | 2688 |

ASR drops: 0 in 3 rounds, 0 in 48 ASR requests. Every round wrote 22
transcripts (16 ASR and 6 joint). The stage log records 88 requests and 0
aborts over the whole window, with a maximum compute time of 2282 ms. The
server log records 64 `transcription done` lines, 0 `client_closed` and 0
`http error`.

`vqa2-val-262189000` scored 1.0000 in all three rounds. VQA accuracy returns to
the run-D value 0.8958. The one remaining failure in every round is the
recorded quality miss on `vqa2-val-262197000`, which answered `1` against six
`2` and four `0` reference answers.

Per-case comparison against run D (`/tmp/mmbench/evidence-v5/round-1/cases.jsonl`):

| case | task | run D | round 1 | round 2 | round 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| asr-en-ls-1089-134686-0000 | asr | 0.0714 | 0.0714 | 0.0714 | 0.0714 |
| asr-en-ls-1089-134691-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1188-133604-0000 | asr | 0.1053 | 0.1053 | 0.1053 | 0.1053 |
| asr-en-ls-121-121726-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-121-123852-0000 | asr | 0.1034 | 0.1034 | 0.1034 | 0.1034 |
| asr-en-ls-121-123859-0000 | asr | 0.0270 | 0.0270 | 0.0270 | 0.0270 |
| asr-en-ls-121-127105-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1221-135766-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1221-135767-0000 | asr | 0.0556 | 0.0556 | 0.0556 | 0.0556 |
| asr-en-ls-1284-1180-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1284-1181-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1284-134647-0000 | asr | 0.0476 | 0.0476 | 0.0476 | 0.0476 |
| asr-en-ls-1320-122612-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1320-122617-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asr-en-ls-1580-141083-0000 | asr | 0.0333 | 0.0333 | 0.0333 | 0.0333 |
| asr-en-ls-1580-141084-0000 | asr | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gsm8k-test-0000 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0001 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0002 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0003 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0004 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0005 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0006 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0007 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0008 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0009 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0010 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0011 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0012 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0013 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0014 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gsm8k-test-0015 | gsm8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-262148000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393225000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393226000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-240301000 | vqa | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| vqa2-val-131089000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-262162000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393243000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-262175000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-131108000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393254000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-42000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-131115000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-262189000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393266000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-393267000 | vqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vqa2-val-262197000 | vqa | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| joint-speechcoco-val-0000 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |
| joint-speechcoco-val-0001 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |
| joint-speechcoco-val-0002 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |
| joint-speechcoco-val-0003 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |
| joint-speechcoco-val-0004 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |
| joint-speechcoco-val-0005 | joint | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 | 0.0000 / 1.000 |

Every value matches run D. No case regressed.

### 15.4 What was not established

- **Run D on `vqa2-val-262189000`.** Its stage log is gone. The fix stands on
  the deterministic reproduction and the arithmetic, not on an explanation of
  that earlier pass.
- **The ASR drop over the original network path.** The three rounds ran inside
  the container. The SSH tunnel from the Mac to the container was not usable
  during this session (the forwarding channel carried no data and bulk
  transfers stalled), so the exact external path that produced the four drops
  was not re-run. The four drops are explained and their mechanism is
  reproduced, but "zero drops over that path" is not demonstrated.
- **The head-timeout close is still invisible in the server log.** It happens
  inside httplib before any handler runs, and this version offers no hook for
  it. A recurrence is now attributable to the absence of a `transcription` line
  for a request the client saw fail, but it is not logged directly.

### 15.5 Cleanup, 2026-11-19 (second window)

| Item | Before start | During run | After stop |
| --- | ---: | ---: | ---: |
| GPU 6 used | 3 MiB | 2387 MiB | 3 MiB |
| GPU 7 used | 3 MiB | 78867 MiB | 3 MiB |
| GPU 0/1 | 3 MiB | 3 MiB, never touched | 3 MiB |
| Port 43310 | free | listening | free |
| Port 43300 | — | never started or stopped | — |

The server, ASR stage and Omni stage were stopped by pid. No process of this
window remains, no listener claims 43310, and the resident 43301 service on
GPUs 2-5 was never touched.

## 16. External-link rounds and Mandarin coverage, 2026-11-19 (second window)

The first window ran the 54 English cases with the client inside the container.
This window restores the original client path and runs the Mandarin cases that
`74ed3f2` pinned but never scored.

Client: Mac, `ssh -L 43310:<container-ip>:43310 <gpu-host>`, measured round trip 0.6 s.
Service: one instance on port 43310, ASR stage GPU 6, Omni stage GPU 7. GPUs 0/1
and port 43300 were not touched.

Mandarin data: FLEURS `cmn_hans_cn` test split, revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, archive 525346466 B, SHA-256
`09d19ad18f5d7e91076880807e866cd16abd924c7052b55f71cdae91714fc166`, CC BY 4.0.
The harness verified the archive, every member and every converted WAV before
the first request.

Deployed bytes for this window (SHA-256, first 20 hex): `umm-infer`
`6fdbf9666f61113c5e4f`, `umm-native-stage-asr` `81ac3287be1fb48d1868`,
`umm-native-stage-omni` `6fb3bd93413c987f7a9c`.

### 16.1 The drop reproduced over the original path

One 54-case round over this path produced 9 infrastructure failures, all
`RemoteDisconnected: Remote end closed connection without response`: 5 on
`/v1/audio/transcriptions` and 4 on `/v1/chat/completions`. The original window
dropped only ASR requests; with the link degraded, chat requests drop too, so
the failure belongs to the connection phase and not to one route.

With the link degraded, a probe of five 400 KB transcriptions failed 5/5: four
received `HTTP 400` after 5.3–29.7 s and one received a bare EOF. The client's
`sendall` returned in about 1 ms each time, so the bytes were in the client
kernel immediately.

Correlated observations for that window:

| Component | Observation |
| --- | --- |
| client | `sendall` returns in ~1 ms; `getresponse` then waits 5–30 s and gets 400 or EOF |
| server error hook | `Failed to read ... method=POST path=/v1/audio/transcriptions body_bytes=0` |
| server `recv` shim (diagnostic only, not shipped) | `ret=-1 errno=11` (`EAGAIN`): the socket read timed out; normal closes show `ret=0` |
| ASR stage | 14 requests logged where 22 were sent; the dropped ones never arrived |
| control: plain TCP, same forward, 800 KB × 20 | 20/20 delivered intact |
| control: `curl` multipart POST over the same forward | 3/3 answered in ~0.6 s |

The plain-TCP control rules out loss or truncation in the forward. The shim
shows the server's read failing with `EAGAIN`, not `EOF`: the server stopped
waiting. The server keeps httplib's default 5 s socket read timeout
(`CPPHTTPLIB_SERVER_READ_TIMEOUT_SECOND`); a stall longer than that aborts the
body read, which httplib maps to `400 Bad Request`.

So the responsible component is the client-to-server transport, not the ASR
engine, the Omni engine, the pipeline or the stage. The server contributes the
failure mode: a valid request whose upload stalls is answered `400`, which the
harness records as `http_4xx_valid_request`. The brief first excluded a read
timeout change; the owner then authorized it, and section 16.5 records the
change and the re-test.

### 16.2 Mandarin (FLEURS zh_cn) first real-model results

The manifest defines the Mandarin metric as `cer` and carries no word-level
reference, so no WER is reported. A character-count WER would misstate the
metric and is not computed.

Four full 70-case rounds were run over the external path:

| Round | Cases | infra | contract | quality | EN ASR valid | EN WER | ZH valid | ZH corpus CER | VQA | GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 70 | 1 | 0 | 1 | 16/16 | 0.0329 | 15/16 | 0.1018 | 0.8958 | 1.0000 |
| 2 | 70 | 2 | 2 | 0 | 16/16 | 0.0329 | 14/16 | 0.1221 | 0.8333 | 1.0000 |
| 3 | 70 | 0 | 2 | 1 | 15/16 | 0.0283 | 16/16 | 0.1106 | 0.8333 | 1.0000 |
| 4 | 70 | 0 | 4 | 1 | 15/16 | 0.0283 | 13/16 | 0.1292 | 0.8958 | 1.0000 |

EN WER and VQA move between rounds only because a failed case leaves the scored
set; every case that produced a transcript matches run D in all four rounds.
No round was free of transport failures on this path.

Per-case Mandarin CER, first successful attempt per case across the four rounds:

| case | CER | chars | char errors | ms | round |
| --- | ---: | ---: | ---: | ---: | --- |
| asr-zh-fleurs-10104380129516909856 | 0.0870 | 46 | 4 | 3145 | 1 |
| asr-zh-fleurs-10325559490685159122 | 0.4000 | 45 | 18 | 2719 | 1 |
| asr-zh-fleurs-10369927130798325266 | 0.0000 | 38 | 0 | 2941 | 1 |
| asr-zh-fleurs-10852536777113131618 | 0.0000 | 38 | 0 | 1880 | 1 |
| asr-zh-fleurs-1177351021583652757 | 0.2647 | 34 | 9 | 3496 | 2 |
| asr-zh-fleurs-14716260585206763911 | 0.2687 | 67 | 18 | 1926 | 1 |
| asr-zh-fleurs-15324743446256875074 | 0.1731 | 52 | 9 | 8557 | 1 |
| asr-zh-fleurs-16106752411261640702 | 0.0000 | 38 | 0 | 8839 | 1 |
| asr-zh-fleurs-16272218284004703131 | 0.0612 | 49 | 3 | 2827 | 1 |
| asr-zh-fleurs-17791005930196814474 | 0.0667 | 45 | 3 | 12165 | 1 |
| asr-zh-fleurs-1868048601472473390 | 0.0357 | 28 | 1 | 2379 | 1 |
| asr-zh-fleurs-4048226328336996391 | 0.0208 | 48 | 1 | 3459 | 1 |
| asr-zh-fleurs-6567034835218108721 | 0.0000 | 23 | 0 | 7338 | 1 |
| asr-zh-fleurs-7669819575583671099 | 0.0526 | 38 | 2 | 2618 | 1 |
| asr-zh-fleurs-8424295226444731846 | 0.0000 | 18 | 0 | 11169 | 1 |
| asr-zh-fleurs-8702090716396029445 | 0.0769 | 26 | 2 | 6100 | 1 |

All 16 cases produced at least one valid transcript across the four rounds.
Corpus CER over those 16 results: **0.1106** (70 character errors / 633
reference characters). Per round, over the cases that produced a transcript:
0.1018 (61/599), 0.1221 (68/557), 0.1106 (70/633) and 0.1292 (66/511).

English ASR over the same rounds: every valid case equals run D; the only
English failure is `asr-en-ls-1221-135767-0000`, answered with `400` in rounds
3 and 4 by the same transport mechanism.

### 16.3 What this window does not establish

- **Three clean 54-case rounds over the external path.** Four full rounds were
  attempted and none was clean; every failure is the transport stall above.
  The three clean 54-case rounds recorded in section 15.3 ran with the client
  inside the container.
- **Mandarin WER at the time of this window.** Not defined then; added in
  section 17.4 with jieba segmentation.
- **A code fix for the stall itself.** The stall is outside the repository.
  The server-side read timeout was aligned with the payload policy after the
  owner authorized it (section 16.5); no retry or connection-reset workaround
  is applied.

### 16.4 Cleanup, second window

| Item | Before the rounds | During the rounds | After stop |
| --- | ---: | ---: | ---: |
| GPU 6 used | 3 MiB | 2389 MiB | 3 MiB |
| GPU 7 used | 3 MiB | 78867 MiB | 3 MiB |
| GPU 0/1 | 3 MiB | 3 MiB, never touched | 3 MiB |
| Port 43310 | free | listening | free |
| Port 43300 | — | never started or stopped | — |

The server, ASR stage and Omni stage were stopped by pid. No process of this
window remains, no listener claims 43310, and the resident 43301 service on
GPUs 2-5 was never touched (`/health` answered 200 after the stop).

### 16.5 Read timeout aligned with the payload policy, and the re-test

Owner decision: the read timeout may be changed. Change in
`src/serve/openai_api.cpp`, deployed as `umm-infer` SHA-256
`8a489f1cf1c5d849e286…`:

```cpp
// How long one socket read may wait for the next byte. The body is read before
// any handler runs, so this value alone decides whether a slow upload survives.
// ... 30 s exceeds a worst-case TCP retransmission backoff and still reclaims
// a dead connection.
constexpr time_t kHttpReadStallSeconds = 30;
server_.set_read_timeout(kHttpReadStallSeconds, 0);
```

The value bounds one stalled read, not the whole transfer. A progressing upload
of any size is unaffected, a dead connection is still reclaimed, and the handler
wait keeps its own 600 s request deadline.

Re-test: six full 70-case rounds over the same external path (three on one
service instance, then a restart, then three more), same client, same data,
same case set.

| Rounds | Cases | infra | contract | quality | EN ASR | EN WER | ZH ASR | ZH corpus CER | VQA | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 70 | 0 | 0 | 1 | 16/16 | 0.0329 | 16/16 | 0.1106 | 0.8958 | 1.0000 |
| 4–6 | 70 | 0 | 0 | 1 | 16/16 | 0.0329 | 16/16 | 0.1106 | 0.8958 | 1.0000 |

All six rounds are identical to five decimal places. Counters over the window:
the ASR stage logged 118 requests with 0 aborts and a maximum compute time of
2280 ms; the server logged 96 `transcription done`, 0 `Failed to read`, 0
`http error` and 0 `client_closed`. ASR drops: 0 in six rounds, 0 in 96
transcription requests and 96 joint ASR legs. `vqa2-val-262189000` scored 1.0
in every round. The single remaining failure is the recorded quality miss on
`vqa2-val-262197000`.

Per case, 68 of 70 match run D in every round. The two exceptions are joint
cases, described in section 16.6. Every English ASR, Mandarin ASR, GSM8K and
VQA case matches run D exactly.

### 16.6 Separate finding: a joint-ASR transcript depends on service history

Two joint cases change transcript after the service has served a full suite,
with identical request bytes:

| Case | Fresh service, first requests | After a full suite | run D |
| --- | --- | --- | --- |
| `joint-speechcoco-val-0000` | `Well dressed man and woman pose for picture.` | `Were dressed men and women posed for picture.` | same as fresh |
| `joint-speechcoco-val-0005` | `A room with blue walls and a white sink and door. Um.` | `A room with three walls and a white sink and door. Um.` | same as fresh |

The request hashes are identical in both conditions (audio
`6f47641905b7…`, image `b654e4127bdd…`, prompt `31779b525b53…` for the first
case). The deviant transcripts repeat exactly in every round after the first
suite, and the run-D transcripts return on a freshly started service, so the
difference follows service history and not the client path.

The deviation predates the timeout change: the four rounds in section 16.2,
run before it, show the same two values. The retained state was located and
fixed afterwards; section 17 gives the cause, the fix and the re-test.

## 17. Request-history dependence in ASR, fixed 2026-11-20

### 17.1 Cause

`d_au_norm` is one workspace used with two row widths. `conv_permute` writes
the permuted conv output as `[tokens][480 * 16]` (7680 BF16 per token), and the
layer-normalised activations reuse the same buffer as `[tokens][896]`. The
allocation was sized for the narrow width:

```cpp
mk(im.d_au_norm, 390 * kAudioD);        // 349,440 elements, 698,880 bytes
```

`tokens = 13 * ceil(valid_frames / 100)`, so a write of `tokens * 7680`
elements overruns that allocation from 4 chunks on, that is from 301 mel
frames, which is audio longer than 3.01 s. The joint SpeechCOCO clips are
2.78 s (279 frames, 3 chunks, 299,520 elements) and stay inside the buffer.
The FLEURS Mandarin clip used for isolation is 10.38 s (1038 frames, 11
chunks): 143 tokens, 1,098,240 elements written into a 349,440-element buffer,
748,800 elements past the end.

`compute-sanitizer --tool memcheck` on the pre-fix build reports the write by
name:

```text
========= Invalid __global__ write of size 2 bytes
=========     at umminfer::native::conv_permute_kernel(...)+0x280
=========     Address 0x7feb0e3ffc00 is out of bounds
=========     and is 1,024 bytes before the nearest allocation at
=========     0x7feb0e400000 of size 2,097,152 bytes
=========     ...
=========     in umminfer::native::AsrEngine::Impl::audio_forward(int, ...)
========= ERROR SUMMARY: 8129 errors
```

The 8,129 reported writes are the part of the overrun that lands in the gap
between allocations. The rest lands inside whatever `cudaMalloc` placed next,
which is why the damage surfaces as a changed decode in a later request rather
than as a fault.

### 17.2 Isolation, on the pre-fix binary

The trigger is audio length, not language. Same engine instance, same English
clip before and after a long request:

| Sequence on the pre-fix binary | Transcript |
| --- | --- |
| English clip, first request | `Well dressed man and woman pose for picture.` |
| 10.38 s Mandarin clip | `这并不是告别，这是一个篇章的结束，也是新篇章的开始。` |
| same English clip after it | `Were dressed, man and woman, pose for picture.` |
| 8.35 s English clip (three copies of the same clip) | decodes all three copies correctly |
| same English clip after that | `Well dressed men and women pose for picture.` |

The Mandarin clip is not special: an 8.35 s English clip leaves the same
damaged state. A repeated short English request does not.

### 17.3 Fix and re-test

The workspace is sized for the wider layout, the row width is one constant, and
a request that would exceed the workspace throws instead of writing past it:

```cpp
constexpr int kMaxAudioTokens = (kMelFrames + 99) / 100 * 13;   // 390
constexpr int kAudioPermRow = kAudioDown * 16;                   // 7680
constexpr int kAuNormElements = kMaxAudioTokens * kAudioPermRow;
...
if (tokens > kMaxAudioTokens)
  throw std::runtime_error("audio tower tokens exceed workspace: ...");
conv_permute(d_conv3, d_au_norm, chunks, stream);
```

On the fixed binary the same sequence returns one transcript for all three
English decodes, with the Mandarin and 8.35 s English requests in between.

Re-test: three full 70-case rounds over the same external path (Mac →
`ssh -L` → container, `127.0.0.1:43310`), one client, one service instance,
evidence in `/tmp/mmfix/ev-hist`, `/tmp/mmfix/ev-h2`, `/tmp/mmfix/ev-h3`:

| Round | Cases | infra | contract | quality | EN WER / CER | ZH WER | ZH CER | VQA | GSM8K | joint WER / hit |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 70 | 0 | 0 | 1 | 0.0188 / 0.0245 | 0.0503 | 0.0916 | 0.8958 | 1.0000 | 0.0000 / 1.0000 |
| 2 | 70 | 0 | 0 | 1 | 0.0188 / 0.0245 | 0.0503 | 0.0916 | 0.8958 | 1.0000 | 0.0000 / 1.0000 |
| 3 | 70 | 0 | 0 | 1 | 0.0188 / 0.0245 | 0.0503 | 0.0916 | 0.8958 | 1.0000 | 0.0000 / 1.0000 |

All three rounds are identical, and their 70 metric records are identical case
by case (0 differing records). The single quality failure is
`vqa2-val-262197000`, which scores 0.0 in run D as well.

Per case against run D, 46 of the 54 baseline cases are identical in all three
rounds and 8 differ, with 0 infrastructure and 0 public-contract failures in
every round. Five English ASR cases improve, one English ASR case regresses on
its auxiliary CER only, and two VQA cases change `detail.matches` without
changing `vqa_accuracy`:

| Case | run D | Rounds 1–3 | Direction |
| --- | --- | --- | --- |
| `asr-en-ls-1089-134686-0000` | WER 2/28 | WER 1/28 | better |
| `asr-en-ls-1188-133604-0000` | WER 2/19, CER 13/109 | WER 1/19, CER 9/109 | better |
| `asr-en-ls-121-123852-0000` | WER 3/29, CER 6/167 | WER 2/29, CER 5/167 | better |
| `asr-en-ls-1221-135767-0000` | WER 4/72, CER 9/396 | WER 2/72, CER 7/396 | better |
| `asr-en-ls-1284-134647-0000` | WER 1/21, CER 2/130 | WER 0/21, CER 1/130 | better |
| `asr-en-ls-1221-135766-0000` | WER 0/37, CER 2/208 | WER 0/37, CER 3/208 | **CER worse by one character** |
| `vqa2-val-393226000` | matches 5, accuracy 1.0 | matches 3, accuracy 1.0 | detail only |
| `vqa2-val-393243000` | matches 3, accuracy 1.0 | matches 7, accuracy 1.0 | detail only |

The accepted exception is the one regression: `asr-en-ls-1221-135766-0000`
keeps WER 0 and moves its auxiliary CER from 2/208 to 3/208, one character.
Apart from that case, no metric regresses against run D.

The two joint cases of section 16.6 return to their run-D transcripts:

| Case | Run D | Pre-fix, after one suite | Fixed, rounds 1–3 |
| --- | --- | --- | --- |
| `joint-speechcoco-val-0000` | WER 0, hit 1.0 | WER 0.5, hit 0.75 | WER 0, hit 1.0 |
| `joint-speechcoco-val-0005` | WER 0, hit 1.0 | WER 0.0833, hit 0.8333 | WER 0, hit 1.0 |

English ASR also improves, because the overrun also damaged the current
request: EN WER 0.0329 → 0.0188 and Mandarin CER 0.1106 → 0.0916.

### 17.4 Mandarin WER

The manifest holds reference sentences, not word boundaries, so the harness
segments both reference and hypothesis with jieba. The segmentation is
dictionary-only (`jieba.cut(text, HMM=False)`), and the segmenter name and
version are written into `report.json` as `zh_segmenter`:

```text
jieba 0.42.1, jieba.cut(HMM=False), default dictionary
```

Both sides are NFKC-normalised with punctuation and spaces removed before
segmentation, the same normalisation the CER metric uses. WER is reported next
to CER; `cer` stays the case metric and no threshold changed. The harness test
`tests/multimodal_benchmark_test.py` asserts the word counts, including that
characters are never counted as words.

Corpus over the 16 Mandarin cases, all three rounds: WER 0.0503 (17/338),
CER 0.0916 (58/633).

| case | CER | WER | char err/ref | word err/ref |
| --- | ---: | ---: | ---: | ---: |
| 10104380129516909856 | 0.0435 | 0.0714 | 2/46 | 2/28 |
| 10325559490685159122 | 0.4000 | 0.1500 | 18/45 | 3/20 |
| 10369927130798325266 | 0.0000 | 0.0000 | 0/38 | 0/22 |
| 10852536777113131618 | 0.0000 | 0.0000 | 0/38 | 0/22 |
| 1177351021583652757 | 0.2647 | 0.0625 | 9/34 | 1/16 |
| 14716260585206763911 | 0.2687 | 0.0476 | 18/67 | 1/21 |
| 15324743446256875074 | 0.0385 | 0.0417 | 2/52 | 1/24 |
| 16106752411261640702 | 0.0000 | 0.0000 | 0/38 | 0/22 |
| 16272218284004703131 | 0.0612 | 0.1154 | 3/49 | 3/26 |
| 17791005930196814474 | 0.0222 | 0.0370 | 1/45 | 1/27 |
| 1868048601472473390 | 0.0000 | 0.0000 | 0/28 | 0/16 |
| 4048226328336996391 | 0.0208 | 0.0323 | 1/48 | 1/31 |
| 6567034835218108721 | 0.0000 | 0.0000 | 0/23 | 0/14 |
| 7669819575583671099 | 0.0789 | 0.1364 | 3/38 | 3/22 |
| 8424295226444731846 | 0.0000 | 0.0000 | 0/18 | 0/10 |
| 8702090716396029445 | 0.0385 | 0.0588 | 1/26 | 1/17 |

## 18. Bounded Qwen3.8 VQA + SpeechCOCO gate, 2026-11-26

This is a **real 22-case evaluation**, not the deterministic 70-case replay and
not a controlled-routing measurement. It ran the existing 16 VQAv2 cases and
six SpeechCOCO joint cases once each through public HTTP/SSE, native
`Qwen3.8-27B`, and native `Qwen3-ASR-1.7B`. The other 32 standalone ASR cases
and 16 GSM8K cases are **NOT_RUN** in this bounded gate.

### 18.1 Result and timing

| Task | Cases | Usable | Completed terminals | Metric | Value | TTFT p50 / p95 ms | Terminal p50 / p95 ms | Total p50 / p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| VQAv2 | 16 | 16 | 16 | official-style VQA accuracy | **0.958333** | 658.5 / 817.8 | 682.8 / 837.0 | 682.8 / 837.0 |
| Joint image + real speech | 6 | 6 | 6 | corpus ASR WER | **0.000000 (0/58 words)** | 665.7 / 718.3 | 1535.8 / 1743.5 | 1535.8 / 1784.3 |
| Joint image + real speech | 6 | 6 | 6 | macro caption-keyword hit | **0.972222** | same rows | same rows | same rows |
| **All** | **22** | **22** | **22** | | | 662.1 / 726.9 | 711.4 / 1657.6 | 711.4 / 1657.6 |

VQA accuracy is the manifest rule: `min(1, matches/3)` per question, averaged
over 16 questions. The points total is 15.333333/16. Joint WER is corpus WER;
keyword hit is the mean of the six per-case rates, not a micro-average.

Failure counts in the accepted run are **0 infrastructure, 0 public-contract,
0 quality-gate, and 0 performance-gate failures**. All cases returned HTTP 200,
`finish_reason=stop`, `[DONE]`, and exactly one unified `complete` terminal at
stage `vlm`. All six joint streams contain exactly one `asr_final`, before the
first VLM delta. `terminal_legs` is `null` in these text-only-output requests:
no image, TTS, or video output legs were requested, so the server omits the
optional array. The harness preserves that `null`; it does not turn it into an
empty array or a fabricated successful leg measurement. Any reported leg must
be `completed`.

The summary verdict is **`BASELINE_REQUIRED`**. This is an explicit 22-of-70
subset without an accepted matching baseline, not a stability PASS and not a
replacement for the full 70-case gate. Output token counts come from public
`usage.completion_tokens`.

### 18.2 Per-case VQA responses

TTFT is `first_delta_ms`; terminal is the unified `complete` observation. Full
raw SSE, responses, requests, input hashes, and scores are in the evidence.

| Case | Real answer | VQA accuracy | TTFT ms | Terminal ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `vqa2-val-262148000` | `down` | 1.0000 | 726.9 | 745.8 | 745.8 |
| `vqa2-val-393225000` | `foodiebaker.com` | 1.0000 | 624.5 | 698.4 | 698.4 |
| `vqa2-val-393226000` | `walking` | 1.0000 | 663.0 | 682.8 | 682.8 |
| `vqa2-val-240301000` | `ventilation` | 0.3333 | 689.6 | 728.0 | 728.0 |
| `vqa2-val-131089000` | `green` | 1.0000 | 626.1 | 644.5 | 644.5 |
| `vqa2-val-262162000` | `no` | 1.0000 | 658.5 | 678.3 | 678.3 |
| `vqa2-val-393243000` | `no` | 1.0000 | 662.1 | 682.4 | 682.4 |
| `vqa2-val-262175000` | `no` | 1.0000 | 691.5 | 711.4 | 711.4 |
| `vqa2-val-131108000` | `no` | 1.0000 | 624.4 | 642.9 | 642.9 |
| `vqa2-val-393254000` | `This is camping` | 1.0000 | 664.8 | 721.5 | 721.5 |
| `vqa2-val-42000` | `white` | 1.0000 | 668.8 | 688.4 | 688.4 |
| `vqa2-val-131115000` | `yes` | 1.0000 | 627.8 | 646.2 | 646.2 |
| `vqa2-val-262189000` | `4` | 1.0000 | 817.8 | 837.0 | 837.0 |
| `vqa2-val-393266000` | `red and yellow` | 1.0000 | 657.4 | 714.1 | 714.2 |
| `vqa2-val-393267000` | `black` | 1.0000 | 625.4 | 643.8 | 643.9 |
| `vqa2-val-262197000` | `2` | 1.0000 | 618.8 | 637.3 | 637.3 |

### 18.3 Per-case joint responses and objective sub-metrics

The transcript below is the actual `asr_final`, not the manifest reference.
Each full VLM answer is saved in `answers/` and `human_review/`; the objective
metrics do not grade its complete semantic quality.

| Case | Real ASR transcript | ASR WER | Keyword hit | ASR final ms | TTFT ms | Terminal ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `joint-speechcoco-val-0000` | `Well-dressed man and woman pose for picture.` | 0.0000 | 1.0000 | 53.1 | 638.9 | 1287.6 | 1287.6 |
| `joint-speechcoco-val-0001` | `A bicycle replica with a clock as the front wheel.` | 0.0000 | 1.0000 | 52.3 | 529.9 | 1743.5 | 1784.3 |
| `joint-speechcoco-val-0002` | `A black Honda motorcycle parked in front of a garage.` | 0.0000 | 1.0000 | 57.5 | 714.8 | 1256.1 | 1256.2 |
| `joint-speechcoco-val-0003` | `A view of train tracks and a train.` | 0.0000 | 1.0000 | 45.3 | 707.8 | 1537.6 | 1537.6 |
| `joint-speechcoco-val-0004` | `Two zebras standing together while one zebra swishes his tail.` | 0.0000 | 1.0000 | 60.2 | 665.7 | 1535.8 | 1535.8 |
| `joint-speechcoco-val-0005` | `A room with blue walls and a white sink and door. Um.` | 0.0000 | 0.8333 | 59.5 | 718.3 | 1657.6 | 1657.6 |

Human-review observations, not additional automatic scores: case 0001 repeats
part of the instruction before the spoken sentence; case 0005 describes the
bathroom but omits the declared `room` keyword. The latter explains the only
sub-1.0 joint keyword row. These answers remain available for human quality
review and are not hidden by the WER or keyword aggregates.

### 18.4 Harness canary and independent audit

An initial real 22-case canary used a new over-strict terminal check and marked
all 22 cases `missing_terminal_legs`. The actual native terminal contract omits
`legs[]` when no optional output legs exist. That canary is **rejected harness
evidence**, is not included in the accepted metrics, and was not repaired by
editing any response. The harness now requires one unified `complete`, records
`terminal_ms` and terminal identity, accepts `null` only for the valid
no-output-leg shape, and rejects a malformed array or any non-completed
reported leg. The deterministic mock was changed to the real no-output-leg
shape, with negative tests for missing, duplicate, malformed, and failed
terminals/legs.

After the fresh final run, an independent stdlib audit (`gate-audit.json`)
rechecked every row against the raw SSE and cached input bytes:

- 22 unique real cases, 22 HTTP 200 responses, 22 completed terminals;
- 22 saved answers and 22 hash-/format-checked image inputs;
- six saved transcripts, six non-empty PCM16 mono 16 kHz SpeechCOCO WAVs with
  manifest matching hashes and durations, and six raw ASR-before-VLM orderings;
- six human-review records; all four failure categories at zero; and
- recomputed VQA accuracy, joint corpus WER, and macro keyword hit equal to the
  runner aggregates.

### 18.5 Source, configuration, data, model, and binary provenance

The requested base was `622183c5edf6a825bf8b2232fe3b85a592ea5add`.
The accepted run source was `45d51a7850f60e57ba9b3cfe83f3f65211073e1d`
(tree `e31259b8d2d16bfa8d441902fb02e9c49695ca46`). Binaries were built at
`0ff873fdb7c99fef219b3b11a1971e8a675595c0`; the accepted run changed only the
Python harness after that build, and `src`, `native`, `include`, and `build.zig`
are identical between the binary-build and run commits.

| Object | SHA-256 or identity |
| --- | --- |
| Accepted-run source archive | `d9ed03f84794eb818133f2c74dd72712ec7e9d0e7821cae2f05dd4c84cf0fb61` |
| `tests/multimodal_benchmark.py` | `f511e7f63bf996f01e6bfcff9cda033e7dc3ae4300e1903c33f675a62f8eb848` |
| `benchmarks/multimodal_manifest.json` | `599963a3851d5ce02d740b8b06c423726bae5c9c1e07d1931708074c83caa9a3` |
| `run-config.json` | `a1b37376a07952e342a0b92471eded64911a5151ed769da5ea78bab156d1d8c3` |
| `data-SHA256SUMS` (28 input files) | `a2cb454479721bcf8a820b7d042d819c385f630814da3acbe90638815bf59c10` |
| `umm-infer` | `8ba9a24c105a6f3a249c45d0719e4f64b49315abf8e3e277a1b197943a590e65` |
| `umm-native-stage-asr` | `c6ed1faf071be49444af3cf054f50ff3bf5cd4e7500d2d108276fac22fdd7d4c` |
| `umm-native-stage-qwen38` | `ec5ac57cbf1b49f5a0fc5df101afa0e3d0acadb613ab379b3d51b61b69cef010` |
| Qwen3.8 `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Qwen3.8 available-weight manifest (18 shards) | `b7a23bc59e1bbdba42ad309d484b319060d0c2623fbfb086778738fa6e461743` |
| Qwen3-ASR `config.json` | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| Qwen3-ASR available-weight manifest (2 shards) | `3b50bdd28d2c87395edfcc7bf111161bb6e9addae8f4f7afdc4aa168538c634b` |

Every individual Safetensors SHA-256, tokenizer/config checksum, binary
checksum, and per-input checksum is recorded in `provenance.json`,
`model-SHA256SUMS`, `binary-SHA256SUMS`, and `data-SHA256SUMS`. These are hashes
of the **available local bytes**, not claims of an upstream revision ID.

The run used NVIDIA H20 SM90, driver 535.161.08, ASR on explicitly idle
physical GPU 1 and Qwen3.8 on explicitly idle physical GPU 7. The server bound
`127.0.0.1` with `--port 0` and observed ephemeral port 44015. IPC paths,
binaries, logs, cache, and evidence were isolated under `/tmp/umm-task25-real`.
GPU 0 was reserved and unused; GPUs 2–6 and the resident service were never
used, contacted, or restarted.

### 18.6 Evidence checksums, cleanup, and remaining limits

The retained local evidence directory is
`/tmp/umm-task25-evidence-local/real-vqa-joint/`. It contains the complete
`cases.jsonl`, raw SSE, answers, transcripts, human-review records, report,
failures, audit, run configuration, provenance, native logs, build logs, GPU
and cleanup records, and `SHA256SUMS`. Both remote and copied-local verification
passed **85/85 files**. `SHA256SUMS` SHA-256 is:

```text
08b0897b50ea37fb4c3e4d6352318a5670cd6481f70d3d1892e3f3f23fd897e4
```

Before stopping, `/health` reported `in_flight=0`; GPU 1/7 used 5117/56365 MiB.
Only the three owned processes were signalled. Afterwards the loopback endpoint
refused connections, no owned process remained active, and both GPUs returned
to **3 MiB**. The resident service was not touched.

Remaining limits:

- standalone English/Mandarin ASR and GSM8K were not part of this task and are
  **NOT_RUN** here; six short English joint captions do not replace the 32-case
  standalone ASR evaluation;
- this is one bounded round, not a latency-stability or regression baseline;
- joint free-form answer quality still needs human review; keyword hit is only
  an objective sub-metric;
- the manifest's documented VQA normalization is used, not the complete
  official toolkit normalization; and
- no mock response or deterministic replay metric contributed to any value in
  this section. The separate 70-case deterministic contract replay is labelled
  `NOT_REAL_EVIDENCE` as before.

## 19. Combined real 70-case coverage audit, 2026-11-26

The newest main used here was `bad6fc3db38104ca76b4d5179f705373fabc218d`.
Commit `e51f1d9098899ed9cb9a4dc9df1576e60b95e097` integrated the accepted real
benchmark final designated as
`4cafdafdc9f269d8ac4025cf66dd2c57e536200f`. The integrated runner and contract
test have the exact SHA-256 values recorded by the accepted run in section 18.
The accepted 16-VQA + 6-joint evidence was **not rerun or relabelled**. This task
added one real attempt for each of the other 32 standalone ASR cases and 16
GSM8K cases. The three disjoint real sets cover all 70 manifest case ids.

### 19.1 All 70 cases have real responses and completed terminals

| Task | Cases | Usable | Completed terminals | Metric | Value | TTFT p50 / p95 ms | Terminal p50 / p95 ms | Total p50 / p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| English ASR | 16 | 16 | 16 | corpus WER / CER | **0.011765 (5/425) / 0.024905 (59/2369)** | 421.6 / 686.9 | 421.7 / 687.0 | 421.7 / 687.0 |
| Mandarin ASR | 16 | 16 | 16 | corpus CER / WER | **0.088468 (56/633) / 0.044379 (15/338)** | 424.9 / 475.1 | 425.0 / 475.2 | 425.0 / 475.2 |
| GSM8K | 16 | 16 | 16 | exact integer accuracy | **0.875000 (14/16)** | 327.1 / 440.7 | 1733.3 / 2232.1 | 1754.3 / 2272.2 |
| VQAv2 (accepted section 18) | 16 | 16 | 16 | VQA accuracy | **0.958333** | 658.5 / 817.8 | 682.8 / 837.0 | 682.8 / 837.0 |
| Joint image + real speech (accepted section 18) | 6 | 6 | 6 | corpus ASR WER / macro keyword hit | **0.000000 (0/58) / 0.972222** | 665.7 / 718.3 | 1535.8 / 1743.5 | 1535.8 / 1784.3 |
| **All** | **70** | **70** | **70** | | | 431.2 / 714.8 | 644.5 / 2021.0 | 644.5 / 2061.2 |

The Mandarin WER uses the pinned `jieba 0.42.1`, `jieba.cut(HMM=False)`, and the
default dictionary. English CER is auxiliary to the declared WER. Mandarin WER
is auxiliary to the declared CER. VQA keeps the manifest's documented
normalization. Joint keyword hit is the mean of six per-case rates; joint ASR
WER is corpus WER. The combined timing rows describe three separate bounded
runs and are not a performance baseline.

All 70 cases returned HTTP 200, `finish_reason=stop`, `[DONE]`, and exactly one
unified `complete` terminal at stage `vlm`. The 38 audio-input streams contain
one `asr_final` before the first VLM delta. `terminal_legs` remains `null`
because no optional image, TTS, or video output legs were requested. It is not
rewritten to `[]`. No metric is missing; numeric zeroes remain zeroes.

Combined failure coverage is **0 infrastructure, 0 public-contract, 2 quality,
and 0 performance failures**. The two quality failures are the GSM8K misses
below. `successes=70` follows the harness definition: a quality miss is not an
infrastructure or public-contract failure. Every source summary and the combined
audit retain **`BASELINE_REQUIRED`**. This is not a three-round stability PASS.

### 19.2 New standalone ASR responses, metrics, and timing

These requests used public `POST /v1/chat/completions` with SSE, native ASR on
GPU 1, `temperature=0`, no retry, and `--asr-route chat`. `max_tokens=1` bounds
only the auxiliary downstream VLM response used to exercise the unified
terminal; it does not bound ASR decoding (`asr_max_new_tokens=256`). The scored
text is the real `asr_final`. `ASR final ms` measures that event. TTFT here is
the first auxiliary VLM delta, not an ASR-partial latency claim. The auxiliary
one-token answers are saved and marked for human review but receive no quality
score.

**English: WER is the case metric; CER is auxiliary.**

| Case | Real ASR transcript | WER | CER | ASR final ms | TTFT ms | Terminal / total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `asr-en-ls-1089-134686-0000` | He hoped there would be stew for dinner, turnips and carrots, and bruised potatoes, and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce. | 0.000000 | 0.044304 | 196.4 | 499.3 | 499.4 / 499.4 |
| `asr-en-ls-1089-134691-0000` | He could wait no longer. | 0.000000 | 0.043478 | 38.3 | 330.7 | 330.7 / 330.8 |
| `asr-en-ls-1188-133604-0000` | You will find me continually speaking of four men—Titian, Holbein, Turner, and Tintoret—in almost the same terms. | 0.000000 | 0.055046 | 129.7 | 427.2 | 427.2 / 427.3 |
| `asr-en-ls-121-121726-0000` | Also, a popular contrivance whereby love-making may be suspended, but not stopped, during the picnic season. | 0.000000 | 0.048077 | 102.8 | 399.3 | 399.3 / 399.3 |
| `asr-en-ls-121-123852-0000` | Those pretty wrongs that liberty commits, when I am some time absent from thy heart, thy beauty and thy years full well befits, for still temptation follows where thou art. | 0.068966 | 0.029940 | 189.5 | 488.4 | 488.4 / 488.4 |
| `asr-en-ls-121-123859-0000` | You are my all the world, and I must strive to know my shames and praises from your tongue. None else to me, nor I to none alive, that my steel'd sense or changes right or wrong. | 0.000000 | 0.028902 | 206.2 | 506.7 | 506.8 / 506.8 |
| `asr-en-ls-121-127105-0000` | It was this observation that drew from Douglas, not immediately, but later in the evening, a reply that had the interesting consequence to which I call attention. | 0.000000 | 0.025316 | 128.1 | 426.0 | 426.1 / 426.1 |
| `asr-en-ls-1221-135766-0000` | How strange it seemed to the sad woman as she watched the growth and the beauty that became every day more brilliant, and the intelligence that threw its quivering sunshine over the tiny features of this child. | 0.000000 | 0.009615 | 163.4 | 463.4 | 463.5 / 463.5 |
| `asr-en-ls-1221-135767-0000` | Hester Prin went one day to the mansion of Governor Bellingham with a pair of gloves which she had fringed and embroidered to his order, and which were to be worn on some great occasion of state. For though the chances of a popular election had caused this former ruler to descend a step or two from the highest rank, he still held an honorable and influential place among the colonial magistracy. | 0.027778 | 0.020202 | 378.3 | 686.9 | 687.0 / 687.0 |
| `asr-en-ls-1284-1180-0000` | He wore blue silk stockings, blue knee pants with gold buckles, a blue ruffled waist, and a jacket of bright blue braided with gold. | 0.000000 | 0.031250 | 116.9 | 414.9 | 414.9 / 414.9 |
| `asr-en-ls-1284-1181-0000` | Ojo examined this curious contrivance with wonder. | 0.000000 | 0.020408 | 55.9 | 349.5 | 349.5 / 349.5 |
| `asr-en-ls-1284-134647-0000` | The grateful applause of the clergy has consecrated the memory of a prince who indulged their passions and promoted their interest. | 0.000000 | 0.007692 | 104.8 | 401.4 | 401.5 / 401.5 |
| `asr-en-ls-1320-122612-0000` | Since the period of our tale, the active spirit of the country has surrounded it with a belt of rich and thriving settlements. Though none but the hunter or the savage is ever known, even now, to penetrate its wild recesses. | 0.000000 | 0.022831 | 186.6 | 487.9 | 488.0 / 488.0 |
| `asr-en-ls-1320-122617-0000` | Notwithstanding the high resolution of Hawkeye, he fully comprehended all the difficulties and danger he was about to incur. | 0.000000 | 0.016393 | 103.3 | 400.1 | 400.1 / 400.1 |
| `asr-en-ls-1580-141083-0000` | I will endeavor in my statement to avoid such terms as would serve to limit the events to any particular place or give a clue as to the people concerned. | 0.033333 | 0.013072 | 123.8 | 421.6 | 421.7 / 421.7 |
| `asr-en-ls-1580-141084-0000` | It was the Indian whose dark silhouette appeared suddenly upon his blind. | 0.000000 | 0.013889 | 64.0 | 358.0 | 358.0 / 358.0 |

**Mandarin: CER is the case metric; jieba WER is auxiliary.**

| Case | Real ASR transcript | CER | WER | ASR final ms | TTFT ms | Terminal / total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `asr-zh-fleurs-10325559490685159122` | 特朗普与土耳其总统雷杰普·塔伊普·埃尔多安通话后，发表了声明。 | 0.400000 | 0.150000 | 136.2 | 432.2 | 432.2 / 432.2 |
| `asr-zh-fleurs-14716260585206763911` | 邓迪大学的 Pamela Ferguson 教授指出，记者如果公布嫌疑人的照片等信息的话，似乎确实存在危害性。 | 0.268657 | 0.047619 | 112.0 | 408.6 | 408.6 / 408.6 |
| `asr-zh-fleurs-10369927130798325266` | 在昆达里尼瑜伽中，昆达里尼能量、启迪能量通过瑜伽姿势、呼吸练习、念语和视觉形象被唤醒。 | 0.000000 | 0.000000 | 163.8 | 461.7 | 461.8 / 461.8 |
| `asr-zh-fleurs-7669819575583671099` | 他受到了新加坡副总理黄根成的欢迎，并与新加坡总理李显龙讨探讨了贸易和恐怖主义问题。 | 0.026316 | 0.045455 | 127.4 | 424.9 | 425.0 / 425.0 |
| `asr-zh-fleurs-17791005930196814474` | 当然，这个传统有其基督神学的解释，但它很可能是基督诞生以前就存在的一种庆祝春天、祈求丰收的仪式。 | 0.022222 | 0.037037 | 151.0 | 448.1 | 448.2 / 448.2 |
| `asr-zh-fleurs-6567034835218108721` | 由于分离和重组，变异在每一代的两个库之间来回变动。 | 0.000000 | 0.000000 | 86.2 | 380.9 | 380.9 / 380.9 |
| `asr-zh-fleurs-4048226328336996391` | 这些理论研究为什么某些人想要做他们所做的事，以及他们所处的环境中哪些事情会让他们做或不做某些事情。 | 0.020833 | 0.032258 | 164.8 | 461.8 | 461.8 / 461.8 |
| `asr-zh-fleurs-16272218284004703131` | 虽然有一种试验性疫苗看似能够降低埃博拉病毒的死亡率，但迄今为止还没有明确证明任何药物都适合治疗现有的感染。 | 0.061224 | 0.115385 | 129.3 | 427.1 | 427.1 / 427.1 |
| `asr-zh-fleurs-10104380129516909856` | 它的长下颚上布满了七十多颗剃刀般锋利的牙齿，上颚上还有一排，这意味着任何与它相遇的东西都无路可逃。 | 0.043478 | 0.071429 | 176.2 | 475.1 | 475.2 / 475.2 |
| `asr-zh-fleurs-10852536777113131618` | 因此，一个组织在共同努力克服障碍的过程中，往往会带来新的创新流程，以满足客户的需求。 | 0.000000 | 0.000000 | 96.5 | 392.3 | 392.3 / 392.3 |
| `asr-zh-fleurs-1868048601472473390` | 他没有设定削减数额，只是表示将根据中国的经济产出来决定数额。 | 0.000000 | 0.000000 | 113.7 | 408.4 | 408.5 / 408.5 |
| `asr-zh-fleurs-8424295226444731846` | 场景在金字塔上展示，不同的金字塔被点亮。 | 0.000000 | 0.000000 | 87.8 | 381.5 | 381.5 / 381.5 |
| `asr-zh-fleurs-15324743446256875074` | 早先于全球发布的华硕EPC个人笔记本，因成本节约及功能性因素，成为2007年台北IT资讯月备受瞩目的话题。 | 0.038462 | 0.041667 | 170.8 | 469.1 | 469.1 / 469.1 |
| `asr-zh-fleurs-8702090716396029445` | 咖喱饭菜是一道以香草和香料为主的菜肴，并配肉类或蔬菜。 | 0.038462 | 0.058824 | 113.2 | 408.1 | 408.1 / 408.1 |
| `asr-zh-fleurs-1177351021583652757` | 除了周三的比赛，卡帕内多还参加了锦标赛的两项个人竞赛。 | 0.264706 | 0.062500 | 87.1 | 382.1 | 382.1 / 382.1 |
| `asr-zh-fleurs-16106752411261640702` | 同理，有了申根签证，你就不必分别向每个申根成员国申请签证，从而节省了时间、金钱和手续。 | 0.000000 | 0.000000 | 133.9 | 431.2 | 431.2 / 431.2 |

All 32 cases have zero infrastructure, public-contract, quality-gate, and
performance-gate failures. High per-case error rates remain visible above; the
aggregate does not hide them.

### 19.3 New GSM8K responses, exact answers, and timing

These 16 requests used public SSE, native Qwen3.8 on GPU 7, `temperature=0`,
`max_tokens=256`, and no retry. Exact accuracy is **14/16 = 0.875000**. The full
real responses are reproduced here and retained in `answers/`.

| Case | Full real response | Predicted / gold | Exact | TTFT ms | Terminal / total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `gsm8k-test-0000` | Janet uses 3 eggs for breakfast and 4 for muffins, totaling 7 eggs consumed. She has 16 - 7 = 9 eggs left to sell. Selling 9 eggs at $2 each gives her 9 * 2 = $18.<br>#### 18 | 18 / 18 | 1 | 418.8 | 1588.4 / 1588.4 |
| `gsm8k-test-0001` | The robe requires 2 bolts of blue fiber and 1 bolt of white fiber, which is half of the blue amount. Adding these together gives a total of 3 bolts.<br>#### 3 | 3 / 3 | 1 | 317.8 | 1048.3 / 1048.3 |
| `gsm8k-test-0002` | Josh's total investment was $130,000, which is the sum of the purchase price and repairs. The house's new value is $200,000, calculated by increasing the original $80,000 by 150%. Subtracting the total cost from the new value gives a profit of $70,000.<br>#### 70000 | 70000 / 70000 | 1 | 326.2 | 1954.6 / 1995.6 |
| `gsm8k-test-0003` | James runs 3 sprints per session, so 3 times 3 equals 9 sprints per week. Each sprint is 60 meters, so 9 times 60 equals 540 meters.<br>#### 540 | 540 / 540 | 1 | 316.9 | 1275.3 / 1275.3 |
| `gsm8k-test-0004` | Wendi's 20 chickens require a total of 60 cups of feed per day, calculated by multiplying 20 chickens by 3 cups each. Since she has already given 15 cups in the morning and 25 cups in the afternoon, she has fed 40 cups so far. Therefore, she needs to give 20 cups in the final meal.<br>#### 20 | 20 / 20 | 1 | 440.7 | 2021.0 / 2061.2 |
| `gsm8k-test-0005` | The first, third, and every odd-numbered glass costs $5, so 8 glasses cost $40. The second, fourth, and every even-numbered glass costs 60% of $5, which is $3, so 8 glasses cost $24. Adding these amounts together gives a total of $64.<br>#### 64 | 64 / 64 | 1 | 323.7 | 1733.3 / 1774.3 |
| `gsm8k-test-0006` | Charleston has 80 sheep, which is 4 times the 20 sheep in Seattle. Toulouse has 160 sheep, which is twice the amount in Charleston. The total number of sheep is 260.<br>#### 260 | 260 / 260 | 1 | 319.1 | 1338.1 / 1338.1 |
| `gsm8k-test-0007` | Carla downloads 80 GB before the restart, which takes 40 minutes. The 20-minute update delay is added to the total time. Since she must restart from the beginning, the total time is the initial 40 minutes plus the 20-minute delay plus the full 100 minutes to download the entire 200 GB file again, totaling 160 minutes.<br>#### 160 | 160 / 160 | 1 | 429.3 | 2117.3 / 2157.4 |
| `gsm8k-test-0008` | John initially drives 180 miles away from home. On the return trip, he covers 15 miles in the first half-hour and 240 miles in the remaining 2.5 hours, for a total return distance of 255 miles. Since he traveled 255 miles back but only needed to cover 180 miles, he is 75 miles past his home.<br>#### 75 | 75 / 45 | 0 | 427.0 | 2120.6 / 2161.2 |
| `gsm8k-test-0009` | Eliza earns $400 for her first 40 regular hours at $10 per hour. She earns $60 for the 5 overtime hours at a rate of $12 per hour. Her total earnings for the week are $460.<br>#### 460 | 460 / 460 | 1 | 327.1 | 1478.0 / 1478.0 |
| `gsm8k-test-0010` | The second month had 180 downloads, which is three times the first month's 60. The third month had 126 downloads, calculated by reducing 180 by 30%. Adding the downloads from all three months gives a total of 366.<br>#### 366 | 366 / 366 | 1 | 328.7 | 1573.3 / 1573.3 |
| `gsm8k-test-0011` | Toula spent 3 times 68 dollars on donuts, 2 times 80 dollars on cupcakes, and 6 times 55 dollars on cheesecakes. Adding these amounts together gives 204 plus 160 plus 330, which equals 694.<br>#### 694 | 694 / 694 | 1 | 419.1 | 1754.3 / 1754.3 |
| `gsm8k-test-0012` | First, calculate the net annual profit by subtracting the yearly maintenance cost from the revenue, which is 7 lemons times $1.50 minus $3, resulting in $7.50 per year. Next, divide the initial planting cost of $90 by the annual net profit of $7.50 to find the break-even point, which equals 12 years.<br>#### 12 | 12 / 13 | 0 | 419.4 | 2015.3 / 2055.4 |
| `gsm8k-test-0013` | Let the starting number be x. After selling a third at the green house, 2x/3 remain; after selling 2 more at the red house, 2x/3 - 2 remain; and after selling half of that at the orange house, (2x/3 - 2)/2 = 5, which simplifies to x/3 - 1 = 5, so x/3 = 6 and x = 18.<br>#### 18 | 18 / 18 | 1 | 324.3 | 2232.1 / 2272.2 |
| `gsm8k-test-0014` | 20% of 20 students is 4, leaving 16 students. 25% of the remaining 16 is 4, so 12 students enrolled in hip-hop. 12 out of 20 is 60%.<br>#### 60 | 60 / 60 | 1 | 326.9 | 1461.6 / 1461.6 |
| `gsm8k-test-0015` | The profit from the jewelry would be 5000 times 0.025, which equals 125 dollars. The profit from the electronic gadgets would be 8000 times 0.012, which equals 96 dollars. Since 125 is greater than 96, the maximum profit is 125 dollars.<br>#### 125 | 125 / 125 | 1 | 425.6 | 2005.4 / 2045.8 |

`gsm8k-test-0008` reasons to `#### 75` instead of `45`. `gsm8k-test-0012`
reasons to `#### 12` instead of `13`. Both are model-quality misses with valid
completed responses. No case, marker, threshold, or response was changed to
improve the score.

### 19.4 Independent audit, input verification, and deterministic replay

`tools/multimodal_coverage_audit.py` does not import the runner's scoring or
aggregation functions. It rebuilt all 70 answers and 38 transcripts from raw
SSE, checked event order and terminal identity, recomputed every per-case metric
and source aggregate, and compared them with `report.json` and `summary.json`.
It also resolved every pinned input again and checked all input hashes. The 38
WAVs decode as non-empty PCM16 mono 16 kHz; the 22 images have valid JPEG/PNG
signatures and matching hashes; all 16 GSM8K prompts match their pinned row
hashes. The audit contains per-case response/transcript hashes and media details.

```sh
/tmp/umm-task27-client-venv/bin/python -I tools/multimodal_coverage_audit.py \
  --source accepted_vqa_joint=/tmp/umm-task27-evidence-local/combined-70/accepted-vqa-joint \
  --source standalone_asr=/tmp/umm-task27-evidence-local/combined-70/standalone-asr \
  --source gsm8k=/tmp/umm-task27-evidence-local/combined-70/gsm8k \
  --cache-dir ~/.cache/umminfer-multimodal-benchmark \
  --output /tmp/umm-task27-coverage-audit-recheck.json
```

The second audit output was byte-identical to the first. The accepted source
passed 85/85 internal checksums. The new ASR and GSM8K sources have 134 and 38
checksummed files respectively. The combined evidence root is
`/tmp/umm-task27-evidence-local/combined-70/`; all **287/287** files pass its
`SHA256SUMS`, whose SHA-256 is:

```text
4aba09f25b9edf62ad89a0316a74c2c158cb9fe48d7de73ef87a6e4547b8e5b7
```

The separate deterministic 70-case contract replay also passes, including
Mandarin WER with the pinned segmenter. It remains labelled
`NOT_REAL_EVIDENCE`; no replay response or metric contributes to this real
coverage result.

### 19.5 New-run source, configuration, model, and binary provenance

The accepted run retains the distinct source and binary identities in section
18.5. The table below binds only the new 48 cases. The combined audit therefore
does not claim that one binary served all 70 cases.

| Object | SHA-256 or identity |
| --- | --- |
| Base main | `bad6fc3db38104ca76b4d5179f705373fabc218d` |
| Accepted final integrated | `4cafdafdc9f269d8ac4025cf66dd2c57e536200f` |
| Binary-build and new-run source | `e51f1d9098899ed9cb9a4dc9df1576e60b95e097` |
| Source archive | `8e21c8bee5cbdcab54d3f33dc010c5314ebebc143defab030e9733d41e97e8d1` |
| `tests/multimodal_benchmark.py` | `f511e7f63bf996f01e6bfcff9cda033e7dc3ae4300e1903c33f675a62f8eb848` |
| `benchmarks/multimodal_manifest.json` | `599963a3851d5ce02d740b8b06c423726bae5c9c1e07d1931708074c83caa9a3` |
| Combined `run-config.json` | `18287c2fa58ff890f1ae80987e54e08565e7f0837676f7e3d7e5900ee77fd95b` |
| Combined `provenance.json` | `466c91f29e85721d672b88f4818a370919903e123a2c6e48b9afa7d70f458f3d` |
| `coverage-audit.json` | `d4514acd07f280d9d7097dac45bd2b8caa85d9298f8caba22102ba28ead5f6a5` |
| `tools/multimodal_coverage_audit.py` | `79b47f07036e25ba5d82a83ec7e23acd7583603558fa2d0728353d3a9a541a16` |
| `umm-infer` | `d2bc15821190c043a734576c422e8dbed61134e79b219de0516b71a10777a5e4` |
| `umm-native-stage-asr` | `cf8962464aa3c44ad0f3f812c9f756dc2939b4038c3da1376cece96e9f234a81` |
| `umm-native-stage-qwen38` | `446661e9a99ebb00688a01f6e52f21dcfc20a995308f4f092664dae151fd3fc5` |
| Qwen3.8 `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Qwen3.8 + ASR available-weight manifest (18 + 2 shards) | `edadb8d2c36afae98dfff947a5749c9b042b7ea9fe123ba7c56ceaf5b8a2944f` |
| Qwen3-ASR `config.json` | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| Model config/tokenizer manifest | `8c3980a7498d8abe3e4b239702a347018d364e229809fb1922129b6d69123fe4` |

`provenance.json` and `run-provenance/model-weights-SHA256SUMS` record every
individual available Safetensors SHA-256. These are hashes of local bytes, not
claims of an upstream revision id. The native server and both stages were built
from the new-run source. The harness changes are Python-only.

### 19.6 Cleanup and remaining limits

The service used NVIDIA H20 SM90, driver 535.161.08, ASR on explicitly idle
physical GPU 1, and Qwen3.8 on explicitly idle physical GPU 7. The server bound
loopback with kernel-assigned port 49141. GPU 0 was reserved and unused. GPUs
2–6 and the resident service were never contacted or restarted.

| GPU | Before start | Before stop | After stop |
| ---: | ---: | ---: | ---: |
| 0 | 3 MiB | 3 MiB | 3 MiB |
| 1 | 3 MiB | 5,117 MiB | **3 MiB** |
| 7 | 3 MiB | 56,365 MiB | **3 MiB** |

Only the three owned server/stage processes were signalled. `/health` reported
`in_flight=0` before shutdown. The isolated endpoint refused connections after
shutdown. The failed client SSH tunnel was also stopped by its owned pid; no
tunnel or isolated benchmark listener remains.

Remaining limits:

- The accepted 22 and new 48 cases are disjoint real evidence from different
  bounded runs. This establishes declared 70-case coverage, not a same-binary
  full round, an accepted regression baseline, or three-round stability.
- GSM8K accuracy is 87.5%, with the two visible quality misses above. Human
  quality review remains necessary for joint free-form answers and the
  auxiliary standalone-ASR VLM outputs.
- Standalone-ASR TTFT is the auxiliary VLM first-delta time. ASR recognition
  timing is reported separately as `asr_final_ms`.
- The VQA normalization limitation in section 1 remains. Joint keyword hit is
  only an objective sub-metric. No human-quality score is inferred.

### 19.7 Final committed-tree validation

The final tree leaves `src`, `native`, `include`, `build.zig`, the manifest, and
the accepted runner/test bytes identical to the real-run source
`e51f1d9098899ed9cb9a4dc9df1576e60b95e097`. The later commits add the
independent auditor, tests, evidence documentation, and AOCI metadata only.

| Check | Result |
| --- | --- |
| Real coverage gate and independent final-tree reaudit | **PASS**, 70 unique real cases, 70 completed terminals, 70 matching inputs; `BASELINE_REQUIRED` retained |
| `umm-test-local` | **PASS**, exit 0; includes unified contracts and voice/client C++ tests |
| Native ASR/Qwen3.8 Release host CTest | **PASS 3/3** (`asr_host_tests`, `qwen38_host_tests`, `qwen38_video_input_tests`) |
| Client boundary with explicit local dependency prefixes | **PASS** chat parsing, serve parsing, URL policy; real MLX/TinyASR intentionally `--skip-real` because they are not this SM90 gate |
| Client media CLI | **PASS 8/8** |
| `tests/narrative_live_e2e.py --self-check` | **PASS**, sequence/terminal/clean-prose/interleaving/PCM/PNG/MP4/HTTP 400 checks; fixtures are not real generation |
| `tests/blackbox_integration.py --self-check` | **PASS** log cap, timeout, and orphan process-group cleanup |
| Deterministic 70-case replay | **PASS**, all metrics match the replay; `NOT_REAL_EVIDENCE` retained |
| Coverage-auditor offline negative tests | **PASS** metric arithmetic, missing-vs-zero, raw SSE, terminal, and checksum rejection |
| Feature-map validation | **PASS**, 189 implemented rows and 843 references at `aa4353458ef55134bf68bd0cfe4fa5a19a754508` |
| Hand-maintained AOCI baseline | **PASS**, 453 files, zero digest mismatches |
| Combined evidence `SHA256SUMS` | **PASS 287/287**, final-tree audit output byte-identical to the sealed audit |

One preliminary native run used an unoptimized build concurrently with the full
C++ suite. It failed the existing performance assertion with `lm_fast=5.992 ms`
against the 2 ms limit. The quiet Release rerun passed all three tests. No source
or threshold was changed. The failed client-prefix invocation was rerun with the
actual local dependency prefixes; it had failed before executing any client case.
Final validation logs are under `/tmp/umm-task27-final-validation/`.
