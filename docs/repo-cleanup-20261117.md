# Repository cleanup — 2026-11-17

Scope: stale branches and worktrees, build artifacts, dead files. Base commit
`1f851a0` on `main`. Branch `cleanup/repo-20261117` in worktree
`/tmp/umminfer-repo-cleanup`.

`main` was rewritten on 2026-11-17 with `git-filter-repo` (author and committer).
Commit ancestry is therefore not evidence of merge state. Every branch below was
judged by content:

- `git diff --numstat main <branch>`: lines the branch adds relative to `main`.
- `git cherry main <branch>`: whether each branch commit's patch already exists
  in `main`.
- Blob test: for every file whose blob differs from `main`, check whether that
  exact blob appears anywhere in `main` history
  (`git rev-list --objects main -- <path>`). A blob absent from `main` history
  is the only candidate for unique content.

## 1. Branch verdicts

Legend: `covered` = every differing blob exists in `main` history.
`superseded` = a blob is absent from `main` history, but it is an older version
of a file that `main` has since replaced.

| Branch | main..branch commits | cherry | unique blobs | Verdict | Replaced by in main |
| --- | --- | --- | --- | --- | --- |
| `cleanup/cpp` | 1 | `-` | none | covered | `922d866` |
| `cleanup/docs` | 1 | `-` | `README.md` (7 lines), `docs/validation.md` (0 lines) | superseded | `65845ec`; `main` README accepts optional audio and string content |
| `cleanup/python` | 1 | `-` | none | covered | `e58e832` |
| `native/self-hosted-backend` | 1 | `-` | none | covered | `04e6215` |
| `phase4/bench` | 1 | `+` | `docs/performance-baseline.md` (11 lines) | superseded | `a6baae3`; the unique lines are the pre-provenance text that `main` replaced with the full provenance section |
| `phase4/cancel-failure` | 1 | `-` | none | covered | `83273ea` |
| `phase4/session-joint-input` | 2 | `+` | `README.md`, `python/umminfer_qwen_omni_worker.py`, `src/serve/openai_api.cpp`, `tests/unified_contract_test.cpp` | superseded | `main` `unified_input_parts` accepts audio-free requests and `session_history`; worker has SGLang `rid` plus `/abort_request`; `tests/unified_contract_test.cpp:2612` has the empty-session rejection |
| `phase4/streaming-coverage` | 2 | `+` / `-` | `README.md`, `docs/roadmap.md` | superseded | `68115b3`; unique text is the 2026-11-13 status snapshot |
| `phase5/mac-unified` | 1 | `-` | none | covered | `6b9c7be` |
| `phase5/model-provenance` | 1 | `-` | `docs/joint-optimization.md` | superseded | `fa14f17`; `main` keeps the same hashes in `docs/performance-baseline.md:97-100` |
| `phase5/session-history` | 1 | `+` | `README.md`, `docs/roadmap.md`, `docs/validation.md` | superseded | `c186198`; branch says H20B multi-turn validation is `NOT_RUN`, `main` records it as passed |
| `phase6/compute-failure` | 1 | `-` | `docs/roadmap.md`, `docs/validation.md` | superseded | `a2cb283`; `docs/validation.md:407` section 5f |
| `phase6/gpu-evidence` | 1 | `+` | `README.md`, `docs/roadmap.md`, `docs/validation.md` | superseded | `4e9e6b2`; `docs/validation.md:320` section 5e, `README.md:657` |
| `phase6/realtime-h20b` | 1 | `-` | none | covered | `f08292b` |
| `phase6/rl-architecture` | 1 | `-` | `README.md` | superseded | `d0eecf5` |
| `phase6/rl-training` | 1 | `-` | `README.md`, `docs/roadmap.md` | superseded | `bc6c4f4`; `docs/roadmap.md:298` |
| `phase7/h20b-service` | 1 | `-` | none | covered | `1d69471` |
| `phase7/multimodal-benchmark` | 1 | `-` | none | covered | `cc2ec6e` |
| `test/blackbox-integration` | 1 | `-` | none | covered | `2852147` |

No branch keeps unique work. Every `+` from `git cherry` is a patch-id
difference caused by the history rewrite; the blob test shows the content is an
older version of a file that `main` has since replaced.

## 2. Deleted branches and worktrees

Removed worktrees:

```
/tmp/umminfer-blackbox-integration
/tmp/umminfer-cleanup-cpp
/tmp/umminfer-cleanup-docs
/tmp/umminfer-cleanup-python
/tmp/umminfer-mixed-output-cpp
/tmp/umminfer-mixed-output-workers
/tmp/umminfer-native-backend
/tmp/umminfer-native-image
/tmp/umminfer-native-integration
/tmp/umminfer-native-omni-perf
/tmp/umminfer-native-tts
/tmp/umminfer-native-vlm
/tmp/umminfer-phase7-benchmark
/tmp/umminfer-phase7-service
```

Kept: `/tmp/umminfer-phase4-20261113` (end-to-end verification in progress) and
`/tmp/umminfer-repo-cleanup` (this worktree).

Removed branches: the 19 branches in section 1, plus 7 already-merged branches
whose worktrees were removed: `mixed-output/cpp`, `mixed-output/workers`,
`native/integration`, `native/omni-perf`, `native/qwen-image`, `native/qwen3-omni`,
`native/qwen3-tts`.

## 3. Untracked files preserved from removed worktrees

These files were untracked in the removed worktrees. They were moved, not
deleted, to `/tmp/umminfer-preserved-worktree-files-20261117/` with
`MANIFEST.sha256` (33 files, 17 MiB):

| Source worktree | Files | Note |
| --- | --- | --- |
| `mixed-output/workers` | `python/umminfer_image_stage.py`, `python/umminfer_tts_stage.py`, two tests | Python stages; `main` serves image and TTS through the native executors |
| `native/self-hosted-backend` | `native/evidence-h20b/` | ASR acceptance evidence: `wer_cer.json`, logs, `SHA256SUMS` |
| `native/tts` | `native/tts_tests/tools/*.cu` | four diagnostic probes |
| `native/qwen3-omni` | `evidence/` (16 MiB), `.scratch/`, `native/omni/mini_*.{json,txt}` | golden preprocessing data; `main` has the `mini_*` fixtures under `native/omni_tests/` |

Nothing in this directory is deleted. Move it somewhere durable if the evidence
must survive `/tmp` cleanup.

## 4. Build artifacts and dead files

- Tracked build artifacts: 0. `git ls-files` matches none of `zig-out/`,
  `.zig-cache/`, `build/`, `*.o`, `*.a`.
- Untracked build output from the verification build: `.zig-cache/`, `zig-out/`,
  `__pycache__/`; all ignored.
- `.gitignore`: added `.venv/` (the README creates it in the repo root) and
  `.scratch/`.
- Dead code and stale markers: no `TODO`, `FIXME`, `XXX`, `HACK` or
  `deprecated` in `src/`, `include/`, `python/`, `service/` or `build.zig`. No
  unreferenced file under `src/` or `python/`. No duplicate document titles.
  No zero-byte tracked files.
- Files with no textual reference, kept because they are runnable tests or
  documentation entry points:
  `tests/mlx_e2e_interleave_perf.py`, `tests/mlx_greedy_perf.py`,
  `tests/mlx_perf_selfcheck.py`, `tests/mlx_perf_compare_selfcheck.py`,
  `tests/qwen_omni_worker_live.py`, `tests/sglang_nan_injection_test.py`,
  `native/stage_tests/collect_evidence.sh`, `native/IMAGE_NATIVE.md`,
  `native/STAGE_NATIVE.md`.

## 5. SGLang / Python glue still on the serving path

Reported only; nothing removed.

| Entry | File | Callers |
| --- | --- | --- |
| `--upstream-kind sglang` proxy mode | `src/main.cpp:289-297`, `src/serve/openai_api.cpp:664-675` | `serve` CLI |
| `ApiConfig.upstream_*` fields | `include/umminfer/serve/openai_api.hpp:33-46` | `src/serve/openai_api.cpp:1610` |
| Qwen ASR worker over SGLang HTTP | `python/umminfer_qwen_asr_stage.py:110-120` | `python/umminfer_service.py:224`, `docs/h20b-service.md:14` |
| Qwen Omni worker over SGLang HTTP | `python/umminfer_qwen_omni_worker.py:70-127` | `python/umminfer_service.py:233`, `docs/h20b-service.md:15` |
| Supervisor Python worker branch | `python/umminfer_service.py:211-242` | used only when the config has no `native_stages` |
| Phase 7 runbook (Python/SGLang, port 43300) | `docs/h20b-service.md` | not referenced by `service/h20b-native-service.sh` |

The deployed native path is `service/h20b-service.json` (`native_stages`,
port 43301) plus `service/h20b-native-service.sh`. The native script starts no
Python or SGLang process; its header comment says so.

## 6. Verification

Run in `/tmp/umminfer-repo-cleanup`:

```
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew   -> exit 0
zig build test -Ddeps-prefix=/opt/homebrew                     -> exit 0
```

## 7. Items left for user decision

- `docs/h20b-service.md`: runbook for the Python/SGLang Phase 7 service
  (port 43300). It overlaps `native/STAGE_NATIVE.md` and
  `service/h20b-native-service.sh` (port 43301). It also records validation
  results. Kept.
- The six unreferenced test scripts in section 4: runnable but invoked by no
  document or build step. Kept.
- Preserved untracked files in `/tmp/umminfer-preserved-worktree-files-20261117/`:
  evidence and superseded Python stages. Kept.
- Eight fully merged branches without worktrees remain:
  `cuda-ipc`, `fix/harness-probe-isolation`, `phase4/integration`,
  `phase5/cuda-stress`, `phase5/final-h20b`, `phase5/joint-optimization`,
  `phase5/streaming-abort`, `rename-umminfer-20261111`. They are outside the
  requested no-merged set and were not deleted.
