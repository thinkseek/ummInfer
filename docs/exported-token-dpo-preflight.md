# Exported-token joint-DPO CPU data and preflight path

Date: 2026-11-26. The original deterministic implementation was integrated
non-destructively with newer main. The integrated real-local-tokenizer CPU
measurement source is `e400caed4250bb8b67e55157745cc3528450ac80`.
The original implementation commit recorded by Task #43 is
`fde70d5a6bb7252d7eddcf71a2f151eba20d6e5c`.
Task #45 executed only tokenization, data packing, preflight, and independent
verification in the previously validated offline CPU runtime. It did not run a
GPU, model inference, a real DPO optimizer, model download, or serving endpoint.
Exported ids are post-hoc supervision ids, not sampled online behavior ids.

## Opt-in contract and compatibility

`--token-source legacy` remains the default. The accepted full
`prompt + "\n" + assistant_target` encoding, independent chosen/rejected masks,
sigmoid-DPO execution path, checkpoint fields, and historical real evidence are
unchanged. A preference pack built without token exports retains byte-identical
JSONL. The pinned legacy two-pair `dpo_train.jsonl` SHA-256 remains
`2e7e15b5143bdba21cb597371849c35b911c07ba03b647ad8b184491617e4ead`.
The accepted SFT adapter and its real evidence files were not modified.

When exported trajectories are packed, DPO chosen and rejected response summaries
additively retain `token_export` and the complete `token_export_source`. GRPO
rollout summaries do not gain these fields. Export-driven DPO requires explicit
`--token-source token_export` and is all-or-nothing across every side of every
`dpo_train` pair:

- missing, mixed, partial, stale, malformed, duplicate, crossed, prompt-mismatched,
  tokenizer-mismatched, or unsupported records fail without per-side fallback;
- structural pair/source validation precedes planner output and all ML imports;
- a fresh tokenizer-only process hides Torch from Transformers discovery and
  rejects actual Torch imports. It rehashes the complete local model/tokenizer
  directory, binds the exact tokenizer id/revision/runtime and file list, and
  strongly revalidates each v1/v2 export before planner or optimizer output;
- only the shared unexported `prompt + "\n"` prefix is tokenized for the training
  pack. Exported chosen/rejected target ids, raw offsets, labels, weights, and
  annotation bindings are used directly. The legacy full encoding is computed
  only as an exact equality reference and never as a fallback;
- pair id, prompt hash/prefix, chosen/rejected export hashes, tokenizer identity,
  packed ids/labels/masks, and frozen-reference model/normalization inputs enter
  run identity. Export checkpoints additionally bind the reference-input digest.
  Legacy/export runs and different preference pairs cannot cross-resume.

`--preflight-only` requires export mode and no resume checkpoint. It writes
`token_export_preflight.json` plus SHA-256 and the ordinary NOT_RUN plan, but no
live manifest, model output, optimizer state, or training checkpoint. Its result
explicitly marks real training, optimizer, inference, and quality `NOT_RUN`.

## Deterministic evidence

The v1/v2 fixtures contain two preference pairs with multilingual text,
NFC-decomposed input, emoji, `<spoken>` spans, structured image/video targets,
completed CAS-backed terminal legs, and explicit `source=slot-0` anchors on the
chosen targets. Both sides are packed independently and must equal the legacy
reference exactly. The CPU numerical fixture uses different chosen/rejected
reference masks and obtains policy/reference scaled margin `0.15` and exact
sigmoid-DPO loss `0.6209570477895321`; it executes no optimizer.

| Item | Format v1 | Format v2 |
| --- | --- | --- |
| Exported trajectory SHA-256 | `2f6ecd8a45305494a5c4e47a8b4088a9cf89abd2be914edad0b317799b846990` | `861ad64feb973d986ea79ed6193d9a54ca93ee3e34b6ecb3bb323af30467bf37` |
| DPO train SHA-256 | `fd2f760941520af6128adfd28e857f4df7bbc3cee293eae0deaf41d7be185833` | `f68d0121dfdb5626b30565a52ba30c1360ba635990c359de8e523d806388cdbe` |
| Raw preflight bundle SHA-256 | `3b7c61f4f1524af2abedad24015a1c67eafceb88a8052695c1848c77b6cab5d5` | `502f23aa391ea09e4c1bc8a0de472a8202d5fb40552d5e1df2cb95674bac721c` |
| Pair ids | `d67b1db95b424b828ba909bbfc7a9a26`, `ffdf7bdc94d1c61f7f0c681b5140a75b` | same pair identities |
| Independent fresh-process reconstruction | **PASS, byte-identical** | **PASS, byte-identical** |

The invalid-input matrix covers missing chosen exports, partial rejected exports,
stale source, malformed ids, crossed pairs/sides, prompt mismatch, duplicate pair
ids, unsupported formats, mixed tokenizer identities, and wrong tokenizer
id/revision/runtime. Every pre-output gate rejects before planner output, Torch
or Transformers import, live manifest, or checkpoint creation. Split-first and
uninterrupted prepared runs have identical run identity and packed schedules;
resume rejects legacy/export, different-pair, and reference-input mismatches.
A tampered bundle fails independent verification even when its sidecar is
re-signed.

## Commands actually run

Implementation-source SHA-256 values for this CPU evidence are:

| File | SHA-256 |
| --- | --- |
| `python/umminfer_joint_dpo_live.py` | `e432c02364681638346a7757bfca0bf4fcc21a22fb526dadfb8b13d4aa735ca2` |
| `python/umminfer_sft_dataset.py` | `ca689792d35d1c7145b54cbd96216d669938b127a65395378a5f816af22898bb` |
| `tests/joint_dpo_live_test.py` | `b10264560d44fddb7b551204f5502b3b52a5dbc0937665767010f29d2069c3c0` |
| `tests/joint_trainer_test.py` | `4a8fcc4ead5ebc13088aef2d80d619b39a228ab19ea3739046d7ee59b948538f` |
| `tests/sft_dataset_test.py` | `7054cbc2b63e1e319318e8fe13febcc3a4f0b74380e515395abf8f09ea6b5a09` |

All commands were local CPU-only and completed successfully:

```sh
python3 -I tests/token_export_test.py
python3 -I tests/sft_dataset_test.py
python3 -I tests/joint_trainer_test.py
python3 -I tests/joint_sft_live_test.py
python3 -I tests/joint_dpo_live_test.py
python3 -I tests/joint_grpo_live_test.py
python3 -I tests/online_rollout_test.py
python3 -I tests/stage_c_pipeline_test.py
python3 -I tests/rl_architecture_test.py
python3 -I tests/sse_replay_test.py
umm-test-local
```

The DPO suite ran 10 tests. The token-export suite passed its deterministic
cases and reported its optional real tokenizer fixture `NOT_RUN`. The DPO suite
itself prints `REAL JOINT DPO OPTIMIZER: NOT_RUN` and
`MODEL QUALITY / GENERALIZED IMPROVEMENT / GRPO: NOT_RUN`.

The following standalone fixture writer/verifier commands are also exercised by
the suite in independent processes for both versions (replace `VERSION` with `1`
or `2`, and use a new temporary directory):

```sh
python3 -I tests/joint_dpo_live_test.py --fixture-bundle-write TMPDIR VERSION
python3 -I tests/joint_dpo_live_test.py \
  --fixture-bundle-verify TMPDIR VERSION TMPDIR/verification.json
```

## Real exact-local-tokenizer gate — PASS

Task #45 supplied the designated local `Qwen/Qwen2.5-0.5B-Instruct` directory to
both standalone commands. Revision remains `UNKNOWN` because upstream revision
verification was not performed. The pinned runtime was Python 3.12.3,
Transformers 5.23.0, and tokenizers 0.23.3. The complete 31-regular-file bundle,
including cache metadata and the 988,097,824-byte weight file, was rehashed; its
file-list SHA-256 is
`23ebb8d9f27833dadcc9047936b3f73a4453de77a3078fff40e74c07f2d68e83`.
The two real v2 pairs contain multilingual text, decomposed `café`, emoji byte
fallback, `<spoken>` spans, structured image/video targets, completed CAS legs,
and chosen-side explicit `source=slot-0` anchors. Chosen/rejected targets differ;
rewards are respectively `1.0` and `0.6666666666666666` in both pairs.

The actual command shape was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" python3 -I python/umminfer_joint_dpo_live.py \
  --dataset-dir EXPORTED_PREFERENCE_PACK \
  --config configs/training/joint_dpo_live_gate.json \
  --model-path EXACT_LOCAL_MODEL_TOKENIZER_DIR \
  --output-dir NEW_CPU_PREFLIGHT_DIR \
  --source-commit SOURCE_40_HEX --max-steps 2 \
  --token-source token_export --preflight-only

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" python3 -I python/umminfer_joint_dpo_live.py \
  --token-export-verify \
  --bundle NEW_CPU_PREFLIGHT_DIR/token_export_preflight.json \
  --dataset-dir EXPORTED_PREFERENCE_PACK \
  --config configs/training/joint_dpo_live_gate.json \
  --model-path EXACT_LOCAL_MODEL_TOKENIZER_DIR \
  --tokenizer-id EXACT_TOKENIZER_ID --tokenizer-revision EXACT_REVISION \
  --output NEW_CPU_PREFLIGHT_DIR/independent_verification.json
```

The verifier independently revalidates the pack and tokenizer, rebuilds the
planner/schedule/run identity in a fresh process, and requires canonical bundle
equality. It does not load model weights for inference or import Torch.

The complete real gate was driven by:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" python3 -I tests/joint_dpo_live_test.py \
  --real-tokenizer-preflight --model-path EXACT_LOCAL_MODEL_TOKENIZER_DIR \
  --evidence-dir NEW_EVIDENCE_DIR --source-commit \
  e400caed4250bb8b67e55157745cc3528450ac80
```

The positive preflight and the independent verifier each returned 0 in a separate
fresh process with `torch_imported=false`. Exact ids, raw offsets, labels,
weights, annotations, shared prompt prefixes, independent policy/reference
inputs, pair/prompt/export-bound identity, and canonical bundle reconstruction
all passed. The legacy unexported pack retained its pinned SHA-256. Real
partial-export, stale-source, malformed-token, crossed-pair, and
tokenizer-mismatch cases each exited 2 before planner, checkpoint, optimizer,
live manifest, preflight bundle, or Torch payload creation.

| Real-gate item | Exact value |
| --- | --- |
| Exported trajectory / DPO train SHA-256 | `3678f744c6cec776e501b70b3895be1c80ec8528141d99c3e4f4782d37a03324` / `ff13bf833b98b82f31e032453998a7d39cbf0ae137fa4d701e9abe50adc5bd21` |
| Legacy DPO train SHA-256 | `2e7e15b5143bdba21cb597371849c35b911c07ba03b647ad8b184491617e4ead` |
| Preflight bundle / run identity SHA-256 | `fc6c1d00eb171c0452e28557d979d69229b71cf44f01dea434c9548e9f86b534` / `30e09a248fdca7d2111a1358a1ea0752c8009892aa234195c8107f8b4eb3352d` |
| Dataset manifest / exact config / resolved config SHA-256 | `e97295de2e8316aaa05bf31fe80ce6042b52622988447ec48e8709895026bff9` / `ca1cfe51e2d25bd5a59546362fc719bf2ff19ed54c79d0d81880eea1c64c3b04` / `497a31c6e056ef5ff45f8573a03c671cec5d509be0ea6f39929314d74a2dd72d` |
| Model config / weight-file SHA-256 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` / `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Independent verifier output SHA-256 | `976bc40cc6614a734939563232b1b909b64a0a5f90f35aeca63ce0289c7d72a3` |
| Real adapter / real harness SHA-256 | `e432c02364681638346a7757bfca0bf4fcc21a22fb526dadfb8b13d4aa735ca2` / `32eabf6b2d1b8fa2521c30385ac71e2a8b3bfb8e4ab7c3dcbf84f42e653da5bb` |
| Acceptance / commands / SHA256SUMS SHA-256 | `92702d0e3652a1e8d673f73b9bd78f4fcab46c8ce6048e119b61ddf6f0349894` / `6f0e28cc118ac3a34f333fbb24c1fcf620093e6a63525f83e5286d915a85882a` / `2cae07a72a0a52bfd69e94f59450e5a5e1383772a21d7deef807c7feb13a3f2a` |

The 44-file raw bundle under `/tmp/umminfer-task45-e400cae-evidence` retains raw
trajectories/exports, CAS fixtures, preference packs, preflight/planner output,
fresh-process verifier output, all positive/negative stdout and stderr, exact
commands, source/runtime/file hashes, and `SHA256SUMS`; all entries passed an
independent checksum run. The compact committed result is
[docs/evidence/exported-token-dpo-real-tokenizer-e400cae.json](evidence/exported-token-dpo-real-tokenizer-e400cae.json).
The first measurement attempt exposed only a harness assumption that scheduled
pair order equals raw pack order. Commit
`e400caed4250bb8b67e55157745cc3528450ac80` corrected that assertion to compare
unique pair sets; no adapter or real-tokenizer incompatibility was found, and the
final frozen source passed end to end.

## Remaining real-gate requirements and boundaries

A future export-driven real DPO optimizer gate must separately prove the same
legacy/export ids, labels, masks, policy/reference log-probabilities, exact
losses, gradients, parameter updates, and byte-identical split/resume checkpoints
on an explicitly approved device. It must record the complete source,
data/tokenizer/model/runtime hashes and independently verify the evidence.
This task supplies none of those real optimizer results.

The historical legacy-only SM90 DPO result in
[joint-training.md](joint-training.md#measured-sm90-dpo-acceptance--2026-11-26)
remains historical evidence for its recorded source, not a result of this task.
Real export-driven DPO optimization, model quality, generalized improvement,
production training, sampled online behavior ids, online rollout parity, model
inference, serving, and vision/media forward execution remain `NOT_RUN`,
`NOT_CLAIMED`, or `NOT_COVERED` as applicable. The real local-tokenizer CLI
preflight and independent verifier are **PASS**; this does not change the real
DPO optimizer boundary.
