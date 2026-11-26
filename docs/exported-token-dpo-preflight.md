# Exported-token joint-DPO CPU data and preflight path

Date: 2026-11-26. Integration base:
`5d04dd006c84edd48913cae5b9b91b91f0a4e12e`, which contains the accepted
exported-token SFT path and the newest then-available mainline integration.
Implementation commit:
`fde70d5a6bb7252d7eddcf71a2f151eba20d6e5c`.
This task adds deterministic CPU-only DPO data/preflight coverage. It did not
run a GPU, model inference, a real DPO optimizer, serving, or remote execution.
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

## Future exact-local-tokenizer gate

No real local tokenizer was supplied to the new DPO commands in this task, so
this standalone CLI gate is **NOT_RUN**. A future CPU-only run must use an
export pack produced by the exact same complete local tokenizer/model directory,
id, revision, and Transformers/tokenizers runtime, with offline loading:

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
Real export-driven DPO optimization, real local-tokenizer CLI execution, model
quality, generalized improvement, production training, sampled online behavior
ids, online rollout parity, model inference, serving, and vision/media forward
execution remain `NOT_RUN`, `NOT_CLAIMED`, or `NOT_COVERED` as applicable.
