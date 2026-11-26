# Exported-token joint-SFT path and real SM90 gate

Date: 2026-11-26. Source under test:
`93214a5261ebaf2fd6108e129aa5e7993af18aa7`.
The gate is an optimizer/data-path acceptance result, not a model-quality or
online-policy result. Exported IDs are post-hoc supervision IDs; they are not
evidence of IDs sampled by online inference.

## Data-path contract

`python/umminfer_joint_sft_live.py --token-source legacy` remains the default and
retains the accepted full `prompt + "\n" + assistant_target` fast-tokenizer
re-encoding. A pack without `token_export` has byte-identical JSONL output. The
pinned regression values are trajectory SHA-256
`f6e2496a2f62b5cfe13288855d10fa2c4bb19b571e921346749f3eb5ba9496fc`
and SFT train SHA-256
`9bf15b0fb13d25c1d7a94084174d759e58351dddcf327514d969ca63d227f24e`.

When a source trajectory has an export, the SFT builder additively retains
`token_export` and `token_export_source`. The latter is the complete source
record without the export block, allowing source-record, provenance, prompt,
target, media-reference, and slot bindings to be checked independently.

`--token-source token_export` is explicit and all-or-nothing for `sft_train`:

- every row must have a supported v1 or v2 export; mixed, partial, stale,
  malformed, or unsupported input fails without per-row legacy fallback;
- a stdlib parent preflight rejects structural errors before planner output,
  Torch/Transformers imports, checkpoints, or optimizer state;
- a fresh tokenizer-only process makes Torch unavailable to Transformers and
  also blocks actual Torch imports. It rehashes every regular file below the
  exact local model/tokenizer directory, checks tokenizer id/revision/runtime,
  re-encodes both streams, and validates v1 raw offsets or v2 NFC/byte-fallback
  provenance, normalized bytes, source mappings, labels, weights, annotations,
  and all hashes;
- only the unexported prompt prefix is tokenized for training. Target IDs, raw
  offsets, labels, weights, and annotations come from the validated export. A
  separate legacy encoding is computed solely as an equality reference and is
  never substituted for those exported values;
- tokenizer identity and per-record export/input/label/mask/annotation/offset
  hashes enter the checkpoint run identity. Legacy and export checkpoints
  therefore cannot cross-resume;
- every successful run writes `token_export_preflight.json` plus SHA-256, raw
  per-step target/export audits, canonical gradient and FP32 parameter-update
  hashes, model/optimizer hashes, and checkpoint sidecars.

The live harness uses format v2 for real Qwen NFC and byte-fallback cases.
Deterministic CPU tests separately cover v1 and v2 fixture-tokenizer equality,
legacy pack bytes, multilingual/spoken/visual/explicit-anchor cases, exact
identity mismatches, and pre-output rejection. The real tokenizer suite covers
multilingual, decomposed NFC, emoji byte fallback, spoken and visual targets,
explicit `source=slot-0`, cancellation, malformed/tampered records, and two
fresh CLI processes.

## Exact commands

The following commands ran from the repository directory containing source
commit `93214a5261ebaf2fd6108e129aa5e7993af18aa7`.

Deterministic local CPU/pre-ML checks:

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

All passed. The default token-export suite honestly reports its optional real
fixture as `NOT_RUN`; the following real-local-tokenizer command then passed
independent of GPU execution:

```sh
PYTHON=/root/rl-validation-20261114/venv/bin/python
MODEL=/root/rl-validation-20261114/models/qwen2.5-0.5b-instruct
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" "$PYTHON" -I tests/token_export_test.py \
  --real-tokenizer-dir "$MODEL" \
  --real-tokenizer-id Qwen/Qwen2.5-0.5B-Instruct \
  --real-tokenizer-revision UNKNOWN \
  --real-tokenization-version transformers-5.23.0_tokenizers-0.23.3
```

The final real optimizer command used only explicitly idle physical GPU 7. No
server, realtime stack, network port, or resident service was started or
contacted:

```sh
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
"$PYTHON" -I tests/joint_sft_live_test.py --live \
  --model-path "$MODEL" \
  --evidence-dir /tmp/umminfer-task40-export-sft-93214a5-cu124-evidence \
  --source-commit 93214a5261ebaf2fd6108e129aa5e7993af18aa7
```

The harness executes four independent training processes: legacy two-step,
export-driven two-step, export split-first (`--stop-after 1`), and fresh-process
export resume. Two additional invalid export processes must exit 2 without a
plan, preflight output, success manifest, checkpoint, or optimizer execution.

Independent verification retained in the evidence directory:

```sh
cd /tmp/umminfer-task40-export-sft-93214a5-cu124-evidence
python3 -I independent_verify.py
sha256sum -c SHA256SUMS
```

The verifier passed every acceptance, per-path arithmetic, preflight,
checkpoint-byte, and resume assertion. All 66 listed files passed
`SHA256SUMS`; large `.pt` payloads are checked independently against their
sidecars by `independent_verify.py`.

## Frozen identities and hashes

| Item | Exact value |
| --- | --- |
| Runtime | Python 3.12.3, torch 2.6.0+cu124, Transformers 5.23.0, tokenizers 0.23.3, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Model/tokenizer | `Qwen/Qwen2.5-0.5B-Instruct`, Qwen2ForCausalLM, revision `UNKNOWN`, 151,665-token ID space, unknown ID `null` |
| Python executable SHA-256 | `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118` |
| Live adapter / live test SHA-256 | `8598dbe01402efa1f0dfcba1e65ae9a6471cab39874e3d026d2d5161b2cc921b` / `d24459f44c4d01e7039a98bb7efa603f65eafc491559de852ef9d33afe05bb0d` |
| Existing trainer / trajectory SHA-256 | `2c0405d4a3a692aae31a4cc94d50e90e169f471038bac17dae94edeacf6725fd` / `d895acc57cc6a7a2ea5ab1cec7272e084667a493ad288bf8fd3541c75ac40305` |
| Dataset builder / token-export source SHA-256 | `c1ed69dfdaf53bb94caf28bca7fc6b9bc00439c34b20da16922fb9a59408b4f7` / `e5cb8ddb1a039d8ce79b79ef65162294c32353dd4a7fd648397be056ae6d826d` |
| Exact / resolved config SHA-256 | `b7c7e0d3503e1ed339cdc18617a7a919709c3357f4ea51992f4c8abd67479f2d` / `3c791a37d2b3cb190ecc2c86d8aeb72b4778892393817cb345347c50ee3b0c64` |
| Model config SHA-256 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| `model.safetensors` | 988,097,824 bytes; `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Available weight-set SHA-256 | `0d79ed576a7c2e115025787de0f0536c260affeaeb73f6c8022141290609eec7` |
| Complete model/tokenizer-directory file-list SHA-256 | `23ebb8d9f27833dadcc9047936b3f73a4453de77a3078fff40e74c07f2d68e83` |
| `tokenizer.json` / `tokenizer_config.json` SHA-256 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` / `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `merges.txt` / `vocab.json` SHA-256 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` / `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| Two exported trajectory JSONL SHA-256 values | `f56c0eb2d7cdcb758572cc0e97a3ae31d5556fb79f6238a70fe404273cf36685` / `026848c8db8df2526c48b7782100e80ffd616036034507381753881bf4f24c29` |
| Dataset manifest / frozen SFT train SHA-256 | `81e723b0cd6265c2798a18926fa784457a1fef5e266170bde12345ce343f1839` / `7653b5cd58bf7aa9f771dd723853b3a74c2cc4c9e3067528f448455eec7c778f` |
| Batch schedule / executed-order SHA-256 | `290fddbe09e6c6b086732f9e1fa9b66d50bb84c9a432fab7acc787b1efb751a9` / `af57077e5ca7a186b1905643d20d872b02edd2398caa85cc156d444636c68a49` |
| Legacy / export run-identity SHA-256 | `1341079ee598830d0cb5fc09a9be45b2429009bee812ed9202fbf214fddba121` / `3bf73e7c26875dd4c5f38a802ce14e77ec118e0e65cd0026c7c8b9f49dee4477` |
| Optimizer seed | `umminfer-stage-c-joint-sft-live-gate-v1`, derived uint31 `1139696663`; fixture seed `joint-sft-live-fixture-v1` |
| Acceptance JSON SHA-256 | `04b38f4f535be8deb3fcb7d9589dab91c3f344fc6d91acb2fbd8cdc09642182b` |
| Final SHA256SUMS SHA-256 | `573dc21e655d326c8bbda8bc94e43999cdb61a2cb031731531350b1671649394` |

The complete regular-file list, including cache metadata and every size/hash,
is in `token_export_preflight.json`; the table does not truncate that binding to
only tokenizer filenames.

## Exact legacy-versus-export arithmetic

Both paths used full-parameter BF16 AdamW, LR `1e-4`, zero weight decay,
gradient clipping `1.0`, one record per step, and two steps. All 290 gradient
tensors were non-zero and all post-step parameters were finite. The legacy and
export paths had zero mismatches for IDs, labels, masks/weights/annotations,
losses, canonical gradients, FP32 updates, model/optimizer hashes, and target
audits.

| Step / frozen target | Loss | Grad norm | Changed tensors / scalar parameters | Parameter delta L2 | Max abs delta | Step wall ms | CUDA peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: multilingual + NFC + emoji + spoken | 8.003159523010254 | 220.0 | 254 / 371,913,250 | 2.076159949852151 | 0.0001220703125 | 3,109.320 | 5,303,966,720 / 5,752,487,936 |
| 2: image+video + explicit `source=slot-0` | 4.829397201538086 | 86.5 | 254 / 321,032,603 | 1.514441026520064 | 0.0001220703125 | 2,754.206 | 5,618,786,304 / 6,889,144,320 |

| Step | Canonical gradient SHA-256 | FP32 parameter-update SHA-256 |
| --- | --- | --- |
| 1 | `e2f32dff8b3f016a86a6b1c26c75d795a8c8b239dd6eb76aab44070779505aaf` | `12c4a29e6ddb9e027f3ad5251e5997582249e362c227c864bfd0571ac11a0be8` |
| 2 | `1560d7672b5624d510e2fdf79e3ecc6553b0380276a2fa27067b24d73d5080dd` | `7596e5013f2ae8ad3819ec7116f19a803f0a779217fba8042fa78766d06dd584` |

Step 1 masked 11 prompt tokens and supervised 23 target tokens; step 2 masked 7
and supervised 33, including 10 route-tag, 17 visual-prompt-body, and two
closing-delimiter tokens. Their exact export evidence is:

| Step | Input IDs / labels / loss weights SHA-256 |
| --- | --- |
| 1 | `2e8a87cd3fb12e3456b3e399823fcf41a8ec6ab0884bb37e4ace6beb8f7e6d00` / `ff969d4e88f40c5323b744693d2c1a1685534f3ef4c52596193f6e645ddf6d54` / `aeb9f12d17325e55ba691bafaadbc1a6c5186959ff616a35e5542c2feb46a3dc` |
| 2 | `d1e6ad8e374e89f5ad4966d3fbbe5024828289c3c654b7587dc85d6cd16b8654` / `f81f9eeab49acf23d0ee38278959beaa18af5e42d7e8bb2023780550c173b381` / `86ed269e464e582eabc23cf9fbad2cf09f3001c35081c7cd1371fe1f504f7764` |

Step 1 annotations/raw offsets/export SHA-256:
`4a52e0681dd878f726664631a4c40d3ffb2895490d232e95e415bb466f09bd7a`,
`2790b32c7905da5ff6a63e87152be3460895cbf7a9888061c00eb2f669c95ea1`,
`19f3d9c4ebaca8087859867b1b12fbce0c65c8b1feff11d79f8fbb72f11d6467`.
Step 2 corresponding values:
`7e2588cb764383f8343f9e655d573a3fa5a434890be0ff8679f5908c5e2476e8`,
`e2c6c3337a8f02a86a5ff8abadf9ae68480887c160878ba76c44f89fbdae320e`,
`a8025cec3d2a0675a54bae63521a20cf1c63aa863847ec64f43cf183dc266dc9`.

Base, step-1, and final all-parameter SHA-256 values:
`4b4c678c452575cbc29e0b80a86698fd07fdb83bc7c1b9710e50cfca13d06b9a`,
`bfd40996bc3286df3c69739abe978652d5c65a6ce4cdeae048b035c3fab7f2b8`,
`0080d0d800a8899b66a442960a15b1c194e8e327b36a826c46ab8513792eaa7a`.
Final optimizer-state SHA-256:
`7cc069c9b78286a79722a61094ad656e4f26158e3f299d93a7f98f44a588ff60`.
Export preflight SHA-256:
`bccafe268d248e15c5babdad19de559dbfbd369d17f3a8bf3990761eb9b6d2f8`.
Total wall times for legacy, export, split-first, and resume were respectively
28,439.454 ms, 29,331.510 ms, 14,555.221 ms, and 14,463.272 ms, including
hashing and checkpoint writes.

## Resume, rejection, and cleanup

Export step-1 checkpoint SHA-256 was
`5fd9d3bff6ff941c423ee1cfc7718eb34ff1f4a994a319b2401167ec7822dc0b`;
step 2 was
`815025399d6285b119e677008183ee9bb9e02ea7d1f31b2bf7ce996d838d4759`.
The independent split-first step-1 checkpoint was byte-identical, and the
fresh-process resumed step-2 checkpoint was byte-identical to the uninterrupted
export run. Every exact resume comparison field matched. Legacy checkpoint
bytes intentionally differ because the data-path identity is checkpoint-bound;
their model, optimizer, gradient, and update values still match the export path.

Re-signed target-binding and partial-export processes each exited 2 before
Torch import or planner/preflight/manifest/checkpoint creation. After all
subprocesses exited, physical GPU 7 (NVIDIA H20, SM90) reported 3 MiB used, 0%
utilization, and no compute process. No task-owned process remained running.
No other GPU was used, and GPUs 2–6 plus the resident service were not
contacted or restarted.

An earlier default-Python preflight used torch `2.11.0+cu130`; the container
driver supports CUDA 12.2, so Torch reported that the driver was too old and
the adapter exited before CUDA initialization or optimizer output. That failed
attempt is preserved separately under
`/tmp/umminfer-task40-export-sft-fdc518b-evidence` and is not counted as an
optimizer result. A second preflight then passed the legacy steps but correctly
stopped export preflight when Transformers 5.23.0 imported Torch. The Torch-
isolation fix and complete four-process rerun above supersede that attempt;
neither earlier directory is presented as final acceptance evidence.

## Remaining boundaries

This gate covers one frozen Qwen2.5-0.5B tokenizer/model/runtime, two examples,
and bounded CPU fixtures plus two real optimizer steps per path. It does not
establish model quality, generalized improvement, production-model training,
DPO/GRPO outcomes, upstream revision verification, vision/media forward
execution, sampled online behavior IDs, online-rollout parity, or serving
behavior. Those remain `NOT_RUN`, `NOT_CLAIMED`, or `NOT_COVERED` as applicable.
Real optimizer and real-tokenizer evidence here is format v2; v1 equality and
strong validation are covered by deterministic fixture tests. No real-Qwen v1
optimizer result is claimed.
