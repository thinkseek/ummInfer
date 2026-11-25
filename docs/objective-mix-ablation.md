# Stage C Ablation #5: Post-Training Objective-Mix & Quality-vs-Latency Pareto Frontier

Realizes Row 5 of [vision.md](vision.md#32-完整消融实验体系ablation-matrix) (`后训练目标配比: 仅 SFT vs SFT + 一致性奖励 / + 模态路由奖励 / + 延迟惩罚` → `综合质量分 vs P95 端到端时延 Pareto 前沿`) as a standalone, single-variable, 4-arm ablation harness built on `CompositeUmmReward` (`python/umminfer_rl_reward.py`).

## 1. 4-Arm Single-Variable Experimental Design

All 4 arms share the exact same base model configuration (`Qwen/Qwen2.5-0.5B-Instruct`, `bfloat16`, `eager` attention, `gradient_checkpointing: false`, `trust_remote_code: false`), RNG seed (`umminfer-stage-c-objective-mix-ablation-v1`), rollout groups, `max_seq_len: 256`, `learning_rate: 5e-5`, `weight_decay: 0.0`, `max_grad_norm: 1.0`, `steps: 2`, `clip_eps: 0.2`, `kl_coef: 0.04`, and `sft_weight: 1.0`.

**Only `objective_mix` changes across arms:**

| Arm (`arm_id`) | `sft_weight` | `grpo_weight` | `routing_f1` | `prose_visual_balance` | `barge_in_stability` | `latency_penalty` | Behavioral Target |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `sft_only` | `1.0` | `0.0` | — (`null`) | — (`null`) | — (`null`) | — (`null`) | Baseline unranked behavior cloning across the rollout pool (no RL advantage contrast). |
| `sft_plus_consistency` | `1.0` | `1.0` | `0.0` | `0.6` | `0.4` | `0.0` | Rewards prose/visual count balance and clean barge-in cancellation; blind to modality-routing F1 and latency. |
| `sft_plus_routing` | `1.0` | `1.0` | `1.0` | `0.0` | `0.0` | `0.0` | Rewards modality-set routing F1 only; blind to redundant duplicate slots, failed barge-in legs, and tail latency. |
| `sft_plus_full_latency_pareto` | `1.0` | `1.0` | `0.4` | `0.3` | `0.3` | `0.2` | Full multi-objective `CompositeUmmReward` jointly optimizing routing, consistency, barge-in stability, and latency budget. |

### Single-Variable Invariance Gate

- `verify_single_variable_invariance()` hashes every fixed field excluding `arm_id`, `description`, and `objective_mix` (`model`, `optimization`, `grpo`, `sft_weight`, `evaluation`, and `rollout_groups_sha256`) into `per_arm_fixed_fields_sha256`.
- Execution fails closed (nonzero exit) if:
  - `per_arm_fixed_fields_sha256` differs across any two arms,
  - `per_arm_objective_mix_sha256` fails to produce 4 distinct hashes,
  - any arm ID or required candidate profile is missing, or
  - any rollout record fails `CompositeUmmReward` schema-v2 validation.

## 2. Deterministic Schema-v2 Rollout Dataset

Embedded in [`configs/training/objective_mix_ablation.json`](../configs/training/objective_mix_ablation.json): 3 prompt groups × 3 candidate rollouts = 9 schema-v2-compatible trajectories covering:

1. `group_black_hole_accretion` (`expected_visual_modalities: ["image"]`, normal completion):
   - `optimal_balanced_fast`: 1 image slot, balanced prose, `terminal_ms = 2800 ms`.
   - `overtriggered_redundant_slow`: 2 duplicate image slots (`routing_f1 = 1.0`, degraded balance), `terminal_ms = 24500 ms`.
   - `undertriggered_text_fast`: text-only (`0` slots), `terminal_ms = 950 ms`.
2. `group_barge_in_storyboard_edit` (`expected_visual_modalities: ["image"]`, full-duplex barge-in interruption):
   - `optimal_balanced_fast`: 1 image slot, clean cancellation (`barge_in_stability = 1.0`), `terminal_ms = 2200 ms`, `barge_in_ms = 85 ms`.
   - `overtriggered_redundant_slow`: 3 slots (`image`, `video`, `image`) with failed in-flight video leg (`barge_in_stability = 0.0`), `terminal_ms = 27800 ms`, `barge_in_ms = 240 ms`.
   - `undertriggered_text_fast`: text-only with clean cancellation (`barge_in_stability = 1.0`), `terminal_ms = 820 ms`, `barge_in_ms = 55 ms`.
3. `group_orbit_mechanics_motion` (`expected_visual_modalities: ["image", "video"]`, static + motion explanation):
   - `optimal_balanced_fast`: 1 image + 1 video slot, `terminal_ms = 4800 ms`.
   - `overtriggered_redundant_slow`: 4 redundant slots (`image`, `video`, `video`, `image`, `routing_f1 = 1.0`), `terminal_ms = 28600 ms`.
   - `undertriggered_text_fast`: 1 image slot only (omits video), `terminal_ms = 1650 ms`.

Because the 3 GRPO objective mixes weigh `routing_f1`, `prose_visual_balance`, `barge_in_stability`, and `latency_penalty` differently, each arm induces a strictly distinct per-group advantage matrix and ranking across the 9 candidates.

## 3. Policy-Weighted Metrics & 2D Pareto Frontier

For each prompt group $g \in \{1, \dots, G\}$ and candidate $i \in \{1, \dots, K\}$:

- **Sequence mean log-probability**: target-token mean causal log-probability $\bar{\ell}_{g,i}$ under the updated policy $\pi_\theta$.
- **Normalized group policy distribution**:
  $$p_{\text{policy}}(g, i) = \frac{\exp(\bar{\ell}_{g,i})}{\sum_{j=1}^{K} \exp(\bar{\ell}_{g,j})}$$
- **Policy-weighted composite quality (`composite_quality_score`, maximize ↑)**:
  $$Q(\pi_\theta) = \frac{1}{G} \sum_{g=1}^{G} \sum_{i=1}^{K} p_{\text{policy}}(g, i) \cdot q_{\text{ref}}(g, i)$$
  where $q_{\text{ref}}(g, i)$ is evaluated via `CompositeUmmReward` with `routing_f1=0.4`, `prose_visual_balance=0.3`, `barge_in_stability=0.3`, `latency_penalty=0.0` (kept latency-free so quality and latency remain orthogonal Pareto axes).
- **Policy-weighted expected P95 terminal latency (`expected_p95_terminal_ms`, minimize ↓)**:
  Computed over all $G \times K$ pooled candidates with weights $w_{g,i} = \frac{1}{G} p_{\text{policy}}(g, i)$ using continuous piecewise-linear inverse-CDF interpolation at $q = 0.95$ (`weighted_quantile_linear`), ensuring smooth sensitivity to upper-tail probability shifts.
- **2D Pareto dominance (`compute_pareto_frontier`)**:
  Arm $B$ dominates Arm $A$ iff:
  $$Q_B \ge Q_A \;\land\; L_B \le L_A \;\land\; (Q_B > Q_A \;\lor\; L_B < L_A)$$
  Each arm reports `is_pareto_optimal` and its exact sorted `dominated_by` list.

### Executed Training Objective (`--live-gpu`)

All 4 arms start from **bit-identical** BF16 initial weights (`initial_parameter_sha256` verified identical before each arm) and identical RNG seed, then run `steps` full-parameter BF16 `AdamW` updates minimizing:

$$\mathcal{L}_{\text{arm}} = w_{\text{sft}} \cdot \mathcal{L}_{\text{SFT}} + w_{\text{grpo}} \cdot \frac{1}{N}\sum_{g,i} \left[ -\min\!\left(r_{g,i} A_{g,i},\; \operatorname{clip}(r_{g,i}, 1-\epsilon, 1+\epsilon) A_{g,i}\right) + \beta_{\text{KL}}\left(e^{\Delta_{g,i}} - \Delta_{g,i} - 1\right) \right]$$

where $r_{g,i} = \exp(\operatorname{clamp}(\bar{\ell}_{g,i} - \bar{\ell}^{\text{init}}_{g,i}, -20, 20))$ and $\Delta_{g,i} = \operatorname{clamp}(\bar{\ell}^{\text{init}}_{g,i} - \bar{\ell}_{g,i}, -20, 20)$.

## 4. Commands

### Deterministic CPU-Only Contract Replay (`--offline-contract`)

Zero `torch`/`transformers` imports; runs cleanly on CPU-only macOS/Linux:

```bash
python3 -I python/umminfer_objective_mix_ablation.py \
  --config configs/training/objective_mix_ablation.json \
  --offline-contract \
  --out /tmp/umm-obj-mix-contract
```

Outputs written to `--out`:
- `report.json` (invariance verification, per-arm summaries, 2D Pareto frontier, pairwise divergence)
- `arms/<arm_id>/report.json` (4 per-arm detailed step & group evaluation reports)
- `SHA256SUMS` (sorted relative paths; byte-identical across repeated runs)

### Real SM90 GPU Multi-Arm Execution (`--live-gpu`)

Verifies exclusive idle GPU (`memory_used_mib <= 100`, `0` compute processes) before loading, hashes local `.safetensors` weights (`weights_sha256`, `file_count`), resets bit-identical initial BF16 weights before each of the 4 arms, verifies finite losses/gradients and positive L2 parameter deltas, frees CUDA memory between arms, and verifies post-run cleanup:

```bash
python3 -I python/umminfer_objective_mix_ablation.py \
  --config configs/training/objective_mix_ablation.json \
  --live-gpu \
  --model-dir /path/to/local/Qwen2.5-0.5B-Instruct \
  --gpu 7 \
  --out /tmp/umm-obj-mix-live
```

### Unit & Contract Tests

```bash
python3 -m unittest -v tests/objective_mix_ablation_test.py
```

## 5. Evidence & Claim Boundaries

| Mode | `evidence_class` | `real_training_evidence` | Permitted Claims | Explicitly `NOT_COVERED` / Forbidden Claims |
| --- | --- | --- | --- | --- |
| `--offline-contract` / `--replay-only` | `CONTRACT_REPLAY` | `NOT_RUN` | Single-variable invariance, `CompositeUmmReward` advantage differentiation, exact 2D Pareto math, byte-identical output determinism & `SHA256SUMS`. | Real GPU gradients, real parameter deltas, or real model policy shifts. |
| `--live-gpu` | `LIVE_GPU_OBJECTIVE_MIX` | `REAL_SM90_OPTIMIZER_GATE` | Real SM90 BF16 AdamW execution from bit-identical initial weights, finite loss/grad/param deltas, multi-arm weight & log-prob divergence, and offline-rollout Pareto frontier on the local base model. | Unseen human-rated generalization, online rollout policy quality, and production model training (`NOT_COVERED`). |

## 6. SM90 Live GPU Execution Record (`LIVE_GPU_OBJECTIVE_MIX / REAL_SM90_OPTIMIZER_GATE`) — 2026-11-26

Executed on isolated SM90 **GPU 7** (`NVIDIA H20`, compute capability `(9, 0)`, PyTorch `2.6.0+cu124`, Transformers `5.6.0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`) with local `Qwen2.5-0.5B-Instruct` (`model.safetensors` `988,097,824 B`, `weights_sha256=0d79ed57...`, 290 BF16 tensors; resident 5-GPU stack on GPUs 2–6 untouched):

- **Isolation & Integrity**:
  - Pre-check GPU 7: `3.0 MiB / 0` compute processes (`exclusive_idle_verified=true`).
  - Post-cleanup inside process: `torch_memory_allocated_bytes=0`, `torch_memory_reserved_bytes=0`, `other_compute_process_count=0`; post-exit GPU 7 returned to `3 MiB`.
  - Shared initial parameter SHA-256 across all 4 arms: `4699d9ff0d18fb8f553f4b0f21de132913af80e5bb4d66763b8576b3f84ba059`.
  - `single_variable_verified=true` (`fixed_fields_sha256=58b62bd9c559a42a...`), `all_arms_distinct_final_parameters=true`, `all_pairs_distinct_delta_logp=true` (`min_pairwise_delta_logp_l2=0.5051`), `total_wall_ms=26,922.2 ms`, `5/5 SHA256SUMS OK`.

| Arm (`objective_mix`) | Composite Quality Score ↑ (`Δ`) | Expected P95 Terminal (ms) ↓ (`Δ`) | Expected Terminal (ms) | Optimal-Fast `Δp_policy` | Redundant-Slow `Δp_policy` | Param `ΔL2` | Final Param SHA-256 | Pareto Frontier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sft_only` | `0.6768` (`-0.0505`) | `28,240.1` (`+1.8`) | `8,942.7` | `-0.0500` | `-0.1455` | `1.3909` | `16155d54...` | Dominated |
| `sft_plus_routing` | `0.8754` (`+0.1480`) | `28,361.0` (`+122.7`) | `13,332.4` | `+0.1966` | `+0.0085` | `1.3702` | `9cdcdbaf...` | Dominated |
| `sft_plus_consistency` | **`0.9365`** (`+0.2092`) | `27,915.0` (`-323.3`) | `7,966.6` | `+0.4128` | `-0.2129` | `1.3466` | `05f39033...` | **Pareto Optimal** |
| `sft_plus_full_latency_pareto` | **`0.9362`** (`+0.2089`) | **`27,813.0`** (`-425.4`) | **`7,943.4`** | **`+0.4150`** | **`-0.2141`** | `1.3463` | `3a006e9c...` | **Pareto Optimal** |

- **Observed Trade-off**: `sft_plus_consistency` and `sft_plus_full_latency_pareto` form a 2-arm non-dominated Pareto frontier strictly dominating both `sft_only` and `sft_plus_routing` (`sft_plus_full_latency_pareto` matches peak quality within `0.0003` while reducing expected P95 terminal latency by an additional `102.1 ms` vs `sft_plus_consistency` and `548.1 ms` vs `sft_plus_routing`).
- **Remaining Claim Boundaries**: `unseen_human_rated_generalization`, `online_rollout_policy_quality`, and `production_model_training` remain explicitly `NOT_COVERED`.
