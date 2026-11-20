# ummInfer 模型支持矩阵与选型方案

> 基线：基于公开 `main`（`1e77d585116b5e94f33710ce0fa42b3eec0c01e9`）整合语音路由、架构修复与文档后的代码树  
> 外部核查截止日期：`2026-11-20`（Hugging Face 官方 API：`https://huggingface.co/api/models/*`）  
> 说明：本文为模型支持状态与选型策略文档，表中未知项均显式标注 `UNKNOWN`，未实现的接入成本均标为**估计**，未经实现不得承诺。

---

## 1. 当前模型与后端支持矩阵

状态口径：**真实模型验证通过**（H20B/Mac 实测有录证）｜**仅代码接入**（已接编译或路由，缺真实验收）｜**仅契约或测试桩**（无真实识别/生成能力）｜**计划中**（仅路线图/规划，无实现）。  
注：原 SGLang `qwen3-asr` / `qwen3-omni` Python worker 已于历史提交 `7542ba0` 退役删除，不再计入当前实现。

| 模型 / 后端 | 实现位置 / Stage | 当前状态 | 权重 Revision | 硬件与显存 | 上下文 / 规模上限 | 模态 | 流式 | 并发 | 许可证（代码 / 权重） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-ASR 0.6B** | `native/umm_asr_native` + `umm-native-stage-asr` | 真实模型验证通过（H20B Case 0001 / L1-L2 / Benchmark Run D WER 0.0329） | `UNKNOWN`（`config.json` SHA-256 `76d3ae46…`） | CUDA SM90 (H20B)，~2.4 GiB BF16 | `kMaxSeq=1024`，16 kHz mono PCM16，最长 30 s | Audio → Text | Stage 非流式；Realtime 靠 200 ms 窗口重算预览 | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen3.8-27B**（默认 VLM） | `native/qwen38` + `umm-native-stage-qwen38` | 真实模型验证通过（H20B 文本/单图对齐 + `batch=1/2/4/8` 实测） | `UNKNOWN` | CUDA SM90 (H20B)，~55.8 GiB（batch=8 增 ~5.5 GiB） | `max_seq` 默认 8192（可配）；单张静态图，视觉上限 4096 pre-merge patch | Text + 1 Image → Text（无音频塔） | 支持（逐 token UTF-8 `vlm_delta`） | 支持 Continuous Batching（`--batch-size` 最大 8，图像 prefill 串行） | Apache-2.0 / Apache-2.0 |
| **Qwen3-Omni Thinker (30B-A3B)**（可选 VLM） | `native/omni` + `umm-native-stage-omni` | 真实模型验证通过（H20B Case 0001 / L2 PrefixCache / Benchmark Run D 54 例） | `UNKNOWN`（`config.json` SHA-256 `eab5093d…`） | CUDA SM90 (H20B)，~78.8 GiB 峰值（权重 57.87 GiB BF16） | `max_seq` 默认 2048（可配）；单张静态图（未加载 audio/talker/code2wav） | Text + 1 Image → Text | 代码已支持逐 token 流式 + 同会话 Prefix KV Cache | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen3-TTS-12Hz-0.6B CustomVoice** | `native/tts` + `umm-native-stage-tts` | 真实模型验证通过（H20B Case 0001 / L2 流式 20 轮 / ASR 回环） | `85e237c12c027371202489a0ec509ded67b5e4b5` | CUDA SM90 (H20B)，~4.9–7.2 GiB（BF16 Talker + FP32 Codec） | `max_frames` 默认 8192（12.5 Hz，~655 s）；固定 2 音色（`vivian`/`ryan`）、中/英 | Text → Audio (24 kHz mono PCM16/WAV) | 支持（首块 2 帧=160 ms，后续每 8 帧=640 ms；跨请求非 bit-exact） | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen-Image-2512**（默认图像） | `native/image` + `umm-native-stage-image --image-backend 2512` | 真实模型验证通过（H20B Case 0001 50 steps 228.8 s / 并发压测 8 steps 50.5 s） | `UNKNOWN` | CUDA SM90 (H20B)，稳态 ~53.0 GiB（首次加载 +~11 GiB） | 文本 `kMaxSeq=512`；Stage 仅放行 512×512 PNG（CLI 支持 1024×1024） | Text → Image (RGB PNG) | 否 | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen-Image-2.1** | `native/image2` + `umm-native-stage-image --image-backend 2.1` | CLI 真实模型验证通过（H20B 40 steps 69.8 s）；**Stage 仅代码接入**（未接入默认服务配置、缺端到端验证） | `UNKNOWN` | CUDA SM90 (H20B)，~14.4 GiB 峰值（14,763 MiB） | 512×512（32 倍数），无 True-CFG / 无 KV cache / 无条件图编辑 | Text → Image (RGBA PNG) | 否 | 1（`kSerial`） | Apache-2.0 / **Qwen Research（非商用）** |
| **MLX-LM（如 Qwen2.5-0.5B-Instruct-4bit）** | `python/umminfer_mlx_stage.py` | 真实模型验证通过（Apple M4 Pro 开发路径，非 H20B 生产路径） | `UNKNOWN`（权重 SHA-256 `ddffab9c…`） | Apple Silicon Metal（48 GiB 统一内存实测） | 取决于所加载 MLX 模型（`UNKNOWN` 统一上限） | Text → Text | 支持（SSE） | 1（串行，无跨请求 KV 复用） | MIT (MLX) / 视模型（示例 Apache-2.0） |
| **Qwen2.5-0.5B-Instruct (RL Phase 2)** | `python/umminfer_rl_{policy_serve,train}.py` | 真实模型验证通过（仅 H20B GPU 4 离线 RL 架构闭环，非推理服务） | `UNKNOWN` | CUDA (PyTorch) | 离线 GSM8K 实验脚本（`UNKNOWN` 上限） | Text → Text | SSE rollout | 离线实验 | Apache-2.0 / Apache-2.0 |
| **Upstream Proxy**（`/v1/images/*`, `/v1/videos*`） | `src/serve/upstream_proxy.cpp` | **仅代码接入**（仅测过配置/路由门禁与连接失败，无真实上游成功转发） | N/A | 无本地 GPU | 依赖外部上游（JSON 64 MiB / 二进制 512 MiB 上限） | Image / Video 透传 | 透传 | 受上游与连接限制 | Apache-2.0 / 视上游 |
| **macOS SFSpeechRecognizer** | `clients/ummmlx/Sources/ummmlx/Speech.swift` | **仅代码接入**（客户端本地听写，单元测试部分覆盖） | 系统内置 | macOS 客户端本机 | 系统接口上限（`UNKNOWN`） | Audio → Text | 支持（`AsyncThrowingStream`） | 客户端单路 | Apple SDK |
| **TinyCTC** | `python/umminfer_tiny_asr_stage.py` | **仅契约或测试桩**（随机初始化权重 seed `20261111`，无识别质量） | 本地生成 | Apple Silicon / CPU (MLX) | 最长 30 s，16 kHz | Audio → Text（桩） | 非流式 | 1 | Apache-2.0 |
| **Echo / Demo Stage & Contract Mocks** | `umm-infer serve --demo`、`tests/unified_contract_test.cpp` | **仅契约或测试桩** | N/A | CPU | 无模型 | 控制面契约 | 模拟事件 | 契约测试 | Apache-2.0 |
| **`/v1/audio/speech` 路由** | `src/serve/openai_api.cpp:787` (`handle_speech`) → `UnifiedEngine::tts_stage_name()` | **仅代码接入**（已集成至控制面，`tests/unified_contract_test.cpp` 契约测试通过；真实 TTS Stage / H20B 端到端验证 `NOT_RUN`；未注册 TTS stage 或固定上游模式仍返回 501） | N/A | 复用已注册 TTS Stage | 支持 `model`/`input`/`voice`（默认 `vivian`）/`response_format`（`wav`/`pcm`），受 `AdmissionGuard` 保护 | Text → Audio (`audio/wav` 或 `audio/pcm`) | 否（单次返回完整音频） | 受 `AdmissionGuard` 与 TTS Stage 并发限制 | Apache-2.0 |
| **Qwen3.5-Omni**（`plus` / `flash`） | `docs/roadmap.md` / `docs/validation.md` | **计划中**（仅规划官方 API，无本地权重，API 权限 `NOT_COVERED`） | `UNKNOWN` | 云端托管 API | `UNKNOWN` | Text+Image+Audio+Video → Text/Speech | 规划支持 | 视官方 API | 专有闭源 API |
| **MiniMaxAI/MiniMax-H3** | `docs/roadmap.md` / `docs/validation.md` | **计划中**（integration planning，无代码接入） | `UNKNOWN` | 规划 H20B BF16（磁盘 ~144.05 GB，非 VRAM） | 本地 H3-Base 仅 768p（2K 非全开源） | Text/Image/Video/Audio → 768p Video + 32 kHz Audio | 否 | `UNKNOWN` | MiniMax H3 Community（含地域/商用限制） |
| **MiniMaxAI/MiniMax-H3-Image** | `docs/roadmap.md` / `docs/validation.md` | **计划中**（integration planning，无代码接入） | `UNKNOWN` | 规划 H20B BF16（磁盘 ~135.38 GB，非 VRAM） | 最高 16 参考图编辑、16MP+ | Text/Image → Image | 否 | `UNKNOWN` | MiniMax H3 Image Community（含地域/商用限制） |

---

## 2. 候选模型外部核查（截至 2026-11-20）

以下数据均取自 Hugging Face 官方模型仓库元数据（`huggingface.co/api/models/*`，采样日期 `2026-11-20`）与官方文档。其中已独立通过 Hugging Face 官方 API 核实许可：`Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-TTS-12Hz-1.7B-Base`、`black-forest-labs/FLUX.2-klein-4B` 为 `apache-2.0`；`Qwen/Qwen-Image-2.1` 为 `qwen-research`（非商用）。**注意：30 天下载量（dl）与点赞数（likes）仅反映社区采用度信号，不作为模型质量证明。**

| 领域 | 候选模型（官方链接） | 发布 / 更新日期 | 许可证 | 核心能力与架构要点 | 采用度信号（2026-11-20 HF） |
| --- | --- | --- | --- | --- | --- |
| **VLM / Omni** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | 2026-08-05 / 2026-11-07 | `apache-2.0` | 27B 混合线性/全注意力 VLM（Gated Delta + Full Attn），长上下文、强多模态推理；已原生接入 | 6,454,419 dl · 20,726 likes |
| **VLM / Omni** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) / [`gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) / [`gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | 2026-03–04 / 2026-07–11 | `apache-2.0` | MoE (26B-A4B) 与 Dense (31B) 支持 256K 上下文、140+ 语种；`E4B` 支持原生音频+图像+文本端侧低显存 | 26B-A4B: 4,889,051 dl · 1,791 likes<br>31B: 6,065,842 dl · 4,430 likes<br>E4B: 4,326,816 dl · 1,881 likes |
| **VLM / Omni** | [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 2026-02-24 / 2026-11-20 | `apache-2.0` | 混合注意力 + 稀疏 MoE（3B 激活），低延迟高吞吐早融合多模态 | 1,345,891 dl · 1,560 likes |
| **VLM / Omni** | [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 2026-02-03 / 2026-11-09 | `apache-2.0` | 端到端全双工 Omni（视听看说并行流式） | 156,059 dl · 1,587 likes |
| **ASR** | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)（对照 [`0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)） | 2026-01-28 / 2026-01-30 | `apache-2.0` | AuT + Qwen3 解码器，50+ 语种/方言，与现有 `native/umm_asr_native` 同架构 | 1.7B: 1,587,140 dl · 1,267 likes<br>0.6B: 233,342 dl · 400 likes |
| **ASR** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | 2026-01-21 / 2026-03-11 | `apache-2.0` | 原生因果流式 ASR（<500 ms 延迟），多语种语音理解与转写 | 1,469,215 dl · 1,064 likes |
| **ASR** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 2025-08-04 / 2026-08-05 | `cc-by-4.0` | FastConformer + TDT 非自回归极速转写，高 RTFx 吞吐，支持多语种 | 531,382 dl · 1,289 likes |
| **ASR** | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 2024-10-01 / 2024-10-04 | `mit` | 传统 Encoder-Decoder 经典基线，生态广但非原生流式、架构偏旧 | 4,713,890 dl · 3,560 likes |
| **TTS** | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) / [`1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 2026-01-21 / 2026-01-29 | `apache-2.0` | 12.5 Hz 多码本超低延迟流式、10 语种、3 秒零样本克隆（`Base`），与 `native/tts` 同架构 | Base: 4,135,218 dl · 609 likes<br>CustomVoice: 1,102,710 dl · 2,246 likes |
| **TTS** | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) | 2024-12-26 / 2025-04-10 | `apache-2.0` | 82M 超轻量开源 TTS，CPU/Metal/低显存极速合成，适合本地与低延迟兜底 | 13,250,520 dl · 7,483 likes |
| **TTS** | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) | 2026-03-09 / 2026-03-11 | `fish-audio-research-license` | 双自回归、细粒度情绪/表达标签，但研究许可非商用且参数偏重 | 43,001 dl · 1,529 likes |
| **图像生成** | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 2026-01-14 / 2026-02-24 | `apache-2.0` | 4B 蒸馏流匹配 DiT，极低步数/低显存（~13 GiB），支持生成与编辑，商用友好 | 439,432 dl · 1,136 likes |
| **图像生成** | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) | 2026-09-14 / 2026-09-30 | `qwen-research`（非商用） | 7B 单流 DiT + Qwen3-VL，原生 RGBA 与 2K/多图编辑；CLI 已接入，但非商用许可 | 291,507 dl · 3,689 likes |
| **图像生成** | [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 2025-11-22 / 2026-11-19 | `flux-non-commercial-license` | 32B 高质量文生图+多参考图编辑，但非商用许可且显存重 | 287,646 dl · 2,603 likes |
| **图像生成** | [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | 2026-11-03 / 2026-11-16 | `minimax-h3-image-community-license-agreement` | 33.1B 6 步蒸馏、16MP+ 与 16 参考图编辑；135.4 GB 磁盘，含地域与商用限制 | 876,986 dl · 1,668 likes |
| **视频生成** | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) / [`Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | 2025-07-18 / 2025-08-07 | `apache-2.0` | 5B 轻量 720p 文/图生视频与 14B MoE 工业级版本，商用无限制，3D VAE 与 `native/image` 同源 | 5B: 21,233 dl · 1,025 likes<br>A14B: 5,139 dl · 633 likes |
| **视频生成** | [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) | 2026-01-03 / 2026-10-22 | `ltx-2-community-license-agreement` | 音视频同步 DiT（单次输出视频+对白/音效），但社区许可有营收上限 | 301,262 dl · 1,816 likes |
| **视频生成** | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 2026-07-28 / 2026-11-20 | `minimax-h3-community-license-agreement` | 768p 视频 + 32 kHz 立体声原生同步，采用度高；但 144 GB 磁盘、2K 非开源且含地域排除条款 | 5,752,082 dl · 7,192 likes |

---

## 3. P0 / P1 / P2 支持优先级决策

选型原则：**优先商用友好（Apache-2.0/MIT）、高采用度、可复用现有 Stage/UnifiedEngine 契约与 CUDA 算子内核的模型**，不为冷门型号或受限许可堆砌手写 C++/CUDA 引擎。

### P0（立即投入：商用友好、优先围绕现有 Stage/UnifiedEngine 主干）

> 注：下表未实现项目的接入成本均为**初步估计**，未经代码实现与真实基准验证不得作为工期或效果承诺。非商用许可模型（如 `Qwen-Image-2.1`，`qwen-research`）不列入 P0。

| 目标项 | 接入成本（相对现有 Stage/UnifiedEngine，均为估计） | 硬件成本（估计） | 是否替代旧模型 | 核心理由与验证边界 |
| --- | --- | --- | --- | --- |
| **巩固 `Qwen3.8-27B` 默认 VLM**（补齐 Prefix KV Cache、采样参数、小-m decode GEMM 优化） | **估计：低**（复用 `native/qwen38` 与 `umm-native-stage-qwen38`，零协议变更） | 单卡 H20B (~56–62 GiB) | 仅在 `qwen38` 的 PrefixCache、功能覆盖、54 例基准质量及性能**全部对齐并通过验证后**，才评估 `native/omni` 退役 | Apache-2.0、6.45M dl、已是默认 VLM 且具备 `batch=8` 能力；不得仅凭单一 PrefixCache 特性就认定可安全淘汰 `native/omni` |
| **升级 ASR 至 `Qwen3-ASR-1.7B`**（保持 0.6B 可选） | **估计：极低–低**（预期基于 `native/umm_asr_native` 参数化 `config.json` 维度复用现有 AuT+Qwen3 内核与 `asr_stage`，实际工作量待适配验证） | 单卡 ~4–5 GiB（可与轻量级 stage 共卡或独占） | 视真实 benchmark 结果决定是否替代 0.6B 为默认高精度 ASR（保留 0.6B 低显存档） | Apache-2.0（HF 官方 API 核实）、1.59M dl；目标是在真实多语种与复杂词 benchmark 上验证相对 0.6B 的 WER 改进（提升幅度为待验证目标，不预设必然或大幅提升） |
| **升级 TTS 至 `Qwen3-TTS-12Hz-1.7B`**（含 `Base` 零样本克隆，复用已合入的 `/v1/audio/speech` 路由） | **估计：低–中**（预期复用 `native/tts` 12.5 Hz Talker/RVQ 架构并新增 speaker encoder/conditioning；`/v1/audio/speech` 控制面路由已合入，仅需补齐真实 TTS Stage E2E 验证，1.7B 实际改动量待实现验证） | 单卡 ~8–10 GiB | 视真实 benchmark 结果决定是否替代 0.6B 为默认高质量 TTS（保留 0.6B 轻量档） | Apache-2.0（HF 官方 API 核实）、4.14M dl（Base）；目标是在真实中/英与零样本克隆 benchmark 上验证相对 0.6B 的音质与多音色能力改进（提升幅度为待验证目标） |

### P1（下一阶段：补齐关键能力短板与商用许可闭环）

| 目标项 | 接入成本（均为估计，未经实现不得承诺） | 硬件成本（估计） | 是否替代旧模型 | 核心理由与验证边界 |
| --- | --- | --- | --- | --- |
| **`FLUX.2-klein-4B`（商用图像 P0/P1 候选：生成/编辑）** | **估计：中**（需新增 `native/image_flux2` 或独立 stage 并适配 `umm-native-stage-image` 契约与 `UnifiedEngine` 图像 leg；未经实现不得承诺接入成本与最终速度） | 单卡 ~13–16 GiB | 验收通过后可作为商用 Apache-2.0 默认图像后端替代 `Qwen-Image-2512` | Apache-2.0（HF 官方 API 核实）+ 4B 蒸馏 + 原生编辑，旨在同时解决 `2512` 过慢（228.8 s）与 `Qwen-Image-2.1` 非商用的痛点；可按排期前移至商用图像 P0 评估 |
| **`Voxtral-Mini-4B-Realtime-2602`（原生流式 ASR）** | **估计：中**（新因果音频编码器 + Mistral 解码器，接入 `asr_stage` 真流式 `StreamMessage`） | 单卡 ~10–12 GiB | 补充（Realtime 场景替代 200 ms 窗口重算） | Apache-2.0、1.47M dl，目标解决全双工 L1/L2 伪流式预览的算力浪费与延迟抖动 |
| **`google/gemma-4-26B-A4B-it` 或 `gemma-4-E4B-it`** | **估计：中–高**（新增 Gemma4 原生 stage，含共享 KV/PLE/MoE；`E4B` 可直接承接原生音频输入） | `E4B` ~10 GiB；`26B-A4B` ~55 GiB | 补充（提供非 Qwen 生态与原生音频理解选项） | Apache-2.0、4.3M–4.9M dl、256K 上下文；避免单一模型家族绑定 |
| **`Kokoro-82M`（超轻量本地/兜底 TTS）** | **估计：低**（可先以独立轻量 stage（CPU/Metal/CUDA）接入 `tts` 契约） | <1 GiB | 补充（Mac 开发/超低延迟语音兜底） | Apache-2.0、13.25M dl，目标补齐 Mac 端及低算力场景轻量 TTS 空白 |
| **`Wan2.2-TI2V-5B`（商用级本地视频 Stage）** | **估计：高**（`UnifiedEngine` 放开 `kVideo` 异步作业生命周期 + 新增视频 stage，可参考 `native/image` Wan-VAE 经验） | 单卡 ~20–24 GiB (720p) | 新增视频能力（补齐 `/v1/videos` 本地后端） | Apache-2.0 商用无限制、单卡可跑，避免引入高风险许可 |

### P2（按需/仅外部 Stage 或上游代理，不手写 C++/CUDA 引擎）

- **`MiniMax-H3` / `MiniMax-H3-Image`**：采用度高（H3 5.75M dl），但 135–144 GB 磁盘体量、社区许可含地域排除（不含 US/EU/UK/KR）与商用条款，且 H3 2K 闭源。建议从“手写原生引擎”降级为**仅通过外部 Python/SGLang/ComfyUI Stage 或 `UpstreamProxy` 按需实验接入**。
- **`Lightricks/LTX-2`**：原生音视频同步能力强，但社区许可有营收上限，按需作外部 Stage。
- **`Qwen3.5-Omni`（`plus`/`flash`）**：纯闭源云端 API，仅通过 `UpstreamProxy` 透传，不投入本地引擎。
- **`MiniCPM-o-4.5`**：若未来将全双工架构从 `ASR→VLM→TTS` 级联改为端到端单一 Omni 模型时再立项。
- **明确不投入**：`Whisper-large-v3-turbo`（已有同级更优的 Qwen3-ASR 1.7B / Voxtral Realtime）、`FLUX.2-dev`（非商用且 32B 过重）、`Fish Speech S2-Pro`（非商用研究许可）。

---

## 4. 现有模型处置建议（本任务不删除代码）

| 处置级别 | 涉及模型 / 后端 | 行动准则 |
| --- | --- | --- |
| **保留并持续投入** | `Qwen3.8-27B`、`Qwen3-ASR 0.6B`（扩至 1.7B）、`Qwen3-TTS 0.6B`（扩至 1.7B） | 作为主干维护，补齐 P0 能力缺口与自动化回归 |
| **限定范围保留** | `Qwen-Image-2.1`（仅研究/非商用兼容路径）、`MLX-LM`（仅 Mac 低资源开发） | `Qwen-Image-2.1` 因 `qwen-research` 许可移出 P0，仅保留为研究/非商用兼容路径（不作为继续投入优先级，配置/文档强提示非商用）；`MLX-LM` 维持现状，不加生产并发特性 |
| **降级为兼容 / 停止新增投入** | `Qwen3-Omni Thinker` (`native/omni`)、`Qwen-Image-2512` (`native/image`)、`Qwen3.5-Omni` 规划、`MiniMax-H3*` 原生引擎规划 | 仅修阻断性缺陷，不再开发新特性或新内核 |
| **后续可评估删除适配** | `native/omni`（前置条件：`qwen38` 的 PrefixCache、功能覆盖、54 例基准质量及性能全部对齐并通过验证后才评估退役）、`native/image`（前置条件：`FLUX.2-klein-4B` 等商用级 Apache-2.0 图像后端通过 Stage 验收） | 满足全部前置条件后单独立项评估移除；**本任务不执行删除** |
| **保持测试桩定位** | `TinyCTC`、`--demo` Echo Stage、`UnifiedContract` Mock、`RL Phase 2` 离线脚本 | 严禁在文档或基准中当作真实模型质量证据 |

---

## 5. 最小真实验证矩阵（Minimal Real-Model Verification Matrix）

任何 P0/P1/P2 模型合入主干前，必须经过真实链路：`真实 CLI/HTTP → umm-infer → 真实 Stage → 真实权重 → 解码产物校验`（禁止假桩、禁止仅凭 HTTP 200 验收）。

| 模态 / 阶段 | 硬件基线 | 最小必测用例 | 必录元数据与门禁 |
| --- | --- | --- | --- |
| **VLM** (`Qwen3.8-27B` / `Gemma-4`) | H20B 单卡 BF16 | 1) 纯文本流式 SSE + 多轮 `session_history`；2) 单图 VQA；3) 并发 `batch=1/4/8` 输出一致性；4) 推理中 `/abort` 后 `in_flight=0` 且下条成功 | 记录 HF commit/revision、`config.json` 与权重 SHA-256、TTFT、tok/s、峰值 VRAM；对比 HF reference 或 Benchmark 54 例无显著退化 |
| **ASR** (`Qwen3-ASR-1.7B` / `Voxtral-Realtime`) | H20B 单卡 BF16 | 1) 中/英 16 kHz WAV 离线 `/v1/audio/transcriptions`；2) `/v1/realtime` 实时流式 + `server_vad` 打断恢复；3) 非法采样率/超长音频拒绝 | LibriSpeech test-clean 16 例 WER ≤ 0.035（1.7B 需优于 0.6B 基线）；Realtime 首字延迟 <1 s；记录音频与输出 SHA-256 |
| **TTS** (`Qwen3-TTS-1.7B` / `Kokoro-82M`) | H20B（Kokoro 另测 Mac CPU/Metal） | 1) `/v1/audio/speech` 非流式 WAV + 流式 PCM16；2) 中/英合成 + 1.7B 零样本音色克隆；3) 合成 WAV 经 ASR 回环转写校验；4) 中途 `/abort` | 校验 WAV 头/采样率/非静音比例/RMS/SHA-256；流式块拼接与终态逐样本一致（`stream_check identical=1`）；ASR 回环词错误率达标 |
| **Image** (`Qwen-Image-2.1` Stage / `FLUX.2-klein-4B`) | H20B 单卡 BF16 | 1) 经 `umm-native-stage-image` 的中/英各 1 条文生图（512×512 及目标分辨率）；2) 固定 seed 两次复现 SHA-256 一致；3) 去噪中途 `/abort` 退出 | 真实 PNG 解码校验宽高/色彩通道（RGB/RGBA）、非纯色检查、峰值 VRAM、总耗时与去噪步时、产物 SHA-256 |
| **Video** (`Wan2.2-TI2V-5B`) | H20B 单卡 BF16 | 1) `/v1/videos` 异步创建→轮询→拉取 MP4；2) 取消任务与显存回收 | 真实解码器校验帧数/分辨率/FPS/时长与文件 SHA-256；无进程与显存泄漏 |

---

## 6. 已知未知项（UNKNOWN）与风险

1. **本地权重精确 Revision 缺失**：除 `Qwen3-TTS-0.6B` 记录了 `85e237c12c027371202489a0ec509ded67b5e4b5` 外，H20B 本地 `asr` / `qwen38` / `omni` / `image` / `image2` 与 Mac `MLX` 目录均无 `.git`，revision 为 `UNKNOWN`（仅部分有 `config.json`/权重 SHA-256）。后续拉取新权重须强制记录 HF commit SHA。
2. **许可证合规风险**：树内已实现的 `Qwen-Image-2.1` 使用 `qwen-research` 非商用许可；路线图中的 `MiniMax-H3` / `H3-Image` 含地域排除与社区商用限制。对外发布商用版本前必须由 Apache-2.0 模型（如 `FLUX.2-klein-4B`、`Wan2.2`）兜底。
3. **测试套件与实测覆盖边界**：`ab51cf5` 曾移除多组历史单测与金标验证脚手架；本集成分支已恢复 `native/{omni,tts,image,image2}` CPU host 单元测试并统一 `README.md` 与 `docs/architecture.md` 口径，但 `/v1/audio/speech` 真实 TTS Stage 链路及所有 H20B 真实模型/GPU 验证仍为 `NOT_RUN`。
4. **GitHub 公开 API 限流**：本次核查中 GitHub REST API 返回 403 rate limit，故采用度信号统一以 Hugging Face 官方 API（`2026-11-20`）为准，GitHub stars 记为 `UNKNOWN`。
