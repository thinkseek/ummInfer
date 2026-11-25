# ummInfer 模型支持矩阵与选型方案

> 基线：基于公开 `main`（`1e77d585116b5e94f33710ce0fa42b3eec0c01e9`）整合语音路由、架构修复与文档后的代码树  
> 外部核查截止日期：`2026-11-20`（Hugging Face 官方 API：`https://huggingface.co/api/models/*`）  
> 说明：本文为模型支持状态与选型策略文档，表中未知项均显式标注 `UNKNOWN`，未实现的接入成本均标为**估计**，未经实现不得承诺。

---

## 1. 当前模型与后端支持矩阵

状态口径：**真实模型验证通过**（SM90/Mac 实测有录证）｜**仅代码接入**（已接编译或路由，缺真实验收）｜**仅契约或测试桩**（无真实识别/生成能力）｜**计划中**（仅路线图/规划，无实现）。
注：原 SGLang `qwen3-asr` / `qwen3-omni` Python worker 已于历史提交 `7542ba0` 退役删除，不再计入当前实现。

| 模型 / 后端 | 实现位置 / Stage | 当前状态 | 权重 Revision | 硬件与显存 | 上下文 / 规模上限 | 模态 | 流式 | 并发 | 许可证（代码 / 权重） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-ASR 0.6B / 1.7B** | `native/umm_asr_native` + `umm-native-stage-asr` | **0.6B 与 1.7B 均真实模型验证通过**（0.6B：SM90 Case 0001 / Benchmark Run D WER 0.0329；1.7B：Wave 5 魔搭直连 2-shard 权重 `config.json` SHA-256 `2e74a751…`，SM90 GPU 1 实测四音频 0.6B/1.7B 成对对比，中文 `301 ms` 逐字一致，英文 `1,453 ms`，`/v1/audio/transcriptions` 0.41–0.50 s 返回 HTTP 200） | 0.6B `UNKNOWN` (`76d3ae46…`)；1.7B ModelScope `master` (`2e74a751…`) | CUDA SM90：0.6B ~2.4 GiB BF16；1.7B 峰值 5.1–6.3 GiB BF16 | `kMaxSeq=1024`，16 kHz mono PCM16，最长 30 s | Audio → Text | Stage 非流式；Realtime 靠 200 ms 窗口重算预览 | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen3.8-27B**（默认 VLM） | `native/qwen38` + `umm-native-stage-qwen38` | 真实模型验证通过（SM90 文本/单图对齐 + `batch=1/2/4/8` + Wave 6 小-m GEMV 寄存器驻留优化：`m=1: 57.2→24.1 ms`、`m=4: 74.8→50.8 ms`、`m=8: 104.0→78.1 ms / 102.4 agg tok/s`；同条件 54 例真实多模态基准 `54/54` 逐字节一致，整体 p95 `6,075→3,244 ms`，GSM8K `87.5%`、VQA `95.83%`、ASR EN WER `1.88%`、Joint WER `0.0% / kw 97.22%`） | `UNKNOWN` | CUDA SM90 (NVIDIA Hopper)，~55.8–63.8 GiB（`batch=8`） | `max_seq` 默认 8192（可配）；单张静态图，视觉上限 4096 pre-merge patch | Text + 1 Image → Text（无音频塔） | 支持（逐 token UTF-8 `vlm_delta`） | 支持 Continuous Batching（`--batch-size` 最大 8，图像 prefill 串行） | Apache-2.0 / Apache-2.0 |
| **Qwen3-Omni Thinker (30B-A3B)**（可选 VLM） | `native/omni` + `umm-native-stage-omni` | 真实模型验证通过（SM90 Case 0001 / L2 PrefixCache / Benchmark Run D 54 例） | `UNKNOWN`（`config.json` SHA-256 `eab5093d…`） | CUDA SM90 (NVIDIA Hopper)，~78.8 GiB 峰值（权重 57.87 GiB BF16） | `max_seq` 默认 2048（可配）；单张静态图（未加载 audio/talker/code2wav） | Text + 1 Image → Text | 代码已支持逐 token 流式 + 同会话 Prefix KV Cache | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen3-TTS-12Hz-0.6B / 1.7B CustomVoice / 1.7B-Base** | `native/tts` + `umm-native-stage-tts` + `umm-tts-native` | **0.6B、1.7B CustomVoice 与 1.7B-Base 零样本克隆均真实模型验证通过**（Wave 5：1.7B CustomVoice SM90 GPU 7 首帧 `126.5–146.9 ms`、流式 `identical=1`、`/v1/audio/speech` 200 WAV/PCM；Wave 6：1.7B-Base 魔搭直连 `config.json` SHA-256 `b4f01752…`，原生 C++ FP32 ECAPA-TDNN 对齐 PyTorch BF16 `cosine=0.999996/0.999997`，SM90 GPU 1 中英零样本克隆 `4.27/5.98 s`、首包 `196/202 ms`、独立说话人相似度 `0.9931/0.9917`、ASR 回环逐字匹配；Wave 7：`umm-native-stage-tts` 与 HTTP `/v1/audio/speech` + `audio.ref_audio` 全链路打通，SM90 GPU 1 实测 `/v1/audio/speech` `ref_audio` 返回 200 WAV `311,084 B` / `6.48 s` / RMS `2859.9`，CustomVoice 0.6B/1.7B 零回归） | 0.6B `85e237c1…`；1.7B-CustomVoice `17a07f52…`；1.7B-Base `b4f01752…` | CUDA SM90：0.6B ~5.0–5.3 GiB；1.7B CustomVoice ~7.0–7.3 GiB；1.7B-Base ~7.4–7.6 GiB | `max_frames` 默认 8192（12.5 Hz，~655 s）；CustomVoice 预设 2 音色（`vivian`/`ryan`），Base 支持 CLI `--ref-audio`、Stage `--default-ref-audio` 与 HTTP `ref_audio` 24 kHz WAV 零样本克隆 | Text (+ Ref Audio) → Audio (24 kHz mono PCM16/WAV) | 支持（首块 2 帧=160 ms，后续每 8 帧=640 ms；流式与非流式 `identical=1`） | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **FLUX.2-klein-4B**（可选商用图像 Stage） | `python/umminfer_flux2_stage.py` + `service/native-service.json` (`optional_stages`) | **真实模型验证通过**（Wave 6 魔搭直连 `model_index.json` SHA-256 `51a76cb1…`，SM90 GPU 7 实测 512×512 4 步 `0.58–0.66 s`、1024×1024 4 步 `1.81 s / 20.6 GiB`、同 seed 字节复现及第 5/50 步 `/abort` 取消；Wave 7 打通公开 HTTP `image.steps` 与 `image.image` data URL 透传，SM90 GPU 7 实测公开 HTTP 文生图 `299,176 B` 与图生图编辑 `304,036 B` `l1_diff=68.51`） | ModelScope `master` (`51a76cb1…`) | CUDA SM90：512×512 峰值 ~16.1 GiB；1024×1024 峰值 ~20.6 GiB（磁盘 15 GiB） | 4 步蒸馏流匹配；支持文生图与公开 HTTP `image.image` data URL 编辑 | Text (+ Image) → Image (PNG) | 否（支持去噪步级 `/abort`） | 1（单请求串行） | Apache-2.0 / Apache-2.0 |
| **Qwen-Image-2512**（默认图像） | `native/image` + `umm-native-stage-image --image-backend 2512` | 真实模型验证通过（SM90 Case 0001 50 steps 228.8 s / 并发压测 8 steps 50.5 s） | `UNKNOWN` | CUDA SM90 (NVIDIA Hopper)，稳态 ~53.0 GiB（首次加载 +~11 GiB） | 文本 `kMaxSeq=512`；Stage 仅放行 512×512 PNG（CLI 支持 1024×1024） | Text → Image (RGB PNG) | 否 | 1（`kSerial`） | Apache-2.0 / Apache-2.0 |
| **Qwen-Image-2.1** | `native/image2` + `umm-native-stage-image --image-backend 2.1` | CLI 真实模型验证通过（SM90 40 steps 69.8 s，本地 `native/image2/IMAGE2_NATIVE.md` 记录较参考 diffusers 6.04 s 慢约 10×）；**Stage 仅代码接入**（未接入默认服务配置、缺端到端验证） | `UNKNOWN` | CUDA SM90 (NVIDIA Hopper)，~14.4 GiB 峰值（14,763 MiB） | 512×512（32 倍数），无 True-CFG / 无 KV cache / 无条件图编辑 | Text → Image (RGBA PNG) | 否 | 1（`kSerial`） | Apache-2.0 / **Qwen Research（非商用）** |
| **MLX-LM（如 Qwen2.5-0.5B-Instruct-4bit）** | `python/umminfer_mlx_stage.py` | 真实模型验证通过（Apple M4 Pro 开发路径，非 SM90 生产路径） | `UNKNOWN`（权重 SHA-256 `ddffab9c…`） | Apple Silicon Metal（48 GiB 统一内存实测） | 取决于所加载 MLX 模型（`UNKNOWN` 统一上限） | Text → Text | 支持（SSE） | 1（串行，无跨请求 KV 复用） | MIT (MLX) / 视模型（示例 Apache-2.0） |
| **Qwen2.5-0.5B-Instruct (RL Phase 2)** | `python/umminfer_rl_{policy_serve,train}.py` | 真实模型验证通过（仅 SM90 GPU 4 离线 RL 架构闭环，非推理服务） | `UNKNOWN` | CUDA (PyTorch) | 离线 GSM8K 实验脚本（`UNKNOWN` 上限） | Text → Text | SSE rollout | 离线实验 | Apache-2.0 / Apache-2.0 |
| **Upstream Proxy**（`/v1/images/*`, `/v1/videos*`） | `src/serve/upstream_proxy.cpp` | **仅代码接入**（兼容旁路代理；仅测过配置/路由门禁与连接失败，无真实上游成功转发） | N/A | 无本地 GPU | 依赖外部上游（JSON 64 MiB / 二进制 512 MiB 上限） | Image / Video 透传 | 透传 | 受上游与连接限制 | Apache-2.0 / 视上游 |
| **UnifiedEngine Video Stage**（`--video-entry` + `Wan-AI/Wan2.2-TI2V-5B-Diffusers` / `umm-native-stage-video`） | `python/umminfer_video_stage.py` + `native/video/*` + `native/stage_cli/video_stage.cpp` + `src/pipeline/unified.cpp` + `src/serve/openai_api.cpp` + `tests/video_live_e2e.py` + `tests/native_video_stage_contract_test.py` | **Diffusers 与纯原生 C++/CUDA HTTP 全链路均真实模型验证通过**（Wave 5：原生 M1 3D-VAE `PSNR=123.38 dB`、M2 UMT5 `cosine=0.99973–0.99998`、M3 DiT 单步 `cosine=0.999879 / PSNR=55.18 dB`；Wave 6：接入 Diffusers 精确对齐的二阶 **UniPC-BH2** 求解器 `<1e-5`，SM90 GPU 6 实测 `umm-infer serve`→`umm-native-stage-video --video-backend cuda` 480×320 13 帧 15 步 `7.08 s`、704×480 29 帧 20 步 `25.64 s`、1280×704 45 帧 10 步 `117.2 s` 与 CFG 5/20 中途 `/abort` 恢复；**外部授权 `/v1/videos` 上游实测仍为 `NOT_RUN`**） | `UNKNOWN`（ModelScope `master`，`model_index.json` SHA-256 `6a72faeb…`） | CUDA (SM90)：Diffusers ~23.5 GiB；纯原生 C++/CUDA 352×256 峰值 ~20.9 GiB，704×480 29 帧峰值 ~41.7–48.4 GiB | 单视频产物上限 48 MiB（单请求聚合上限 60 MiB），`video/mp4`，宽高需 32 像素对齐、帧数 `4k+1` | Text → Video（与 Text/Image/Audio 同请求同会话统一编排） | SSE 分片 `kArtifact` / 非流式 `artifacts` | 1（单请求串行；受 `UnifiedEngine` 准入保护） | Apache-2.0 / Apache-2.0 |
| **macOS SFSpeechRecognizer** | `clients/ummmlx/Sources/ummmlx/Speech.swift` | **仅代码接入**（客户端本地听写，单元测试部分覆盖） | 系统内置 | macOS 客户端本机 | 系统接口上限（`UNKNOWN`） | Audio → Text | 支持（`AsyncThrowingStream`） | 客户端单路 | Apple SDK |
| **TinyCTC** | `python/umminfer_tiny_asr_stage.py` | **仅契约或测试桩**（随机初始化权重 seed `20261111`，无识别质量） | 本地生成 | Apple Silicon / CPU (MLX) | 最长 30 s，16 kHz | Audio → Text（桩） | 非流式 | 1 | Apache-2.0 |
| **Echo / Demo Stage & Contract Mocks** | `umm-infer serve --demo`、`tests/unified_contract_test.cpp` | **仅契约或测试桩** | N/A | CPU | 无模型 | 控制面契约 | 模拟事件 | 契约测试 | Apache-2.0 |
| **`/v1/audio/speech` 路由** | `src/serve/openai_api.cpp` (`handle_speech`) → `UnifiedEngine::tts_stage_name()` | **真实模型验证通过**（离线契约测试通过；Linux SM90 空闲 GPU 7 独立栈实测 `umm-native-stage-tts` 返回 200 `audio/wav` `207,404 B` SHA-256 `f9f7ae18…` 24 kHz mono PCM16 非静音 RMS `1760.7` 与 `audio/pcm` `207,360 B`，回收后显存回落 `3 MiB`；未注册 TTS stage 或固定上游模式返回 501） | N/A | 复用已注册 TTS Stage | 支持 `model`/`input`/`voice`（默认 `vivian`）/`response_format`（`wav`/`pcm`），受 `AdmissionGuard` 保护 | Text → Audio (`audio/wav` 或 `audio/pcm`) | 否（单次返回完整音频） | 受 `AdmissionGuard` 与 TTS Stage 并发限制 | Apache-2.0 |
| **Qwen3.5-Omni**（`plus` / `flash`） | `docs/roadmap.md` / `docs/validation.md` | **计划中**（仅规划官方 API，无本地权重，API 权限 `NOT_COVERED`） | `UNKNOWN` | 云端托管 API | `UNKNOWN` | Text+Image+Audio+Video → Text/Speech | 规划支持 | 视官方 API | 专有闭源 API |
| **MiniMaxAI/MiniMax-H3** | `docs/roadmap.md` / `docs/validation.md` | **计划中**（integration planning，无代码接入） | `UNKNOWN` | 规划 SM90 BF16（磁盘 ~144.05 GB，非 VRAM） | 本地 H3-Base 仅 768p（2K 非全开源） | Text/Image/Video/Audio → 768p Video + 32 kHz Audio | 否 | `UNKNOWN` | MiniMax H3 Community（含地域/商用限制） |
| **MiniMaxAI/MiniMax-H3-Image** | `docs/roadmap.md` / `docs/validation.md` | **计划中**（integration planning，无代码接入） | `UNKNOWN` | 规划 SM90 BF16（磁盘 ~135.38 GB，非 VRAM） | 最高 16 参考图编辑、16MP+ | Text/Image → Image | 否 | `UNKNOWN` | MiniMax H3 Image Community（含地域/商用限制） |

---

## 2. 候选模型外部核查（截至 2026-11-20）

以下数据均取自 Hugging Face 官方模型仓库元数据（`huggingface.co/api/models/*`，采样日期 `2026-11-20`）与官方模型卡。其中已独立通过 Hugging Face 官方 API 核实许可：`Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-TTS-12Hz-1.7B-Base`、`black-forest-labs/FLUX.2-klein-4B` 为 `apache-2.0`；`Qwen/Qwen-Image-2.1` 为 `qwen-research`（非商用）。下文引用的官方模型卡指标均属外部公开披露，**不代表本机新实测**。**注意：30 天下载量（dl）与点赞数（likes）仅作为生态活跃度与维护风险的辅助信号，不作为模型质量或产品收益论据。**

| 领域 | 候选模型（官方链接） | 发布 / 更新日期 | 许可证 | 核心能力与架构要点 | 生态辅助信号（2026-11-20 HF） |
| --- | --- | --- | --- | --- | --- |
| **VLM / Omni** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | 2026-08-05 / 2026-11-07 | `apache-2.0` | 27B 混合线性/全注意力 VLM（Gated Delta + Full Attn），长上下文、多模态推理；已原生接入 | 6,454,419 dl · 20,726 likes |
| **VLM / Omni** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) / [`gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) / [`gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | 2026-03–04 / 2026-07–11 | `apache-2.0` | `26B-A4B` 为 MoE、256K 上下文、**无原生音频**；`31B` 为 Dense、256K；`E4B` 为 **Dense**、**128K** 上下文、支持**原生音频**+图像+文本（BF16 权重文件 `15,992,595,884 bytes = 14.894 GiB`） | 26B-A4B: 4,889,051 dl · 1,791 likes<br>31B: 6,065,842 dl · 4,430 likes<br>E4B: 4,326,816 dl · 1,881 likes |
| **VLM / Omni** | [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 2026-02-24 / 2026-11-20 | `apache-2.0` | 混合注意力 + 稀疏 MoE（3B 激活），低延迟高吞吐早融合多模态 | 1,345,891 dl · 1,560 likes |
| **VLM / Omni** | [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 2026-02-03 / 2026-11-09 | `apache-2.0` | 端到端全双工 Omni（视听看说并行流式） | 156,059 dl · 1,587 likes |
| **ASR** | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)（对照 [`0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)） | 2026-01-28 / 2026-01-30 | `apache-2.0` | AuT + Qwen3 解码器，50+ 语种/方言；BF16 权重文件 `4.375 GiB`；官方卡 LibriSpeech clean WER `2.11→1.63`，但 TED-LIUM `3.85→4.50`（非全基准单调提升） | 1.7B: 1,587,140 dl · 1,267 likes<br>0.6B: 233,342 dl · 400 likes |
| **ASR** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | 2026-01-21 / 2026-03-11 | `apache-2.0` | 原生因果流式 ASR（<500 ms 延迟），多语种语音理解与转写 | 1,469,215 dl · 1,064 likes |
| **ASR** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 2025-08-04 / 2026-08-05 | `cc-by-4.0` | **FastConformer + TDT (Token-and-Duration Transducer)**，联合预测 token 与 duration 以减少解码步数，高 RTFx 吞吐，支持多语种 | 531,382 dl · 1,289 likes |
| **ASR** | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 2024-10-01 / 2024-10-04 | `mit` | 传统 Encoder-Decoder 经典基线，生态广但非原生流式、架构偏旧 | 4,713,890 dl · 3,560 likes |
| **TTS** | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) / [`1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 2026-01-21 / 2026-01-29 | `apache-2.0` | 12.5 Hz 多码本低延迟流式、10 语种；`Base` 需 speaker encoder 与参考音频（通常含参考文本）克隆。官方卡非全面提质：CustomVoice 西语 WER `1.154→1.330`；Base 英语 speaker similarity `0.829→0.775` | Base: 4,135,218 dl · 609 likes<br>CustomVoice: 1,102,710 dl · 2,246 likes |
| **TTS** | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) | 2024-12-26 / 2025-04-10 | `apache-2.0` | 82M 超轻量开源 TTS，CPU/Metal/低显存极速合成，适合本地与低延迟兜底 | 13,250,520 dl · 7,483 likes |
| **TTS** | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) | 2026-03-09 / 2026-03-11 | `fish-audio-research-license` | 双自回归、细粒度情绪/表达标签，但研究许可非商用且参数偏重 | 43,001 dl · 1,529 likes |
| **图像生成** | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 2026-01-14 / 2026-02-24 | `apache-2.0` | 4B 蒸馏流匹配 DiT，低步数蒸馏、支持生成与编辑，商用友好（峰值 VRAM 待原型实测） | 439,432 dl · 1,136 likes |
| **图像生成** | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) | 2026-09-14 / 2026-09-30 | `qwen-research`（非商用） | 7B 单流 DiT + Qwen3-VL，原生 RGBA 与 2K/多图编辑；CLI 已接入，但非商用许可 | 291,507 dl · 3,689 likes |
| **图像生成** | [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 2025-11-22 / 2026-11-19 | `flux-non-commercial-license` | 32B 高质量文生图+多参考图编辑，但非商用许可且显存重 | 287,646 dl · 2,603 likes |
| **图像生成** | [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | 2026-11-03 / 2026-11-16 | `minimax-h3-image-community-license-agreement` | 33.1B 6 步蒸馏、16MP+ 与 16 参考图编辑；135.4 GB 磁盘，含地域与商用限制 | 876,986 dl · 1,668 likes |
| **视频生成** | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) / [`Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | 2025-07-18 / 2025-08-07 | `apache-2.0` | 5B 轻量 720p 文/图生视频与 14B MoE 工业级版本，商用无限制，3D VAE 与 `native/image` 同源 | 5B: 21,233 dl · 1,025 likes<br>A14B: 5,139 dl · 633 likes |
| **视频生成** | [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) | 2026-01-03 / 2026-10-22 | `ltx-2-community-license-agreement` | 音视频同步 DiT（单次输出视频+对白/音效），但社区许可有营收上限 | 301,262 dl · 1,816 likes |
| **视频生成** | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 2026-07-28 / 2026-11-20 | `minimax-h3-community-license-agreement` | 768p 视频 + 32 kHz 立体声原生同步；144 GB 磁盘、2K 非开源且含地域排除与社区商用限制 | 5,752,082 dl · 7,192 likes |

---

## 3. P0 / P1 / P2 支持优先级决策

选型原则：**核心排序依据为目标质量、真实业务需求、延迟、显存（区分权重文件体积与峰值 VRAM）、剩余工程量与商用许可（优先 Apache-2.0/MIT）**；下载量仅作生态/维护风险辅助信号，不为受限许可或未经质量门禁的模型投入主干 C++/CUDA 引擎。

### P0（立即投入：商用友好、优先围绕现有 Stage/UnifiedEngine 主干）

> 注：下表未实现项目的接入成本均为**初步估计或待原型评估**，未经代码实现与真实基准验证不得作为工期或效果承诺。非商用许可模型（如 `Qwen-Image-2.1`，`qwen-research`）降为 P2 研究维护，不列入 P0。

| 目标项 | 接入成本（相对现有 Stage/UnifiedEngine） | 硬件与显存（区分权重体积与峰值 VRAM） | 是否替代旧模型 | 核心理由与验证边界 |
| --- | --- | --- | --- | --- |
| **巩固 `Qwen3.8-27B` 默认 VLM**（Prefix KV Cache、采样/有符号 seed、小-m decode GEMV 优化与 54 例多模态基准） | **核心项已完成并真机实测通过**（Wave 2–6：多轮轮次边界 Prefix Cache 提速 2.23×–4.21×、完整 `temperature`/`top_p`/有符号 `seed` 透传、小-m 寄存器驻留 GEMV 实测 `m=1: 57.2→24.1 ms`、`m=4: 74.8→50.8 ms`、`m=8: 104.0→78.1 ms / 102.4 agg tok/s`，并完成 54 例真实多模态基准 `54/54` 逐字节一致与 GSM8K `87.5%` / VQA `95.83%`） | 单卡 SM90（稳态 ~55.8–63.8 GiB，`batch=8`） | 仅在补齐扩展图像集非劣效与配置迁移/回滚演练后才评估 `native/omni` 退役（见第 4 节） | Apache-2.0、已是默认 VLM 且完成 `batch=8` 破百吞吐与 54 例多模态质量/延迟验证 |
| **评估并接入 `Qwen3-ASR-1.7B`**（保持 0.6B 可选） | **已完成并真权重实测通过**（`native/umm_asr_native` 支持 `config.json` 动态维度、2-shard `model.safetensors.index.json` 与权重 shape 校验；SM90 GPU 1 魔搭真权重四音频 0.6B/1.7B 成对对比与 `/v1/audio/transcriptions` 实测通过） | BF16 2-shard 权重 `4.375 GiB`；SM90 GPU 1 实测峰值显存 **`5,103–6,276 MiB`**（vs 0.6B `2,387 MiB`） | 保留 0.6B 低显存档，1.7B 可按需切换为高精度档 | Apache-2.0；真权重实测通过，中文逐字一致、英文补全前导语气词，可在业务音频集上持续成对评估 |
| **TTS 拆项 A：`/v1/audio/speech` 路由验收** | **真实模型验证通过**（0.6B 与 1.7B 均已完成 SM90 GPU 7 `/v1/audio/speech` 端到端 200 WAV/PCM 与 ASR 回环验收） | 复用已注册 TTS Stage（无新增显存） | 补齐 OpenAI Speech 公开接口 | 0.6B 与 1.7B 真实 Stage 端到端均已验收通过 |
| **TTS 拆项 B：`Qwen3-TTS-12Hz-1.7B-CustomVoice` 升级** | **已完成并真权重实测通过**（新增 `native/tts/model_config.hpp` 与 `small_to_mtp_projection`，修复 `kEps=0` 与 `past_hidden` 未 norm；SM90 GPU 7 实测 0.6B 与 1.7B 中英合成、流式 `identical=1`、`/v1/audio/speech` 与 ASR 回环 100% 逐字通过） | 权重 `4.3 GiB`；SM90 GPU 7 实测峰值 VRAM **`7,053–7,333 MiB`**（vs 0.6B `5,069–5,309 MiB`） | 保留 0.6B 轻量档，1.7B CustomVoice 可按需直接切换 | Apache-2.0；中英 ASR 回环 100% 逐字一致，建议按目标语种补充人工听评 |
| **TTS 拆项 C：`Qwen3-TTS-12Hz-1.7B-Base` 零样本克隆链路** | **CLI、Stage 与公开 HTTP 全链路均已完成并真权重实测通过**（Wave 6 新增原生 C++ FP32 ECAPA-TDNN `SpeakerEncoder`、`TtsEngine::extract_speaker_embedding`/`synthesize_clone` 与 `--ref-audio`；Wave 7 接通 `umm-native-stage-tts` `--default-ref-audio` 与 HTTP `/v1/audio/speech` + `audio.ref_audio` 透传，SM90 GPU 1 实测 `/v1/audio/speech` 返回 200 WAV `311,084 B` / `6.48 s` / RMS `2859.9`，CustomVoice 0.6B/1.7B 零回归） | 权重 `4.3 GiB`；SM90 GPU 1 实测峰值 VRAM **`7,367–7,597 MiB`**（CPU FP32 speaker encoder 耗时 `357–553 ms`） | 补充（解除“仅 2 固定音色”限制，CustomVoice 0.6B/1.7B 保持零回归） | Apache-2.0；已完成数值对齐、ASR 可懂度、独立说话人相似度与 HTTP 端到端验收，后续可按目标语种补充人工听评 |

### P1（下一阶段：商用图像优先评估与关键能力短板补齐）

| 目标项 | 接入成本（均为估计，未经实现不得承诺） | 硬件与显存（区分权重与峰值） | 是否替代旧模型 | 核心理由与验证边界 |
| --- | --- | --- | --- | --- |
| **`FLUX.2-klein-4B`（商用图像 Stage：文生图与编辑）** | **Stage 与公开 HTTP 全链路均已完成并真权重实测通过**（Wave 6 新增 `python/umminfer_flux2_stage.py`、`tests/flux2_stage_contract_test.py` 与 `umminfer_service.py` `flux2` 启动支持；Wave 7 补齐公开 HTTP `image.steps`/`image.image` 透传与 Web UI，SM90 GPU 7 实测公开 HTTP 文生图 `299,176 B`、图生图编辑 `304,036 B` `l1_diff=68.51` 与 1024×1024 `1.81 s / 20.6 GiB`） | 权重磁盘 `15 GiB`；SM90 GPU 7 实测峰值 VRAM **`16.1 GiB`（512×512） / `20.6 GiB`（1024×1024）** | 已可作为默认商用图像 Stage 上线替换 `Qwen-Image-2512`（保留 `native/image` 作回退） | Apache-2.0 + 4 步蒸馏 + 原生图生图编辑，1024×1024 实测 `1.81 s / 20.6 GiB`（vs `2512` 50 步 `228.8 s / 53 GiB`） |
| **`Voxtral-Mini-4B-Realtime-2602`（原生流式 ASR）** | **估计：中**（新因果音频编码器 + Mistral 解码器，接入 `asr_stage` 真流式 `StreamMessage`） | 权重与峰值 VRAM **待原型实测** | 补充（Realtime 场景替代 200 ms 窗口重算） | Apache-2.0，目标解决全双工 L1/L2 伪流式预览的算力浪费与延迟抖动 |
| **`google/gemma-4-26B-A4B-it` 或 `gemma-4-E4B-it`** | **估计：中–高**（新增 Gemma4 原生 stage；`26B-A4B` 为 MoE/256K/无原生音频；`E4B` 为 Dense/128K/支持原生音频） | `E4B` BF16 权重 **`14.894 GiB`**（`15,992,595,884 bytes`，峰值 VRAM 另计；若采用量化须指明格式）；`26B-A4B` 峰值 **待原型实测** | 补充（提供非 Qwen 生态与 `E4B` 原生音频理解选项） | Apache-2.0，按实际需求提供多生态备选与原生音频理解路径 |
| **`Kokoro-82M`（超轻量本地/兜底 TTS）** | **估计：低**（可先以独立轻量 stage（CPU/Metal/CUDA）接入 `tts` 契约） | <1 GiB（82M 参数） | 补充（Mac 开发/超低延迟语音兜底） | Apache-2.0，目标补齐 Mac 端及低算力场景轻量 TTS 空白 |
| **`Wan2.2-TI2V-5B`（商用级本地视频 Stage）** | **Python Diffusers、纯原生 C++/CUDA（含流式 3D-VAE）与公开 HTTP 全链路均已实测通过并上线 `43301`**（`native/video/*` 已完成 M1 3D-VAE、M2 UMT5-XXL `cosine=0.99973–0.99998`、M3 30 层 BF16 DiT 单步 `cosine=0.999879 / PSNR=55.18 dB`、Wave 6 二阶 `UniPC-BH2` `<1e-5`、Wave 7 HTTP `video.{num_frames,steps,guidance_scale}` 透传与 Wave 8 逐潜帧精确因果流式 3D-VAE：384×256 13 帧对齐 Diffusers `PSNR=128.43 dB`、1280×704 77 帧峰值仅 `14.21 GiB` 无 OOM；SM90 `43301` GPU 6 已上线实测 480×320 13 帧 `7.74 s`） | SM90 单卡实测峰值：Diffusers **23,463 MiB**；纯原生 C++/CUDA 流式 3D-VAE 384×256 13 帧 **4.28 GiB**，**1280×704 77 帧 14.21 GiB**（稳态服务驻留 **20.4 GiB**） | 已作为默认原生零 Python 商用视频 Stage 上线（复用 `UnifiedEngine` 统一控制面） | Apache-2.0 商用无限制、单卡可跑；长视频 1280×704 77 帧 OOM 已彻底解决，外部授权 `/v1/videos` 上游实测仍为 `NOT_RUN` |

### P2（研究维护 / 按需外部 Stage 或上游代理，不投入主干 C++/CUDA 引擎）

- **`Qwen-Image-2.1`（降为研究维护 / P2）**：因 `qwen-research` 非商用许可移出 P0/P1，仅保留为研究/非商用兼容路径。树内已有 `native/image2` 代码只降低剩余工程成本，**不能证明产品优先级**；其 SM90 实测 40 steps 69.8 s 与 `Qwen-Image-2512` 50 steps 228.8 s 仅属**不同工作点的墙钟耗时对比**，不能称为同质量 3.3×；且本地 `native/image2/IMAGE2_NATIVE.md` 记录原生路径仍比参考 diffusers（6.04 s）慢约 10×，KV Cache 等优化收益均**待消融实测**，不预设逼近参考。
- **`MiniMax-H3` / `MiniMax-H3-Image`**：135–144 GB 磁盘体量、社区许可含地域排除（不含 US/EU/UK/KR）与商用条款，且 H3 2K 闭源。降级为**仅通过外部 Python/SGLang/ComfyUI Stage 或 `UpstreamProxy` 按需实验接入**。
- **`Lightricks/LTX-2`**：社区许可有营收上限，按需作外部 Stage。
- **`Qwen3.5-Omni`（`plus`/`flash`）**：纯闭源云端 API，仅通过 `UpstreamProxy` 透传，不投入本地引擎。
- **`MiniCPM-o-4.5`**：若未来将全双工架构从 `ASR→VLM→TTS` 级联改为端到端单一 Omni 模型时再立项。
- **明确不投入**：`Whisper-large-v3-turbo`（非原生流式且架构偏旧）、`FLUX.2-dev`（非商用且 32B 过重）、`Fish Speech S2-Pro`（非商用研究许可）。

---

## 4. 现有模型处置建议（本任务不删除代码）

| 处置级别 | 涉及模型 / 后端 | 行动准则 |
| --- | --- | --- |
| **保留并持续投入** | `Qwen3.8-27B`、`Qwen3-ASR 0.6B`（评估扩至 1.7B）、`Qwen3-TTS 0.6B`（分项评估 1.7B CustomVoice / Base 及 `/v1/audio/speech` 真实验收） | 作为主干维护，按真实门禁补齐能力缺口与自动化回归 |
| **研究维护 / 限定范围保留（P2）** | `Qwen-Image-2.1`（仅研究/非商用兼容路径）、`MLX-LM`（仅 Mac 低资源开发） | `Qwen-Image-2.1` 降为 P2 研究维护（`qwen-research` 非商用，已有代码只降低剩余成本不证明产品优先级，配置/文档强提示非商用）；`MLX-LM` 维持现状，不加生产并发特性 |
| **降级为兼容 / 停止新增投入** | `Qwen3-Omni Thinker` (`native/omni`)、`Qwen-Image-2512` (`native/image`)、`Qwen3.5-Omni` 规划、`MiniMax-H3*` 原生引擎规划 | 仅修阻断性缺陷，不再开发新特性或新内核 |
| **后续可评估删除适配（退役就绪度检查）** | `native/omni`（当前 `43301` GPU 3 已由 `Qwen3.8-27B` 承载且 54 例基准/多轮 PrefixCache/连续批处理均已验，`omni` 仅保留在 `optional_stages` 作回滚备份；最终删除前剩余项：扩展图像集非劣效对照与一次 `qwen38↔omni` 热切回滚演练）、`native/image`（当前 `43301` GPU 4 已由 Apache-2.0 `FLUX.2-klein-4B` 承载且支持 1024×1024 文生图与 HTTP `image.image` 编辑，`image-2512` 仅保留在 `optional_stages` 作纯 C++ 无 Python 兜底；最终删除前剩余项：业务提示词人工视觉质量盲测确认） | 满足剩余门禁后再单独立项移除；**当前保留作回滚备份，不删除代码** |
| **保持测试桩定位** | `TinyCTC`、`--demo` Echo Stage、`UnifiedContract` Mock、`RL Phase 2` 离线脚本 | 严禁在文档或基准中当作真实模型质量证据 |

---

## 5. 最小真实验证矩阵（Minimal Real-Model Verification Matrix）

任何 P0/P1/P2 模型合入主干前，必须经过真实链路：`真实 CLI/HTTP → umm-infer → 真实 Stage → 真实权重 → 解码产物校验`（禁止假桩、禁止仅凭 HTTP 200 验收）。

| 模态 / 阶段 | 硬件基线 | 最小必测用例 | 必录元数据与门禁 |
| --- | --- | --- | --- |
| **VLM** (`Qwen3.8-27B` / `Gemma-4`) | SM90 单卡 BF16（量化须明示格式） | 1) 纯文本流式 SSE + 多轮 `session_history` 与 PrefixCache 命中/失效；2) 单图 VQA 与大图边界；3) 并发 `batch=1/4/8` 输出一致性；4) 推理中 `/abort` 后 `in_flight=0` 且下条成功 | 记录 HF commit/revision、`config.json` 与权重 SHA-256、权重体积与峰值 VRAM、TTFT、tok/s；评估退役 `native/omni` 前须完成同 revision、同 54 例及扩展图像集非劣效对照、配置迁移与回滚演练 |
| **ASR** (`Qwen3-ASR-1.7B` / `Voxtral-Realtime`) | SM90 单卡 BF16 | 1) 目标语种/方言/噪声/专名成对回归（与 0.6B 同集对照）；2) `/v1/audio/transcriptions` 与 `/v1/realtime` + `server_vad` 打断恢复；3) 非法采样率/超长音频拒绝 | 区分权重体积与峰值 VRAM；按目标语种/方言/噪声/专名分组报告 WER/CER（避免只看单集而掩盖如 TED-LIUM 类退化）；Realtime 首字延迟 <1 s；记录音频与输出 SHA-256 |
| **TTS** (`Qwen3-TTS-1.7B CustomVoice / Base` / `Kokoro-82M`) | SM90（Kokoro 另测 Mac CPU/Metal） | 1) `/v1/audio/speech` 非流式 WAV/PCM 与统一链路流式 PCM16；2) 分语种合成（CustomVoice）及参考音频+参考文本零样本克隆（Base）；3) 合成 WAV 经 ASR 回环转写 + speaker similarity + 人工听评；4) 中途 `/abort` | 校验 WAV 头/采样率/非静音比例/RMS/SHA-256 与 `stream_check identical=1`；必录**分语种 WER**、**speaker similarity** 与**人工听评**（明确：**ASR 回环仅证明内容可懂度，不能证明自然度与音色保真度**） |
| **Image** (`FLUX.2-klein-4B` 商用候选 / `Qwen-Image-2.1` P2 研究) | SM90 单卡 BF16 | 1) 经 `umm-native-stage-image` 的中/英多提示词文生图与图像编辑（含目标分辨率）；2) 固定 seed 两次复现 SHA-256 一致；3) 去噪中途 `/abort` 退出；4) 优化前后消融对比 | 真实 PNG 解码校验宽高/通道/非纯色/SHA-256、权重体积与峰值 VRAM；默认替代 `2512` 前必过 **prompt alignment**、**人工质量**、**编辑能力** 与 **质量相当工作点延迟** 门禁（禁止将不同步数/工作点的墙钟比当作同质量加速比） |
| **Video** (`Wan2.2-TI2V-5B` 或首个外部视频 Stage) | SM90 单卡 BF16（或真实外部上游） | 1) 经 `UnifiedEngine`（`POST /v1/chat/completions` `modalities=["text","video"]` + `--video-entry`）及 `POST /v1/visual/render`（T2V/I2V）生成 MP4 并校验 SSE 分片重组与 SHA-256；2) 同会话取消 `/abort`（含跨会话 403）与显存/资源回收；3) 兼容校验 `/v1/videos*` 代理（若启用） | 真实解码器校验帧数/分辨率/FPS/时长与文件 SHA-256；无进程与显存泄漏；**本机 SM90 原生 C++/CUDA Wan2.2 T2V/I2V 已实测上线（卡 6）；外部授权 `/v1/videos` 上游代理仍为 `NOT_RUN`** |

---

## 6. 已知未知项（UNKNOWN）与风险

1. **本地权重精确 Revision 与已归档 SHA-256 指纹（2026-11-25 核验）**：SM90 `<model-root>` 目录均无 `.git`（Git revision 为 `UNKNOWN`），Wave 8 已对全部 7 套本地权重归档关键配置 SHA-256 与分片数以便可复现审计：
   - `qwen3.8-27b`：`51.77 GiB`（18 shards），`config.json` `191e0af232104ed8…`
   - `Qwen3-ASR-1.7B`：`4.38 GiB`（2 shards），`config.json` `2e74a751548b8ad7…`，`configuration.json` `c57f6a580d63f746…`
   - `FLUX.2-klein-4B`：`14.89 GiB`（4 shards），`model_index.json` `51a76cb1cf3ed374…`，`transformer/config.json` `09733c74a3da6d17…`，`vae/config.json` `0d6dfb69ae95a5e2…`
   - `Qwen3-TTS-12Hz-1.7B-CustomVoice`：`4.21 GiB`（2 shards），`config.json` `17a07f527a1c25ea…`
   - `Qwen3-TTS-12Hz-1.7B-Base`：`4.23 GiB`（2 shards），`config.json` `b4f01752d15a488a…`
   - `Wan2.2-TI2V-5B-Diffusers`：`31.85 GiB`（9 shards），`model_index.json` `6a72faeb564b0e89…`，`transformer/config.json` `dc00d9866e72cf77…`，`vae/config.json` `d996c340fe9a7df5…`
   - `qwen-image-2.1`：`40.68 GiB`（7 shards，非商用），`model_index.json` `cf1ecd104ea09085…`，`transformer/config.json` `56ae3281c4e6c2d1…`，`vae/config.json` `9785d527b278cb8b…`
   后续拉取新权重须继续记录源仓库 commit SHA 与配置摘要。
2. **许可证合规风险**：树内已实现的 `Qwen-Image-2.1` 使用 `qwen-research` 非商用许可；路线图中的 `MiniMax-H3` / `H3-Image` 含地域排除与社区商用限制。对外发布商用版本前必须由 Apache-2.0 模型（如 `FLUX.2-klein-4B`、`Wan2.2`）兜底。
3. **测试套件与实测覆盖边界**：`ab51cf5` 曾移除多组历史单测与金标验证脚手架；当前主干已覆盖 `native/{omni,tts,image,image2,video}` CPU host 单元测试、`UnifiedEngine` 文本/图像/音频/视频四模态统一契约测试、macOS Swift 客户端测试与 SM90 五卡原生真模型实测（ASR/VLM/FLUX.2/TTS/Wan2.2 T2V&I2V，详见 [feature-map.md](feature-map.md) 与 [native-service.md](native-service.md)）；外部授权 `/v1/videos` 上游代理与真模型负载下的 CUDA IPC 零拷贝仍为 `NOT_RUN` / `NOT_COVERED`。
4. **GitHub 公开 API 限流**：本次核查中 GitHub REST API 返回 403 rate limit，故生态/维护风险辅助信号统一以 Hugging Face 官方 API（`2026-11-20`）为准，GitHub stars 记为 `UNKNOWN`。
