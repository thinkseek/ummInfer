# ummInfer 权威架构与验证说明

> 本文档是 ummInfer 的单一权威架构文档，涵盖真实源码核对后的系统拓扑、端到端调用链（同步/流式、取消、错误、会话与资源生命周期）、本机轻量验证矩阵以及已记录的技术债边界。
> 代码锚点索引见 [docs/feature-map.md](feature-map.md)；模型支持矩阵与选型见 [docs/model-support.md](model-support.md)；阶段门控与规划见 [docs/roadmap.md](roadmap.md)。

---

## 1. 整体架构图

```mermaid
flowchart TB
  subgraph Clients["公开入口 / 客户端"]
    CLI["umm-infer CLI<br/>chat / speech / transcribe / realtime / voice / image / video<br/>(src/main.cpp, src/client/cli.cpp, voice.cpp)"]
    CPP["C++ Client Library<br/>OpenAIClient / RealtimeSession<br/>(src/client/openai.cpp, realtime.cpp, http.cpp, websocket.cpp)"]
    SWIFT["macOS App (Swift)<br/>ummmlx (clients/ummmlx/)"]
    WEB["Web Client<br/>web/index.html (--web-root)"]
  end

  subgraph Server["控制面进程: umm-infer serve (src/main.cpp)"]
    HTTP["OpenAIApi (cpp-httplib, 线程池随 max_in_flight 自适应)<br/>/health /ready /metrics /v1/models<br/>/v1/chat/completions /v1/audio/transcriptions /v1/audio/speech /abort<br/>(src/serve/openai_api.cpp, openai_chat.cpp, openai_audio.cpp, openai_helpers.cpp)"]
    PROXY["UpstreamProxy (固定上游兼容代理)<br/>Qwen chat / SGLang image & video<br/>熔断器 + AdmissionGuard<br/>(src/serve/upstream_proxy.cpp)"]
    WS["RealtimeApi (libwebsockets)<br/>WS /v1/realtime + EnergyVad + O(1) 活跃请求索引<br/>(src/serve/realtime_api.cpp)"]
    UE["UnifiedEngine (统一四模态编排)<br/>准入限流 max_in_flight · 契约校验 · SessionHistoryStore<br/>ASR→VLM→Image∥Audio(TTS)∥Video · 统一 LegState 生命周期<br/>(src/pipeline/unified.cpp, unified_schema.cpp, session_store.cpp)"]
    COORD["Coordinator<br/>stage 注册/心跳幂等重注册/就绪门控 · pending_submits_<br/>事件泵 · ZMQ 断连监控 stage_down<br/>(src/pipeline/pipeline.cpp)"]
  end

  subgraph Transport["进程间控制面与数据面"]
    ZMQ["ZeroMQ + MessagePack (src/comm/comm.cpp, src/proto/)<br/>PUSH→PULL (submit / data_ready / data_ack / admin / shutdown)<br/>PUSH→PULL (stage_ready / stream / complete / admin_result)<br/>PUB→SUB + PUSH (abort 双通道广播)"]
    RELAY["CommRouter + Relay (src/comm/comm.cpp, src/relay/relay.cpp)<br/>POSIX SHM (CreditAllocator + shm_open/mmap)<br/>可选 CUDA IPC (仅 C++ Stage 间, 传 host-serialized bytes)"]
  end

  subgraph NativeStages["线上服务后端: C++ Stage Host / 原生 CUDA Stage"]
    STAGE["C++ Stage + SimpleScheduler<br/>(src/pipeline/pipeline.cpp, src/scheduling/simple_scheduler.cpp)<br/>inbox/outbox · abort 墓碑 · kSerial / kThreadSafe<br/>thread-local emit_stage_stream / bound_request_params"]
    BIN_ASR["umm-native-stage-asr (native/umm_asr_native)"]
    BIN_VLM["umm-native-stage-qwen38 / omni (native/qwen38, native/omni)"]
    BIN_IMG["umm-native-stage-image (native/image 2512 / native/image2 2.1)"]
    BIN_TTS["umm-native-stage-tts (native/tts)"]
    BIN_VID["umm-native-stage-video (native/video Wan2.2 CUDA) / 可替换 Python Stage (flux2 · video)"]
    DEMO["serve --demo / umm-infer stage (echo_compute)"]
  end

  subgraph PythonWorkers["本机 / 离线 Python Workers (非 SM90 线上路径)"]
    MLX["MLXStage / TinyASRStage<br/>(python/umminfer_mlx_stage.py, umminfer_tiny_asr_stage.py)<br/>直连 ZMQ+Msgpack · ThreadPoolExecutor<br/>不经过 C++ Stage / SimpleScheduler / SHM Relay"]
    SUP["umminfer_service.py (纯 stdlib 进程守护)<br/>umminfer_rl_*.py (独立 RL 训练/评测工具)"]
  end

  CLI --> CPP
  CPP --> HTTP
  CPP --> WS
  SWIFT --> HTTP
  WEB --> HTTP

  HTTP -->|"fixed upstream"| PROXY
  HTTP -->|"unified mode"| UE
  HTTP -->|"legacy / transcription"| COORD
  WS -->|"ASR preview / final"| COORD
  WS -->|"audio_output=true"| UE
  UE --> COORD

  COORD <--> ZMQ
  ZMQ <--> STAGE
  STAGE <-->|"多段 C++ Stage 间"| RELAY
  STAGE --- BIN_ASR
  STAGE --- BIN_VLM
  STAGE --- BIN_IMG
  STAGE --- BIN_TTS
  STAGE --- BIN_VID
  STAGE --- DEMO

  ZMQ <-->|"直连控制帧 (无 SHM)"| MLX
```

---

## 2. 关键调用链与生命周期

| 维度 | 真实代码路径与行为 | 关键源码定位 |
|---|---|---|
| **同步请求（非流式）** | `POST /v1/chat/completions` (`stream=false`) → 校验通过后 `submit_chat()` 调 `UnifiedEngine::submit()`（无音频直投 `vlm_stage`，有音频先投 `asr_stage`）或 `Coordinator::submit()`（分配 `req-N`）→ `while (!settled && !req.is_connection_closed()) wait_chat(id, 0.1)` 轮询等待 → `release_chat(id)` 清理记录并返回 JSON；带可选多模态输出时由 `handle_nonstream_mixed()` 收集统一产物事件（单片直编码、多片预分配拼装）并返回含 `artifacts`/`legs` 的 JSON。`/v1/audio/transcriptions` 绕过 `UnifiedEngine` 直接走 `Coordinator::submit(..., entry_stage)`；`/v1/audio/speech` 复用 `UnifiedEngine::tts_stage_name()` 注册的 TTS stage，经 `AdmissionGuard` 准入后同步返回 `audio/wav` 或 `audio/pcm`（未注册 TTS stage 或固定上游模式返回 501）。 | `src/serve/openai_chat.cpp:46-203`, `454-518`; `src/serve/openai_audio.cpp:18-385`; `src/pipeline/unified.cpp:183-260` |
| **流式请求（SSE / WS）** | **HTTP SSE**：`OpenAIApi` 按 `max_in_flight` 配置 `cpp-httplib` 线程池并分配 `http-N` 注册 `SseSink` → `set_chunked_content_provider` 消费 `SseSink::pop()`。`Coordinator::run()` 单线程泵收到 `StreamMessage`/`CompleteMessage` → `UnifiedEngine::CallbackGate` → `handle_stream`/`handle_complete` 入 `dispatch_queue_` 串行化派发（避免重入死锁）→ `on_event` 推送到 `SseSink`。文本/图像/音频/视频共用统一 `UnifiedModality` 与 `LegState`（`pending`/`running`/`completed`/`failed`/`aborted`，可通过 `legs_snapshot()` 查询），图像（`image/png`/`jpeg`）、音频（`audio/wav`/`pcm`）与视频（`video/mp4`，单产物上限 48 MiB、单请求聚合上限 60 MiB）产物统一经 `slice_artifact_events` 按 `2 MiB` 分片并在 HTTP provider 线程做 base64 与末片 SHA-256。**流式 TTS**：`audio.stream=true` 且无 image/video 输出时，`handle_stream` 在 VLM delta 到达时调用 `drain_sentences_locked`（带异常保护，提交失败转为 `kLegError` 而不中断 VLM 或事件泵）按句切分（首句 8 codepoints，后续 48 codepoints，最多 32 个子 leg `id:tts:N`）边生成边推 `audio/pcm` `kArtifact`。**WebSocket**：`RealtimeApi` 在 `libwebsockets` 线程每 10ms `poll_sessions()` 消费 `partial_queue_`（经 `active_request_sessions_` `O(1)` 查找归属会话）与 `unified_queue_`，VAD 触发 `speech_started`/`speech_stopped`，`audio_output=true` 时把转写结果交给 `UnifiedEngine`。 | `src/serve/openai_api.cpp:29-56`, `334-424`; `src/serve/openai_chat.cpp:520-695`; `src/pipeline/unified.cpp:459-610`, `1007-1110`; `src/pipeline/unified_schema.cpp:358-420`; `src/serve/realtime_api.cpp:238-270`, `861-1150` |
| **取消与中止** | **触发源**：客户端断连（`req.is_connection_closed()` / `!out_sink.is_writable()` / WS `LWS_CALLBACK_CLOSED`）、超时（`timeout`）、barge-in（WS `on_speech_started`）、显式 `POST /abort {request_id, session_id}`（校验归属，不匹配返回 403）、进程退出（`Coordinator::stop()` 标记 `shutdown`）。**传播**：`UnifiedEngine::cancel_for_session` 在 `mu_` 内置 `cancel_requested=true`，收集所有已提交子 leg（`vlm_id` / `:image` / `:video` / `:tts*`），锁外逐个调 `Coordinator::abort` → `CoordinatorControlPlane::broadcast_abort` 通过 **PUSH（可靠有序）+ PUB（低延迟）双通道**广播 `AbortMessage` → `Stage` 的 `abort_thread_` 和 `run()` 写入 `SimpleScheduler::aborted_` 墓碑（保留 10 分钟，防 abort-before-submit 乱序）并清理 `pending_ops_` → 计算函数在安全边界检查 `AbortCheck` 抛 `RequestAborted`。未扇出的 `pending` 输出 leg（含 `video`）被 `cancel_requested` 阻断并转为 `aborted`；终态 `legs[]` 对已取消的 `image`、`tts`、`video` leg 均如实报告 `state: "aborted"`。 | `src/serve/openai_api.cpp:230-279`; `src/serve/openai_chat.cpp:548-556`; `src/pipeline/unified.cpp:282-320`, `1110-1200`; `src/pipeline/pipeline.cpp:215-265`, `1000-1035`; `src/comm/comm.cpp:535-560`; `src/scheduling/simple_scheduler.cpp:82-121`, `228-244` |
| **错误传播与容错** | **输入/准入错误**：HTTP 层校验返回 400（含未注册 `--image-entry`/`--tts-entry`/`--video-entry` 或 `stream=false` 请求 `audio.stream=true`）；`AdmissionGuard` 或 `UnifiedEngine` `max_in_flight` 满抛 `QueueFullError` 返回 `503 + Retry-After: 1`；固定上游代理带连续失败熔断器（`UpstreamProxy::check_circuit_breaker`）。**执行/进程错误与网关热重启**：Stage 计算抛异常 → `SimpleScheduler::emit_error` → `Stage::fail_request` 发 `CompleteMessage(success=false)`；Stage 进程崩溃 → `PushSocket` ZMQ monitor 捕获 `ZMQ_EVENT_DISCONNECTED` → `Coordinator::stage_down` 立即将归属该 stage 的 pending/running 请求置 `failed` 并清除 `ready_stages_`（使 `/ready` 变 503）。相反，当网关进程（`umm-infer serve`）重启时，C++ `Stage`（`ready_heartbeat_ms=1000` 非阻塞 `send_dontwait`）与 Python `VideoStage`（`READY_HEARTBEAT_S=1.0` `zmq.NOBLOCK`）周期性重发 `stage_ready`，新 `Coordinator` 幂等重注册并排空待发请求；`umminfer_service.py` 默认仅重启网关而不杀已加载 GPU 权重的 Stage（超出 `ready_grace_s` 仍未就绪时才触发单次保底重启）。**可选输出 leg 隔离**：Image/TTS/Video 子 leg 提交失败、执行失败或超出产物字节上限均发 `kLegError` 但不中断主 VLM 终态；主请求在所有子 leg 结算后仍发 `kComplete` 并在 `legs[]` 报告各 leg 状态。 | `src/comm/comm.cpp:97-170`, `398-408`; `src/pipeline/pipeline.cpp:232-265`, `865-915`; `python/umminfer_service.py:295-330`; `src/pipeline/unified.cpp:459-480`, `750-805`, `1146-1220`; `src/serve/upstream_proxy.cpp` |
| **会话与资源生命周期** | **会话**：默认 `session_id` 仅作事件分组标签与取消归属边界；`--session-history` 开启进程内 `SessionHistoryStore`（默认 256 会话 × 8 轮 / 16384 字符三级淘汰，仅在 `kComplete` 成功时入库，取消/失败不入库，重启清空）。**请求记录**：`UnifiedEngine::release` 清理 `records_`、`child_to_unified_`、`admission_`（并刷新 `unified_admission` 队列深度指标），级联 `Coordinator::release` 所有子 id（含 `:vlm`/`:image`/`:tts*`/`:video`）。**回调退场**：`UnifiedEngine::~UnifiedEngine` 通过 `CallbackGate::retire()` 等待活跃回调退出并将后续事件无缝回退到前序 sink。**Relay 信用与内存**：`ShmPutOperation` 析构自动 `shm_unlink` 并释放 `CreditAllocator` 信用；`Stage::reaper_thread_` 按 `transfer_timeout_ms`（默认 30s）回收无 ACK 传输；`CudaIpcPutOperation` 在被放弃时退休 device slot 并将 relay 标为不健康，防止复用接收端仍在映射的显存。 | `src/pipeline/session_store.cpp:17-71`; `src/pipeline/unified.cpp:41-119`, `380-430`; `src/relay/relay.cpp:80-120`, `226-286`, `451-551`; `src/pipeline/pipeline.cpp:315-346` |

---

### 2.1 原生 Qwen3.8 视频输入与有界时序关键帧

`video_url` 默认仍沿用原有确定性均匀抽帧：FFmpeg 单次缩放、最多 8 帧、
temporal-2 patch packing、按历史公式生成 4 个时间片时间戳，随后复用同一
`expand_video_pads`、3D MRoPE、批量多片 ViT 以及串行/批处理生成路径。未提供
`frame_selection` 时，HTTP/Stage wire shape 不增加字段，解码与归一化像素保持
逐字节一致；`{"policy":"uniform"}` 与缺省等价。

显式 `video_url.frame_selection={"policy":"temporal",
"motion_threshold":0.02,"scene_threshold":0.25}` 才启用新路径：

1. 输入仍先经过公开 32 MiB data-URL 上限；可选整数 `max_frames` 只接受 4/8/16，
   缺省严格为 8。解码器在分配 stdout 前固定候选 64 帧、所选预算和 3584 pre-merge
   patch 硬上限；bool/float/string/零/负数/其他整数/溢出值均 fail closed。
2. 容器有帧数/帧率元数据时，以源帧序号 `select` 均匀覆盖全片并保持 PTS
   passthrough；未知元数据时使用有界 `fps` 候选回退。`showinfo` 提供源时间戳，
   缺失或非有限 PTS 才使用确定性名义时间戳；短解码不补造新帧。
3. CPU 在固定 16×16 Rec.601 luma 网格上计算相邻帧平均绝对运动差和 16-bin
   直方图场景差。首末候选是覆盖锚点；相邻场景检测合并为一个代表，先选场景
   边界再选运动峰值，同分按较早解码序号。严格阈值使重复帧即使在零阈值下也
   不消耗优先槽位；静态、重复、全部超过阈值或全部低于阈值均有确定性均匀补位。
4. 结果只按解码顺序输出，绝不按时间戳重排；非单调有限 PTS 原样保留。每个
   temporal-2 时间片使用所选帧对第一帧的源时间戳。选择结果仍一次性进入
   `preprocess_video_frames`，不是互相独立的静图，因此 Wave 16 多片 ViT、视频
   模板、mRoPE、prefix media hash 与 serial/batched parity 均不改变。

离线验证由 `native/qwen38_tests/test_video_input.cpp`、Stage schema、公开 HTTP JSON/SSE
与 C++ CLI/Client 契约覆盖；`tests/video_temporal_live.py` 保留 SM90 同视频同问题
A/B 门禁。初始 temporal 接入未访问 GPU；后续独立受控实测见下文与专题文档。

长视频扩展在 `benchmarks/video-temporal-long/` 固定 4 个 449 帧 / 224.5 秒
H.264 源与逐帧内容/包序证据，独立 CPU oracle 覆盖预算 4/8/16；历史 R4 live 只运行 8。
`tests/video_temporal_long_live.py`
复用短门禁的 SSE/native-log 契约，以 seeded interleaved warmup+3 重复比较 serial
与 batch-four、冻结源码/二进制/模型/数据/请求拓扑，并强制冷前缀、设备独占、
默认 uniform 兼容、精确源内容/PTS、temporal-2/mRoPE、完整产物与 SHA-256 门禁。
新增的 `selected_frame_content_fnv1a64` 仅用于诊断 native 输入与独立解码内容的
一致性，不改变默认 uniform 像素或历史 nominal mRoPE 时间戳。Task #44 R4 已在
`77215ce` 用隔离 SM90 GPU 0/1 完成 96/96 请求、冷前缀、serial/batch-four、源内容/
PTS/temporal-2/mRoPE 与完整证据门禁，获 `CONTROLLED_LIVE`；default/uniform/temporal
显式答案命中分别仅 6/24、6/24、0/24，感知质量与泛化提升仍 `NOT_RUN`；该历史 R4 的
预算 4/16 仅 CPU-oracle。详见 [video-temporal-long.md](video-temporal-long.md)；fake
endpoint 只验证合同。

Task #47 的官方根因审计进一步证明：R4 default/uniform 选择缺少原预期答案锚点、
彩色 source-binding tag 会污染颜色问题、部分 prompt 直接泄露答案；像素/PTS/mRoPE
对照未证明 native 实现缺陷。Task #51 新增独立的 `temporal-clean-r1` 夹具，不改冻结
R4/R5 或 native runtime：仅用 shape/pattern/luminance 语义，预注册并 hash-freeze
问题、答案与评分，独立渲染锚点并把 late-anchor selection-miss controls 排除在质量
准确率之外。隔离 SM90 GPU 0 serial / GPU 1 batch-four 完成 192/192 请求和 208/208
parity；default/uniform/temporal 质量用例命中为 18/24、18/24、12/24，所有 miss 与
control 原始答案均保留且未新增别名。证据支持有界模型局限，未证明运行时缺陷；
`PERCEPTUAL_QUALITY: NOT_RUN`。详见 [video-temporal-clean.md](video-temporal-clean.md)。

Task #54 在不改冻结 R4/R5/clean 字节、prompt、答案或评分的前提下加入显式 4/8/16：
default 与 uniform-8 逐字节等价；uniform 4/16 以 rounded source-index `select` 覆盖首末
源帧并保留真实 PTS，temporal 沿用冻结候选和优先级/fallback。SM90 GPU 0 serial / GPU 1
batch-four 完成 448/448 请求（112 热身 + 336 实测），400/400 parity、全冷前缀、精确 RGB/
源序号/PTS/temporal-2/mRoPE、948 项 SHA-256 与双端新进程验证通过。4 帧降低时延但命中
下降；16 帧提高部分 temporal 命中但增加 prefill/ViT 时延，不能据此声称泛化质量。详见
[video-temporal-clean.md](video-temporal-clean.md#task-54-explicit-4816-frame-budgets)。

Task #61 的受控单因子扩展只改变显式 temporal 的运动优先级：相邻超阈值 motion
候选先合并为一个事件，每个 run 以 motion、scene、最早解码序选出唯一代表，再沿用冻结
的全局排序与 uniform 补位；首末锚点、阈值、候选 stride、PTS、temporal-2/mRoPE 和
default/uniform 均不变。在同一空闲 SM90 物理 GPU 1 上顺序运行 serial 与 batch-four，
896/896 请求、每阶段 400/400 parity 和 256/256 default/uniform 跨阶段恒等通过；temporal-8
冻结质量命中从 12/24 提升到 18/24，仅追平 default/uniform-8，4/16 无回退，时延门禁通过。
两个无 producer scoring import 的新进程独立重解码 1,796 源帧并复核全部记录。配对符号检验
`p=0.5`，故仅为四例合成夹具的受控改进，感知质量与泛化仍 `NOT_RUN`/`NOT_COVERED`。
详见 [video-temporal-motion-run.md](video-temporal-motion-run.md)。

Task #66 随后在不改 accepted selector、阈值或 runtime 的前提下，冻结 8 个新 procedural
synthetic 449 帧源与 10 个多语言质量问题，在同一空闲 SM90 物理 GPU 1 顺序运行 serial 与
batch-four，完成 560/560 请求和 500/500 parity。primary temporal-8 与 uniform-8 均为
18/60，paired case/mode 为 2 改进 / 2 回退，精确符号检验 `p=1.0`；temporal-4 为 30/60
（uniform-4 12/60，`p=0.109375`），temporal-16 与 uniform-16 均 24/60。四个 Task #61
用例及分数保持不变且不进入新分母，历史日文 `円` miss 不新增别名。独立新进程重解码
3,592 帧并复核全部 RGB/source/PTS/temporal-2/mRoPE、答案、聚合与校验和。该结果不支持
泛化提升；自然视频与感知质量仍 `NOT_COVERED`/`NOT_RUN`，最终 runtime 不变。详见
[video-temporal-heldout.md](video-temporal-heldout.md)。

Task #72 在不改 accepted selector 或 runtime 的前提下，将同一仓库历史 Apache-2.0
real-footage 源切成 8 个不重叠自然视频片段，冻结 8 种语言的视觉可判定问题、源帧范围、
RGB/PTS/B-frame 顺序、评分、population 与分析后，在同一空闲 SM90 物理 GPU 1 顺序运行
serial 与 batch-four。accepted r2 完成 320/320 native-valid 请求、304/304 parity；
primary temporal-8 与 uniform-8 均 42/48，16 个 case/mode 无改进或回退，`p=1.0`；
budget-4 双臂均 48/48。`natural-07` 的 18 个八帧 `Pappina` miss 全部保留且不新增别名，
latency outlier 为 0。双端独立新进程重解码 165 帧并复核全部 RGB/source/PTS、首末锚点、
temporal-2/mRoPE、Task #66 控制与校验和。早期 duration oracle r1 被排除但完整保留；
自然视频结果仍限于单源三类场景，感知质量 `NOT_RUN`，不支持泛化提升。详见
[video-temporal-natural.md](video-temporal-natural.md)。

---

### 2.2 显式叙事图像槽位 → I2V 引用与内容寻址绑定

Wave 21 的 `narrative.continuity_mode="i2v_chain"` 仍负责统一跨槽位调度，默认
`off` 的请求、自动标签路由、SSE 与隐式 latest-image/T2V fallback 均保持原行为。
本扩展只增加模型可选的显式指令后缀：

```text
[[video: <motion/camera prompt> | source=slot-N]]
```

`NarrativeStreamParser` 仅保留 `| source` 后缀语法；普通 `|` 提示词不变。解析器从
render prompt、clean prose、会话历史与流式 TTS 中移除整个指令及引用后缀。
`UnifiedEngine::dispatch_narrative_slot_locked` 将显式 `slot-N` 映射到同一请求的
`anchor_slot_index`，并设置 `anchor_explicit=true`；不会创建第二套 continuity
mode、跨请求图片注册表或隐式“上一张图”引用。

显式引用只允许 `i2v_chain` 中指向当前请求更早的 image slot。视频 leg 在源图
pending/running 时保持 pending 并计入 `legs_remaining`，源图 completed 后才沿既有
`release_chained_slots_locked` 派发。完成处理在源 `OutputLeg` 中保存不可变字节、
MIME、data URL 与 SHA-256；真正提交前 `bind_chained_anchor_locked` 重算 SHA-256
并核对内容寻址绑定，再以二进制 `inputs.image` 绑定同一份字节。请求原有
size/seed/num_frames/steps/guidance 等设置原样保留，`anchor_crop="center_cover"`
记录 Wan2.2 的保比例中心裁切策略。原生 Wan `parse_video_work` 在执行前再次核对
`anchor_explicit` 与 `anchor_sha256`：显式无图、wrong-hash 或 tampered bytes 均 fail closed；
未携带这些可选字段的既有 direct T2V/I2V 输入保持原行为。

缺失、重复、 malformed、forward/self、cross-session、wrong-modality、冲突
`video.image`、失败或取消的显式引用均不产生 T2V fallback 或陈旧字节；依赖 leg
以 slot-attributed `leg_error` 失败/中止，取消仍沿既有 Coordinator abort 传播。
隐式 mainline 链的 latest-image 选择与失败回退 T2V 行为有意不变。

`visual_slot` 只携带 `continuity_mode`、`anchor_slot_index` 与可选
`anchor_explicit`，不携带时序相关的哈希；解析后的视频 artifact 和 terminal leg
才携带 `anchor_sha256`。轨迹 v2 仅在字段存在时投影这些元数据，媒体仍只落 CAS；
已解析链接强制核对更早图像槽位的 CAS SHA-256，失败/取消的未解析引用可作为负例
回放。`tests/narrative_slot_reference_live.py` 保留公开叙事链接的确定性契约与负例；
Task #62 的 `tests/flux2_i2v_continuity_live.py` 另在隔离 SM90 物理 GPU 1 上完成
**1 次真实 FLUX.2 源图 + 3 次原生 I2V + 3 次同设置 T2V 对照**，逐阶段卸载并校验精确
PNG bytes/SHA-256、原生 `i2v=1/0` 日志与完整模型/二进制/请求/媒体/RGB 证据。冻结
64×40 center-cover RGB 代理的 I2V 首帧 MAE 为 `2.286328`、首尾运动 MAE 为
`3.466276`（T2V 分别 `85.225521` / `15.288021`）；wrong-hash 与 tampered-source
均在生成前拒绝，新进程独立复核通过。Task #62 当时的完整公共 VLM 叙事编排和感知质量
均 `NOT_RUN`，详见 [flux2-i2v-continuity.md](flux2-i2v-continuity.md)。

Task #65 已在同一官方基线的隔离物理 GPU 1 上补齐真实公共 VLM 叙事门禁：一次文本
请求产生 `slot-0` FLUX.2 PNG、显式 `source=slot-0` 的原生 Wan I2V 与干净流式语音。
串行 VLM 常驻，FLUX→TTS→Wan 每次只驻留一个输出 Stage。公开流保留 69 个 unified
事件、32 个完整产物（30 个 PCM chunk）、raw SSE/CAS 与逐事件单调 tick；只读 tap 证明
Wan 收到同一不可变 PNG bytes，五个 TTS leg 的 native PCM→public PCM→terminal WAV
逐字节闭合。源图/视频 SHA-256 为 `01af1c29…150b` / `49c6ae4e…10b60`，冻结首帧/运动
MAE 为 `2.259115` / `5.124479`，新进程双端重放通过，GPU 1 清理至 `3 MiB`。
此外，原始取消回归先暴露 running source 在 terminal 中未结算，最小修复让 cancellation
同时将 pending/running slot 置为 aborted；late bytes/events 仍被抑制。隐式 off、
prompt_anchor 与 default fallback 不变。完整证据见
[public-narrative-i2v.md](public-narrative-i2v.md)；感知质量与泛化仍 `NOT_RUN`/`NOT_COVERED`。

Task #67 在官方 Wave 91 基线的隔离物理 GPU 7 上完成真实原生 Wan2.2 取消/故障门禁：
1 个先生成的有效控制、4 个取消窗口（serial queued、denoise step 4/20、MP4 emission、
post-terminal）、1 个原生非法尺寸拒绝及 4 个紧随的精确同 hash 恢复；共 6 个有效真实
MP4 均为 `b1a4d7d1…7c21`。raw MessagePack 先暴露 PUSH+PUB 对同一 request 各发一次
native abort terminal 的缺陷，`Stage::abort_request` 现以有界 10 分钟墓碑合并双通道终态；
确定性回归由 6 个重复终态降为 1 个且后续请求成功。公开 ACK/terminal 与 CUDA cooperative
exit 分开计时，实测 denoise 退出区间为 432.578–433.578 ms，不声称即时设备抢占。
VAE 内部取消仍 `NOT_RUN`。完整原始证据、0 late event/artifact、`in_flight=0`、双端新进程
复核和清理见 [native-wan-cancel-fault.md](native-wan-cancel-fault.md)。

Task #74 在官方 `a4320e7` 基线的隔离 GPU 1 上通过真实公共
`/v1/visual/render` 覆盖 FLUX.2 宽幅 704×480、竖幅 480×704 与方形
512×512→704×480 center-cover 锚定。现有单次等比缩放/中心裁切无需运行时修复；
原生日志新增实际 source_size 与 decode/resize/VAE encode/DiT/VAE decode/MP4 分项
计时，公开 schema 不变。实测 DiT 中位 `8811.12 ms` 为最大分项。历史候选
`76fe857` 曾临时加入 pinned velocity view；真实 CUDA 6144 元素逐位一致，九对
public A/B 及 profile MP4 全部字节一致，但公共完整产物配对中位差仅
`−3.687044 ms`（`−0.0331393%`），6/9 为负差、paired MAD `7.398493 ms`，且
candidate IQR/MAD 增大，不能证明用户可感知收益。因此该候选 **NOT_PROMOTED**；
`56ddc30` 已移除 candidate-only API、copy switch、测试夹具与默认调用点，最终运行时
仅保留几何验证和低开销分项计时，不声明任何性能提升。双端新进程复核及最终原始
baseline 复验见 [i2v-geometry-latency.md](i2v-geometry-latency.md)；感知质量、
角色/风格一致性与部署均未验证。

---

### 2.3 可选请求级原生分配器遥测与受控消融边界

公开聊天请求的 `telemetry.native_allocator=true` 是唯一启用入口；默认请求不增加
Stage 参数、结果字段、SSE/JSON 字段或遥测日志。启用后，`UnifiedEngine` 将布尔参数
随 ASR/VLM/输出子 leg 透传，原生 Stage 包装器在单次 compute 外建立独立
allocation-ownership epoch，终态结果经 `request_telemetry` 回到统一终态信封。

当前串行 Qwen3.8、串行 Qwen3-ASR、串行 Wan2.2（`cuda` / `cuda-scaffold`）、串行原生
Qwen3-TTS 与串行原生 Qwen-Image-2.1 把显式 `cudaMalloc`/`cudaFree` 接入所有权跟踪。峰值只累计该 epoch
首次拥有的直接 CUDA 块；加载期常驻权重、CUDA context/driver、外部库内分配不在
范围内。直接 runtime 块的 allocated 与 reserved 相等，不从设备显存、常驻基线或
其他请求做减法。跨请求晚到的 free 只修改原 owner，不能污染新 epoch；重叠串行
作用域会把双方降级为 `NOT_COVERED`。批处理、并发共享执行、未接入后端、
Python/外部代理、缺失或 malformed 数据均显式 `NOT_COVERED`，字节峰值为 null
而非伪造的零。

TTS 的 Talker、Code Predictor、causal codec/KV、控制/scratch 与双 CUDA Graph
均在 model load 时分配或捕获，属于排除的预请求池；请求执行只复用这些固定地址。
ASR 的 STFT/log-mel 在 CPU 执行；权重、encoder/conv、decoder/KV、scratch 与
decode graph 均在请求前创建。因此 COVERED 的 TTS 与 ASR 请求若只复用预请求池且无新直接分配，
可以真实地同时报告 `0 B` 与 `0 alloc / 0 free`，但不得把缺失覆盖写成零。Graph runtime
内部分配与 pinned host staging 不属于该直接设备块指标，standalone TTS/ASR CLI
仍走原始 CUDA 调用且无请求遥测；独立 CUDA fixture 用非零小块验证计数、峰值、晚 free、失败与取消，
不能把该 fixture 峰值冒充模型请求峰值。

Image2 的 BF16/FP32 DeviceTensor、文本输入 id 与 DiT target mask 均通过同一 owner
adapter；standalone CLI 保持直接 CUDA 调用，2512/Omni 仍 `backend_not_instrumented`。
Task #69 先以 telemetry-off 请求完成真实模型加载，再以 `load_ms=0` 的 warm 请求证明
预请求权重不进入后续 epoch；成功的直接请求分配仍需非零计量，不能照搬 TTS/ASR 的
pool-reuse 零事件结论。GPU 1 实测四个成功 epoch 的 owned peak 均为 `607,110,944 B`、
final `0`、299 alloc/299 free；失败、取消与恢复边界及独立 NVML/媒体校验见
[native-service.md](native-service.md#sm90-qwen-image-21-validation-gate--pass-2026-11-27)。

终态信封保留公共 request id、子 stage request id、公共/原生 stage 身份、串行/批处理
属性、epoch reset 语义、事件样本覆盖和 completed/failed/cancelled 状态。立即取消若先于
原生终态到达，则以 `stage_terminal_snapshot_missing` 结束，不回写已发送事件。公开投影
采用固定字段 allow-list，最多 64 个 stage 项，不接收模型路径、IPC 路径或自由格式后端
诊断。锁与指针表仅服务于启用的覆盖 epoch，无逐帧采样或无界日志。

请求级峰值与 `/metrics` NVML、消融 `measurement.resources` 的
`whole-device-used-memory` 严格分离。受控测量可选
`request_memory={enabled,required}`：enabled 注入公开请求选项并保存逐请求报告；
required 要求所有测量请求完全 COVERED，否则撤销 `CONTROLLED_LIVE`。详细协议和后续
SM90 门禁见 [native-service.md](native-service.md#opt-in-request-native-allocator-telemetry)。
串行 Qwen3.8/Wan2.2 的 GPU 0 门禁与受控双臂消融已通过；Task #56 的串行
Qwen3.8+CustomVoice TTS GPU 7 门禁（双提示、warm、off/on JSON/SSE/PCM、失败/取消与独立 NVML/清理）
及 Task #57 的串行 Qwen3-ASR GPU 7 门禁（中英 PCM16、warm epoch、off/on 协议与转写一致性、原生失败/取消、
独立 50 ms whole-device NVML 与清理）均已通过；Task #69 的串行 Qwen-Image-2.1 GPU 1
门禁（9 个公开请求、off/on JSON/SSE/PNG、warm、失败/取消/恢复、独立双端新进程验证与清理）
也已通过。Base TTS live telemetry、批处理/重叠执行、2512/Omni、准确率、感知质量、
生产部署与泛化性能仍不在该门禁声明范围内。

---

### 2.4 原生 Qwen3-TTS continuation 重复惩罚确定性

Task #59 修复了 Talker group-0 历史重复惩罚的 CUDA 竞态。旧核函数让每个
history entry 对同一 `logits[id]` 做非同步 read/modify/write；重复 id 可能被
多次惩罚并改变 argmax、EOS 帧数与完整 PCM。现实现先以 shared-memory bitset
幂等标记出现过的 id，再由每个 logit 的单独 owner 应用一次惩罚，保持 HF
gather/scatter 语义且不改变公开请求、流式 PCM/WAV、allocator、取消或调度契约。

聚焦真实 CUDA fixture 修复前失败、修复后通过；最终两轮 fresh-process
108-request direct-native A/A 的 codec frames 与 PCM 全部逐字节一致，两轮真实
native-stage replay 同时验证 stream/terminal、nonstream/stream、cancel/retry 和
seeded fragmentation。完整双 GPU R4、fair-priority 候选、感知质量均未重跑或
解锁。根因、哈希与独立校验见 [realtime-first-audio](realtime-first-audio.md#task-59-native-continuation-determinism-closeout)
及 [证据](evidence/tts-continuation-determinism-sm90.json)。

---

### 2.5 公开 HTTP 语音与原生 PCM 逐字节门禁

Task #63 在隔离 SM90 物理 GPU 0 上同时运行统一 Qwen3.8 VLM 与原生 TTS
entries，保留 HTTP/SSE/JSON、直接 `/v1/audio/speech`、Realtime WebSocket 与
ZMQ/MessagePack 原始记录。独立新进程把每个公开 PCM chunk 绑定到实际 native
TTS chunk、提交文本和 UTF-8 游标，覆盖 fresh repeats、取消/重试、延迟读取与最终
`response.playback_ack`。直接精确文本控制复现 Task #59 continuation PCM，但不把
VLM 实际生成的不同拼写改写成该控制文本。

Realtime 背压只以 native TTS audio chunks 减去公开 `audio.delta` 帧数证明至少
4,096 条待交付音频事件；VLM 文本、ASR、ACK 与终态事件均不计入释放条件。至少
一个 reader 正在暂停且达到该条件后才恢复所有当前暂停 reader，未暂停 reader 继续
消费以避免后续 VLM/text 阻塞首音；暂停 reader 由独立线程每 30 秒发送既有
WebSocket PING，PING/PONG 不计入音频背压且不改变 response deadline；这些 heartbeat
仍未阻止后续 reset，相关 timeout 实验未进入最终源码。最终证据为 HTTP 独立 PASS，
realtime 4,096 audio backlog 达成但仅 17 completed/12 reset，完整 parity blocked。
最终 29 个响应仍要求逐字节 PCM、游标、终态和
`in_flight=0` 全部成立，否则证据 fail closed。实现仅新增验证驱动、独立校验器与
回归测试，未改变 serving schema、scheduler、阈值或 native TTS kernel。完整双 GPU
R4、物理播放与听评仍 `NOT_RUN`。详见
[public-speech-pcm-gate](public-speech-pcm-gate.md)。

Task #70 将 12 个 reset 精确定位到 libwebsockets 默认 300/310 秒
connection-validity hangup：client PING 不会确认 validity，延迟读取在响应仍活跃时
也可能被误判为 idle transport。`RealtimeApi` 现在仅在 ASR/response ownership 或
generation-complete 输出仍排队时刷新 transport validity，绝不刷新
`response_deadline`；ownership 结束后恢复普通 idle 清理。真实 loopback 的 1/2 秒
门禁修复前在 2.001 秒丢失终态，修复后 PCM/最终 ACK 完整，10 秒 stall timeout 与
idle-only close 均保持原语义。执行提交 `2b49b5b` 合入 Wave 94。真实 GPU 0 R2
最终达到 4,125−29=4,096 精确 audio backlog，29/29 responses 均一个终态与最终 ACK，
493 legs / 4,176 chunks / 115,034,880 PCM bytes 逐字节闭合，25 个 HTTP 控制保持通过。
两个新进程验证与 1,266 文件 SHA256SUMS 均通过，GPU 0 清理至 3 MiB。结果与证据哈希见
[public-speech-pcm-gate](public-speech-pcm-gate.md#task-70-real-sm90-outcome-pass) 及
[机器证据](evidence/realtime-transport-validity-sm90.json)。

### 2.6 原生 TTS 首音公平交替与双 GPU R4

Task #73 在新冻结合约下为串行 native TTS 增加内部 `request_priority`：
`:tts:0` 与一次性 `:tts` 进入 first-audio FIFO 类，后续 `:tts:N` 和普通请求
留在原 FIFO 类；两类同时非空时交替取出，绝不抢占正在执行的 CUDA compute。
未安装 hook 的 scheduler 保持严格 FIFO，batch/concurrent 配置拒绝该 hook。
公开 schema、取消墓碑、响应 deadline、队列上限、采样默认值和 PCM/cursor 均未改变。

四轮 `fair → FIFO → fair → FIFO` 完整 R4 均通过七臂、所有回归和独立原始证据
校验，共 324 个 measured responses。两组 A/B 的 primary median 分别降低
46.161%/45.921%，p95 降低 60.263%/60.048%，serial 变化在 ±0.05% 内；跨策略与
同策略 A/A 的 transcript/text/prompt/cursor/PCM 全部精确相等。native compute
提前与 VLM 共享 GPU 的额外开销及 alternate arm 无收益均保留，不作泛化声明。
候选另通过 30-response 真实 network/reconnect 门禁和 25 HTTP + 29/29 delayed
reader 公共 PCM 门禁，终态、最终 ACK 与 4,125−29=4,096 精确释放均保持。
ACK-anchored cancel→silence 是数字虚拟播放队列清空时间，不是物理扬声器实测。
完整实现边界、逐臂数据、原始证据哈希和负结果见
[realtime-r4-post-determinism](realtime-r4-post-determinism.md)。

### 2.7 跨槽位视觉连续性、无损 Web 分镜与原生 TTS 长流生命周期（Tasks #76, #77, #81）

- **跨槽位引用（Task #76）**：`UnifiedEngine` 将 `source=slot-N` 从显式 I2V 扩展到 FLUX.2 图像编辑（`slot-0` 静图 → `slot-1` 编辑图 → `slot-2` Wan2.2 I2V），基于不可变 CAS 字节与 SHA-256 绑定，失败源递归结算传递依赖，取消/主失败同时终结 `pending/running` 槽位；详见 [cross-slot-visual-continuity](cross-slot-visual-continuity.md)。
- **零手工控制 Web 交互分镜与无损回放（Task #77）**：Web UI 默认折叠高级控件，支持源字节绑定 Revise、取消/重试终态精确落地以及默认 raw-SSE/media 无损 JSON 导出与导入重建；详见 [storyboard-showcase](storyboard-showcase.md)。
- **原生 TTS Markdown 过滤与 32-active 长流无损背压（Task #81）**：`sentence_split.hpp` 在送入 native Talker 前剥离所支持的 Markdown 表现分隔符并保留公开文本/UTF-8 游标不变，EOF 抑制未闭合保留指令，32 活跃 TTS 子请求按终态释放且独立流式 PCM 不计入 60 MiB 终态聚合上限，`realtime_api.cpp` 在既有 1,024/4,096 边界下保序重排队；详见 [ns01-tts-lifecycle](ns01-tts-lifecycle.md)。
- **统一 NS-01…NS-08 自动化评测与诊断基线（Tasks #75, #78, #79, #80, #82, #83, #86）**：落地统一注册与机器计分板 `tools/umminfer_ns_scorecard.py`、`VideoDecodeProfile` 分项计时以及 NS-01…NS-08 冻结配对实测套件；未达硬门槛的候选（T75 二次缩放、T78 prompt 路由、T79 TeaCache 0.25、T80 端侧默认折叠、T82/T83 全路径打断、T86 boundary2 TeaCache）全部严格 `NOT_PROMOTED` 并从默认运行时移除，仅保留可复现评测与紧凑证据。T86 三几何逐次 `<5 s`、取消/恢复全过，但 15/15 首帧 RGB 不精确且竖屏 SSIM `0.896092<0.90`，详见 [ns02-fidelity-speedup](ns02-fidelity-speedup.md)。

---

## 3. 本机轻量构建与测试验证矩阵

> 环境：macOS 25.3.0 (ARM64), Zig 0.16.0, AppleClang 21.0.0, CMake, Swift, Python 3.13.7（系统 Python 无 `msgpack`/`pyzmq`/`mlx`/`websockets` 第三方包），无 NVIDIA CUDA GPU。
> 规则：本矩阵仅记录本机无 GPU 轻量构建与测试；真实大模型与 SM90 线上验证均明确标为 `NOT_RUN`。

| 序号 | 套件 / 构建目标 | 命令 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | 控制面 Debug 构建 | `zig build -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl` | **PASS** | 退出码 0，产出 `zig-out/bin/umm-infer` |
| 2 | 控制面 ReleaseFast 构建 | `zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl` | **PASS** | 退出码 0 |
| 3 | C++ 统一编排契约测试 | `zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl` | **PASS** | 退出码 0，`unified_contract_test` 全套（含文本/图像/音频回归及新视频统一契约：`pending/running/completed/failed/aborted`、取消、会话归属、资源清理、SSE 5 MiB 分片重组）通过；**仅验证测试 Stage 契约，不计作真实生视频** |
| 4 | HTTP 与进程生命周期冒烟 | `zig build smoke -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl` | **PASS** | 退出码 0；demo/external stage/disconnect/SIGTERM/CLI 全部通过 |
| 5 | 客户端 CLI 解析与 URL 策略边界检查 | `python3 -I tests/client_boundary_e2e.py zig-out/bin/umm-infer --skip-real` | **PASS** | 退出码 0；`chat_cli_parsing`、`serve_stage_parsing`、`url_policy` 三项 PASS，真实回归按 `--skip-real` 跳过 |
| 6 | 功能映射锚点校验 | `python3 -I tools/check_feature_map.py` | **PASS** | 退出码 0；逐行与基线 commit 核对通过 |
| 7 | 离线多模态基准契约测试 | `python3 -I tests/multimodal_benchmark_test.py` | **PASS** | 退出码 0；70 用例回放与指标一致性通过 |
| 8 | 黑盒门禁自检 | `python3 -I tests/blackbox_integration.py --self-check` | **PASS** | 退出码 0；`log_cap`、`timeout`、`orphan` 三项进程组清理自检通过 |
| 9 | macOS Swift 客户端测试 | `(cd clients/ummmlx && swift test)` | **PASS** | 退出码 0；71 tests, 0 failures（含 `video/mp4` 多分片产物重组） |
| 10 | `native/stage_cli` host 构建与 ctest | `cmake -S native/stage_cli -B /tmp/umminfer-stage-host -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build /tmp/umminfer-stage-host -j && ctest --test-dir /tmp/umminfer-stage-host --output-on-failure` | **PASS** | 退出码 0；`tts_tail_trim_test` 1/1 通过 |
| 11 | `native/` 根 host 库构建 | `cmake -S native -B /tmp/umminfer-native-root -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build /tmp/umminfer-native-root -j` | **PASS** | 退出码 0；`libqwen38_host.a`、`libumm_asr_host.a` 编译通过 |
| 12 | `native/{omni,tts,image,image2}` host 构建与 ctest | `cmake -S native/<kind> -B /tmp/umminfer-<kind>-host -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build ... -j && ctest ...` | **PASS** | 4/4 子项目 host 测试恢复并全部通过（`omni_host_tests`、`tts_host_tests`、`image_host_tests`、`image2_host_tests`；`tts_host_tests` 无权重目录时跳过 `test_tokenizer` 并打印 `SKIP`） |
| 13 | MLX Stage 结果元数据单测 | `python3 -I tests/mlx_stage_result_test.py` | **FAIL (环境缺失)** | `ModuleNotFoundError: No module named 'msgpack'` |
| 14 | Python 端到端真实模型套件 | `tests/client_boundary_e2e.py`（真实 chat/ASR 部分）及 `tests/{realtime_contract_e2e,realtime_client_contract_e2e,client_retry_e2e,mac_unified_validation}.py` | **NOT_RUN** | 需要真实 MLX/TinyASR 模型及 `mlx`/`msgpack`/`zmq`/`websockets` 依赖 |
| 15 | CUDA IPC / 原生 CUDA stage / 真实视频生成后端 / SM90 线上与压测 | `zig build cuda-bench -Dcuda`, `native/*` `-DUMM_*_CUDA=ON`, 真实视频 Stage/上游生成, `tests/{sm90_service_live,full_duplex_live,full_duplex_l2_live,soak_native,concurrency_load,multimodal_benchmark}.py` | **NOT_RUN** | 需要 Linux CUDA 主机、真实模型权重或真实视频上游；本机不执行，测试 Stage 桩不计作真实生视频验收 |

---

## 4. 已知技术债与治理边界

> 本轮已完成修复：
> - **P1**：`UnifiedEngine::settle_output_leg_locked` 取消 TTS leg 误报 `failed` 修复与回归测试（`src/pipeline/unified.cpp`、`tests/unified_contract_test.cpp`，已做 Red/Green 验证）；
> - **P1**：`UnifiedEngine::handle_stream` 中 `drain_sentences_locked` 流式 TTS 句切分提交异常边界保护与回归测试（`src/pipeline/unified.cpp`、`tests/unified_contract_test.cpp`，已做 Red/Green 验证）；
> - **P1**：恢复 `native/{omni,tts,image,image2}` CPU host 单元测试并删除 `CMakeLists.txt` 中指向已删除测试/工具的失效 target（已做 Red/Green 验证）；
> - **P2**：`RealtimeApi::drain_unified` 移除 `http-` 前缀硬编码耦合，改为显式 `unified_owned_` 所有权集合追踪；`RealtimeApi::drain_partials` 改为 `active_request_sessions_` `O(1)` 索引查找（`src/serve/realtime_api.cpp`）；
> - **P2**：对齐 HTTP 边界与 `unified_schema` 校验规则（统一单 text part 与 `image.output_format=png` 约束，`src/serve/openai_helpers.cpp`、`src/pipeline/unified_schema.cpp`），并拆分 `src/serve/openai_api.cpp` 为 `openai_internal.hpp`/`openai_helpers.cpp`/`openai_audio.cpp`/`openai_chat.cpp`、`cpp-httplib` 线程池随 `max_in_flight` 自适应；
> - **P2**：补齐 `UnifiedEngine` 取消竞态 `record_primary_leg("aborted")` 记账与 `release()` 非终态 `unified_admission` 指标刷新及回归测试（`src/pipeline/unified.cpp`、`tests/unified_contract_test.cpp`）；
> - **P2**：Wave 3 架构加固完成——(1) C++/Python Stage 周期性非阻塞 `stage_ready` 心跳与 `Coordinator` 幂等重注册，网关重启默认保留运行中的 GPU Stage 进程；(2) `Qwen3.8-27B` 连续批处理接入混合注意力 Prefix 快照复用（SM90 GPU 0 4.21× prefill 提速）与 12-bit 浮点分桶快速 top-p 采样（248k 词表 16.9× 加速）；(3) `native/stage_cli` 将模型专属 CLI 参数下沉至各 `*_stage.cpp` 钩子（`stage_usage_flags`/`parse_stage_flag`），并在 CUDA 12.4 SM90 下验证全部 6 个原生 Stage 联编通过；
> - **P2**：Wave 4 能力补齐与原生演进完成——(1) `POST /v1/chat/completions` 开放 `top_p` `(0,1]` 与 `seed` (`int64`) 校验并全链路透传到 VLM stage（ASR leg 自动剥离）；(2) `Qwen3.8-27B` 在 `<|im_end|>\n` 轮次边界切分快照，打通真实多轮聊天 Prefix Cache 复用（SM90 GPU 0 串行与 `--batch-size 4` T2–T5 prefill 提速 2.23×–2.71×，4 组配置贪心 100% 一致）；(3) 原生 C++/CUDA Wan2.2 **M2** 落地多 shard Safetensors 加载、CPU Unigram `Umt5Tokenizer` 与 CUDA BF16 **UMT5-XXL 文本编码器**（SM90 GPU 6 中/英/空三提示词 `token_exact=true`、`cosine=0.99973–0.99998`）；(4) `native/umm_asr_native` 重构为读取 `config.json` 的配置驱动 `AsrModelConfig`（兼容 0.6B 与 1.7B，SM90 GPU 1 0.6B 中英双音频逐字零回归）；
> - **P2**：Wave 5 全栈真权重闭环与原生 Wan2.2 端到端落地——(1) `native/stage_cli` 提取 `parse_sampling_params` 修复有符号 `int64` `seed` 契约并在 SM90 GPU 0 实测 `/v1/chat/completions`（串行与 `--batch-size 4`）正负 seed 100% 复现且互异、三轮 `prefix_reused` 命中 `13→51`；(2) `native/tts` 改为 `config.json` 驱动（新增 `TtsModelConfig` 与 1.7B `small_to_mtp_projection`、修复 `kEps=0` 截断与 `past_hidden` 未 norm），SM90 GPU 7 实测魔搭 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 与 0.6B 中英合成、流式 `identical=1`、`/v1/audio/speech` 及 ASR 回环逐字通过；(3) `native/umm_asr_native` 新增 `model.safetensors.index.json` 多 shard 加载，SM90 GPU 1 实测魔搭 2-shard `Qwen3-ASR-1.7B` vs 0.6B 四音频对照与 `/v1/audio/transcriptions` 端到端；(4) 原生 C++/CUDA Wan2.2 **M3 DiT 30 层 BF16 主干**（SM90 GPU 6 单步对齐 Diffusers `cosine=0.999879`、`PSNR=55.18 dB`、`235 ms`、`9,604 MiB`）与 `VideoEngine::kCuda`（UMT5-XXL + DiT CFG FlowMatch + 3D Causal VAE + H.264 MP4）全原生端到端视频生成实测通过（352×256 9 帧 12 步 `4.28 s / 20,857 MiB`），并已接入 `umm-native-stage-video` CUDA 构建；
> - **P2**：Wave 6 生产化打磨与四模态纵深升级——(1) `native/video` 落地 Diffusers 精确对齐的二阶 `UniPC-BH2` 流匹配求解器（6 步 `<1e-5`）并打通 `umm-infer serve`→`umm-native-stage-video --video-backend cuda` 全 HTTP 链路（SM90 GPU 6 实测 480×320 `7.08 s`、704×480 29 帧 `25.64 s` 与 CFG 5/20 中途 `/abort` 恢复）；(2) 新增 Apache-2.0 `python/umminfer_flux2_stage.py`（SM90 GPU 7 实测 `FLUX.2-klein-4B` 512×512 `0.58–0.66 s`、1024×1024 `1.81 s / 20.6 GiB`、i2i 编辑 `1.69 s` 与第 5/50 步 `/abort`），并在 `python/umminfer_service.py` 补齐 `flux2`/`video` `STAGE_KINDS`、Python `launcher` 启动路径与 `--video-entry` 自动连线；(3) `native/qwen38/qwen38_gemm.cu` 小-m 解码 GEMV 改为 1 warp/行 + 输入寄存器驻留（`m=1..8` 逐元素 `maxabs=0`，SM90 GPU 0 步长 `m=1: 57.2→24.1 ms`、`m=4: 74.8→50.8 ms`、`m=8: 104.0→78.1 ms / 102.4 agg tok/s`，54 例多模态基准 `54/54` 逐字节一致且整体 p95 `6,075→3,244 ms`）；(4) `native/tts` 落地原生 C++ FP32 ECAPA-TDNN `SpeakerEncoder` 与 `Qwen3-TTS-12Hz-1.7B-Base` 零样本语音克隆（SM90 GPU 1 对齐 PyTorch BF16 `cosine=0.999996/0.999997`，中英克隆说话人相似度 `0.9931/0.9917`、ASR 回环逐字匹配，CustomVoice 0.6B/1.7B 零回归）；
> - **P2**：Wave 7 三层架构收敛与 HTTP 最后一公里打通——(1) **控制面层**：`UnifiedEngine` 收敛单产物 leg 提交流（`submit_single_output_leg_locked` 与 `Record::for_each_single_output_leg`），保持流式多句 TTS 路径独立，`emit_tts_chunk_locked` 动态透传 `sample_rate`/`channels`；HTTP 边界抽取 `decode_wav_ref_audio` 与 `run_sync_coordinator_request`，将 `valid_mixed_output` 拆为三模态独立解析器并全链路透传 `image.{steps,image}`、`audio.ref_audio`、`/v1/audio/speech` `ref_audio` 与 `video.{num_frames,steps,guidance_scale}`；(2) **Stage/Worker 层**：`native/tts` 抽取内存 WAV 解析 `read_wav_pcm16_mono_bytes` 与 `TtsEngine::is_base()`，`umm-native-stage-tts` 接通 `--default-ref-audio` 与 Base 零样本克隆分流，`stage_common.hpp` 提供 `log_stage_abort` 并在 `image_stage.cpp` 收敛 2512/2.1 请求信封（同时明确拒绝 `inputs.image` 编辑并指引 `flux2`）；(3) **Web UI**：`web/index.html` 新增折叠高级选项面板透传采样、FLUX.2 图生图、TTS `ref_audio` 与视频参数；SM90 空闲 GPU 1/7 实测通过 `/v1/audio/speech` `Qwen3-TTS-1.7B-Base` `ref_audio` 克隆与 `/v1/chat/completions` `FLUX.2` 文生图/图生图编辑；
> - **P2**：Wave 8 工程打磨与 SM90 常驻服务 `43301` 五阶段全栈上线——(1) `native/video/wan_vae_kernels.cu` 将 Wan2.2 3D-VAE 改为逐潜帧精确因果流式解码（每层 `kd=3` 卷积维护 2 帧 `CausalCache`，与 Diffusers `AutoencoderKLWan._decode` 数学等价且零拼接缝），SM90 GPU 7 实测 384×256 13 帧 `PSNR=128.43 dB`（峰值 `4.28 GiB`），1280×704 77 帧峰值收敛至 `14.21 GiB` 彻底消除 OOM；(2) SM90 常驻服务 `43301` 完成带时间戳备份的灰度上线（GPU 2 `Qwen3-ASR-1.7B`、GPU 3 `Qwen3.8-27B` `--batch-size 8`、GPU 4 `FLUX.2-klein-4B`、GPU 5 `Qwen3-TTS-1.7B-CustomVoice`、GPU 6 `Wan2.2` 原生 CUDA 含流式 3D-VAE，挂载 `--session-history --web-root`），`native_service_live.py` 15/15 与五模态在线验收全通过；(3) 完成 7 套 SM90 模型权重配置 SHA-256 溯源归档与 `native/omni` / `native/image` 退役就绪度核对；
> - **P2**：Wave 9 全客户端表面契约对齐与线上热升级/77 帧闭环实测——(1) C++ `OpenAIClient`（新增 `speech()` 及 `ImageOutput{steps,image}`/`AudioOutput{ref_audio}`/`VideoOutput{num_frames,steps,guidance_scale}`）、`umm-infer` CLI（新增 `speech` 子命令与 `chat` 全模态扩展参数）、macOS `ummmlx` `ChatClient`（`ChatStreamOptions` 与三模态结构）及 `service/native-service.json`（补齐原生 `video-native` / `tts-base` 可选 Stage）全面对齐 Wave 7/8 服务端契约；(2) SM90 线上 `43301` 实测原生 `umm-infer` CLI（`speech`→`transcribe` 逐字回环、`chat` FLUX.2 文生图/图生图与 Wan2.2 视频）、仅替换网关二进制 `3 s` 热重启且全部 5 个 GPU Stage 零重启（`restarts=0`），以及空闲 GPU 7 全流程 `1280×704 77 帧` 原生 CUDA Wan2.2 端到端生成（`92.66 s`、H.264 MP4 `270,505 B`）；
> - **P2**：Wave 10 交错多模态叙事编排、全局流式语音与原位图↔视互换——(1) 新增独立零依赖 `NarrativeStreamParser`（`src/pipeline/narrative_parser.{hpp,cpp}`）与 `UnifiedEngine` 交错叙事模式（`narrative: {enabled, visual_mode, max_slots}`），在 VLM 生成途中剥离 `[[image|video|visual:...]]` 标记、实时发 `visual_slot` 并派发 `:slot:N`、解除 `audio.stream=true` 与图/视互斥，同时新增 `POST /v1/visual/render` 单阶段渲染端点复用 `AdmissionGuard` + `run_sync_coordinator_request`；(2) Web UI（`web/index.html`）与 macOS 原生 App（`clients/ummmlx`，以 AppKit `AVPlayerView` 替代 `_AVKit_SwiftUI` 杜绝 SwiftPM 崩溃）均升级为段落与内联视觉槽交错排版、卡片原位 `→ Video / → Image` 切换、顶部全局流式语音控条与 VLM 传图/音频输入；(3) SM90 线上 `43301` `3 s` 零 Stage 重启热升级并通过 `tests/narrative_live_e2e.py` 实测（TTFT `0.58 s`、全局语音首包 TTFA `0.78 s`、FLUX.2 `slot-0` 中途 `1.46 s` 出图、Wan2.2 `slot-1` `7.96 s` 出 MP4、原位图转视频 H.264 `ffprobe` PASS）；
> - **P2**：Wave 11 原生 C++/CUDA Wan2.2-TI2V-5B I2V（让 FLUX.2 插图原位动起来）——(1) `native/video/wan_vae_kernels.cu` 实现 FP32 单帧 3D-VAE 编码器 `encode_wan_vae_image_fp32`（SM90 GPU 7 对拍 Diffusers `cosine=1.000000` / `PSNR=134.07 dB` / `217 ms`），`wan_dit_kernels.cu` 与 `engine.cpp` 落地 `expand_timesteps` 逐帧时间步与首帧潜变量锚定（T2V 路径逐字节零回归、704×480 29 帧首帧保真 `PSNR=34.05 dB` / `frame0_to_last_l1=75.17`）；(2) 控制面抽取共享 `validate_image_data_url` 统一打通 `video.image`、`POST /v1/visual/render` (`kind=video` + `image`) 与 CLI `--video-image`；(3) Web UI 与 macOS `ummmlx` 卡片一键 **`Animate → Video`** 自动将已生成的 FLUX.2 PNG 送入原生 I2V，并在 SM90 线上 `43301` 仅热替换网关与 GPU 6（GPUs 2–5 零重启）完成全链路实测验收；
> - **P3**：移除私有死代码 `UnifiedEngine::SessionTurn`（`include/umminfer/pipeline/unified.hpp`）。

剩余未在当前范围改动的架构与整洁性债务（后续优先级见 [docs/roadmap.md#next-priorities](roadmap.md#next-priorities)）：

- **P3 · 有意保留项与测试覆盖边界**：
  - `include/umminfer/pipeline/unified.hpp:214-224` 两组等价公开访问器（`image_output_enabled`/`image_enabled`、`tts_output_enabled`/`tts_enabled`）为保持公开头文件兼容而保留；
  - `native/{omni,qwen38}/{stb_image.h,unicode_ranges.hpp}` 仅为 2~3 行 `#include "../common/..."` 转发壳，为兼容各引擎独立 include 路径而保留；
  - `src/proto/messages.cpp:40-81` 的旧版 `DataRef.header` 宽松 Base64 解码器接受无填充（如 `Zg`）与非规范尾位输入，为保持对旧发送端的公开协议兼容而保留，未替换为严格的 `util::base64_decode`；
  - `src/main.cpp:154` 与 `native/stage_cli/main.cpp:43` 的 15 行 `ensure_ipc_parent` 位于两个独立可执行文件入口，避免为少量代码引入额外公共头耦合；
  - `SimpleScheduler` 批处理/墓碑、流式 TTS 句切分与字节上限中止、跨会话 `cancel_for_session` 403、`/v1/models`、`--web-root`、上游媒体代理、控制面消息解析、`EnergyVad`、`umminfer_service.py` 守护进程、CLI `image`/`video`、macOS `Theme` 与 Web UI 均已有离线自动化回归；`/v1/audio/speech` 已完成 Linux SM90 GPU 真实 `umm-native-stage-tts` 端到端实测（WAV/PCM 均 200 且非静音）。
