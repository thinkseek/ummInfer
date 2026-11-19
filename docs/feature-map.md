# ummInfer 功能映射

基线 commit `a213514ea12b32bb971ed49607fde06bc1a6f993`（当前 main）。功能清点在 `973e280` 完成；`a213514` 只在 `src/serve/openai_api.cpp` 与 `tests/multimodal_benchmark.py` 增删行，锚点已按 `a213514` 重新定位。下表全部 `文件:行号` 只在 `a213514` 下有效；后续提交会使行号漂移。

校验：`python3 -I tools/check_feature_map.py`。脚本逐条打开文件，确认行号存在、非空，且反引号内符号确实出现在该行；同时校验「验证」列引用的文件与行号。

状态：`已验证` = 仓库内有测试文件对该行为做了断言；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。GPU 与真实模型用例本轮未运行，只按测试文件举证。

状态：`已验证` = 仓库内有测试文件对该行为做出断言；`部分` = 测试只覆盖该功能的一部分；`已实现未验` = 无测试断言；`未实现` = 代码中缺少该能力。本轮不执行需要 GPU 或权重的验证，状态只引用仓库内已有测试文件。

状态定义：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分路径；`已实现未验` = 无测试断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 仓库内无测试断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。本轮不运行需要 GPU 的用例。

状态定义：`已验证` = 仓库内有测试文件对该行为做了断言；`部分` = 测试只覆盖其中一部分；`已实现未验` = 仓库内没有测试断言该行为。需要 GPU 或模型权重的测试只按测试文件举证，本轮未运行。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖该行为的一部分；`已实现未验` = 无测试断言该行为。判定只看仓库内测试文件的断言，不看代码能否跑通；需要 GPU 或模型权重的测试本轮未执行。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试文件断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖该行为的一部分；`已实现未验` = 无测试断言该行为。

状态：`已验证` = 仓库内有测试文件对该行为做了断言；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为；`未实现` = 只有占位或返回 501 的路由。

状态定义：`已验证` = 仓库内有测试文件对该行为做了断言；`部分` = 测试只覆盖该行为的一部分；`已实现未验` = 无测试断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。

状态：`已验证` 仓库内有测试文件断言该行为；`部分` 测试只覆盖该行为的一部分；`已实现未验` 无测试断言该行为。

状态取值：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖该功能的一部分；`已实现未验` = 仓库内无测试断言该行为；`未实现` = 路由或入口明确返回未实现。

状态定义：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。



状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 仓库内无测试断言该行为；`未实现` = 代码未提供该能力。

状态判定只看仓库内的测试文件：`已验证` 表示有测试断言该行为；`部分` 表示测试只覆盖其中一部分；`已实现未验` 表示没有任何测试断言该行为。需要 GPU 或模型权重的用例本轮未执行，只按测试源码举证。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言。需要 GPU 或模型权重的验证本轮未执行，仅按测试文件举证。

状态：`已验证` = 仓库内有测试文件断言该行为；`部分` = 测试只覆盖其中一部分；`已实现未验` = 无测试断言该行为。

## 1. 对外接口

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 健康与就绪探针 /health、/ready | `src/serve/openai_api.cpp:581` `server_.Get("/health", [this](const httplib::Request&, h` | `src/serve/openai_api.cpp:598` `const bool entry_ready = !config_.external_entry_stage` | GET /health -> {status:ok,in_flight}; GET /ready -> 200/503 | tests/http_smoke.py:77 | 已验证 |
| Prometheus 指标导出 /metrics | `src/serve/openai_api.cpp:592` `server_.Get("/metrics", [](const httplib::Request&, http` | `src/serve/openai_api.cpp:593` `res.set_content(obs::Metrics::instance().render()` | GET /metrics, text/plain; version=0.0.4 | tests/soak_native.py:263 | 已验证 |
| Bearer API Key 鉴权 | `src/serve/openai_api.cpp:564` `server_.set_pre_routing_handler(` | `src/serve/openai_api.cpp:571` `if (util::constant_time_equals(auth, expected))` | Authorization: Bearer <key>, 失败 401 auth_error | tests/http_smoke.py:76 | 已验证 |
| 过载 503 与 Retry-After | `src/serve/openai_api.cpp:91` `class AdmissionGuard` | `src/serve/openai_api.cpp:527` `void reject_overloaded(httplib::Response& res)` | 503 + Retry-After: 1 + type=overloaded | tests/unified_contract_test.cpp:1385 | 已验证 |
| 聊天补全非流式 JSON | `src/serve/openai_api.cpp:617` `server_.Post("/v1/chat/completions"` | `src/serve/openai_api.cpp:1369` `json out =` | POST /v1/chat/completions, stream=false | tests/http_smoke.py:97 | 已验证 |
| 聊天补全 SSE 流式 | `src/serve/openai_api.cpp:617` `server_.Post("/v1/chat/completions"` | `src/serve/openai_api.cpp:1420` `res.set_chunked_content_provider(` | stream=true; choice delta; 末尾 data: [DONE] | tests/http_smoke.py:103 | 已验证 |
| 模型列表 /v1/models | `src/serve/openai_api.cpp:608` `server_.Get("/v1/models", [this](const httplib::Request&` | `src/serve/openai_api.cpp:609` `json body =` | GET /v1/models -> object=list,data[].id | 无 | 已实现未验 |
| 语音转写 /v1/audio/transcriptions | `src/serve/openai_api.cpp:862` `void handle_transcription(const httplib::Request& req, h` | `src/serve/openai_api.cpp:986` `request_id = coordinator_.submit(std::move(payload), con` | POST /v1/audio/transcriptions multipart file,model,stream | tests/multimodal_benchmark.py:273 | 已验证 |
| 请求中止 /abort | `src/serve/openai_api.cpp:735` `server_.Post("/abort", [this](const httplib::Request& re` | `src/serve/openai_api.cpp:768` `unified_engine_->cancel_for_session(request_id, session_` | POST /abort {request_id,session_id} -> {success:true}; 403 会话不符 | tests/unified_contract_test.cpp:3158 | 已验证 |
| 语音合成路由 /v1/audio/speech 返回 501 | `src/serve/openai_api.cpp:786` `server_.Post("/v1/audio/speech"` | `src/serve/openai_api.cpp:788` `res.status = 501` | POST /v1/audio/speech -> 501 speech is not implemented | 无 | 已实现未验 |
| 图片生成/编辑与视频任务上游代理 | `src/serve/openai_api.cpp:645` `server_.Post("/v1/images/generations"` | `src/serve/openai_api.cpp:657` `proxy_.handle_image_generation(req, res)` | POST /v1/images/generations/edits; POST/GET/DELETE /v1/videos | 无 | 已实现未验 |
| 静态站点托管 --web-root | `src/main.cpp:310` `api_config.web_root = flag(flags, "web-root")` | `src/serve/openai_api.cpp:798` `!server_.set_mount_point("/", config_.web_root))` | --web-root DIR 挂载 /；CORS 头 - | 无 | 部分 |
| WebSocket 实时握手校验 | `src/serve/realtime_api.cpp:115` `bool valid_handshake(lws* wsi, const std::string& served` | `src/serve/realtime_api.cpp:118` `if (path != "/v1/realtime")` | GET /v1/realtime?intent=transcription&model=<served>; 否则 400 | tests/realtime_contract_e2e.py:313 | 已验证 |
| 实时会话配置与转写事件 | `src/serve/realtime_api.cpp:571` `void update_session(lws* wsi, RealtimeSession* session` | `src/serve/realtime_api.cpp:816` `json completed =` | session.update -> transcription_session.updated; transcription.segment/completed | tests/realtime_contract_e2e.py:220 | 已验证 |
| 实时语音回复 audio.delta 推流 | `src/serve/realtime_api.cpp:842` `void start_response(RealtimeSession* session, const std:` | `src/serve/realtime_api.cpp:893` `{{"type", "audio.delta"}` | audio.delta {segment_id,format:pcm16,sample_rate,channels,audio(base64)} | tests/full_duplex_l2_live.py:290 | 已验证 |
| SSE 事件携带 session_id 与请求字段解析 | `src/serve/openai_api.cpp:278` `json unified_event_json(const UnifiedEvent& event)` + `src/serve/openai_api.cpp:617` `server_.Post("/v1/chat/completions"` | `src/serve/openai_api.cpp:282` `{"session_id", event.ses` + `src/serve/openai_api.cpp:1279` `inputs["session_id"] = Value(body["session_id"].get<std:` | SSE field session_id | tests/unified_contract_test.cpp:2730 | 已验证 |

## 2. 请求编排

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 统一请求提交与准入上限 | `src/pipeline/unified.cpp:172` `std::string UnifiedEngine::submit(StagePayload payload)` | `src/pipeline/unified.cpp:189` `if (admission_.size() >= config_.max_in_flight)` | inputs.messages / modalities / output_modalities; max_in_flight; 超限抛 QueueFullError | tests/unified_contract_test.cpp:1287 | 已验证 |
| 请求契约校验与非法模态拒绝 | `src/pipeline/unified_schema.cpp:81` `ValidatedSubmission validate_submission(const Value& inp` | `src/pipeline/unified_schema.cpp:98` `if (*modality == UnifiedModality::kVideo)` | image_url.url 非空、至多一个 image_url、audio_options.voice 必填、session_history 需显式 session_id | tests/unified_contract_test.cpp:1581 | 已验证 |
| ASR/VLM 增量事件、无缺口 seq 与自动交接 VLM | `src/pipeline/unified.cpp:364` `void UnifiedEngine::handle_stream(const StreamMessage& m` + `src/pipeline/unified.cpp:496` `void UnifiedEngine::start_vlm_locked(const std::string&` | `src/pipeline/unified.cpp:385` `event.seq = record.next_seq++` + `src/pipeline/unified.cpp:516` `coordinator_.submit(vlm_payload, config_.vlm_stage)` | asr_partial / asr_final / vlm_delta; chunk.text, chunk.is_final, event.seq | tests/unified_contract_test.cpp:789 | 已验证 |
| 取消请求并传播到子 leg | `src/pipeline/unified.cpp:259` `UnifiedEngine::CancelStatus UnifiedEngine::cancel_for_se` | `src/pipeline/unified.cpp:285` `for (const std::string& id : active_ids) coordinator_.ab` | cancel(request_id); abort 广播到 vlm/image/tts 全部已提交子请求 | tests/unified_contract_test.cpp:1151 | 已验证 |
| 按会话归属校验取消权限 | `src/pipeline/unified.cpp:259` `UnifiedEngine::CancelStatus UnifiedEngine::cancel_for_se` | `src/pipeline/unified.cpp:268` `it->second.session_id != expected_session_id)` | request_id, session_id, CancelStatus::kSessionMismatch | 无 | 已实现未验 |
| 终态等待与请求记录释放 | `src/pipeline/unified.cpp:308` `bool UnifiedEngine::release(const std::string& request_i` | `src/pipeline/unified.cpp:341` `coordinator_.release(request_id)` | wait(request_id, timeout_s) 阻塞到终态回调完成；release() 删除记录与子 leg | tests/unified_contract_test.cpp:3979 | 已验证 |
| 图片与语音输出并行扇出 | `src/pipeline/unified.cpp:866` `void UnifiedEngine::start_outputs_locked(const std::stri` | `src/pipeline/unified.cpp:871` `if (record.image_leg.enabled) start_image_locked(request` | output_modalities 含 image/audio；子请求 id 为 <id>:image 与 <id>:tts | tests/unified_contract_test.cpp:3790 | 已验证 |
| 非流式产物按 2 MiB 分片 | `src/pipeline/unified_schema.cpp:319` `std::vector<UnifiedEvent> slice_artifact_events(` | `src/pipeline/unified_schema.cpp:371` `event.kind = UnifiedEvent::Kind::kArtifact` | artifact_id, index, total; kMaxArtifactImageBytes / kMaxArtifactAudioBytes / kMaxArtifactAggregateBytes | tests/unified_contract_test.cpp:4028 | 已验证 |
| Coordinator 提交到入口 stage | `src/pipeline/pipeline.cpp:751` `std::string Coordinator::submit(StagePayload payload` | `src/pipeline/pipeline.cpp:812` `control_plane_->submit_to_stage(entry_stage, endpoint, M` | entry_stage 名; 自动 request_id = req-N; 未 ready 时进入 pending_submits_ | tests/unified_contract_test.cpp:789 | 已验证 |
| Stage 接收提交、交给调度器并回传失败 | `src/pipeline/pipeline.cpp:348` `void Stage::on_submit(const SubmitMessage& msg)` + `src/pipeline/pipeline.cpp:444` `void Stage::fail_request(const std::string& request_id` | `src/pipeline/pipeline.cpp:381` `scheduler_->enqueue(IncomingMessage` + `src/pipeline/pipeline.cpp:457` `control_plane_->send_complete(done)` | submit message → new_request；payload.data 为空时用 request.inputs | 无 | 已实现未验 |
| Coordinator 事件回调转发 | `src/pipeline/pipeline.cpp:852` `void Coordinator::handle_event(const Message& msg)` | `src/pipeline/pipeline.cpp:940` `if (on_complete) on_complete(snapshot)` | on_stream / on_complete sink；仅 pending/running 请求可收敛 | tests/unified_contract_test.cpp:2148 | 已验证 |
| 调度器批量收集与取消墓碑 | `src/scheduling/simple_scheduler.cpp:128` `void SimpleScheduler::collect_batch(std::vector<Incoming` | `src/scheduling/simple_scheduler.cpp:175` `batch.push_back(std::move(*next))` | max_batch_size, max_batch_wait_ms, max_batch_cost；abort 墓碑 abort_retention | 无 | 已实现未验 |
| 控制面消息解析与版本校验 | `src/proto/messages.cpp:889` `Message parse_message(const Value& v)` | `src/proto/messages.cpp:896` `if (type == "data_ready") return DataReadyMessage::from_` | type=data_ready/stream/complete/submit/abort/stage_ready/admin; protocol_version=1 | tests/unified_contract_test.cpp:789 | 部分 |

## 3.1 原生 stage：ASR

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 启动时加载 Qwen3-ASR 模型 | `native/stage_cli/asr_stage.cpp:31` `CancellableComputeFn make_compute(const StageOptions& op` | `native/stage_cli/asr_stage.cpp:34` `auto engine = std::make_shared<native::AsrEngine>(option` | --kind asr --model DIR --gpu N; logs "ready kind=asr" | tests/full_duplex_live.py:169 | 已验证 |
| 单请求 ASR 推理编排与音频输入解析 | `native/stage_cli/asr_stage.cpp:40` `return [engine, tokenizer, frontend, max_new_tokens, gpu` + `native/stage_cli/stage_common.cpp:181` `AsrWork parse_asr_work(const Value& work)` | `native/stage_cli/asr_stage.cpp:54` `ids = engine->transcribe_tokens(mel, max_new_tokens` + `native/stage_cli/stage_common.cpp:195` `native::PcmAudio parsed = native::load_wav_pcm16(bytes)` | inputs.audio -> {"text","generated_token_count","finish_reason"} | tests/full_duplex_live.py:234 | 部分 |
| WAV 容器解析与 128 维 log-mel 特征 | `native/umm_asr_native/wav.cpp:24` `PcmAudio load_wav_pcm16(std::string_view wav)` + `native/umm_asr_native/feature_frontend.cpp:97` `MelFeatures WhisperMel::compute(const std::vector<float>` | `native/umm_asr_native/wav.cpp:70` `int16_t s = static_cast<int16_t>(read_u16(wav, data_off` + `native/umm_asr_native/feature_frontend.cpp:153` `acc += static_cast<double>(mel_bank_[b * kMels + m]) *` | 仅接受 PCM format 1 / mono / 16-bit / 16 kHz | tests/full_duplex_live.py:162 | 部分 |
| 词表加载与字节级文本反解码 | `native/umm_asr_native/tokenizer.cpp:47` `QwenTokenizer::QwenTokenizer(const std::string& vocab_js` | `native/umm_asr_native/tokenizer.cpp:93` `if (it != unicode_to_byte_.end()) bytes.push_back(static` | vocab.json 映射 token->id；special id 被跳过 | 无 | 已实现未验 |
| Whisper 音频塔 18 层与 28 层文本解码前向 | `native/umm_asr_native/engine.cpp:114` `int audio_forward(int valid_frames, const AbortCheck& ab` + `native/umm_asr_native/engine.cpp:180` `void text_layer(int layer, int m, int history)` | `native/umm_asr_native/engine.cpp:146` `gqa_attention(d_au_q, d_au_k, d_au_v, d_au_norm, valid_t` + `native/umm_asr_native/engine.cpp:204` `gqa_attention(d_q, kv + kv_offset(layer, false, 0)` | window_len 104, 双向非因果 | tests/full_duplex_live.py:162 | 部分 |
| 贪心解码循环与 ASR 提示词 | `native/umm_asr_native/engine.cpp:355` `std::vector<int> AsrEngine::transcribe_tokens(const MelF` | `native/umm_asr_native/engine.cpp:395` `for (int step = 0` | max_new_tokens 默认 256；finish_reason length/stop | tests/full_duplex_live.py:234 | 部分 |
| ASR 请求取消 | `native/umm_asr_native/engine.cpp:163` `if (abort && abort())` | `native/umm_asr_native/engine.cpp:164` `throw RequestAborted("aborted in audio tower layer " + s` | abort endpoint；异常信息 "aborted ..." | tests/full_duplex_live.py:282 | 部分 |
| ASR 结果字段组装与每请求耗时日志 | `native/stage_cli/stage_common.cpp:436` `Value asr_result(const std::string& text, int generated_` + `native/stage_cli/asr_stage.cpp:73` `const auto elapsed = std::chrono::duration_cast<std::chr` | `native/stage_cli/stage_common.cpp:439` `out["text"] = Value(text)` + `native/stage_cli/asr_stage.cpp:75` `std::cerr << "asr request request_id=" << req_id` | text / generated_token_count / finish_reason | tests/full_duplex_live.py:296 | 部分 |
| ASR CUDA 核：GQA/窗口注意力、BF16 GEMM、conv3x3s2+mel | `native/umm_asr_native/cuda_kernels.cu:413` `void gqa_attention(const Bf16* q, const Bf16* k, const B` + `native/umm_asr_native/cuda_kernels.cu:498` `void gemm_bf16_naive(Bf16* y, const Bf16* x, const Bf16*` + `native/umm_asr_native/cuda_kernels.cu:353` `void conv3x3s2_gelu(const Bf16* in, const Bf16* weight` | `native/umm_asr_native/cuda_kernels.cu:189` `__global__ void gqa_attn_kernel(const Bf16* q, const Bf1` + `native/umm_asr_native/cuda_kernels.cu:474` `__global__ void gemm_naive_kernel(Bf16* y, const Bf16* x` + `native/umm_asr_native/cuda_kernels.cu:160` `__global__ void conv_kernel(const Bf16* in, const Bf16*` | causal=true 解码；false 时按 window_len 分窗 | tests/full_duplex_live.py:162 | 部分 |

## 3.2 原生 stage：Omni / VLM

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| VLM 单图问答请求 | `native/stage_cli/omni_stage.cpp:76` `CancellableComputeFn make_compute(const StageOptions& op` | `native/omni/omni_engine.cu:218` `OmniEngine::Result OmniEngine::generate(const std::vecto` | messages[].content[].type="image_url" | tests/soak_native.py:422 | 部分 |
| 视觉塔 32 层前向与特征池化 | `native/omni/vision_forward.cu:61` `void vision_forward(const EngineWeights& w, EngineScratc` | `native/omni/vision_forward.cu:118` `for (int blk = 0` | thinker.visual.*; pooler[rows/4,2048] | 无 | 已实现未验 |
| 图片解码与 smart resize 归一化 | `native/omni/image_preprocess.cpp:129` `PreprocessedImage preprocess_rgb(const uint8_t* rgb, int` | `native/omni/image_preprocess.cpp:97` `void smart_resize(int height, int width, int factor, int` | pixel_values[grid_t*grid_h*grid_w,1536]; 1/127.5-1 归一化 | 无 | 已实现未验 |
| 图像区 M-RoPE 位置编码 | `native/omni/text_decoder.cu:158` `PositionIds build_positions(const std::vector<int>& ids` | `native/omni/text_decoder.cu:182` `p.h[img_start + k] = st_idx + k / llm_gw` | image_pad/vision_start/vision_end token ids | 无 | 已实现未验 |
| MoE 文本解码层推理 | `native/omni/text_decoder.cu:45` `void text_layer(const EngineWeights& w, EngineScratch& s` | `native/omni/text_decoder.cu:97` `moe_route(s.router_logits, s.router_idx, s.router_w, m` | thinker.model.layers.N.*; top-k router | tests/soak_native.py:422 | 部分 |
| 逐 token UTF-8 安全流式输出 | `native/stage_cli/omni_stage.cpp:140` `auto emit_token = [&tokenizer, &pending_utf8](int id)` | `native/stage_cli/omni_stage.cpp:146` `emit_stage_stream(Value(std::move(chunk)))` | stage stream chunk {"text": ...} | tests/soak_native.py:422 | 部分 |
| omni 请求取消与中止传播 | `native/omni/omni_engine.cu:313` `if (abort && abort())` | `native/stage_cli/omni_stage.cpp:171` `throw RequestAborted(error.what())` | RequestAborted; 日志 abort_epoch_ms= | 无 | 已实现未验 |

## 3.3 原生 stage：Image

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 文生图请求与去噪采样 | `native/stage_cli/image_stage.cpp:41` `CancellableComputeFn make_compute(const StageOptions& op` | `native/image/engine.cpp:161` `for (int step = 0` | modalities=["text","image"]; image.prompt/size/seed | tests/soak_native.py:450 | 部分 |
| 图像后端 2512 / 2.1 选择 | `native/stage_cli/image_stage.cpp:42` `const ImageBackend backend = parse_image_backend(options` | `native/stage_cli/image_stage.cpp:48` `if (backend == ImageBackend::kQwen21)` | --image-backend 2512/2.1 | 无 | 已实现未验 |
| Flow-matching 噪声调度与 DiT 条件去噪 | `native/image/scheduler.cpp:10` `FlowMatchSchedule build_schedule(const SchedulerConfig&` + `native/image/dit.cpp:112` `DeviceTensor DiT::forward(DeviceTensor& img_packed, cons` | `native/image/scheduler.cpp:15` `const double m = (cfg.max_shift - cfg.base_shift) /` + `native/image/engine.cpp:169` `cfg_combine(vc.data, vc.data, neg_v.data, gp.guidance, p` | sigmas/timesteps; cfg.shift_terminal | 无 | 已实现未验 |
| VAE latent 解码与 PNG 产物编码 | `native/image/vae.cpp:94` `DeviceTensor VaeDecoder::decode(const DeviceTensor& lat` + `native/image/png.cpp:35` `std::string encode_png(const RgbImage& img)` | `native/image/engine.cpp:204` `DeviceTensor rgb = vae.decode(lat_denorm, latent_h, late` + `native/stage_cli/image_stage.cpp:144` `std::string png = image::encode_png(rgb)` | latents_mean/latents_std; 16ch -> RGB | 无 | 已实现未验 |
| Qwen-Image-2.1 后端推理 | `native/image2/engine.cpp:77` `GenerateResult Image2Engine::generate(const GeneratePara` | `native/image2/engine.cpp:228` `DeviceTensor rgb = vae.decode(denorm, gh, gw)` | --image-backend 2.1; vae/config.json latents_mean/latents_std | 无 | 已实现未验 |
| qwen38 VLM 连续批处理 | `native/stage_cli/qwen38_stage.cpp:59` `CancellableComputeFn make_compute(const StageOptions& op` | `native/qwen38/engine.cu:1323` `GenResult Engine::generate_batched(const BatchRequest& r` | --batch-size N; BatchRequest | 无 | 已实现未验 |

## 3.4 原生 stage：TTS

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| TTS Stage 计算入口 | `native/stage_cli/tts_stage.cpp:31` `CancellableComputeFn make_compute(const StageOptions& op` | `native/stage_cli/tts_stage.cpp:97` `samples = engine->synthesize(` | inputs: text/voice/stream; params: max_frames, gpu | 无 | 已实现未验 |
| 音色选择、非法音色回退与中英语种判定 | `native/stage_cli/tts_stage.cpp:51` `std::string voice = input.voice.empty() ? "vivian" : inp` + `native/stage_cli/stage_common.cpp:383` `bool text_is_mostly_cjk(const std::string& text)` | `native/stage_cli/tts_stage.cpp:60` `const int language = text_is_mostly_cjk(input.text) ? na` | voice: vivian/ryan; 非法值时回退 vivian 并写 stderr | 无 | 已实现未验 |
| 流式 PCM16 音频分块输出 | `native/stage_cli/tts_stage.cpp:72` `auto sink = [&chunks, &streamed_samples](const std::vect` | `native/stage_cli/tts_stage.cpp:92` `emit_stage_stream(Value(std::move(out)))` | chunk fields: audio,index,first_frame,frames,sample_rate=24000,channels=1,format=pcm16 | 无 | 已实现未验 |
| 流式块与终态波形逐样本一致校验 | `native/stage_cli/tts_stage.cpp:112` `if (input.stream)` | `native/stage_cli/tts_stage.cpp:119` `native::stream_matches(streamed_samples, samples, &first` | 日志 tts stream_check identical=1; 不一致抛 ProtocolError | 无 | 已实现未验 |
| 合成尾部噪声裁剪与淡出 | `native/tts/tail_trim.cpp:75` `std::size_t trailing_noise_cut(const std::vector<float>&` | `native/tts/tail_trim.cpp:117` `if (high_band_ratio(samples.data() + cursor, end - curso` | 阈值 RMS<60 PCM16, ZCR>=0.4, 噪声 30-600 ms, 高频带>=0.21 | native/tts/tail_trim_test.cpp:51 | 已验证 |
| 文本 BPE 编码与 TTS prompt 序列构建 | `native/tts/bpe_tokenizer.cpp:180` `std::vector<int> Qwen2BpeTokenizer::encode(const std::st` + `native/tts/prompt.cpp:23` `std::vector<int> c =` | `native/tts/bpe_tokenizer.cpp:209` `for (const std::string& chunk : pretokenize(run))` + `native/tts/prompt.cpp:40` `plan.rows.push_back(` | vocab.json / merges.txt / added_tokens_decoder | 无 | 已实现未验 |
| Talker 自回归生成与流式分块节奏 | `native/tts/engine.cpp:334` `std::vector<float> TtsEngine::synthesize(const std::vect` + `native/tts/engine.cpp:508` `auto emit = [&](const std::vector<float>& chunk, int fir` | `native/tts/engine.cpp:606` `const int next = select_c0(allow_eos, step + 1, ban_id)` + `native/tts/engine.cpp:568` `if (on_chunk && frame_count >= next_stream_frame)` | codec vocab 3072; EOS(kCodecEos) 停止; max_frames 默认 8192 | 无 | 已实现未验 |
| 合成中途取消 | `native/tts/engine.cpp:554` `if (abort && abort())` | `native/tts/engine.cpp:555` `throw RequestAborted("aborted at frame " + std::to_strin` | RequestAborted boundary="aborted at frame N"; 日志 tts request aborted | 无 | 已实现未验 |
| RVQ codec 波形解码 | `native/tts/codec.cpp:135` `std::vector<float> CodecDecoder::decode(const std::vecto` | `native/tts/codec.cpp:416` `conv1d_causal_f32(im.b, im.wt("decoder.decoder.6.conv.we` | codes 必须 [T,16], id∈[0,2047]; 输出 T*1920 个 float | 无 | 已实现未验 |
| WAV 封装与 TTS artifact 描述符 | `native/tts/wav_io.cpp:25` `std::string write_wav_pcm16_mono(const std::vector<float` + `native/stage_cli/stage_common.cpp:458` `Value tts_result(std::string wav_bytes, int sample_rate` | `native/tts/wav_io.cpp:44` `long s = std::lrintf(x * 32767.0f)` + `native/stage_cli/stage_common.cpp:463` `out["media_type"] = Value("audio")` | RIFF/WAVE, PCM16, mono | 无 | 已实现未验 |
| TTS 命令行单次合成 | `native/tts_cli/main.cpp:29` `int main(int argc, char** argv)` | `native/tts_cli/main.cpp:63` `const std::vector<float> wav = engine.synthesize(` | --model --text --out --speaker --language --gpu --max-frames --stream | 无 | 已实现未验 |

## 3.5 原生 stage：中继

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 共享内存 / CUDA IPC 张量中继 | `src/relay/relay.cpp:291` `class CudaIpcRelay : public Relay` | `src/relay/relay.cpp:349` `cuda_check(cudaMemcpy(buffer, tensor.bytes.data()` | --relay shm/cuda_ipc | tests/cuda_bench.cpp:330 | 部分 |

## 4. 多模态输出

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| TTS leg 启动与 :tts 子请求 | `src/pipeline/unified.cpp:795` `void UnifiedEngine::start_tts_locked(const std::string&` | `src/pipeline/unified.cpp:816` `coordinator_.submit(payload, leg.stage_name)` | child_id = request_id+":tts" 或 ":tts:N"; inputs.text / inputs.stream | tests/unified_contract_test.cpp:3817 | 已验证 |
| 按 VLM 文本派发 TTS | `src/pipeline/unified.cpp:866` `void UnifiedEngine::start_outputs_locked(const std::stri` | `src/pipeline/unified.cpp:876` `start_tts_locked(request_id, record, record.vlm_text, tr` | output_modalities 含 audio; 非流式用整段 vlm_text | tests/unified_contract_test.cpp:3790 | 已验证 |
| 流式句切分 fan-out 与尾句 flush | `src/pipeline/unified.cpp:825` `void UnifiedEngine::drain_sentences_locked(const std::st` | `src/pipeline/unified.cpp:832` `size_t cut = sentence_cut(record.sentence_buf, kMaxSente` | kMaxSentenceCodepoints=48, kFirstClauseCodepoints=8, kMaxStreamedTtsLegs=32 | 无 | 已实现未验 |
| 流式音频块转 artifact 事件（含字节上限中止） | `src/pipeline/unified.cpp:882` `void UnifiedEngine::emit_tts_chunk_locked(` + `src/pipeline/unified.cpp:889` `if (size > kMaxArtifactAudioBytes` | `src/pipeline/unified.cpp:918` `event.artifact_bytes = std::make_shared<const std::strin` + `src/pipeline/unified.cpp:893` `collect_tts_aborts_locked(record, "artifact_limit", defe` | artifact: media_type=audio, mime_type=audio/pcm, format=pcm16, sample_rate=24000, channels=1, stream_index | 无 | 已实现未验 |
| TTS leg 结算与 leg_error 事件 | `src/pipeline/unified.cpp:946` `void UnifiedEngine::settle_output_leg_locked(` | `src/pipeline/unified.cpp:979` `event.kind = UnifiedEvent::Kind::kLegError` | legs state: completed/failed/aborted; 失败发 leg_error(error) | tests/unified_contract_test.cpp:3886 | 已验证 |
| 终态 legs 数组汇总 | `src/pipeline/unified.cpp:990` `Value UnifiedEngine::legs_value_locked(const Record& rec` | `src/pipeline/unified.cpp:1008` `item["state"] = Value(record.tts_failed ? "failed"` | terminal legs[]: {stage:"tts", state, error} | tests/unified_contract_test.cpp:3886 | 已验证 |
| 取消传播到全部未结算 TTS leg | `src/pipeline/unified.cpp:924` `void UnifiedEngine::collect_tts_aborts_locked(` | `src/pipeline/unified.cpp:929` `out.emplace_back(leg.child_id, reason)` | abort reason: client / vlm_failure / artifact_limit | tests/unified_contract_test.cpp:3934 | 已验证 |

## 5. 会话

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| session_id 标记事件分组 | `src/pipeline/unified.cpp:172` `std::string UnifiedEngine::submit(StagePayload payload)` | `src/pipeline/unified.cpp:193` `record.session_id = std::move(session_id)` | session_id | tests/unified_contract_test.cpp:1789 | 已验证 |
| --session-history 开启多轮历史 | `src/main.cpp:441` `unified_config.session_history.enabled = flags.count("se` | `src/pipeline/unified.cpp:156` `session_store_(to_history_policy(config_.session_history` | --session-history | tests/full_duplex_l2_live.py:184 | 部分 |
| 历史注入 VLM prompt 与仅成功轮次入库 | `src/pipeline/unified.cpp:496` `void UnifiedEngine::start_vlm_locked(const std::string&` + `src/pipeline/unified.cpp:1019` `void UnifiedEngine::on_legs_drained_locked(const std::st` + `src/pipeline/unified.cpp:519` `void UnifiedEngine::record_session_turn_locked(const Rec` | `src/pipeline/unified.cpp:507` `session_store_.prepend_history(` + `src/pipeline/unified.cpp:1028` `if (kind == UnifiedEvent::Kind::kComplete && record.hist` + `src/pipeline/unified.cpp:521` `record.transcript.empty() ? record.user_text : record.tr` | session_history, messages | tests/unified_contract_test.cpp:3317 | 已验证 |
| 会话历史按 session_id 隔离且需显式 opt-in | `src/pipeline/session_store.cpp:17` `Value::Array SessionHistoryStore::prepend_history(` + `src/pipeline/unified_schema.cpp:81` `ValidatedSubmission validate_submission(const Value& inp` | `src/pipeline/session_store.cpp:21` `const auto it = sessions_.find(session_id)` + `src/pipeline/unified_schema.cpp:224` `if (out.history_requested && !out.explicit_session)` | session_id | tests/unified_contract_test.cpp:3371 | 已验证 |
| 历史三级淘汰（轮次/字符/会话） | `src/pipeline/session_store.cpp:39` `void SessionHistoryStore::record_turn(const std::string&` | `src/pipeline/session_store.cpp:52` `while ((turns.size() > policy_.max_turns` | max_turns, max_chars, max_sessions | tests/unified_contract_test.cpp:3431 | 已验证 |
| 历史为进程内状态，重启即丢弃 | `include/umminfer/pipeline/session_store.hpp:27` `class SessionHistoryStore` | `src/pipeline/session_store.cpp:43` `std::deque<Turn>& turns = sessions_[session_id]` | - | tests/unified_contract_test.cpp:3567 | 已验证 |
| 前缀 KV 复用公共 token 前缀 | `native/stage_cli/omni_stage.cpp:92` `struct PrefixCache` | `native/stage_cli/omni_stage.cpp:135` `prefix = static_cast<int>(std::min(common, max_reusable)` | env UMM_OMNI_PREFIX_CACHE, log prefix_reused | tests/full_duplex_l2_live.py:400 | 已验证 |
| 客户端 realtime 建链 session.update | `src/client/realtime.cpp:136` `struct RealtimeSession::Impl` | `src/client/realtime.cpp:180` `socket.send_text(nlohmann::json` | WS session.update / transcription_session.updated | tests/realtime_client_contract_e2e.py:623 | 已验证 |

## 6. 客户端

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| macOS：SSE 流解析与事件切分 | `clients/ummmlx/Sources/ummmlx/ChatClient.swift:73` `struct SSEParser` | `clients/ummmlx/Sources/ummmlx/ChatClient.swift:102` `guard parts.first == "data" else` | data: 前缀 / [DONE] | clients/ummmlx/Tests/ummmlxTests/ChatClientTests.swift:7 | 已验证 |
| macOS：统一事件解码、seq 校验与产物重组 | `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:144` `static func decode(_ data: Data) throws -> UnifiedEvent` + `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:263` `mutating func ingest(_ chunk: ArtifactChunk) throws -> G` | `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:154` `let kind = UnifiedKind(rawValue: kindText) ?? .unknown` + `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:346` `let digest = Self.sha256Hex(slot.data)` | unified.seq / unified.kind | clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:45 | 已验证 |
| macOS：流式 TTS PCM 合并为 WAV | `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:428` `mutating func append(_ artifact: GeneratedArtifact) thro` | `clients/ummmlx/Sources/ummmlx/UnifiedStream.swift:444` `pcm.append(artifact.data)` | audio/pcm -> audio/wav | clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:280 | 已验证 |
| macOS：抖动缓冲与无缝播放 | `clients/ummmlx/Sources/ummmlx/AudioPlayback.swift:128` `func enqueue(_ chunk: AudioStreamChunk)` | `clients/ummmlx/Sources/ummmlx/AudioPlayback.swift:190` `node.scheduleBuffer(buffer, completionCallbackType: .dat` | UMM_AUDIO_DEBUG / AudioStreamChunk | clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:315 | 部分 |
| macOS：本地语音识别（麦克风/文件） | `clients/ummmlx/Sources/ummmlx/Speech.swift:74` `func transcribeMicrophone() -> AsyncThrowingStream<Speec` | `clients/ummmlx/Sources/ummmlx/Speech.swift:92` `input.installTap(onBus: 0, bufferSize: 4096, format: for` | SFSpeechRecognizer / requiresOnDeviceRecognition | clients/ummmlx/Tests/ummmlxTests/SpeechControllerTests.swift:45 | 部分 |
| macOS：深色/浅色主题 | `clients/ummmlx/Sources/ummmlx/Theme.swift:8` `struct Theme` | `clients/ummmlx/Sources/ummmlx/Theme.swift:12` `Color(hex: dark ? darkValue : light)` | - | 无 | 已实现未验 |
| Web：文本流式对话 | `web/index.html:570` `async function send()` | `web/index.html:646` `for await (const line of readLines(res))` | POST /v1/chat/completions / session_id | 无 | 已实现未验 |
| Web：PCM 音频播放与图片/WAV 产物渲染 | `web/index.html:314` `push(bytes, rate, channels)` + `web/index.html:448` `function deliver(chunk, bytes)` | `web/index.html:336` `source.start(at)` + `web/index.html:455` `const url = URL.createObjectURL(new Blob([bytes]` | audio/pcm / sample_rate / channels | 无 | 已实现未验 |
| CLI：chat 与多模态输出 | `src/client/cli.cpp:378` `int chat(int argc, char** argv, std::atomic<bool>* cance` | `src/client/cli.cpp:498` `const ChatResponse response = client.chat(request)` | chat --prompt --url --model --stream --audio-out --image-out | tests/client_boundary_e2e.py:395 | 已验证 |
| CLI：transcribe 离线转写 | `src/client/cli.cpp:547` `int transcribe(int argc, char** argv, std::atomic<bool>*` | `src/client/cli.cpp:565` `std::cout << client.transcribe(request).text << '\n'` | transcribe --url --model --file --format | tests/client_boundary_e2e.py:508 | 已验证 |
| CLI：realtime 实时转写 | `src/client/cli.cpp:569` `int realtime(int argc, char** argv, std::atomic<bool>* c` | `src/client/cli.cpp:595` `auto session = client.realtime(request)` | realtime --file --live-result --chunk-ms / WS /v1/realtime | tests/realtime_client_contract_e2e.py:676 | 已验证 |
| CLI：image / video 子命令 | `src/client/cli.cpp:622` `int image(int argc, char** argv, std::atomic<bool>* canc` | `src/client/cli.cpp:669` `response = client.generate_image(request)` | image generate/edit / video create/status/content/delete | 无 | 已实现未验 |

## 7. 构建与发布

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| zig build 产出 umm-infer 服务端 | `build.zig:92` `const exe = b.addExecutable(.` | `build.zig:103` `b.installArtifact(exe)` | zig build -Doptimize=ReleaseFast | tests/http_smoke.py:281 | 部分 |
| 构建期依赖前缀与系统库链接 | `build.zig:52` `const deps_prefix = b.option([]const u8, "deps-prefix"` | `build.zig:33` `mod.linkSystemLibrary("zmq", .` | -Ddeps-prefix -Dcurl-prefix -Dcuda-prefix；zmq,msgpackc,curl,websockets,cudart,dl | 无 | 已实现未验 |
| zig build test / smoke 跑契约套件与 HTTP 冒烟 | `build.zig:105` `const test_step = b.step("test", "Run the end-to-end con` + `build.zig:143` `const smoke = b.addSystemCommand(&.` | `build.zig:119` `test_step.dependOn(&run_test.step)` + `build.zig:147` `smoke_step.dependOn(&smoke.step)` | zig build test；zig build unified-test | tests/unified_contract_test.cpp:4616 + tests/http_smoke.py:275 | 部分 |
| zig build cuda-bench 延迟基准 | `build.zig:123` `const bench_step = b.step("cuda-bench", "Measure SHM and` | `build.zig:137` `bench_step.dependOn(&run.step)` | zig build cuda-bench（需 -Dcuda，否则打印提示） | 无 | 已实现未验 |
| CMake 构建原生 CLI 与 5 个 stage 可执行文件 | `native/CMakeLists.txt:36` `add_executable(umm-asr-native` + `native/stage_cli/CMakeLists.txt:79` `if(UMM_STAGE_CUDA)` | `native/CMakeLists.txt:44` `set_target_properties(umm-asr-native PROPERTIES CUDA_ARC` + `native/stage_cli/CMakeLists.txt:84` `add_executable(umm-native-stage-asr` | -DUMM_ASR_CUDA=ON -DUMM_QWEN38_CUDA=ON；umm-asr-native, umm-qwen38-native；CUDA_ARCHITECTURES 90 | 无 | 已实现未验 |
| 各原生子项目独立 CMake 构建 | `native/image/CMakeLists.txt:52` `add_executable(umm-image-native` + `native/image2/CMakeLists.txt:50` `add_executable(umm-image2-native` + `native/tts/CMakeLists.txt:62` `add_executable(umm-tts-native` + `native/omni/CMakeLists.txt:46` `add_subdirectory($` | `native/image/CMakeLists.txt:65` `set_target_properties(umm-image-native PROPERTIES CUDA_A` + `native/image2/CMakeLists.txt:63` `set_target_properties(umm-image2-native PROPERTIES CUDA_` + `native/tts/CMakeLists.txt:71` `set_target_properties(umm-tts-native PROPERTIES CUDA_ARC` + `native/omni_cli/CMakeLists.txt:21` `add_executable(umm-omni-native $` | -DUMM_IMAGE_CUDA=ON；umm-image-native | 无 | 部分 |
| stage CMake 打包两个图像后端 | `native/stage_cli/CMakeLists.txt:158` `add_library(umm_image2512_backend OBJECT` | `native/stage_cli/CMakeLists.txt:181` `set_target_properties(umm_image2512_backend PROPERTIES C` | umm_image2512_backend, umm_image21_backend → umm-native-stage-image；CUDA_ARCHITECTURES 90 | 无 | 已实现未验 |
| ctest 尾音裁剪自检 | `native/stage_cli/CMakeLists.txt:71` `add_executable(tts_tail_trim_test` | `native/stage_cli/CMakeLists.txt:76` `add_test(NAME tts_tail_trim_test COMMAND tts_tail_trim_t` | ctest -R tts_tail_trim_test | native/tts/tail_trim_test.cpp:51 | 已验证 |
| H20B 五进程服务启停与守护 | `service/h20b-native-service.sh:62` `start()` | `service/h20b-native-service.sh:102` `( cd "$ROOT" && nohup "$binary" --kind "$kind" --name "$` | start/stop/status；端口 43301；--vlm-entry/--image-entry/--tts-entry；UMM_NATIVE_GPU_*；h20b-service.json | tests/h20b_service_live.py:165 | 部分 |
| 并发压测独立服务栈启停 | `service/concurrency-stack.sh:32` `start()` | `service/concurrency-stack.sh:34` `nohup "$BIN/umm-infer" serve --host "$HOST" --port "$POR` | start/stop/status；端口 43302；MAX_IN_FLIGHT/IMAGE_STEPS/TTS_FRAMES | 无 | 已实现未验 |
| 发布产物构建、校验和与推送 | `tools/publish-release.sh:91` `log "building umm-infer on $PLATFORM"` + `tools/publish-release.sh:196` `find . -type f ! -name SHA256SUMS ! -name MANIFEST.md -p` + `tools/publish-release.sh:264` `git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_E` | `tools/publish-release.sh:94` `"$ZIG" build -Doptimize=ReleaseFast -Ddeps-prefix="$DEPS` + `tools/publish-release.sh:198` `) > "$DIST/SHA256SUMS"` + `tools/publish-release.sh:266` `git push origin HEAD:main` | publish-release.sh vX.Y.Z；STAGE_BUILD=local/remote/auto；dist/<ver>/linux-x86_64；patchelf $ORIGIN/lib | 无 | 已实现未验 |
| 已发布产物一条命令校验 | `release/tools/verify.sh:17` `echo "== checksums ($VERSION) =="` | `release/tools/verify.sh:55` `BODY=$(curl -sS "http://127.0.0.1:$PORT/v1/chat/completi` | ./tools/verify.sh [vX.Y.Z]；sha256sum -c；serve --help；serve --demo；断言 "content":"echo: hello" | 无 | 已实现未验 |
| macOS 客户端构建、打包与签名 | `clients/ummmlx/Scripts/build.sh:6` `swift build -c "$CONFIG"` | `clients/ummmlx/Scripts/build.sh:18` `codesign --force --sign - --identifier dev.umminfer.ummm` | Scripts/build.sh [release]；build/ummmlx.app；Package.swift: executableTarget ummlx + testTarget ummmlxTests；bundle id dev.umminfer.ummmlx | clients/ummmlx/Tests/ummmlxTests/ChatClientTests.swift:7 | 部分 |

## 8. 测试与基准

| 功能 | 入口 | 核心实现 | 协议/契约 | 验证 | 状态 |
|---|---|---|---|---|---|
| 统一编排契约套件 unified_contract_test | `tests/unified_contract_test.cpp:4616` `int main()` | `tests/unified_contract_test.cpp:4622` `RUN(test_streaming_asr_to_vlm())` | unified kind 事件；/v1/chat/completions；ipc:// entry/completion/abort | tests/unified_contract_test.cpp:4622 | 已验证 |
| HTTP 服务与进程生命周期冒烟 | `tests/http_smoke.py:275` `def main():` | `tests/http_smoke.py:281` `check_demo(binary, root, log)` | /health；/v1/chat/completions；--demo；--api-key；data: [DONE] | tests/http_smoke.py:67 | 已验证 |
| 本地黑盒集成门禁 | `tests/blackbox_integration.py:101` `def main():` | `tests/blackbox_integration.py:203` `code, _, _ = run_suite(name, command, project, env, args` | --suites；zig build test/smoke/unified-test；LOG_CAP 日志上限 | tests/blackbox_integration.py:179 | 已验证 |
| Realtime 服务端契约 | `tests/realtime_contract_e2e.py:610` `def main():` | `tests/realtime_contract_e2e.py:322` `require(actual == status, f"` | /v1/realtime；intent=transcription；input_audio_buffer.append/speech_started/speech_stopped；transcription.completed | tests/realtime_contract_e2e.py:232 | 已验证 |
| Realtime CLI 客户端契约 | `tests/realtime_client_contract_e2e.py:691` `def main():` | `tests/realtime_client_contract_e2e.py:859` `verify_handshake(active_events, client_first=True)` | umm-infer realtime --url/--model/--file；--max-new-tokens/--final-timeout-ms/--stall-ms；--audio-layout | tests/realtime_client_contract_e2e.py:800 | 已验证 |
| 客户端 CLI 边界、URL 策略与重试链路 | `tests/client_boundary_e2e.py:653` `def main():` + `tests/client_retry_e2e.py:676` `def main():` | `tests/client_boundary_e2e.py:689` `check_chat_parsing(binary)` + `tests/client_retry_e2e.py:707` `run_chat_cases(binary, stage, model, env)` | chat/transcribe/serve 子命令参数；URL 允许/拒绝策略 | tests/client_boundary_e2e.py:89 | 部分 |
| 并发压测与吞吐统计 | `tests/concurrency_load.py:159` `def main():` | `tests/concurrency_load.py:192` `threads = [threading.Thread(target=worker) for _ in rang` | --kind text/image/tts；--concurrency；--retry-503；JSON 吞吐/P50/P95 | tests/concurrency_load.py:159 | 已实现未验 |
| 长稳与故障恢复 | `tests/soak_native.py:969` `def main():` | `tests/soak_native.py:1023` `workers = [threading.Thread(target=load.worker, daemon=T` | /health；/metrics；--mode soak/faults；--duration-s；--gpus | tests/soak_native.py:670 | 已实现未验 |
| 全双工语音链路 L1/L2 实测 | `tests/full_duplex_l2_live.py:455` `def main():` | `tests/full_duplex_l2_live.py:494` `run_realtime(args, http_port, ws_port, pcm, evidence)` | /v1/realtime；audio.stream=true/false；--mode realtime/pipeline/prefix | tests/full_duplex_l2_live.py:261 | 已实现未验 |
| H20B 常驻服务在线验收 | `tests/h20b_service_live.py:147` `def main():` | `tests/h20b_service_live.py:185` `status, frames = stream(chat_url, body)` | /health；/ready；/v1/chat/completions；UMMINFER_LIVE_OPT_IN=1 | tests/h20b_service_live.py:70 | 已实现未验 |
| macOS 统一引擎低资源复现 | `tests/mac_unified_validation.py:587` `def main():` | `tests/mac_unified_validation.py:653` `server = start_server(binary, paths)` | /v1/chat/completions；ipc:// asr/vlm/done/abort；--evidence-dir | tests/mac_unified_validation.py:512 | 已实现未验 |
| MLX 三级性能与 token 计数 | `tests/mlx_perf.py:1160` `def main():` | `tests/mlx_perf.py:1240` `results["stage"] = run_stage(stage, model, args.max_toke` | --mode direct/stage/e2e/all；--samples/--warmup；prompt_tokens/generated_token_count | tests/mlx_perf.py:1160 | 已实现未验 |
| 多模态基准与证据回放 | `tests/multimodal_benchmark.py:1184` `def main(argv=None):` | `tests/multimodal_benchmark.py:1280` `report = run_round(manifest, cases, resolved, args, roun` | /v1/chat/completions；/v1/audio/transcriptions；--verify-data；--accept-baseline；--baseline | tests/multimodal_benchmark.py:931 | 已实现未验 |
| 基准 harness 离线契约测试 | `tests/multimodal_benchmark_test.py:208` `def main():` | `tests/multimodal_benchmark_test.py:220` `completed = subprocess.run(cmd, capture_output=True, tex` | --evidence-class contract-test；stability_verdict=NOT_REAL_EVIDENCE | tests/multimodal_benchmark_test.py:144 | 已验证 |
| CUDA/SHM 中继延迟基准 | `tests/cuda_bench.cpp:267` `int main(int argc, char** argv)` | `tests/cuda_bench.cpp:330` `auto op = relay->put(tensor, "bench")` | --role bench；CSV 列 relay,direct_d2h,size_bytes,median_us,p95_us,mib_per_s,hash_ok | tests/cuda_bench.cpp:330 | 已实现未验 |
| macOS 端测试：统一流解码与语音状态机 | `clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:45` `func testDecodesEveryDocumentedKind() throws` + `clients/ummmlx/Tests/ummmlxTests/ChatClientTests.swift:7` `func testSplitMultilineAndComments() throws` | `clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:168` `let artifact = try collector.ingest(ArtifactChunk(` + `clients/ummmlx/Tests/ummmlxTests/SpeechControllerTests.swift:52` `let text = try await controller.transcribeFile(URL(fileU` | kind: asr_partial/asr_final/vlm_delta/artifact/leg_error/complete/error/cancelled；sha256/bytes/base64 | clients/ummmlx/Tests/ummmlxTests/UnifiedStreamTests.swift:158 | 已验证 |

共 119 行。
