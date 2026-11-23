// SPDX-License-Identifier: Apache-2.0
#include "openai_internal.hpp"

#include <algorithm>
#include <ctime>
#include <utility>
#include <vector>

#include "umminfer/util/codec.hpp"

namespace umminfer {

using namespace openai_detail;

std::string OpenAIApi::Impl::submit_chat(StagePayload payload) {
  if (unified_engine_) return unified_engine_->submit(std::move(payload));
  return coordinator_.submit(std::move(payload), config_.entry_stage);
}

bool OpenAIApi::Impl::wait_chat(const std::string& request_id, double timeout_s) {
  return unified_engine_ ? unified_engine_->wait(request_id, timeout_s)
                         : coordinator_.wait(request_id, timeout_s);
}

std::optional<RequestInfo> OpenAIApi::Impl::info_chat(
    const std::string& request_id) {
  return unified_engine_ ? unified_engine_->info(request_id)
                         : coordinator_.info(request_id);
}

void OpenAIApi::Impl::cancel_chat(const std::string& request_id,
                                  const char* source) {
  if (unified_engine_) unified_engine_->cancel(request_id, source);
  else coordinator_.abort(request_id, source);
}

void OpenAIApi::Impl::release_chat(const std::string& request_id) {
  if (unified_engine_) unified_engine_->release(request_id);
  else coordinator_.release(request_id);
}

// stream=false with optional output legs. Collects the same unified events
// the SSE path forwards, reassembles each complete artifact, and answers with one
// chat.completion JSON body. A failed optional leg keeps HTTP 200; a failed
// primary leg answers the same error shape the text-only path uses.
void OpenAIApi::Impl::handle_nonstream_mixed(const httplib::Request& req,
                            httplib::Response& res, StagePayload payload,
                            int64_t created) {
  const std::string request_id = "http-" + std::to_string(++next_request_id_);
  payload.request_id = request_id;
  auto sink = std::make_shared<SseSink>();
  {
    std::lock_guard<std::mutex> lock(streams_mu_);
    streams_[request_id] = sink;
  }
  try {
    submit_chat(std::move(payload));
  } catch (const QueueFullError&) {
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_.erase(request_id);
    }
    reject_overloaded(res);
    return;
  } catch (const std::exception& exc) {
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_.erase(request_id);
    }
    res.status = 503;
    res.set_content(
        json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
            .dump(),
        "application/json");
    return;
  }
  // Cancels and releases on every exit, including a client disconnect.
  auto cleanup = std::shared_ptr<void>(
      nullptr, [this, request_id](void*) {
        cancel_chat(request_id);
        {
          std::lock_guard<std::mutex> lock(streams_mu_);
          streams_.erase(request_id);
        }
        release_chat(request_id);
      });

  bool done = false;
  std::string stream_error;
  std::optional<UnifiedEvent> terminal;
  std::vector<UnifiedEvent> artifacts;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(config_.request_timeout_s);
  while (!done && !req.is_connection_closed()) {
    const double remaining = std::chrono::duration<double>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
    if (remaining <= 0) break;
    SseItem item;
    if (sink->pop(item, std::min(0.1, remaining))) {
      if (auto* event = std::get_if<UnifiedEvent>(&item);
          event != nullptr &&
          event->kind == UnifiedEvent::Kind::kArtifact) {
        artifacts.push_back(*event);
      }
      continue;
    }
    std::lock_guard<std::mutex> lock(sink->mu);
    done = sink->done;
    if (done) {
      stream_error = sink->error;
      terminal = sink->terminal;
    }
  }
  if (req.is_connection_closed()) return;
  if (!done) {
    cancel_chat(request_id, "timeout");
    res.status = 504;
    res.set_content(json{{"error", {{"message", "request timed out"}}}}.dump(),
                    "application/json");
    return;
  }
  if (!stream_error.empty() || !terminal ||
      terminal->kind == UnifiedEvent::Kind::kError ||
      terminal->kind == UnifiedEvent::Kind::kCancelled) {
    const auto info = info_chat(request_id);
    std::string message = stream_error;
    if (message.empty() && info) message = info->error;
    if (message.empty()) message = "request failed";
    res.status = 500;
    res.set_content(
        json{{"error", {{"message", message}, {"type", "generation_error"}}}}
            .dump(),
        "application/json");
    return;
  }

  // One artifact is one whole decoded blob. Its chunks arrive in index order,
  // so the descriptor of the first chunk plus every chunk's bytes is enough.
  struct Pending {
    json descriptor;
    std::string bytes;
    bool started = false;
  };
  std::map<std::string, Pending> pending;
  json artifacts_json = json::array();
  for (const UnifiedEvent& event : artifacts) {
    if (!event.artifact_bytes) continue;
    const json descriptor = value_to_json(event.artifact);
    const std::string id = descriptor.value("artifact_id", std::string());
    const int64_t index = descriptor.value("index", int64_t{-1});
    const int64_t total = descriptor.value("total", int64_t{-1});
    Pending& slot = pending[id];
    if (!slot.started) {
      slot.descriptor = descriptor;
      if (total > 1) slot.bytes.reserve(event.artifact_bytes->size());
      slot.started = true;
    }
    const bool final = index + 1 == total;
    if (!final || total != 1) {
      slot.bytes.append(event.artifact_bytes->data() + event.artifact_offset,
                        event.artifact_length);
    }
    if (!final) continue;
    json out = std::move(slot.descriptor);
    out.erase("index");
    out.erase("total");
    out["encoding"] = "base64";
    if (total == 1) {
      const std::string_view bytes(event.artifact_bytes->data() +
                                       event.artifact_offset,
                                   event.artifact_length);
      out["data"] = util::base64_encode(bytes);
      out["bytes"] = static_cast<int64_t>(bytes.size());
      out["sha256"] = util::sha256_hex(bytes);
    } else {
      out["data"] = util::base64_encode(slot.bytes);
      out["bytes"] = static_cast<int64_t>(slot.bytes.size());
      out["sha256"] = util::sha256_hex(slot.bytes);
    }
    pending.erase(id);
    artifacts_json.push_back(std::move(out));
  }

  const Value& result = terminal->result;
  std::string text = terminal->text;
  if (!result.is_nil()) text = result_text(result);
  const std::string finish = result_finish_reason(result);
  json out = {
      {"id", request_id},
      {"object", "chat.completion"},
      {"created", created},
      {"model", config_.served_model_name},
      {"session_id", terminal->session_id},
      {"choices",
       {{{"index", 0},
         {"message", {{"role", "assistant"}, {"content", text}}},
         {"finish_reason", finish.empty() ? json(nullptr) : json(finish)}}}},
      {"artifacts", std::move(artifacts_json)},
      {"legs", terminal->legs.is_array() ? value_to_json(terminal->legs)
                                         : json::array()}};
  res.set_content(out.dump(), "application/json");
}

void OpenAIApi::Impl::handle_chat(const httplib::Request& req, httplib::Response& res) {
  RouteMetric route_metric{"chat", &res};
  const json body = json::parse(req.body, nullptr, false);
  auto reject = [&res](const std::string& message) {
    res.status = 400;
    res.set_content(
        json{{"error", {{"message", message}, {"type", "invalid_request"}}}}
            .dump(),
        "application/json");
  };
  if (body.is_discarded() || !body.is_object() || !body.contains("messages") ||
      !body["messages"].is_array() || body["messages"].empty()) {
    reject("messages must be a non-empty array");
    return;
  }
  for (const auto& message : body["messages"]) {
    if (!message.is_object() || !message.contains("role") ||
        !message["role"].is_string() || !message.contains("content")) {
      reject("each message requires a string role and content");
      return;
    }
    if (!message["content"].is_string() && !message["content"].is_array()) {
      reject("message content must be a text string or content part array");
      return;
    }
  }
  if (body.contains("stream") && !body["stream"].is_boolean()) {
    reject("stream must be a boolean");
    return;
  }
  const bool stream = body.value("stream", false);
  const int64_t created = static_cast<int64_t>(std::time(nullptr));
  if (body.contains("model") && !body["model"].is_null()) {
    if (!body["model"].is_string()) {
      reject("model must be a string");
      return;
    }
    if (body["model"].get<std::string>() != config_.served_model_name) {
      reject("unsupported model: " + body["model"].get<std::string>() +
             "; this server serves " + config_.served_model_name);
      return;
    }
  }
  if (body.contains("top_p") && !body["top_p"].is_null()) {
    reject("top_p is not implemented by this server");
    return;
  }
  if (body.contains("seed") && !body["seed"].is_null()) {
    reject("seed is not implemented by this server");
    return;
  }
  if (body.contains("stop") && !body["stop"].is_null()) {
    const json& stop = body["stop"];
    const bool empty = (stop.is_string() && stop.get<std::string>().empty()) ||
                       (stop.is_array() && stop.empty());
    if (!empty) {
      reject("stop is not implemented by this server");
      return;
    }
  }
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    const json& format = body["response_format"];
    if (!format.is_object() || !format.contains("type") ||
        !format["type"].is_string() ||
        format["type"].get<std::string>() != "text") {
      reject("only response_format type text is supported");
      return;
    }
  }
  static const std::set<std::string> kAllowedFields = {
      "model", "messages", "stream", "temperature", "max_tokens",
      "max_completion_tokens", "top_p", "seed", "stop", "response_format",
      "stream_options", "session_id", "session_history", "modalities",
      "image", "audio", "video"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!kAllowedFields.count(it.key()) && !it.value().is_null()) {
      reject("unsupported field: " + it.key());
      return;
    }
  }
  if (body.contains("session_id") && !body["session_id"].is_null() &&
      !unified_engine_) {
    reject("session_id requires unified mode");
    return;
  }
  if (body.contains("session_history") && !body["session_history"].is_null()) {
    if (!body["session_history"].is_boolean()) {
      reject("session_history must be a boolean");
      return;
    }
    if (!unified_engine_) {
      reject("session_history requires unified mode");
      return;
    }
    if (body["session_history"].get<bool>() &&
        !unified_engine_->history_enabled()) {
      reject("session_history is not enabled on this server");
      return;
    }
  }
  if (body.contains("stream_options") && !body["stream_options"].is_null()) {
    const json& options = body["stream_options"];
    if (!stream) {
      reject("stream_options requires stream=true");
      return;
    }
    if (!options.is_object()) {
      reject("stream_options must be an object");
      return;
    }
    for (auto it = options.begin(); it != options.end(); ++it) {
      if (it.key() != "include_usage" &&
          it.key() != "include_asr_partials") {
        reject("unsupported stream_options field: " + it.key());
        return;
      }
    }
    if (options.contains("include_usage") &&
        !options["include_usage"].is_boolean()) {
      reject("stream_options.include_usage must be a boolean");
      return;
    }
    if (options.value("include_usage", false)) {
      reject("stream_options.include_usage is not implemented by this server");
      return;
    }
    if (options.contains("include_asr_partials") &&
        !options["include_asr_partials"].is_boolean()) {
      reject("stream_options.include_asr_partials must be a boolean");
      return;
    }
    if (options.value("include_asr_partials", false) && !unified_engine_) {
      reject("stream_options.include_asr_partials requires unified mode");
      return;
    }
  }
  if (body.contains("temperature") && !body["temperature"].is_null()) {
    if (!body["temperature"].is_number()) {
      reject("temperature must be a number");
      return;
    }
    const double temperature = body["temperature"].get<double>();
    if (temperature < 0 || temperature > 2) {
      reject("temperature must be between 0 and 2");
      return;
    }
  }
  const json* token_limit = nullptr;
  for (const char* key : {"max_tokens", "max_completion_tokens"}) {
    if (!body.contains(key) || body[key].is_null()) continue;
    const json& value = body[key];
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<uint64_t>() >
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ||
        value.get<int64_t>() < 1) {
      reject(std::string(key) + " must be a positive integer");
      return;
    }
    token_limit = &value;
  }
  StagePayload payload;
  Value::Map inputs;
  Value::Map mixed_engine_inputs;
  if (unified_engine_) {
    std::string mixed_error;
    if (!valid_mixed_output(
            body, unified_engine_->image_output_enabled(),
            unified_engine_->tts_output_enabled(),
            unified_engine_->video_output_enabled(), mixed_engine_inputs,
            mixed_error)) {
      reject(mixed_error);
      return;
    }
  } else if ((body.contains("modalities") && !body["modalities"].is_null()) ||
             (body.contains("image") && !body["image"].is_null()) ||
             (body.contains("audio") && !body["audio"].is_null()) ||
             (body.contains("video") && !body["video"].is_null())) {
    reject("mixed output requires unified mode");
    return;
  }
  // At least one optional output leg was requested; the non-streaming path
  // then collects the unified events into one JSON body instead of one
  // terminal result.
  const bool mixed_output =
      mixed_engine_inputs.find("output_modalities") != mixed_engine_inputs.end();
  try {
    inputs["messages"] = json_to_value(body["messages"]);
    if (unified_engine_) {
      std::string pcm;
      bool has_audio = false;
      std::string error;
      if (!unified_input_parts(body, pcm, has_audio, error)) {
        reject(error);
        return;
      }
      if (has_audio) inputs["audio"] = Value::bin(std::move(pcm));
      if (body.contains("stream_options") &&
          body["stream_options"].value("include_asr_partials", false)) {
        inputs["asr_partials"] = Value(true);
      }
      if (body.contains("session_id") && !body["session_id"].is_null()) {
        if (!body["session_id"].is_string() ||
            body["session_id"].get<std::string>().empty() ||
            body["session_id"].get<std::string>().size() > 128) {
          reject("session_id must be a nonempty string of at most 128 characters");
          return;
        }
        inputs["session_id"] = Value(body["session_id"].get<std::string>());
      }
      if (body.contains("session_history") &&
          !body["session_history"].is_null()) {
        if (body["session_history"].get<bool>() &&
            !(body.contains("session_id") &&
              body["session_id"].is_string() &&
              !body["session_id"].get<std::string>().empty())) {
          reject("session_history requires a nonempty session_id");
          return;
        }
        inputs["session_history"] =
            Value(body["session_history"].get<bool>());
      }
    }
  } catch (const std::exception& exc) {
    reject(exc.what());
    return;
  }
  for (auto& [key, value] : mixed_engine_inputs) {
    inputs[key] = std::move(value);
  }
  payload.request.inputs = Value(std::move(inputs));
  Value::Map params;
  if (token_limit) params["max_new_tokens"] = Value(token_limit->get<int64_t>());
  if (body.contains("temperature") && !body["temperature"].is_null()) {
    params["temperature"] = Value(body["temperature"].get<double>());
  }
  params["stream"] = Value(stream);
  payload.request.params = std::move(params);

  std::shared_ptr<AdmissionGuard> non_unified_guard;
  if (!unified_engine_) {
    non_unified_guard = std::make_shared<AdmissionGuard>(
        active_http_requests_, config_.max_in_flight);
    if (!non_unified_guard->admitted()) {
      reject_overloaded(res);
      return;
    }
  }

  if (!stream) {
    if (unified_engine_ && mixed_output) {
      handle_nonstream_mixed(req, res, std::move(payload), created);
      return;
    }
    std::string request_id;
    try {
      request_id = submit_chat(std::move(payload));
    } catch (const QueueFullError&) {
      reject_overloaded(res);
      return;
    } catch (const std::exception& exc) {
      res.status = 503;
      res.set_content(
          json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
              .dump(),
          "application/json");
      return;
    }
    bool settled = false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(config_.request_timeout_s);
    while (!settled && !req.is_connection_closed()) {
      const auto remaining = deadline - std::chrono::steady_clock::now();
      if (remaining <= std::chrono::steady_clock::duration::zero()) break;
      const double wait_s = std::min(
          0.1, std::chrono::duration<double>(remaining).count());
      settled = wait_chat(request_id, wait_s);
    }
    if (req.is_connection_closed()) {
      cancel_chat(request_id);
      release_chat(request_id);
      return;
    }
    auto info = info_chat(request_id);
    if (!settled || !info) {
      cancel_chat(request_id, "timeout");
      release_chat(request_id);
      res.status = 504;
      res.set_content(json{{"error", {{"message", "request timed out"}}}}.dump(),
                      "application/json");
      return;
    }
    release_chat(request_id);
    if (info->state != "completed") {
      res.status = 500;
      res.set_content(
          json{{"error", {{"message", info->error}, {"type", "generation_error"}}}}
              .dump(),
          "application/json");
      return;
    }
    const std::string finish = result_finish_reason(info->result);
    json out = {{"id", request_id},
                {"object", "chat.completion"},
                {"created", created},
                {"model", config_.served_model_name},
                {"choices",
                 {{{"index", 0},
                   {"message", {{"role", "assistant"},
                                {"content", result_text(info->result)}}},
                   {"finish_reason", finish.empty() ? json(nullptr) : json(finish)}}}}};
    res.set_content(out.dump(), "application/json");
    return;
  }

  const std::string request_id = "http-" + std::to_string(++next_request_id_);
  payload.request_id = request_id;
  auto sink = std::make_shared<SseSink>();
  {
    std::lock_guard<std::mutex> lock(streams_mu_);
    streams_[request_id] = sink;
  }
  try {
    submit_chat(std::move(payload));
  } catch (const QueueFullError&) {
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_.erase(request_id);
    }
    reject_overloaded(res);
    return;
  } catch (const std::exception& exc) {
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_.erase(request_id);
    }
    res.status = 503;
    res.set_content(
        json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
            .dump(),
        "application/json");
    return;
  }
  auto cleanup = std::shared_ptr<void>(
      nullptr, [this, request_id, non_unified_guard](void*) {
        cancel_chat(request_id);
        {
          std::lock_guard<std::mutex> lock(streams_mu_);
          streams_.erase(request_id);
        }
        release_chat(request_id);
      });
  const std::string model = config_.served_model_name;
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, sink, request_id, model, created, cleanup](size_t,
                                                        httplib::DataSink& out_sink) {
        auto send_chunk = [&](const json& payload) {
          const std::string data = "data: " + payload.dump() + "\n\n";
          return out_sink.write(data.data(), data.size());
        };
        if (!send_chunk({{"id", request_id},
                         {"object", "chat.completion.chunk"},
                         {"created", created},
                         {"model", model},
                         {"choices", {{{"index", 0},
                                      {"delta", {{"role", "assistant"}}},
                                      {"finish_reason", nullptr}}}}})) {
          cancel_chat(request_id);
          return false;
        }
        std::string stream_error;
        std::string finish;
        std::optional<UnifiedEvent> terminal;
        std::optional<Value> sink_result;
        bool settled = false;
        bool disconnected = false;
        const auto timeout = std::chrono::duration<double>(config_.request_timeout_s);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
          if (!out_sink.is_writable()) {
            disconnected = true;
            break;
          }
          const double remaining =
              std::chrono::duration<double>(
                  deadline - std::chrono::steady_clock::now())
                  .count();
          if (remaining <= 0) {
            stream_error = "stream timed out";
            break;
          }
          SseItem item;
          if (sink->pop(item, std::min(0.2, remaining))) {
            json chunk_body = {{"id", request_id},
                               {"object", "chat.completion.chunk"},
                               {"created", created},
                               {"model", model}};
            if (const auto* chunk = std::get_if<Value>(&item)) {
              chunk_body["choices"] = {
                  {{"index", 0},
                   {"delta", {{"content", result_text(*chunk)}}},
                   {"finish_reason", nullptr}}};
            } else {
              const auto& event = std::get<UnifiedEvent>(item);
              json delta = json::object();
              if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
                delta["content"] = event.text;
              }
              chunk_body["choices"] = {{{"index", 0},
                                        {"delta", std::move(delta)},
                                        {"finish_reason", nullptr}}};
              chunk_body["unified"] = unified_event_json(event);
              if (event.kind == UnifiedEvent::Kind::kArtifact) {
                json& artifact = chunk_body["unified"]["artifact"];
                const std::string& raw = *event.artifact_bytes;
                const std::string_view slice(
                    raw.data() + event.artifact_offset,
                    event.artifact_length);
                artifact["encoding"] = "base64";
                artifact["data"] = util::base64_encode(slice);
                const int64_t index = artifact.value("index", int64_t{-1});
                const int64_t total = artifact.value("total", int64_t{-1});
                if (index + 1 == total) {
                  artifact["bytes"] = static_cast<int64_t>(raw.size());
                  artifact["sha256"] = util::sha256_hex(raw);
                }
              }
            }
            const std::string frame_payload = chunk_body.dump();
            if (frame_payload.size() + 8 > kMaxSseFrameWireBytes) {
              stream_error = "artifact SSE frame exceeds its size limit";
              break;
            }
            if (!send_chunk(chunk_body)) {
              disconnected = true;
              break;
            }
            deadline = std::chrono::steady_clock::now() + timeout;
            continue;
          }
          {
            std::lock_guard<std::mutex> lock(sink->mu);
            settled = sink->done;
            if (settled) {
              stream_error = sink->error;
              finish = sink->finish_reason;
              terminal = sink->terminal;
              sink_result = sink->result;
            }
          }
          if (settled) break;
        }
        if (!settled) cancel_chat(request_id);
        if (!disconnected) {
          if (stream_error.empty()) {
            json final_chunk = {
                {"id", request_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model},
                {"choices",
                 {{{"index", 0},
                   {"delta", json::object()},
                   {"finish_reason",
                    finish.empty() ? json(nullptr) : json(finish)}}}}};
            if (terminal) final_chunk["unified"] = unified_event_json(*terminal);
            // The usage carrier rides on the last chunk, before [DONE]. A
            // client that does not know the key ignores it.
            const Value* usage_source =
                terminal ? &terminal->result
                         : (sink_result ? &*sink_result : nullptr);
            if (usage_source != nullptr) {
              json usage = result_usage(*usage_source);
              if (!usage.is_null()) final_chunk["usage"] = std::move(usage);
            }
            send_chunk(final_chunk);
          } else {
            json error_chunk = {{"error", {{"message", stream_error},
                                           {"type", "generation_error"},
                                           {"code", 500}}}};
            if (terminal) error_chunk["unified"] = unified_event_json(*terminal);
            send_chunk(error_chunk);
          }
          static const std::string kDone = "data: [DONE]\n\n";
          out_sink.write(kDone.data(), kDone.size());
          out_sink.done();
        }
        return !disconnected;
      });
}

}  // namespace umminfer
