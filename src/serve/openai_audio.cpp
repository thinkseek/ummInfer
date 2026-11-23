// SPDX-License-Identifier: Apache-2.0
#include "openai_internal.hpp"

#include <charconv>
#include <iostream>
#include <set>
#include <utility>

#include "asr_common.hpp"
#include "umminfer/audio/audio.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer {

using namespace openai_detail;

void OpenAIApi::Impl::handle_transcription(const httplib::Request& req, httplib::Response& res) {
  RouteMetric route_metric{"transcriptions", &res};
  AsrRequestMetric request_metric;
  // The handler starts after httplib has read the whole body, so this clock
  // separates upload time (client total minus this) from engine time.
  const auto handler_started = std::chrono::steady_clock::now();
  auto reject = [&res](int status, const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", {{"message", util::utf8_sanitized(message)},
                                    {"type", "transcription_error"}}}}
                        .dump(),
                    "application/json");
  };
  if (!req.is_multipart_form_data()) {
    reject(400, "one multipart WAV file is required");
    return;
  }
  for (const auto& part : req.form.files) {
    if (part.first != "file") {
      reject(400, "unsupported multipart file part: " + part.first);
      return;
    }
  }
  if (req.form.get_file_count("file") != 1) {
    reject(400, "one multipart WAV file is required");
    return;
  }
  std::map<std::string, int> field_counts;
  for (const auto& part : req.form.fields) ++field_counts[part.first];
  for (const auto& [name, count] : field_counts) {
    if (count > 1) {
      reject(400, "duplicate multipart field: " + name);
      return;
    }
  }
  static const std::set<std::string> kAllowedFields = {
      "model", "stream", "response_format", "include_generation_metadata",
      "language", "prompt", "max_new_tokens", "temperature"};
  for (const auto& [name, count] : field_counts) {
    if (!kAllowedFields.count(name)) {
      reject(400, "unsupported multipart field: " + name);
      return;
    }
  }
  const std::string requested_model = req.form.get_field("model");
  if (requested_model.empty()) {
    reject(400, "model is required");
    return;
  }
  if (requested_model != config_.served_model_name) {
    reject(400, "unsupported model: " + requested_model +
                    "; this server serves " + config_.served_model_name);
    return;
  }
  if (req.form.has_field("language")) {
    reject(400, "language is not supported by this server");
    return;
  }
  if (req.form.has_field("prompt")) {
    reject(400, "prompt is not supported by this server");
    return;
  }
  if (req.form.has_field("temperature") && req.form.get_field("temperature") != "0") {
    reject(400, "temperature is fixed at 0 by this server");
    return;
  }
  bool stream = false;
  if (req.form.has_field("stream")) {
    const std::string value = req.form.get_field("stream");
    if (value != "true" && value != "false") {
      reject(400, "stream must be true or false");
      return;
    }
    stream = value == "true";
  }
  const std::string response_format = req.form.has_field("response_format")
                                          ? req.form.get_field("response_format")
                                          : std::string("json");
  if (response_format != "json") {
    reject(400, "unsupported response_format: " + response_format +
                    "; only json is supported, for both streaming and non-streaming");
    return;
  }
  bool want_metadata = false;
  if (req.form.has_field("include_generation_metadata")) {
    const std::string value = req.form.get_field("include_generation_metadata");
    if (value != "true" && value != "false") {
      reject(400, "include_generation_metadata must be true or false");
      return;
    }
    want_metadata = value == "true";
  }
  int64_t max_tokens = 128;
  if (req.form.has_field("max_new_tokens")) {
    const std::string text = req.form.get_field("max_new_tokens");
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), max_tokens);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        max_tokens < 1 || max_tokens > 4096) {
      reject(400, "max_new_tokens must be between 1 and 4096");
      return;
    }
  }
  std::string pcm;
  try {
    pcm = client::pcm16_from_wav(req.form.get_file("file").content);
  } catch (const std::exception& error) {
    reject(400, error.what());
    return;
  }
  StagePayload payload;
  try {
    payload = serve_detail::make_asr_payload(std::move(pcm), max_tokens, "", "");
  } catch (const std::exception& error) {
    reject(400, error.what());
    return;
  }
  AdmissionGuard guard(active_http_requests_, config_.max_in_flight);
  if (!guard.admitted()) {
    reject_overloaded(res);
    return;
  }
  std::string request_id;
  try {
    request_id = coordinator_.submit(std::move(payload), config_.entry_stage);
  } catch (const std::exception& error) {
    reject(503, error.what());
    return;
  }
  // Submit is synchronous only in the sense that it enqueues; the split
  // between this and the wait below is what tells a slow engine from a slow
  // client-to-server transfer.
  const auto submit_done = std::chrono::steady_clock::now();
  obs::Metrics::instance().trace_start(request_id, "transcription", "asr",
                                       request_metric.start_epoch_ms);
  request_metric.request_id = request_id;
  request_metric.counted = true;
  request_metric.status = "aborted";
  bool settled = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(config_.request_timeout_s);
  while (!settled && !req.is_connection_closed()) {
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) break;
    settled = coordinator_.wait(
        request_id,
        std::min(0.1, std::chrono::duration<double>(remaining).count()));
  }
  if (req.is_connection_closed()) {
    std::cerr << "transcription client_closed request_id=" << request_id
              << " body_bytes=" << req.body.size() << " submit_ms="
              << static_cast<long>(
                     std::chrono::duration<double, std::milli>(
                         submit_done - handler_started).count())
              << " handler_ms="
              << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
              << " settled=" << (settled ? 1 : 0) << "\n";
    coordinator_.abort(request_id, "client");
    coordinator_.release(request_id);
    // An abandoned request must not leave res.status unset: httplib fills a
    // missing status with 200, so a client that is still reading would see a
    // successful transcription with an empty body.
    reject(503, "client closed the connection before the transcription finished");
    return;
  }
  auto info = coordinator_.info(request_id);
  if (!settled || !info) {
    std::cerr << "transcription timeout request_id=" << request_id
              << " handler_ms="
              << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
              << "\n";
    request_metric.status = "timeout";
    coordinator_.abort(request_id, "timeout");
    coordinator_.release(request_id);
    reject(504, "transcription timed out");
    return;
  }
  coordinator_.release(request_id);
  request_metric.status = info->state;
  std::cerr << "transcription done request_id=" << request_id
            << " body_bytes=" << req.body.size() << " handler_ms="
            << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
            << " state=" << info->state << "\n";
  if (info->state != "completed") {
    reject(500, info->error);
    return;
  }
  std::string text;
  try {
    text = serve_detail::asr_result_text(info->result);
  } catch (const std::exception& error) {
    reject(500, error.what());
    return;
  }
  serve_detail::AsrGenerationMetadata metadata;
  const bool has_metadata =
      serve_detail::asr_generation_metadata(info->result, metadata);
  if (want_metadata && !has_metadata) {
    reject(500, "the ASR stage did not report generation metadata");
    return;
  }
  json out = {{"text", text}};
  if (has_metadata) {
    out["generation_metadata"] = {
        {"generated_token_count", metadata.generated_token_count},
        {"finish_reason", metadata.finish_reason}};
  }
  if (!stream) {
    res.set_content(out.dump(), "application/json");
    return;
  }
  json event = {{"type", "transcript.text.done"}, {"text", text}};
  if (has_metadata) event["generation_metadata"] = out["generation_metadata"];
  res.set_content("data: " + event.dump() + "\n\ndata: [DONE]\n\n",
                  "text/event-stream");
}

// POST /v1/audio/speech. One text-to-speech request against the TTS stage
// the unified engine already registered: the same Coordinator submit/abort
// path, no second protocol. The stage speaks WAV only, so a pcm response
// drops the RIFF container here and the stage stays one format.
void OpenAIApi::Impl::handle_speech(const httplib::Request& req, httplib::Response& res) {
  RouteMetric route_metric{"speech", &res};
  auto reject = [&res](int status, const std::string& message,
                       const char* type) {
    res.status = status;
    res.set_content(
        json{{"error", {{"message", util::utf8_sanitized(message)},
                        {"type", type}}}}
            .dump(),
        "application/json");
  };
  if (!config_.upstream_kind.empty()) {
    proxy_.reject_route(res, "audio speech");
    return;
  }
  if (unified_engine_ == nullptr || !unified_engine_->tts_enabled()) {
    reject(501,
           "speech is not implemented by this server: no TTS stage is registered",
           "unsupported_route");
    return;
  }
  const json body = json::parse(req.body, nullptr, false);
  if (body.is_discarded() || !body.is_object()) {
    reject(400, "request body must be a JSON object", "invalid_request");
    return;
  }
  static const std::set<std::string> kAllowedFields = {
      "model", "input", "voice", "response_format"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!kAllowedFields.count(it.key())) {
      reject(400, "unsupported field: " + it.key(), "invalid_request");
      return;
    }
  }
  if (body.contains("model") && !body["model"].is_null()) {
    if (!body["model"].is_string()) {
      reject(400, "model must be a string", "invalid_request");
      return;
    }
    const std::string model = body["model"].get<std::string>();
    if (model != config_.served_model_name) {
      reject(400, "unsupported model: " + model + "; this server serves " +
                      config_.served_model_name,
             "invalid_request");
      return;
    }
  }
  if (!body.contains("input") || !body["input"].is_string() ||
      body["input"].get_ref<const std::string&>().empty()) {
    reject(400, "input must be a nonempty string", "invalid_request");
    return;
  }
  const std::string input = body["input"].get<std::string>();
  // The stage treats voice as a free label and defaults to vivian; only the
  // empty string is a contract error here.
  std::string voice = "vivian";
  if (body.contains("voice") && !body["voice"].is_null()) {
    if (!body["voice"].is_string() ||
        body["voice"].get_ref<const std::string&>().empty()) {
      reject(400, "voice must be a nonempty string", "invalid_request");
      return;
    }
    voice = body["voice"].get<std::string>();
  }
  std::string response_format = "wav";
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    if (!body["response_format"].is_string()) {
      reject(400, "response_format must be a string", "invalid_request");
      return;
    }
    response_format = body["response_format"].get<std::string>();
  }
  if (response_format != "wav" && response_format != "pcm") {
    reject(400,
           "unsupported response_format: " + response_format +
               "; this server serves wav and pcm",
           "invalid_request");
    return;
  }

  AdmissionGuard guard(active_http_requests_, config_.max_in_flight);
  if (!guard.admitted()) {
    reject_overloaded(res);
    return;
  }
  StagePayload payload;
  Value::Map inputs;
  inputs["text"] = Value(input);
  inputs["voice"] = Value(voice);
  inputs["format"] = Value("wav");
  payload.request.inputs = Value(std::move(inputs));
  std::string request_id;
  try {
    request_id = coordinator_.submit(std::move(payload),
                                     unified_engine_->tts_stage_name());
  } catch (const std::exception& error) {
    reject(503, error.what(), "pipeline_error");
    return;
  }
  bool settled = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(config_.request_timeout_s);
  while (!settled && !req.is_connection_closed()) {
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) break;
    settled = coordinator_.wait(
        request_id,
        std::min(0.1, std::chrono::duration<double>(remaining).count()));
  }
  if (req.is_connection_closed()) {
    coordinator_.abort(request_id, "client");
    coordinator_.release(request_id);
    reject(503, "client closed the connection before the speech finished",
           "speech_error");
    return;
  }
  auto info = coordinator_.info(request_id);
  if (!settled || !info) {
    coordinator_.abort(request_id, "timeout");
    coordinator_.release(request_id);
    reject(504, "speech timed out", "speech_error");
    return;
  }
  coordinator_.release(request_id);
  if (info->state != "completed") {
    reject(500, info->error.empty() ? "speech failed" : info->error,
           "speech_error");
    return;
  }
  const Value* data =
      info->result.is_map() ? info->result.find("data") : nullptr;
  if (data == nullptr || !data->is_bin() || data->as_bin().empty()) {
    reject(500, "the TTS stage returned no audio bytes", "speech_error");
    return;
  }
  if (response_format == "wav") {
    res.set_content(data->as_bin(), "audio/wav");
    return;
  }
  try {
    const std::string pcm(audio::parse_wav_pcm16(data->as_bin()).pcm);
    res.set_content(pcm, "audio/pcm");
  } catch (const std::exception& error) {
    reject(500, std::string("invalid TTS WAV container: ") + error.what(),
           "speech_error");
  }
}

}  // namespace umminfer
