// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "harness.hpp"

namespace {

// Real HTTP/SSE and Coordinator, with the same fake stage sockets as the other
// contract tests. This checks public wiring only; it is not a model result.
void test_public_http_unified_sse() {
  Harness h("publichttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public unified server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  nlohmann::json body = {{"model", "qwen3-unified"},
                         {"messages", nlohmann::json::array({message})},
                         {"stream", true},
                         {"max_tokens", 37},
                         {"temperature", 0.25}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public HTTP did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "public WAV was not converted to the ASR PCM-only contract");
    check(!payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.0 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "ASR parameters were not fixed without losing max_new_tokens");
    h.complete(asr_submit->request_id, "asr", true, text_result("real transcript"));
  }

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public HTTP did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "real transcript" &&
              payload.request.inputs.find("audio") == nullptr,
          "VLM did not receive exactly the ASR transcript");
    check(payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.25 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "VLM did not preserve the public sampling parameters");
    h.stream(vlm_submit->request_id, "vlm", text_chunk("answer", false));
    auto result = text_result("answer");
    result["finish_reason"] = Value("stop");
    h.complete(vlm_submit->request_id, "vlm", true, Value(std::move(result)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public unified SSE returned an HTTP error");
  const size_t asr_pos = response.body.find("\"kind\":\"asr_final\"");
  const size_t vlm_pos = response.body.find("\"kind\":\"vlm_delta\"");
  const size_t complete_pos = response.body.find("\"kind\":\"complete\"");
  check(asr_pos != std::string::npos && vlm_pos != std::string::npos &&
            complete_pos != std::string::npos && asr_pos < vlm_pos &&
            vlm_pos < complete_pos,
        "public SSE order is not asr_final, vlm_delta, complete");
  check(response.body.find("\"text\":\"real transcript\"") != std::string::npos &&
            response.body.find("\"content\":\"answer\"") != std::string::npos,
        "public SSE lost the transcript or answer");
  check(response.body.find("\"kind\":\"complete\"", complete_pos + 1) ==
            std::string::npos,
        "public SSE emitted more than one successful terminal");
  wait_until([&] { return engine.in_flight() == 0; },
             "public unified request stayed in flight after completion");
}

// Public joint request: one instruction, one image, one WAV. The ASR leg sees
// PCM only; the VLM leg sees instruction, image, transcript in that order and
// every SSE unified event carries the public session label.
void test_public_http_joint_input() {
  Harness h("publicjoint");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public joint server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json instruction = {{"type", "text"}, {"text", "describe this image"}};
  nlohmann::json image = {{"type", "image_url"},
                          {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "public-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"},
           {"content", nlohmann::json::array({instruction, image, audio_part})}}})},
      {"stream", true}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public joint request did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "the ASR leg received more than PCM");
    h.complete(asr_submit->request_id, "asr", true, text_result("spoken"));
  }
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public joint request did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 3,
          "the VLM prompt is not instruction + image + transcript");
    if (content.is_array() && content.as_array().size() == 3) {
      check(content.as_array()[0].find("text")->as_str() == "describe this image" &&
                content.as_array()[1].find("type")->as_str() == "image_url" &&
                content.as_array()[2].find("text")->as_str() == "spoken",
            "the VLM part order or content is wrong");
    }
    h.complete(vlm_submit->request_id, "vlm", true, text_result("answer"));
  }
  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public joint SSE returned an HTTP error");
  check(response.body.find("\"session_id\":\"public-session\"") !=
            std::string::npos,
        "public SSE lost the session label");
}

// Text-only and image+text requests without an input_audio part are valid
// unified turns: the HTTP boundary accepts both a plain string content and a
// content part array, no ASR leg is submitted, and the VLM answers directly.
void test_public_http_vlm_only() {
  Harness h("publicvlmonly");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public VLM-only server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  auto post = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    return client::detail::perform(request);
  };
  auto body_with = [](nlohmann::json content) {
    return nlohmann::json{{"messages", nlohmann::json::array({nlohmann::json{
                               {"role", "user"}, {"content", std::move(content)}}})}};
  };

  // GSM8K shape: a plain string content.
  auto string_future = std::async(std::launch::async, [&] {
    return post(body_with("what is two plus two"));
  });
  check(h.no_submit(h.asr_recv), "string content submitted an ASR leg");
  auto string_vlm = h.next_submit(h.vlm_recv);
  check(string_vlm.has_value(), "string content did not reach the VLM");
  if (string_vlm) {
    StagePayload payload = StagePayload::from_value(string_vlm->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() &&
              content.as_array()[0].find("text")->as_str() ==
                  "what is two plus two",
          "the VLM lost the string content text");
    h.complete(string_vlm->request_id, "vlm", true, text_result("four"));
  }
  const auto string_response = string_future.get();
  check(string_response.status == 200 &&
            string_response.body.find("\"content\":\"four\"") !=
                std::string::npos,
        "the string-content turn did not return the VLM answer");

  // VQA shape: text part plus image part, no audio.
  const nlohmann::json image = {{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  const nlohmann::json instruction = {{"type", "text"},
                                      {"text", "describe this image"}};
  auto image_future = std::async(std::launch::async, [&] {
    return post(body_with(nlohmann::json::array({instruction, image})));
  });
  check(h.no_submit(h.asr_recv), "image+text content submitted an ASR leg");
  auto image_vlm = h.next_submit(h.vlm_recv);
  check(image_vlm.has_value(), "image+text content did not reach the VLM");
  if (image_vlm) {
    StagePayload payload = StagePayload::from_value(image_vlm->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2 &&
              content.as_array()[1].find("type")->as_str() == "image_url",
          "the VLM lost the image part on the audio-free turn");
    h.complete(image_vlm->request_id, "vlm", true,
               text_result("a settings screen"));
  }
  const auto image_response = image_future.get();
  check(image_response.status == 200 &&
            image_response.body.find("a settings screen") != std::string::npos,
        "the image+text turn did not return the VLM answer");

  // Two audio parts still fail the at-most-one-audio contract before submit.
  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)},
                       {"format", "wav"}}}};
  bool two_audio_rejected = false;
  try {
    post(body_with(nlohmann::json::array({audio_part, audio_part})));
  } catch (const client::Error& exc) {
    two_audio_rejected =
        std::string(exc.what()).find("HTTP 400") != std::string::npos;
  }
  check(two_audio_rejected, "two input_audio parts were accepted");
  check(h.no_submit(h.asr_recv),
        "a rejected two-audio request reached the ASR stage");

  // An audio-free request with no text or image is an empty prompt and must be
  // rejected before any stage sees it. Both an empty content array and an empty
  // string content are covered.
  auto rejected_with_400 = [&](nlohmann::json content) {
    try {
      post(body_with(std::move(content)));
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  check(rejected_with_400(nlohmann::json::array()),
        "an empty content array was accepted as an empty prompt");
  check(rejected_with_400(""),
        "an empty string content was accepted as an empty prompt");
  check(h.no_submit(h.vlm_recv),
        "a rejected empty-prompt request reached the VLM stage");

  api.stop();
  server_thread.join();
  wait_until([&] { return engine.in_flight() == 0; },
             "VLM-only requests stayed in flight after completion");
}

// The streaming VLM-only path must keep the pre-allocated HTTP request id on
// the single VLM leg. A generated id orphans the SSE sink: the client then
// sees only the role frame and hangs until the stream timeout.
void test_public_http_vlm_only_stream() {
  Harness h("publicvlmstream");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public VLM-only stream server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"}, {"content", "what is two plus two"}}})},
      {"stream", true},
      {"max_tokens", 16}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  check(h.no_submit(h.asr_recv), "a streaming VLM-only request reached ASR");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "the streaming VLM-only request missed the VLM");
  if (vlm_submit) {
    // The leg must run under the HTTP id, not a Coordinator-generated one.
    const std::string leg_id = vlm_submit->request_id;
    check(leg_id.rfind("http-", 0) == 0,
          "the VLM-only leg lost the public HTTP request id");
    h.stream(leg_id, "vlm", text_chunk("four", false));
    auto result = text_result("four");
    result["finish_reason"] = Value("stop");
    h.complete(leg_id, "vlm", true, Value(std::move(result)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200,
        "the streaming VLM-only turn returned an HTTP error");
  check(response.body.find("\"kind\":\"vlm_delta\"") != std::string::npos &&
            response.body.find("\"content\":\"four\"") != std::string::npos,
        "the streaming VLM-only turn delivered no vlm_delta content");
  check(response.body.find("\"kind\":\"complete\"") != std::string::npos &&
            response.body.find("data: [DONE]") != std::string::npos,
        "the streaming VLM-only turn lost its terminal or [DONE]");
  wait_until([&] { return engine.in_flight() == 0; },
             "the streaming VLM-only request stayed in flight");
}

// Bad unified media fields are rejected before any stage sees the work.
void test_public_http_unified_errors() {
  Harness h("publicbad");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public error-path server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  const nlohmann::json image = {{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  // The public client throws on a non-2xx response; a 400 is the expected result.
  auto rejected = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      client::detail::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  auto body_with = [](nlohmann::json content) {
    return nlohmann::json{{"messages", nlohmann::json::array({nlohmann::json{
                               {"role", "user"}, {"content", std::move(content)}}})}};
  };

  auto two_images = body_with(nlohmann::json::array({image, image, audio_part}));
  check(rejected(two_images), "two image parts were accepted");
  auto two_texts = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", "a"}},
       nlohmann::json{{"type", "text"}, {"text", "b"}}, image, audio_part}));
  check(rejected(two_texts), "two text parts were accepted");
  auto empty_text = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", ""}}, image, audio_part}));
  check(rejected(empty_text), "an empty text part was accepted");
  auto empty_session = body_with(nlohmann::json::array({image, audio_part}));
  empty_session["session_id"] = "";
  check(rejected(empty_session), "an empty session_id was accepted");
  check(h.no_submit(h.asr_recv), "an invalid unified request reached the ASR stage");
  api.stop();
  server_thread.join();
}

size_t occurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

// Public chat body with one WAV input_audio part. The fake ASR stage receives
// the decoded PCM; no model runs in this file.
nlohmann::json public_audio_body(int max_tokens = 37) {
  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  return {{"model", "qwen3-unified"},
          {"messages", nlohmann::json::array({message})},
          {"stream", true},
          {"max_tokens", max_tokens},
          {"temperature", 0.0}};
}

// Public HTTP abort while the VLM leg is already streaming. The deltas that
// were sequenced before the abort arrive, then exactly one cancelled terminal.
// A late delta and a late success completion are dropped, the VLM stage sees the
// abort, in_flight returns to zero, and the next request completes. A
// cancellation is not a generation failure, so the stream carries no error
// frame.
void test_public_http_cancel_during_vlm() {
  Harness h("httpcancelvlm");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "cancel server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  std::mutex body_mu;
  std::string body;
  std::string stream_request_id;
  std::promise<void> saw_two_deltas;
  auto saw_two_deltas_future = saw_two_deltas.get_future();
  bool signaled = false;
  auto stream_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    request.on_body = [&](std::string_view bytes) {
      std::lock_guard<std::mutex> lock(body_mu);
      body.append(bytes);
      if (stream_request_id.empty()) {
        const size_t id_pos = body.find("\"id\":\"");
        if (id_pos != std::string::npos) {
          const size_t begin = id_pos + 6;
          const size_t end = body.find('"', begin);
          if (end != std::string::npos) {
            stream_request_id = body.substr(begin, end - begin);
          }
        }
      }
      if (!signaled && occurrences(body, "\"kind\":\"vlm_delta\"") >= 2) {
        signaled = true;
        saw_two_deltas.set_value();
      }
      return true;
    };
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "cancel request did not reach the ASR stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", true, text_result("transcript"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "cancel request did not reach the VLM stage");
  const std::string child = vlm_submit ? vlm_submit->request_id : id + ":vlm";
  h.stream(child, "vlm", text_chunk("first ", false));
  h.stream(child, "vlm", text_chunk("second", false));

  check(saw_two_deltas_future.wait_for(std::chrono::seconds(5)) ==
            std::future_status::ready,
        "two VLM deltas never reached the public stream");
  check(!stream_request_id.empty() && stream_request_id == id,
        "the public stream id did not match the submitted request");

  client::detail::HttpRequest abort_request;
  abort_request.url = base + "/abort";
  abort_request.body =
      nlohmann::json{{"request_id", stream_request_id.empty() ? id
                                                              : stream_request_id}}
          .dump();
  abort_request.headers.emplace("Content-Type", "application/json");
  abort_request.timeout = client::Milliseconds{5000};
  const auto abort_response = client::detail::perform(abort_request);
  check(abort_response.status == 200,
        "public abort did not accept the active request");

  // Tokens already in flight at the real worker can still arrive, and the
  // worker can still report success. Both are stale after the abort.
  h.stream(child, "vlm", text_chunk("late", false));
  h.complete(child, "vlm", true, text_result("late answer"));

  const auto response = stream_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "cancelled SSE returned an HTTP error");
  check(occurrences(body, "\"kind\":\"cancelled\"") == 1,
        "cancel during VLM did not produce exactly one cancelled terminal");
  check(occurrences(body, "\"kind\":\"complete\"") == 0 &&
            occurrences(body, "\"kind\":\"error\"") == 0,
        "cancel during VLM produced a second terminal");
  check(occurrences(body, "\"kind\":\"vlm_delta\"") == 2,
        "a VLM delta delivered after the abort reached the public stream");
  check(body.find("late") == std::string::npos,
        "late VLM text reached the public stream");
  check(occurrences(body, "[DONE]") == 1,
        "the cancelled stream did not end with exactly one [DONE]");
  const size_t cancelled_pos = body.find("\"kind\":\"cancelled\"");
  const size_t last_delta = body.rfind("\"kind\":\"vlm_delta\"");
  check(cancelled_pos != std::string::npos &&
            (last_delta == std::string::npos || last_delta < cancelled_pos),
        "a vlm_delta was written after the cancelled terminal");
  check(occurrences(body, "\"type\":\"generation_error\"") == 0,
        "a client-requested cancel was reported as a generation error");
  check(log.terminal_count(id) == 1,
        "the engine reported more than one terminal for one cancellation");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after a VLM cancellation");
  check(h.wait_abort(child), "the VLM stage never received the abort");

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after a cancellation did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("second transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after a cancellation did not reach the VLM stage");
  h.complete(next + ":vlm", "vlm", true, text_result("second answer"));
  check(engine.wait(next, 5.0),
        "the request after a cancellation did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after a cancellation did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

// Public HTTP request whose ASR leg fails in the stage, not at request
// validation. The stream carries one error attributed to the ASR stage, no ASR
// final, no VLM delta, no VLM submit, in_flight returns to zero, and the next
// request completes.
void test_public_http_asr_model_failure() {
  Harness h("httpasrfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "ASR failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR failure request did not reach the stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", false, Value{},
             std::string("ASR model backend unavailable"));

  const auto response = response_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "ASR failure SSE returned an HTTP error");
  check(occurrences(response.body, "\"kind\":\"error\"") == 1,
        "the ASR stage failure did not produce exactly one error terminal");
  check(occurrences(response.body, "\"kind\":\"asr_final\"") == 0 &&
            occurrences(response.body, "\"kind\":\"vlm_delta\"") == 0 &&
            occurrences(response.body, "\"kind\":\"complete\"") == 0,
        "the ASR stage failure emitted success or VLM events");
  check(occurrences(response.body, "\"type\":\"generation_error\"") == 1,
        "the ASR stage failure did not produce exactly one error frame");
  check(response.body.find("\"stage\":\"asr\"") != std::string::npos,
        "the error was not attributed to the ASR stage");
  check(response.body.find("ASR model backend unavailable") !=
            std::string::npos,
        "the stage failure reason was not propagated");
  check(occurrences(response.body, "[DONE]") == 1,
        "the failed stream did not end with exactly one [DONE]");
  check(log.terminal_count(id) == 1, "the ASR failure produced duplicate terminals");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->stage_id == kAsrStageId,
        "the ASR failure terminal did not name the ASR stage");
  check(h.no_submit(h.vlm_recv),
        "the VLM leg started after an ASR stage failure");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after an ASR stage failure");
  engine.release(id);

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after an ASR failure did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("recovered transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after an ASR failure did not reach the VLM stage");
  h.stream(next + ":vlm", "vlm", text_chunk("recovered", false));
  h.complete(next + ":vlm", "vlm", true, text_result("recovered answer"));
  check(engine.wait(next, 5.0),
        "the request after an ASR failure did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after an ASR failure did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

}  // namespace
