// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "harness.hpp"

namespace {

// Their Bin results are contract fixtures, not real model artifacts.

// Drives a VLM-only mixed request to the output barrier and returns the two
// leg submit payloads in arrival order. A VLM-only turn keeps the unified id
// on its single VLM leg, so the completion uses the observed submit id.
std::vector<StagePayload> reach_output_barrier(
    Harness& h, UnifiedEngine& engine, const std::string& id,
    const std::string& answer, bool expect_image = true,
    bool expect_tts = true) {
  (void)engine;
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "mixed request did not reach the VLM");
  h.complete(vlm ? vlm->request_id : id, "vlm", true,
             text_result(answer));
  std::vector<StagePayload> legs;
  if (expect_image) {
    if (auto submit = h.next_submit(h.image_recv)) {
      legs.push_back(StagePayload::from_value(submit->data));
    }
  }
  if (expect_tts) {
    if (auto submit = h.next_submit(h.tts_recv)) {
      legs.push_back(StagePayload::from_value(submit->data));
    }
  }
  return legs;
}

void test_mixed_fanout_join_and_prompt_rules() {
  Harness h("mixedjoin");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id =
      engine.submit(mixed_request(true, true, "explicit prompt"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "no VLM submit");
  h.complete(vlm_submit->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  check(image_submit && image_submit->request_id == id + ":image",
        "image leg id is not the derived :image child id");
  check(tts_submit && tts_submit->request_id == id + ":tts",
        "tts leg id is not the derived :tts child id");
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "explicit prompt",
          "explicit image prompt did not win over the VLM answer");
    check(p.request.inputs.find("messages") == nullptr,
          "image leg received VLM-only messages");
  }
  if (tts_submit) {
    StagePayload p = StagePayload::from_value(tts_submit->data);
    check(p.request.inputs.find("text")->as_str() == "the answer",
          "tts input is not the aggregated VLM answer");
    check(p.request.inputs.find("voice")->as_str() == "Tina" &&
              p.request.inputs.find("sample_rate")->as_int() == 24000,
          "tts options were not forwarded");
  }

  const std::string image_bytes(10, '\x89');
  const std::string audio_bytes(20, '\x52');
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", image_bytes));
  h.complete(id + ":tts", "tts", true,
             artifact_result("audio", "audio/wav", audio_bytes));
  check(engine.wait(id, 5.0), "mixed request did not complete");

  const auto events = log.snapshot();
  int64_t expected_seq = 0;
  size_t artifacts = 0;
  for (const UnifiedEvent& e : events) {
    check(e.seq == expected_seq++, "mixed event seq has a gap");
    if (e.kind == UnifiedEvent::Kind::kArtifact) {
      ++artifacts;
      check(e.artifact_bytes != nullptr &&
                e.artifact_bytes->substr(e.artifact_offset,
                                         e.artifact_length)
                        .size() == e.artifact_length,
            "artifact chunk does not cover its declared slice");
      check(e.artifact.find("artifact_id") != nullptr,
            "artifact descriptor missing artifact_id");
    }
  }
  check(artifacts == 2, "expected one artifact event per small artifact");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            terminal->legs.is_array() &&
            terminal->legs.as_array().size() == 2,
        "mixed terminal is not one kComplete with a two-entry legs array");
  if (terminal) {
    for (const Value& leg : terminal->legs.as_array()) {
      check(leg.find("state")->as_str() == "completed",
            "a successful leg is not marked completed");
    }
  }
  check(engine.in_flight() == 0, "mixed request stayed in flight");
  engine.release(id);
}

void test_mixed_image_prompt_falls_back_to_answer() {
  Harness h("mixedprompt");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, false));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  h.complete(vlm_submit->request_id, "vlm", true,
             text_result("fallback answer"));
  auto image_submit = h.next_submit(h.image_recv);
  check(image_submit.has_value(), "image leg did not start");
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "fallback answer",
          "image leg did not fall back to the VLM answer");
  }
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(5, 'x')));
  check(engine.wait(id, 5.0), "image-only mixed request did not complete");
  engine.release(id);
}

void test_mixed_partial_leg_failure_keeps_artifacts() {
  Harness h("mixedpartial");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(7, 'p')));
  h.complete(id + ":tts", "tts", false, Value(),
             std::string("tts worker down"));
  check(engine.wait(id, 5.0), "partial-failure request did not settle");

  const auto events = log.snapshot();
  size_t leg_errors = 0;
  size_t artifacts = 0;
  for (const UnifiedEvent& e : events) {
    if (e.kind == UnifiedEvent::Kind::kLegError) {
      ++leg_errors;
      check(e.stage_id == kTtsStageId && e.error == "tts worker down",
            "leg_error does not name the failed tts leg");
    }
    if (e.kind == UnifiedEvent::Kind::kArtifact) ++artifacts;
  }
  check(leg_errors == 1 && artifacts == 1,
        "partial failure did not keep the delivered image artifact");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
            log.terminal_count(id) == 1,
        "a leg failure produced the wrong terminal or more than one");
  if (terminal) {
    const auto& legs = terminal->legs.as_array();
    check(legs.size() == 2, "terminal legs array does not cover both legs");
    bool tts_failed = false;
    for (const Value& leg : legs) {
      if (leg.find("stage")->as_str() == kTtsStageId) {
        tts_failed = leg.find("state")->as_str() == "failed" &&
                     leg.find("error") != nullptr;
      }
    }
    check(tts_failed, "terminal does not report the failed tts leg");
  }
  engine.release(id);
}

void test_mixed_cancel_aborts_every_submitted_leg() {
  Harness h("mixedcancel");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  check(engine.cancel(id), "cancel rejected an active mixed request");
  check(h.wait_abort(id + ":image"), "image leg never received abort");
  check(h.wait_abort(id + ":tts"), "tts leg never received abort");
  // Abort completions arrive through the same pump; both legs must settle
  // before the single terminal.
  h.complete(id + ":image", "image", false, Value(),
             std::string("request aborted"));
  h.complete(id + ":tts", "tts", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled mixed request did not settle");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled &&
            log.terminal_count(id) == 1,
        "mixed cancel did not produce exactly one cancelled terminal");
  engine.release(id);
  check(engine.in_flight() == 0, "in_flight nonzero after cancel and release");
}

void test_mixed_cancel_before_fanout_blocks_legs() {
  Harness h("mixednofanout");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, true));
  auto vlm = h.next_submit(h.vlm_recv);
  check(engine.cancel(id), "cancel rejected the VLM-phase request");
  check(h.wait_abort(vlm ? vlm->request_id : id),
        "the VLM leg never received abort");
  h.complete(vlm->request_id, "vlm", false, Value(),
             std::string("request aborted"));
  check(engine.wait(id, 5.0), "pre-fanout cancel did not settle");
  check(h.no_submit(h.image_recv), "an image leg started after cancel");
  check(h.no_submit(h.tts_recv), "a tts leg started after cancel");
  engine.release(id);
}

void test_mixed_join_release_then_late_complete() {
  Harness h("mixedrelease");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  const std::string id = engine.submit(mixed_request(true, true));
  reach_output_barrier(h, engine, id, "answer");
  // Release while legs are still running, then let both complete late. The
  // orphan completions must be ignored without a crash and the engine stays
  // usable with in_flight at zero.
  check(engine.release(id), "release rejected the in-flight record");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(4, 'q')));
  h.complete(id + ":tts", "tts", true,
             artifact_result("audio", "audio/wav", std::string(6, 'w')));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  check(engine.in_flight() == 0, "late completions resurrected a record");
  const std::string next = engine.submit(mixed_request(false, false));
  auto next_vlm = h.next_submit(h.vlm_recv);
  h.complete(next_vlm->request_id, "vlm", true, text_result("recovered"));
  check(engine.wait(next, 5.0), "the request after release did not settle");
  check(engine.in_flight() == 0, "in_flight nonzero after recovery");
  engine.release(next);
}

void test_mixed_artifact_limits_become_leg_errors() {
  Harness h("mixedlimits");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", true, false);
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png",
                             std::string(kMaxArtifactImageBytes + 1, '\x01')));
  check(engine.wait(id, 5.0), "oversized request did not settle");
  bool saw_limit_error = false;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kLegError &&
        e.error.find("limit") != std::string::npos) {
      saw_limit_error = true;
    }
  }
  check(saw_limit_error, "an oversized artifact was not rejected");
  engine.release(id);
}

void test_mixed_chunk_seq_reservation() {
  Harness h("mixedchunks");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", true, false);
  // 5 MiB of fake bytes must split into three 2 MiB-target chunks.
  const std::string bytes(5u << 20, '\xab');
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", bytes));
  check(engine.wait(id, 5.0), "chunked request did not complete");
  std::vector<UnifiedEvent> chunks;
  for (const UnifiedEvent& e : log.snapshot()) {
    if (e.kind == UnifiedEvent::Kind::kArtifact) chunks.push_back(e);
  }
  check(chunks.size() == 3, "5 MiB did not produce three artifact chunks");
  if (chunks.size() == 3) {
    check(chunks[0].seq + 1 == chunks[1].seq &&
              chunks[1].seq + 1 == chunks[2].seq,
          "artifact chunk seq values are not contiguous");
    size_t reassembled = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      check(chunks[i].artifact.find("index")->as_int() ==
                    static_cast<int64_t>(i) &&
                chunks[i].artifact.find("total")->as_int() == 3,
            "artifact chunk index/total is wrong");
      reassembled += chunks[i].artifact_length;
    }
    check(reassembled == bytes.size(),
          "artifact chunks do not reassemble to the full byte count");
  }
  engine.release(id);
}

void test_mixed_previous_stream_tolerates_bin_chunk() {
  Harness h("mixedbin");
  std::promise<const StreamMessage*> forwarded;
  auto future = forwarded.get_future();
  // Attached before engine construction, exactly like the realtime listener:
  // the engine exchanges sinks and this probe becomes its previous_stream.
  h.coordinator.on_stream = [&](const StreamMessage& msg) {
    forwarded.set_value(&msg);
  };
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();

  const std::string id = engine.submit(mixed_request(true, false));
  reach_output_barrier(h, engine, id, "answer", true, false);
  h.stream(id + ":image", "image", Value::bin(std::string(12, '\x7f')));
  check(future.wait_for(std::chrono::seconds(2)) ==
            std::future_status::ready,
        "a Bin chunk did not pass through the previous_stream chain");
  h.complete(id + ":image", "image", true,
             artifact_result("image", "image/png", std::string(3, 'z')));
  check(engine.wait(id, 5.0), "bin-tolerance request did not complete");
  engine.release(id);
}

// Splits an SSE response body into its data payload strings.
using nlohmann::json;

std::vector<std::string> sse_data_payloads(const std::string& body) {
  std::vector<std::string> payloads;
  size_t pos = 0;
  while (pos < body.size()) {
    const size_t line_end = body.find('\n', pos);
    std::string line = body.substr(
        pos, line_end == std::string::npos ? std::string::npos
                                          : line_end - pos);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("data: ", 0) == 0) {
      std::string data = line.substr(6);
      if (data != "[DONE]") payloads.push_back(std::move(data));
    }
    if (line_end == std::string::npos) break;
    pos = line_end + 1;
  }
  return payloads;
}

// Full HTTP path: real OpenAIApi plus the fake image/tts PULL stages. The raw
// SSE body is parsed to prove chunking, base64, sha256, gap-free seq and the
// single terminal end to end.
void test_public_http_mixed_artifact_stream() {
  Harness h("publicmixed");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "mixed HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  // 5 MiB image forces three chunks; the audio stays in one small chunk.
  const std::string image_bytes(5u << 20, '\x89');
  const std::string audio_bytes(4096, '\x52');
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image", "audio"})},
      {"image", {{"output_format", "png"}, {"size", "1024x1024"}}},
      {"audio", {{"voice", "Tina"}, {"format", "wav"},
                 {"sample_rate", 24000}}}};

  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return client::detail::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "mixed HTTP request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("the answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  if (image_submit) {
    StagePayload p = StagePayload::from_value(image_submit->data);
    check(p.request.inputs.find("prompt")->as_str() == "the answer",
          "HTTP image leg did not use the VLM answer as its prompt");
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", image_bytes));
  }
  if (tts_submit) {
    StagePayload p = StagePayload::from_value(tts_submit->data);
    check(p.request.inputs.find("text")->as_str() == "the answer",
          "HTTP tts leg did not receive the aggregated VLM answer");
    Value::Map extra;
    extra["sample_rate"] = Value(int64_t{24000});
    extra["channels"] = Value(int64_t{1});
    h.complete(tts_submit->request_id, "tts", true,
               artifact_result("audio", "audio/wav", audio_bytes,
                               std::move(extra)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "mixed SSE returned an HTTP error");
  check(response.body.rfind("data: [DONE]") != std::string::npos &&
            occurrences(response.body, "data: [DONE]") == 1,
        "mixed SSE does not end with exactly one [DONE]");

  std::map<std::string, std::string> reassembled;
  std::map<std::string, std::string> digests;
  std::map<std::string, int64_t> expected_bytes;
  int64_t last_seq = -1;
  size_t terminals = 0;
  size_t artifact_frames = 0;
  for (const std::string& payload : sse_data_payloads(response.body)) {
    const json frame = json::parse(payload);
    if (!frame.contains("unified")) continue;
    const json& unified = frame["unified"];
    const int64_t seq = unified["seq"].get<int64_t>();
    check(seq == last_seq + 1, "HTTP unified seq is not gap-free");
    last_seq = seq;
    const std::string kind = unified["kind"].get<std::string>();
    if (kind == "artifact") {
      ++artifact_frames;
      const json& a = unified["artifact"];
      const std::string id = a["artifact_id"].get<std::string>();
      const std::string data = a["data"].get<std::string>();
      check(data.size() % 4 == 0, "an artifact chunk is not aligned base64");
      reassembled[id] += client::detail::base64_decode(
          data, kArtifactChunkTargetBytes + 16);
      if (a.contains("sha256")) {
        digests[id] = a["sha256"].get<std::string>();
        expected_bytes[id] = a["bytes"].get<int64_t>();
      }
    }
    if (kind == "complete" || kind == "error" || kind == "cancelled") {
      ++terminals;
      check(unified.contains("legs") &&
                unified["legs"].size() == 2,
            "the mixed terminal does not carry two legs");
    }
  }
  check(terminals == 1, "mixed SSE has not exactly one terminal");
  check(artifact_frames == 4, "expected 3 image chunks plus 1 audio chunk");
  check(reassembled.size() == 2, "expected two artifacts");
  for (auto& [id, bytes] : reassembled) {
    check(digests.count(id) == 1 &&
              digests[id] == client::detail::sha256_hex(bytes),
          "an artifact sha256 does not match its reassembled bytes");
    check(expected_bytes[id] == static_cast<int64_t>(bytes.size()),
          "an artifact byte count does not match");
  }
  check(reassembled.begin()->second.size() == image_bytes.size() ||
            reassembled.rbegin()->second.size() == image_bytes.size(),
        "the 5 MiB image was not reassembled intact");
  wait_until([&] { return engine.in_flight() == 0; },
             "mixed HTTP request stayed in flight");
}

// HTTP edge for partial failure: the image artifact is delivered while the tts
// leg fails. The SSE stream must carry leg_error, keep the artifact, and end
// with one complete terminal naming both leg states.
void test_public_http_mixed_partial_failure() {
  Harness h("publicmixedfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "partial-failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "hi"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image", "audio"})},
      {"image", nlohmann::json::object()},
      {"audio", {{"voice", "Tina"}, {"format", "wav"}}}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{15000};
    return client::detail::perform(request);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "partial-failure HTTP request did not reach the VLM");
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("answer"));
  auto image_submit = h.next_submit(h.image_recv);
  auto tts_submit = h.next_submit(h.tts_recv);
  if (image_submit) {
    h.complete(image_submit->request_id, "image", true,
               artifact_result("image", "image/png", std::string(40, '\x90')));
  }
  if (tts_submit) {
    h.complete(tts_submit->request_id, "tts", false, Value(),
               std::string("tts unavailable"));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "partial-failure SSE returned an HTTP error");
  check(response.body.find("\"kind\":\"leg_error\"") != std::string::npos &&
            response.body.find("tts unavailable") != std::string::npos,
        "the tts leg_error did not reach the public SSE stream");
  check(response.body.find("\"kind\":\"artifact\"") != std::string::npos,
        "the successful image artifact was dropped on partial failure");
  check(occurrences(response.body, "\"kind\":\"complete\"") == 1 &&
            occurrences(response.body, "data: [DONE]") == 1,
        "partial failure did not end with one complete and one [DONE]");
  check(response.body.find("\"state\":\"failed\"") != std::string::npos &&
            response.body.find("\"state\":\"completed\"") != std::string::npos,
        "the terminal legs do not name both states");
  wait_until([&] { return engine.in_flight() == 0; },
             "partial-failure request stayed in flight");
}


// Streamed TTS sentence legs:
// 1. VLM failure while a sentence TTS leg is active aborts the TTS leg outside
//    UnifiedEngine::mu_ without self-deadlocking.
// 2. Successful streamed TTS completion carrying "streamed": true does not
//    re-emit a duplicate WAV artifact after PCM16 chunks.
// 3. Adversarial sentence storms cap child TTS fan-out at kMaxStreamedTtsLegs.
void test_streamed_tts_omits_duplicate_wav_and_caps_fanout() {
  Harness h("ttsstream");
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& e) { log.add(e); };
  engine.on_terminal = [&](const UnifiedEvent& e) { log.finish(e); };

  auto make_streamed_tts_req = [] {
    StagePayload req = mixed_request(false, true);
    Value::Map inputs = req.request.inputs.as_map();
    Value::Map audio_opts = inputs["audio_options"].as_map();
    audio_opts["stream"] = Value(true);
    inputs["audio_options"] = Value(std::move(audio_opts));
    req.request.inputs = Value(std::move(inputs));
    return req;
  };

  // Case A: VLM fails after streaming one sentence -> active TTS leg is aborted
  // cleanly without self-deadlocking on UnifiedEngine::mu_.
  const std::string fail_id = engine.submit(make_streamed_tts_req());
  auto vlm1 = h.next_submit(h.vlm_recv);
  check(vlm1.has_value(), "streamed TTS VLM submit missing");
  if (vlm1) {
    h.stream(vlm1->request_id, "vlm", text_chunk("First sentence here.", false));
    auto tts1 = h.next_submit(h.tts_recv);
    check(tts1.has_value(), "first sentence did not trigger streamed TTS leg");
    h.complete(vlm1->request_id, "vlm", false, Value{},
               std::string("simulated VLM failure"));
    check(engine.wait(fail_id, 5.0),
          "VLM failure after streamed TTS leg deadlocked or timed out");
    if (tts1) {
      check(h.wait_abort(tts1->request_id),
            "active TTS leg did not receive abort after VLM failure");
    }
  }
  engine.release(fail_id);

  // Case B: 40 short sentences cap at kMaxStreamedTtsLegs (32) child submits,
  // and terminal WAV with streamed=true produces zero extra WAV artifacts.
  const std::string ok_id = engine.submit(make_streamed_tts_req());
  auto vlm2 = h.next_submit(h.vlm_recv);
  check(vlm2.has_value(), "second streamed TTS VLM submit missing");
  if (vlm2) {
    std::string burst;
    for (int i = 0; i < 40; ++i) {
      burst += "Sentence number " + std::to_string(i) + " ends here. ";
    }
    h.stream(vlm2->request_id, "vlm", text_chunk(burst, false));
    h.complete(vlm2->request_id, "vlm", true, text_result(burst));
    size_t tts_submits = 0;
    for (size_t i = 0; i < kMaxStreamedTtsLegs; ++i) {
      auto sub = h.next_submit(h.tts_recv);
      if (!sub) break;
      ++tts_submits;
      Value::Map chunk;
      chunk["audio"] = Value::bin(std::string(32, '\x01'));
      chunk["index"] = Value(int64_t{0});
      h.stream(sub->request_id, "tts", Value(std::move(chunk)));
      Value::Map extra;
      extra["streamed"] = Value(true);
      extra["sample_rate"] = Value(int64_t{24000});
      extra["channels"] = Value(int64_t{1});
      h.complete(sub->request_id, "tts", true,
                 artifact_result("audio", "audio/wav",
                                 std::string(64, 'W'), std::move(extra)));
    }
    check(tts_submits == kMaxStreamedTtsLegs,
          "streamed TTS sentence fan-out did not cap at kMaxStreamedTtsLegs");
    check(h.no_submit(h.tts_recv),
          "streamed TTS submitted more than kMaxStreamedTtsLegs child legs");
    check(engine.wait(ok_id, 5.0), "capped streamed TTS request did not settle");
    size_t pcm_artifacts = 0;
    size_t wav_artifacts = 0;
    for (const UnifiedEvent& ev : log.snapshot()) {
      if (ev.request_id != ok_id || ev.kind != UnifiedEvent::Kind::kArtifact) {
        continue;
      }
      const Value* mime = ev.artifact.find("mime_type");
      if (mime && mime->is_str() && mime->as_str() == "audio/pcm") ++pcm_artifacts;
      if (mime && mime->is_str() && mime->as_str() == "audio/wav") ++wav_artifacts;
    }
    check(pcm_artifacts == kMaxStreamedTtsLegs && wav_artifacts == 0,
          "streamed=true TTS result still emitted duplicate WAV artifacts");
  }
  engine.release(ok_id);
}

// Public HTTP: omitted image.size defaults to 512x512, and POST /abort enforces
// optional session_id ownership (403 on mismatch, 200 on match).
void test_public_http_default_512_image_and_session_abort() {
  Harness h("http512abort");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 10.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, h.mixed_config());
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "default-512 HTTP server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "owner-session"},
      {"messages", nlohmann::json::array(
                       {nlohmann::json{{"role", "user"}, {"content", "draw"}}})},
      {"stream", true},
      {"modalities", nlohmann::json::array({"text", "image"})}};

  std::mutex mu;
  std::string seen_id;
  std::promise<void> got_first_frame;
  auto got_first_future = got_first_frame.get_future();
  bool notified = false;
  auto stream_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest req;
    req.url = base + "/v1/chat/completions";
    req.body = body.dump();
    req.headers.emplace("Content-Type", "application/json");
    req.timeout = client::Milliseconds{10000};
    req.on_body = [&](std::string_view bytes) {
      std::lock_guard<std::mutex> lock(mu);
      const std::string chunk(bytes);
      const size_t pos = chunk.find("\"id\":\"");
      if (seen_id.empty() && pos != std::string::npos) {
        const size_t b = pos + 6;
        const size_t e = chunk.find('"', b);
        if (e != std::string::npos) seen_id = chunk.substr(b, e - b);
      }
      if (!notified && !seen_id.empty()) {
        notified = true;
        got_first_frame.set_value();
      }
      return true;
    };
    return client::detail::perform(req);
  });

  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "default-512 VLM submit missing");
  (void)wait_future_ready(got_first_future, "SSE role frame timed out");

  // Cross-session /abort must be rejected with HTTP 403 without cancelling VLM.
  {
    client::detail::HttpRequest bad_abort;
    bad_abort.url = base + "/abort";
    bad_abort.body = nlohmann::json{{"request_id", seen_id},
                                    {"session_id", "wrong-session"}}
                         .dump();
    bad_abort.headers.emplace("Content-Type", "application/json");
    bad_abort.timeout = client::Milliseconds{2000};
    int bad_status = 0;
    try {
      (void)client::detail::perform(bad_abort);
    } catch (const client::Error& err) {
      bad_status = static_cast<int>(err.status());
    }
    check(bad_status == 403,
          "POST /abort with mismatched session_id did not return 403");
  }

  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("a fox"));
  auto img = h.next_submit(h.image_recv);
  check(img.has_value(), "default-512 image submit missing");
  if (img) {
    StagePayload p = StagePayload::from_value(img->data);
    const Value* sz = p.request.inputs.find("size");
    check(sz != nullptr && sz->is_str() && sz->as_str() == "512x512",
          "omitted image.size did not default to 512x512");
    // Now cancel with the matching session_id and verify 200 OK.
    client::detail::HttpRequest ok_abort;
    ok_abort.url = base + "/abort";
    ok_abort.body = nlohmann::json{{"request_id", seen_id},
                                   {"session_id", "owner-session"}}
                        .dump();
    ok_abort.headers.emplace("Content-Type", "application/json");
    ok_abort.timeout = client::Milliseconds{2000};
    const auto abort_res = client::detail::perform(ok_abort);
    check(abort_res.status == 200,
          "POST /abort with matching session_id did not succeed");
    h.complete(img->request_id, "image", false, Value{},
               std::string("request aborted"));
  }
  if (wait_future_ready(stream_future, "stream_future timed out")) {
    (void)stream_future.get();
  }
  api.stop();
  server_thread.join();
}

// UMM_TRACE=1 emits request_start on submit (before completion!), leg_end on
// each leg, and request_end on terminal.
void test_unified_trace_records_start_legs_and_end() {
  const std::string trace_path =
      "/tmp/umm-unified-trace-" + std::to_string(::getpid()) + ".jsonl";
  std::remove(trace_path.c_str());
  ::setenv("UMM_TRACE", "1", 1);
  ::setenv("UMM_TRACE_FILE", trace_path.c_str(), 1);
  obs::Metrics::instance().reset_trace_for_test();

  Harness h("trace");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();

  const std::string id = engine.submit(audio_leg_request(false));
  auto asr = h.next_submit(h.asr_recv);
  check(asr.has_value(), "trace test ASR submit missing");
  {
    std::ifstream in(trace_path);
    const std::string early((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    check(early.find("\"event\":\"request_start\"") != std::string::npos &&
              early.find(id) != std::string::npos,
          "request_start was not written to trace while request was in flight");
  }
  if (asr) h.complete(asr->request_id, "asr", true, text_result("hello"));
  auto vlm = h.next_submit(h.vlm_recv);
  if (vlm) h.complete(vlm->request_id, "vlm", true, text_result("world"));
  check(engine.wait(id, 5.0), "trace request did not complete");
  engine.release(id);

  {
    std::ifstream in(trace_path);
    const std::string full((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    check(full.find("\"event\":\"leg_end\"") != std::string::npos &&
              full.find("\"leg\":\"asr\"") != std::string::npos &&
              full.find("\"leg\":\"vlm\"") != std::string::npos &&
              full.find("\"event\":\"request_end\"") != std::string::npos,
          "trace did not record leg_end and request_end events");
  }

  ::unsetenv("UMM_TRACE");
  ::unsetenv("UMM_TRACE_FILE");
  obs::Metrics::instance().reset_trace_for_test();
  std::remove(trace_path.c_str());
}

}  // namespace
