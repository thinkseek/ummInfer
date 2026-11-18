// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "harness.hpp"

namespace {

void test_streaming_asr_to_vlm() {
  Harness h("stream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.stream(id, "asr", text_chunk("hello world", true));
  wait_until([&] { return log.size() == 3; },
             "ASR partial/final events were not delivered");

  // The VLM leg starts when the ASR completion arrives, not on the final chunk.
  h.complete(id, "asr", true, text_result("hello world"));
  // The final transcript replaces the concatenated partial deltas, and the VLM
  // leg starts with exactly that text as its prompt.
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    check(vlm_submit->request_id == id + ":vlm", "VLM leg id is not a child id");
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    const Value& first = messages.as_array().at(0);
    check(first.find("content")->as_str() == "hello world",
          "VLM prompt is not the final ASR transcript");
  }

  const auto asr_leg_info = h.coordinator.info(id);
  check(asr_leg_info && asr_leg_info->state == "completed",
        "the ASR parent leg did not complete before the VLM barrier");
  const auto vlm_phase_info = engine.info(id);
  check(vlm_phase_info && vlm_phase_info->request_id == id &&
            vlm_phase_info->state == "running" &&
            vlm_phase_info->current_stage == kVlmStageId,
        "info() exposed the completed ASR parent leg instead of unified VLM state");

  h.stream(id + ":vlm", "vlm", text_chunk("hi", false));
  h.stream(id + ":vlm", "vlm", text_chunk(" there", false));
  h.complete(id + ":vlm", "vlm", true, text_result("hi there"));
  check(engine.wait(id, 5.0), "unified request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 6) {
    check(false, "wrong event count for the full ASR -> VLM flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected_kinds[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal,   UnifiedEvent::Kind::kVlmDelta,
      UnifiedEvent::Kind::kVlmDelta,   UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "unified event sequence is not gap-free from zero");
    check(events[i].kind == expected_kinds[i], "unified event kind out of order");
    check(events[i].request_id == id, "an event leaked the internal child id");
  }
  check(events[2].is_final && events[2].text == "hello world",
        "ASR final event has the wrong text");
  check(events[5].text == "hi there", "complete event lost the VLM aggregation");
  const auto final_info = engine.info(id);
  check(final_info && final_info->state == "completed" &&
            final_info->current_stage == kVlmStageId &&
            final_info->result.find("text") != nullptr &&
            final_info->result.find("text")->as_str() == "hi there",
        "info() did not retain the unified VLM terminal state");
  engine.release(id);
}

// Opt-in ASR partials: the ASR leg request streams, partial deltas precede one
// final, and the final replaces the concatenated partials for the VLM prompt.
void test_asr_partials_opt_in() {
  Harness h("partials");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "opt-in ASR partial request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(sent.request.params.at("stream").as_bool(),
          "opt-in request did not ask the ASR stage to stream");
    check(sent.request.inputs.find("asr_partials") == nullptr,
          "the ASR stage received the opt-in hint instead of audio only");
  }

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.complete(id, "asr", true, text_result("hello"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "opt-in ASR leg did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload sent = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *sent.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "hello",
          "partial deltas were not replaced by the final transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "opt-in ASR partial request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 4) {
    check(false, "wrong event count for the ASR partial flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal, UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "ASR partial sequence is not gap-free");
    check(events[i].kind == expected[i], "ASR partial event out of order");
    check(events[i].request_id == id, "ASR partial leaked the child id");
  }
  check(events[0].text == "hel" && events[1].text == "lo" &&
            !events[0].is_final && !events[1].is_final,
        "ASR partial events do not carry the delta text");
  check(events[2].is_final && events[2].text == "hello",
        "ASR final does not replace the partial text");
  engine.release(id);
}

// Default request: the ASR leg is not asked to stream and no partial is emitted.
void test_asr_partials_default_off() {
  Harness h("partialsoff");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(false));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "default ASR request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(!sent.request.params.at("stream").as_bool(),
          "a default request asked the ASR stage to stream");
  }
  h.complete(id, "asr", true, text_result("hello"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "default ASR leg did not reach the VLM leg");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "default request did not complete");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kAsrPartial,
          "a default request emitted an ASR partial");
  }
  engine.release(id);
}

// Cancel after a partial: exactly one cancelled terminal, no VLM submit.
void test_asr_partials_cancel() {
  Harness h("partialscancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) {
    log.finish(event);
  };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  wait_until([&] { return log.size() == 1; },
             "the ASR partial event was not delivered before the cancel");
  check(engine.cancel(id), "cancel was rejected before the ASR leg finished");
  check(h.wait_abort(id), "the ASR stage did not observe the abort");
  h.complete(id, "asr", false, Value{}, std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled request did not settle");
  h.no_submit(h.vlm_recv);

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
        "cancel after a partial did not produce a cancelled terminal");
  check(log.terminal_count(id) == 1,
        "cancel after a partial produced more than one terminal");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kVlmDelta,
          "a cancelled partial request emitted a VLM delta");
  }
  check(engine.in_flight() == 0, "cancelled request stayed in flight");
  engine.release(id);
}

// An ASR stream failure after a partial: exactly one error terminal.
void test_asr_partials_error() {
  Harness h("partialserror");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  h.complete(id, "asr", false, Value{}, std::string("backend reset"));
  check(engine.wait(id, 5.0), "failed partial request did not settle");

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->error == "backend reset",
        "ASR stream failure did not produce the error terminal");
  check(log.terminal_count(id) == 1,
        "ASR stream failure produced more than one terminal");
  check(h.no_submit(h.vlm_recv), "a failed ASR leg started the VLM leg");
  engine.release(id);
}

// A non-streaming ASR stage emits only a completion; the engine synthesizes
// the ASR final from its result text.
void test_nonstream_asr_synthesizes_final() {
  Harness h("nonstream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("spoken text"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "non-streaming request did not complete");

  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "synthesized final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a non-streaming result");
  engine.release(id);
}

// The ASR contract accepts a plain string result, matching
// serve_detail::asr_result_text. The engine must extract it and use it as the
// VLM prompt.
void test_asr_plain_string_result() {
  Harness h("plainstring");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, Value("spoken text"));  // plain string, no map

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "plain-string result did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "spoken text",
          "the VLM prompt is not the plain-string ASR transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "plain-string request did not complete");
  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "plain-string final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a plain-string result");
  engine.release(id);
}

// A result with no usable text is a contract failure: the engine must emit
// kError on the ASR stage and must never start the VLM leg, rather than
// forwarding an empty prompt.
void test_asr_malformed_result_fails() {
  std::vector<Value> bad_results;
  bad_results.emplace_back(Value::Map{});
  bad_results.emplace_back(Value(42));
  for (const Value& bad : bad_results) {
    Harness h("malformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, bad);
    check(engine.wait(id, 5.0), "a malformed ASR result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal.has_value() &&
              terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId,
          "a malformed ASR result did not produce an ASR-stage kError");
    check(terminal && !terminal->error.empty(),
          "the malformed-result error carried no message");
    check(h.no_submit(h.vlm_recv),
          "a malformed ASR result still started the VLM leg");
    engine.release(id);
  }
}

// A malformed streaming final must fail before the completion can start VLM.
void test_streaming_asr_malformed_final_fails() {
  std::vector<Value> bad_finals;
  Value::Map missing_text;
  missing_text["is_final"] = Value(true);
  bad_finals.emplace_back(std::move(missing_text));
  Value::Map numeric_text;
  numeric_text["is_final"] = Value(true);
  numeric_text["text"] = Value(42);
  bad_finals.emplace_back(std::move(numeric_text));

  for (const Value& bad : bad_finals) {
    Harness h("streammalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.stream(id, "asr", bad);
    // The same PUSH -> PULL queue makes this completion a FIFO barrier: the old
    // code treats the malformed final as valid and starts VLM here.
    h.complete(id, "asr", true, text_result("must not run"));
    check(engine.wait(id, 5.0), "a malformed streaming final did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId && !terminal->error.empty(),
          "a malformed streaming final did not produce one ASR kError");
    check(log.terminal_count(id) == 1,
          "a malformed streaming final produced duplicate terminals");
    check(log.size() == 1 && log.snapshot()[0].kind == UnifiedEvent::Kind::kError,
          "a malformed streaming final emitted an ASR final or success event");
    check(h.no_submit(h.vlm_recv),
          "a malformed streaming final still started the VLM leg");
    engine.release(id);
  }
}

void test_unknown_id_wait_fails() {
  Harness h("unknownwait");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  check(!engine.wait("unknown", 0.0), "wait() reported success for an unknown id");
  check(!engine.info("unknown"), "info() returned state for an unknown id");
}

// cancel() broadcasts an abort to both stages and settles the request.
void test_cancel_propagates_to_both_stages() {
  Harness h("cancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(engine.cancel(id), "cancel rejected an active request");
  check(h.wait_abort(id), "abort did not reach the ASR stage");
  wait_until([&] { return log.has_terminal(); },
             "cancellation did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kCancelled,
        "cancel produced the wrong terminal event");
  check(engine.info(id)->state == "aborted", "coordinator state is not aborted");
  check(!engine.cancel(id), "a second cancel was not idempotent");
  engine.release(id);

  // Second request: cancel during the VLM leg aborts the VLM child id.
  const std::string id2 = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id2, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  check(engine.cancel(id2), "cancel rejected the VLM-leg request");
  check(h.wait_abort(id2 + ":vlm"),
        "VLM-leg cancel did not abort the active stage leg");
  engine.release(id2);
}

// An ASR failure stops the flow before any VLM submit.
void test_asr_failure() {
  Harness h("asrfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", false, Value{}, std::string("asr broke"));
  check(engine.wait(id, 5.0), "ASR failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kAsrStageId &&
            log.terminal_event()->error == "asr broke",
        "ASR failure produced the wrong terminal event");
  check(h.no_submit(h.vlm_recv),
        "a failed ASR leg still submitted to the VLM stage");
  engine.release(id);
}

// A VLM failure is observable with the VLM stage id.
void test_vlm_failure() {
  Harness h("vlmfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", false, Value{}, std::string("vlm broke"));
  check(engine.wait(id, 5.0), "VLM failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kVlmStageId &&
            log.terminal_event()->error == "vlm broke",
        "VLM failure produced the wrong terminal event");
  engine.release(id);
}

// VLM results accept exactly the public bare-string/text/content shapes.
void test_vlm_result_text_contract() {
  std::vector<std::pair<Value, std::string>> valid_results;
  valid_results.emplace_back(Value("bare answer"), "bare answer");
  valid_results.emplace_back(Value(text_result("text answer")), "text answer");
  Value::Map content_result;
  content_result["content"] = Value("content answer");
  valid_results.emplace_back(Value(std::move(content_result)), "content answer");

  for (const auto& [result, expected] : valid_results) {
    Harness h("vlmresult");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.complete(id + ":vlm", "vlm", true, result);
    check(engine.wait(id, 5.0), "a valid VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->text == expected,
          "a valid VLM result shape lost its text");
    check(log.terminal_count(id) == 1,
          "a valid VLM result produced duplicate terminals");
    engine.release(id);
  }

  std::vector<Value> bad_results;
  bad_results.emplace_back(Value(42));
  bad_results.emplace_back(Value::Map{});
  Value::Map numeric_text;
  numeric_text["text"] = Value(42);
  bad_results.emplace_back(std::move(numeric_text));
  for (const Value& bad : bad_results) {
    Harness h("vlmmalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.stream(id + ":vlm", "vlm", text_chunk("delta", false));
    h.complete(id + ":vlm", "vlm", true, bad);
    check(engine.wait(id, 5.0), "a malformed VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kVlmStageId && !terminal->error.empty(),
          "a malformed VLM result did not produce exactly one VLM kError");
    check(log.terminal_count(id) == 1,
          "a malformed VLM result produced duplicate terminals");
    engine.release(id);
  }
}

// The admission cap is a hard bound: overflow throws, a finished slot reopens.
void test_queue_cap_backpressure() {
  Harness h("queue");
  UnifiedConfig config;
  config.max_in_flight = 2;
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  const std::string a = engine.submit(audio_request());
  const std::string b = engine.submit(audio_request());
  check(engine.in_flight() == 2, "admission count is wrong");
  bool full = false;
  try {
    engine.submit(audio_request());
  } catch (const QueueFullError&) {
    full = true;
  }
  check(full, "the queue cap did not reject an overflow request");

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(a, "asr", true, text_result("t"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(a + ":vlm", "vlm", true, text_result("r"));
  check(engine.wait(a, 5.0), "request did not complete");
  engine.release(a);
  wait_until([&] { return engine.in_flight() == 1; },
             "a finished request did not free an admission slot");
  const std::string c = engine.submit(audio_request());
  check(engine.in_flight() == 2, "a freed slot did not admit a new request");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  engine.release(b);
  engine.release(c);
}

// Concurrent submissions share one engine mutex. The admission cap must stay a
// hard bound under a burst, and every admitted request must still complete with
// the same result the serial path produces.
void test_queue_cap_concurrent_submit() {
  Harness h("queueburst");
  UnifiedConfig config;
  config.max_in_flight = 2;
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  constexpr int kBurst = 8;
  std::atomic<int> admitted{0};
  std::atomic<int> rejected{0};
  std::atomic<bool> go{false};
  std::mutex ids_mu;
  std::vector<std::string> ids;
  std::vector<std::thread> threads;
  threads.reserve(kBurst);
  for (int i = 0; i < kBurst; ++i) {
    threads.emplace_back([&] {
      while (!go.load()) std::this_thread::yield();
      try {
        const std::string id = engine.submit(mixed_request(false, false));
        {
          std::lock_guard<std::mutex> lock(ids_mu);
          ids.push_back(id);
        }
        ++admitted;
      } catch (const QueueFullError&) {
        ++rejected;
      }
    });
  }
  go = true;
  for (auto& thread : threads) thread.join();

  check(admitted.load() == 2, "the admission cap admitted more than its limit");
  check(rejected.load() == kBurst - 2,
        "a burst over the cap did not reject the excess requests");
  check(engine.in_flight() == 2, "the admission count does not match the cap");

  for (int i = 0; i < 2; ++i) {
    auto submit = h.next_submit(h.vlm_recv);
    check(submit.has_value(), "an admitted request never reached the VLM stage");
    if (!submit) continue;
    h.stream(submit->request_id, "vlm", text_chunk("ok", false));
    h.complete(submit->request_id, "vlm", true, text_result("ok"));
  }
  for (const auto& id : ids) {
    check(engine.wait(id, 5.0), "an admitted request did not complete");
    const auto info = engine.info(id);
    check(info && info->state == "completed",
          "an admitted request did not reach the completed state");
    check(info && info->result.find("text") != nullptr &&
              info->result.find("text")->as_str() == "ok",
          "a concurrent request returned a different result than the serial path");
  }
  for (const auto& id : ids) engine.release(id);
  wait_until([&] { return engine.in_flight() == 0; },
             "released requests did not free their admission slots");
}

// Overload contract: when every admission slot is in use, the HTTP boundary
// answers 503 with Retry-After instead of queueing the request without bound.
void test_public_http_overload_503() {
  Harness h("publicoverload");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedConfig engine_config;
  engine_config.max_in_flight = 1;
  UnifiedEngine engine(h.coordinator, engine_config);
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "overload server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  const std::string body =
      R"({"model":"qwen3-unified","messages":[{"role":"user","content":"hello"}],"stream":false})";
  auto send = [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body;
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  };

  // The first request holds the only slot until its VLM leg completes.
  auto first = std::async(std::launch::async, send);
  auto submit = h.next_submit(h.vlm_recv);
  check(submit.has_value(), "the first request did not reach the VLM stage");

  // The HTTP client turns a non-2xx into client::Error, so the 503 contract is
  // read from the error object: status, overloaded marker and Retry-After.
  bool overloaded_503 = false;
  bool named_overload = false;
  bool retry_after = false;
  try {
    const auto response = send();
    overloaded_503 = response.status == 503L;
    named_overload =
        response.body.find("\"type\":\"overloaded\"") != std::string::npos;
    retry_after = client::detail::retry_after(response.headers).count() > 0;
  } catch (const client::Error& error) {
    overloaded_503 = error.status() == 503;
    named_overload = std::string(error.what()).find("overloaded") != std::string::npos;
    retry_after = error.retry_after().count() > 0;
  }
  check(overloaded_503, "an overloaded engine did not answer 503");
  check(named_overload, "the overload response did not name the overload type");
  check(retry_after, "the overload response did not carry Retry-After");

  if (submit) {
    h.complete(submit->request_id, "vlm", true, text_result("answer"));
  }
  const auto completed = first.get();
  api.stop();
  server_thread.join();
  check(completed.status == 200L,
        "the admitted request did not complete after the overload was rejected");
  wait_until([&] { return engine.in_flight() == 0; },
             "the admitted request did not free its slot");
}

// A repeated ASR final is idempotent; an out-of-order chunk is dropped.
void test_duplicate_final_and_out_of_order() {
  Harness h("dup");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("first", true));
  h.stream(id, "asr", text_chunk("second", true));  // duplicate final
  wait_until([&] { return log.size() == 1; },
             "the duplicate ASR final was not idempotent");
  check(log.snapshot()[0].text == "first",
        "the duplicate final overwrote the transcript");

  // A VLM-stage chunk before the VLM leg exists is out of order and dropped.
  h.stream(id, "vlm", text_chunk("early", false));
  h.complete(id, "asr", true, text_result("first"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  // An ASR chunk after the VLM leg started is late and dropped.
  h.stream(id, "asr", text_chunk("late", false));
  h.stream(id + ":vlm", "vlm", text_chunk("answer", false));
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "request did not complete");

  size_t finals = 0, deltas = 0;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) ++finals;
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) ++deltas;
  }
  check(finals == 1, "more than one ASR final event was emitted");
  check(deltas == 1, "an out-of-order chunk leaked into the event stream");
  engine.release(id);
}

// Image + audio: the ASR leg receives PCM only, the VLM leg receives the
// original image part plus the final transcript as a text part. Video stays
// rejected.
void test_joint_image_audio() {
  Harness h("joint");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(joint_request());
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR stage received no submit");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.find("audio") != nullptr &&
              payload.request.inputs.find("messages") == nullptr,
          "the ASR leg did not receive PCM only");
  }
  h.complete(id, "asr", true, text_result("what is on screen"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2,
          "the VLM prompt is not image part + transcript text part");
    if (content.is_array() && content.as_array().size() == 2) {
      const Value& image = content.as_array()[0];
      const Value& text = content.as_array()[1];
      check(image.find("type")->as_str() == "image_url" &&
                image.find("image_url")->find("url")->as_str() ==
                    "data:image/png;base64,AA",
            "the VLM lost the original image part");
      check(text.find("type")->as_str() == "text" &&
                text.find("text")->as_str() == "what is on screen",
            "the VLM lost the ASR transcript");
    }
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 2 &&
              modalities.as_array()[0].as_str() == "text" &&
              modalities.as_array()[1].as_str() == "image",
          "the VLM did not declare both text and image modalities");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("a settings screen"));
  check(engine.wait(id, 5.0), "joint request did not complete");
  const auto events = log.snapshot();
  check(!events.empty() &&
            events.front().kind == UnifiedEvent::Kind::kAsrFinal &&
            events.back().kind == UnifiedEvent::Kind::kComplete,
        "joint event order is not asr_final .. complete");
  engine.release(id);
}

// A remote HTTP image URL is a protocol value the engine must forward
// verbatim. The engine never fetches it; only the VLM worker does.
void test_joint_image_remote_url() {
  Harness h("jointurl");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string url = "http://172.17.0.1:18080/man_ironing.png";
  const std::string id = engine.submit(joint_request("", url));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("what is on screen"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2,
          "the remote-URL VLM prompt is not image part + transcript text part");
    if (content.is_array() && content.as_array().size() == 2) {
      const Value& image = content.as_array()[0];
      check(image.find("type")->as_str() == "image_url" &&
                image.find("image_url")->find("url")->as_str() == url,
            "the VLM lost or rewrote the remote image URL");
    }
  }
  h.complete(id + ":vlm", "vlm", true, text_result("a man ironing"));
  check(engine.wait(id, 5.0), "remote-URL joint request did not complete");
  engine.release(id);
}

// Malformed media contracts are rejected before any stage sees the work.
void test_media_contract_errors() {
  Harness h("mediaerr");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});

  auto rejects = [&](StagePayload payload, const char* message) {
    bool rejected = false;
    try {
      engine.submit(std::move(payload));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, message);
  };

  rejects(audio_request("video"), "a video modality was accepted");
  rejects(image_part_request(""), "an empty image URL was accepted");

  StagePayload no_url = image_part_request("data:image/png;base64,AA");
  no_url.request.inputs =
      Value(Value::Map{{"messages", Value::Array{Value::Map{
                          {"role", Value("user")},
                          {"content", Value::Array{part("image_url")}}}}}});
  rejects(std::move(no_url), "an image part without image_url.url was accepted");

  StagePayload two_images;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value::Array{image_part(), image_part(), Value(part("input_audio"))};
  two_images.request.inputs = Value::Map{{"messages", Value::Array{Value(std::move(message))}}};
  rejects(std::move(two_images), "a request with two image parts was accepted");

  StagePayload empty;
  rejects(std::move(empty), "a request without messages was accepted");
}

// A VLM-only request built from one public content shape. When as_string is
// true the content is a plain OpenAI string; otherwise it is a text part plus
// an optional image part. No binary audio input is attached.
StagePayload vlm_only_request(bool as_string, bool with_image) {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  if (as_string) {
    message["content"] = Value("what is two plus two");
  } else {
    Value::Array content;
    Value::Map text;
    text["type"] = Value("text");
    text["text"] = Value("describe this image");
    content.push_back(Value(std::move(text)));
    if (with_image) content.push_back(image_part());
    message["content"] = Value(std::move(content));
  }
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// Plain string content without audio: no ASR leg exists, the VLM leg receives
// the text directly and the event stream carries no ASR events.
void test_vlm_only_string_content() {
  Harness h("vlmstring");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(vlm_only_request(true, false));
  check(h.no_submit(h.asr_recv), "a VLM-only request reached the ASR stage");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value() && vlm_submit->request_id == id,
        "the VLM-only request did not run under the public request id");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    check(payload.request.inputs.find("audio") == nullptr,
          "the VLM-only payload carried an audio input");
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 1 &&
              content.as_array()[0].find("text")->as_str() ==
                  "what is two plus two",
          "the VLM lost the string content");
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 1 &&
              modalities.as_array()[0].as_str() == "text",
          "the VLM-only request declared non-text modalities");
    h.stream(vlm_submit->request_id, "vlm", text_chunk("four", false));
    h.complete(vlm_submit->request_id, "vlm", true, text_result("four"));
  }
  check(engine.wait(id, 5.0), "VLM-only string request did not complete");
  const auto events = log.snapshot();
  check(!events.empty() &&
            std::none_of(events.begin(), events.end(),
                         [](const UnifiedEvent& event) {
                           return event.kind ==
                                      UnifiedEvent::Kind::kAsrFinal ||
                                  event.kind ==
                                      UnifiedEvent::Kind::kAsrPartial;
                         }) &&
            events.front().kind == UnifiedEvent::Kind::kVlmDelta &&
            events.back().kind == UnifiedEvent::Kind::kComplete,
        "VLM-only events must start at vlm_delta with no ASR events");
  engine.release(id);
}

// Image + text without audio: the image rides straight to the VLM leg, the
// modalities declare text and image, and no ASR leg is submitted.
void test_vlm_only_image_text() {
  Harness h("vlmimage");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(vlm_only_request(false, true));
  check(h.no_submit(h.asr_recv), "an image+text request reached the ASR stage");
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "the image+text request did not reach the VLM");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2 &&
              content.as_array()[0].find("text")->as_str() ==
                  "describe this image" &&
              content.as_array()[1].find("type")->as_str() == "image_url",
          "the VLM prompt is not instruction + image");
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 2 &&
              modalities.as_array()[0].as_str() == "text" &&
              modalities.as_array()[1].as_str() == "image",
          "the image+text request did not declare text and image");
    h.complete(vlm_submit->request_id, "vlm", true,
               text_result("a settings screen"));
  }
  check(engine.wait(id, 5.0), "image+text request did not complete");
  const auto image_events = log.snapshot();
  check(!image_events.empty() &&
            image_events.back().kind == UnifiedEvent::Kind::kComplete,
        "image+text request did not complete successfully");
  engine.release(id);
}

// Session history on VLM-only turns must keep the real user text. There is no
// ASR transcript on these turns, so the recorded history user side is the
// string content or text part, never an empty message.
void test_vlm_only_session_history() {
  Harness h("vlmhist");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  auto with_history = [](StagePayload request, const std::string& session) {
    Value::Map inputs = request.request.inputs.as_map();
    inputs["session_id"] = Value(session);
    inputs["session_history"] = Value(true);
    request.request.inputs = Value(std::move(inputs));
    return request;
  };
  // Drives one VLM-only turn and returns its VLM leg payload.
  auto run_turn = [&](StagePayload request, const std::string& answer) {
    const std::string id = engine.submit(std::move(request));
    check(h.no_submit(h.asr_recv), "a history VLM-only turn reached ASR");
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value() && vlm->request_id == id,
          "the VLM-only history leg lost the request id");
    StagePayload payload;
    if (vlm) payload = StagePayload::from_value(vlm->data);
    h.complete(id, "vlm", true, text_result(answer));
    check(engine.wait(id, 5.0), "a VLM-only history turn did not complete");
    engine.release(id);
    return payload;
  };

  // Turn 1: plain string content.
  StagePayload first = run_turn(
      with_history(vlm_only_request(true, false), "vlm-sess"), "four");
  check(prompt_roles(first).size() == 1,
        "the first VLM-only turn already carried history");

  // Turn 2: image+text in the same session. It must see turn 1's real user
  // string and answer, then its own text part (the image is not user text).
  StagePayload second = run_turn(
      with_history(vlm_only_request(false, true), "vlm-sess"),
      "a settings screen");
  const auto roles = prompt_roles(second);
  if (roles.size() != 3) {
    check(false, "the second VLM-only turn did not carry one prior turn");
  } else {
    check(roles[0].first == "user" &&
              roles[0].second == "what is two plus two" &&
              roles[1].first == "assistant" && roles[1].second == "four" &&
              roles[2].first == "user" &&
              roles[2].second == "describe this image",
          "VLM-only history lost the real user text or answer");
  }
}

// Two sequential requests in one session. session_id is a grouping label only:
// ids, sequences and terminals stay per request, and neither request inherits
// the other's transcript or VLM answer.
void test_session_reuse() {
  Harness h("session");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  auto run = [&](const std::string& transcript, const std::string& answer) {
    const std::string id = engine.submit(joint_request("session-1"));
    auto asr = h.next_submit(h.asr_recv);
    check(asr && asr->request_id == id, "ASR leg id does not match the request");
    h.complete(id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm && vlm->request_id == id + ":vlm", "VLM leg id is not a child id");
    if (vlm) {
      // Keep the payload alive: content points into it.
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const Value& content = *payload.request.inputs.find("messages")
                                  ->as_array()
                                  .at(0)
                                  .find("content");
      check(content.is_array() && content.as_array().size() == 2 &&
                content.as_array()[1].find("text")->as_str() == transcript,
            "the VLM leg did not receive this request's transcript");
    }
    h.stream(id + ":vlm", "vlm", text_chunk(answer, false));
    h.complete(id + ":vlm", "vlm", true, text_result(answer));
    check(engine.wait(id, 5.0), "session request did not complete");
    return id;
  };

  const std::string first = run("first transcript", "first answer");
  const std::string second = run("second transcript", "second answer");
  check(first != second, "two session requests shared a request id");

  for (const std::string& id : {first, second}) {
    int64_t next_seq = 0;
    int terminals = 0;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-1", "event lost the session label");
      check(event.seq == next_seq++, "event sequence is not gap-free per request");
      if (event.is_final) ++terminals;
    }
    check(next_seq == 3, "unexpected event count for one session request");
    check(terminals == 2, "a session request did not emit asr_final and complete");
  }
  engine.release(first);
  engine.release(second);
}

// Concurrent requests in the same session interleave. Each event keeps its own
// request id and sequence; no delta or terminal crosses over.
void test_session_reuse_concurrent() {
  Harness h("sessionconc");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string a = engine.submit(joint_request("session-2"));
  const std::string b = engine.submit(joint_request("session-2"));
  auto asr_a = h.next_submit(h.asr_recv);
  auto asr_b = h.next_submit(h.asr_recv);
  check(asr_a && asr_b && asr_a->request_id != asr_b->request_id,
        "concurrent session requests did not reach the ASR stage separately");

  h.complete(a, "asr", true, text_result("alpha"));
  h.complete(b, "asr", true, text_result("beta"));
  auto vlm_a = h.next_submit(h.vlm_recv);
  auto vlm_b = h.next_submit(h.vlm_recv);
  check(vlm_a && vlm_b, "both concurrent VLM legs did not start");

  h.stream(a + ":vlm", "vlm", text_chunk("A", false));
  h.stream(b + ":vlm", "vlm", text_chunk("B", false));
  h.complete(b + ":vlm", "vlm", true, text_result("B"));
  h.complete(a + ":vlm", "vlm", true, text_result("A"));
  check(engine.wait(a, 5.0) && engine.wait(b, 5.0),
        "concurrent session requests did not settle");

  for (const std::string& id : {a, b}) {
    const std::string expected = id == a ? "A" : "B";
    int64_t next_seq = 0;
    std::string delta, terminal;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-2", "event lost the session label");
      check(event.seq == next_seq++, "concurrent event sequence is not per request");
      if (event.kind == UnifiedEvent::Kind::kVlmDelta) delta = event.text;
      if (event.kind == UnifiedEvent::Kind::kComplete) terminal = event.text;
    }
    check(delta == expected, "a VLM delta crossed between session requests");
    check(terminal == expected, "a terminal crossed between session requests");
  }
  engine.release(a);
  engine.release(b);
}

// Cancel before the queued ASR completion is pumped. With no pump thread the
// Coordinator still marks the request aborted synchronously inside cancel(),
// then the stale success completion must be ignored: one kCancelled, no VLM.

}  // namespace
