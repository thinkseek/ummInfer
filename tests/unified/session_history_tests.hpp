// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "harness.hpp"

namespace {

// Opt-in history: the second turn's VLM prompt prepends the first completed
// turn as user/assistant messages, in order, ahead of the current user turn.
void test_session_history_order() {
  Harness h("historder");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  StagePayload first = complete_turn(h, engine, history_request("sess"),
                                     "first transcript", "first answer");
  check(prompt_roles(first).size() == 1,
        "the first history turn already carried prior context");

  StagePayload second = complete_turn(h, engine, history_request("sess"),
                                      "second transcript", "second answer");
  const auto roles = prompt_roles(second);
  if (roles.size() != 3) {
    check(false, "the second turn did not carry one prior turn plus itself");
  } else {
    check(roles[0].first == "user" && roles[0].second == "first transcript" &&
              roles[1].first == "assistant" && roles[1].second == "first answer" &&
              roles[2].first == "user" && roles[2].second == "second transcript",
          "history order or content is wrong");
  }
}

// Default off at two levels: an engine without the feature rejects the opt-in,
// and an enabled engine leaves a plain session_id request without history.
void test_session_history_default_off() {
  {
    Harness h("histoff");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    h.start_pump();
    bool rejected = false;
    try {
      engine.submit(history_request("sess"));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, "a default engine accepted session_history");
    check(h.no_submit(h.asr_recv), "a rejected history request reached ASR");
  }
  {
    Harness h("histlabel");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    // A request carrying only the label: no opt-in, no history on the next
    // turn even though the engine supports it.
    complete_turn(h, engine, joint_request("sess"), "t1", "a1");
    StagePayload next = complete_turn(h, engine, joint_request("sess"), "t2", "a2");
    check(prompt_roles(next).size() == 1,
          "a label-only request inherited history without opt-in");
  }
}

// Sessions never share history, and an opt-in without an explicit session_id
// is rejected rather than silently attached to a generated label.
void test_session_history_isolation() {
  Harness h("histiso");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();

  complete_turn(h, engine, history_request("sess-a"), "a transcript", "a answer");
  StagePayload other =
      complete_turn(h, engine, history_request("sess-b"), "b transcript", "b answer");
  check(prompt_roles(other).size() == 1,
        "history leaked between two sessions");

  StagePayload no_label = history_request("sess-x");
  no_label.request.inputs.as_map_mut().erase("session_id");
  bool rejected = false;
  try {
    engine.submit(no_label);
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "session_history without session_id was accepted");
}

// Cancelled and failed requests never enter history: the next turn sees only
// the successful turns, never a failed transcript or answer.
void test_session_history_cancel_error_excluded() {
  Harness h("histfail");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  // A cancelled turn.
  const std::string cancelled = engine.submit(history_request("sess"));
  check(h.next_submit(h.asr_recv).has_value(), "cancel turn: no ASR submit");
  check(engine.cancel(cancelled), "cancel rejected an active history request");
  h.complete(cancelled, "asr", false, Value{}, std::string("aborted"));
  check(engine.wait(cancelled, 5.0), "cancelled turn did not settle");
  check(log.terminal_for(cancelled)->kind == UnifiedEvent::Kind::kCancelled,
        "the cancelled turn did not report kCancelled");
  engine.release(cancelled);

  // A failed turn: ASR stage failure, no VLM leg.
  const std::string failed = engine.submit(history_request("sess"));
  check(h.next_submit(h.asr_recv).has_value(), "fail turn: no ASR submit");
  h.complete(failed, "asr", false, Value{}, std::string("backend reset"));
  check(engine.wait(failed, 5.0), "failed turn did not settle");
  check(log.terminal_for(failed)->kind == UnifiedEvent::Kind::kError,
        "the failed turn did not report kError");
  check(h.no_submit(h.vlm_recv), "a failed turn started the VLM leg");
  engine.release(failed);

  StagePayload good =
      complete_turn(h, engine, history_request("sess"), "good transcript", "good answer");
  check(prompt_roles(good).size() == 1,
        "a cancelled or failed turn entered session history");
}

// Turn-count bound: only the newest max_turns completed turns stay. Eviction
// runs when a turn is recorded, so a turn still sees the turns present at its
// own start; the oldest turn disappears one turn later.
void test_session_history_turn_eviction() {
  Harness h("histturns");
  UnifiedEngine engine(h.coordinator, history_config(2, 16384));
  h.start_pump();

  complete_turn(h, engine, history_request("sess"), "t1", "a1");
  complete_turn(h, engine, history_request("sess"), "t2", "a2");
  // The third turn starts while both prior turns are still retained.
  StagePayload third =
      complete_turn(h, engine, history_request("sess"), "t3", "a3");
  auto roles = prompt_roles(third);
  check(roles.size() == 5, "third turn prompt has the wrong size");
  if (roles.size() == 5) {
    check(roles[0].second == "t1" && roles[2].second == "t2" &&
              roles[4].second == "t3",
          "the third turn did not see both retained turns in order");
  }

  // Recording the third turn evicted the first; the fourth sees t2 and t3.
  StagePayload fourth =
      complete_turn(h, engine, history_request("sess"), "t4", "a4");
  roles = prompt_roles(fourth);
  check(roles.size() == 5 && roles[0].second == "t2" && roles[4].second == "t4",
        "turn eviction order is not FIFO");
}

// Character bound: a small budget evicts the oldest turns when a new turn is
// recorded, and one oversized single turn evicts itself rather than staying
// unbounded.
void test_session_history_char_eviction() {
  Harness h("histchars");
  UnifiedEngine engine(h.coordinator, history_config(8, 20));
  h.start_pump();

  complete_turn(h, engine, history_request("sess"), "0123456789", "0123456789");
  // The second turn starts while the 20-char first turn is still retained.
  StagePayload second =
      complete_turn(h, engine, history_request("sess"), "new", "new");
  auto roles = prompt_roles(second);
  check(roles.size() == 3 && roles[0].second == "0123456789" &&
            roles[2].second == "new",
        "the second turn did not see the retained first turn");

  // Recording the second turn overflowed the 20-char budget and evicted the
  // first. One turn larger than the whole budget is then evicted on recording,
  // so the next turn starts with empty history.
  complete_turn(h, engine, history_request("sess"),
                std::string(30, 'x'), std::string(30, 'y'));
  StagePayload after =
      complete_turn(h, engine, history_request("sess"), "z", "z");
  check(prompt_roles(after).size() == 1,
        "an oversized turn survived the character budget");
}

// Global session cap: with max_sessions=2, recording a third session evicts
// the session with the oldest last turn, even though each session holds only
// one turn.
void test_session_history_session_eviction() {
  UnifiedConfig config = history_config(8, 16384);
  config.session_history.max_sessions = 2;
  Harness h("histsessions");
  UnifiedEngine engine(h.coordinator, config);
  h.start_pump();

  complete_turn(h, engine, history_request("s1"), "t1", "a1");
  complete_turn(h, engine, history_request("s2"), "t2", "a2");
  // A new turn on s1 makes s2 the least recently used session.
  complete_turn(h, engine, history_request("s1"), "t1b", "a1b");
  complete_turn(h, engine, history_request("s3"), "t3", "a3");

  // Check the retained session first: recording a fresh s2 turn below would
  // itself evict the current oldest session.
  StagePayload s1_later =
      complete_turn(h, engine, history_request("s1"), "t1c", "a1c");
  check(prompt_roles(s1_later).size() == 5,
        "a retained session lost its turns after another session was evicted");
  StagePayload s2_later =
      complete_turn(h, engine, history_request("s2"), "t2b", "a2b");
  check(prompt_roles(s2_later).size() == 1,
        "the evicted session retained its history");
}

// Concurrent requests in one history-enabled session keep independent request
// ids, per-request seq from zero, and one terminal each. A turn is visible to
// later turns only after its own successful terminal.
void test_session_history_concurrent() {
  Harness h("histconc");
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string a = engine.submit(history_request("sess"));
  const std::string b = engine.submit(history_request("sess"));
  check(a != b, "concurrent turns shared a request id");
  auto asr_a = h.next_submit(h.asr_recv);
  auto asr_b = h.next_submit(h.asr_recv);
  check(asr_a && asr_b, "concurrent turns did not both reach ASR");

  // Neither turn sees the other: both were submitted before either completed.
  h.complete(a, "asr", true, text_result("ta"));
  h.complete(b, "asr", true, text_result("tb"));
  auto vlm_a = h.next_submit(h.vlm_recv);
  auto vlm_b = h.next_submit(h.vlm_recv);
  check(vlm_a && vlm_b, "concurrent VLM legs did not start");
  for (const auto* submit : {&vlm_a, &vlm_b}) {
    StagePayload payload = StagePayload::from_value((*submit)->data);
    check(payload.request.inputs.find("messages")->as_array().size() == 1,
          "an in-flight turn observed another turn's uncommitted history");
  }

  h.complete(a + ":vlm", "vlm", true, text_result("aa"));
  h.complete(b + ":vlm", "vlm", true, text_result("ab"));
  check(engine.wait(a, 5.0) && engine.wait(b, 5.0),
        "concurrent turns did not settle");
  for (const std::string& id : {a, b}) {
    int64_t next_seq = 0;
    int terminals = 0;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.seq == next_seq++, "concurrent history seq is not per request");
      if (event.is_final) ++terminals;
    }
    check(terminals == 2, "a concurrent turn lost its asr_final or terminal");
    engine.release(id);
  }

  // A later turn sees exactly the two committed turns.
  StagePayload later =
      complete_turn(h, engine, history_request("sess"), "tc", "ac");
  const auto roles = prompt_roles(later);
  check(roles.size() == 5, "the later turn did not see both committed turns");
}

// History is process-local: an engine on a fresh Coordinator starts with empty
// history. The store has no persistence path.
void test_session_history_restart_drops() {
  StagePayload later;
  {
    Harness h("histrestart");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    complete_turn(h, engine, history_request("sess"), "old transcript", "old answer");
  }
  {
    Harness h("histrestart2");
    UnifiedEngine engine(h.coordinator, history_config(8, 16384));
    h.start_pump();
    later = complete_turn(h, engine, history_request("sess"),
                          "new transcript", "new answer");
  }
  check(prompt_roles(later).size() == 1,
        "history survived an engine restart");
}

// Public HTTP contract for the session_history opt-in: rejected outside
// unified mode and on a default engine, accepted and carried into the VLM
// prompt only when the server enabled it alongside an explicit session_id.
void test_public_http_session_history() {
  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  auto body = [&](nlohmann::json extra) {
    extra["model"] = "qwen3-unified";
    extra["stream"] = false;
    extra["messages"] = nlohmann::json::array(
        {nlohmann::json{{"role", "user"},
                        {"content", nlohmann::json::array({audio_part})}}});
    return extra;
  };
  auto post = [](OpenAIApi& api, const nlohmann::json& payload) {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = payload.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      client::detail::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what());
    }
    return std::string();
  };

  // Non-unified server: 400, never accepted and dropped.
  {
    Harness h("histhttpplain");
    ApiConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model = "qwen3-unified";
    config.request_timeout_s = 5.0;
    OpenAIApi api(h.coordinator, config);
    check(api.bind(), "plain history server did not bind");
    h.start_pump();
    std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
    const std::string error = post(api, body({{"session_history", true}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("requires unified mode") != std::string::npos,
          "non-unified server accepted session_history");
    check(h.no_submit(h.asr_recv),
          "a rejected session_history request reached the ASR stage");
    api.stop();
    server_thread.join();
  }

  // Unified engine started without the switch: the opt-in is refused.
  {
    Harness h("histhttpdisabled");
    ApiConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model = "qwen3-unified";
    config.request_timeout_s = 5.0;
    OpenAIApi api(h.coordinator, config);
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    api.set_unified_engine(engine, "vlm");
    check(api.bind(), "disabled-history server did not bind");
    h.start_pump();
    std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());
    std::string error =
        post(api, body({{"session_history", true}, {"session_id", "sess"}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("not enabled") != std::string::npos,
          "a default engine accepted the history opt-in over HTTP");
    error = post(api, body({{"session_history", "yes"}}));
    check(error.find("HTTP 400") != std::string::npos &&
              error.find("must be a boolean") != std::string::npos,
          "a non-boolean session_history was accepted");
    api.stop();
    server_thread.join();
  }

  // Enabled engine: missing session_id is 400; with the label the second turn
  // carries the first turn in its VLM prompt.
  Harness h("histhttpok");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, history_config(8, 16384));
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "history-enabled server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  std::string error = post(api, body({{"session_history", true}}));
  check(error.find("HTTP 400") != std::string::npos &&
            error.find("requires a nonempty session_id") != std::string::npos,
        "session_history without session_id was accepted");

  auto run_turn = [&](const std::string& transcript, const std::string& answer) {
    auto response_future = std::async(std::launch::async, [&] {
      return post(api, body({{"session_history", true},
                             {"session_id", "http-session"}}));
    });
    auto asr = h.next_submit(h.asr_recv);
    check(asr.has_value(), "HTTP history turn: no ASR submit");
    const std::string id = asr ? asr->request_id : std::string();
    h.complete(id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm.has_value(), "HTTP history turn: no VLM submit");
    StagePayload vlm_payload;
    if (vlm) vlm_payload = StagePayload::from_value(vlm->data);
    h.complete(id + ":vlm", "vlm", true, text_result(answer));
    check(response_future.get().empty(), "a successful history turn returned an HTTP error");
    wait_until([&] { return engine.in_flight() == 0; },
               "HTTP history turn stayed in flight");
    return vlm_payload;
  };

  StagePayload first = run_turn("first transcript", "first answer");
  check(prompt_roles(first).size() == 1,
        "the first HTTP history turn already carried context");
  StagePayload second = run_turn("second transcript", "second answer");
  const auto roles = prompt_roles(second);
  check(roles.size() == 3 && roles[0].second == "first transcript" &&
            roles[1].second == "first answer" &&
            roles[2].second == "second transcript",
        "the second HTTP turn did not receive the prior turn as history");

  api.stop();
  server_thread.join();
}

// Without a unified engine, session_id must be rejected, not accepted and
// dropped. This keeps the server-wide unknown-field contract intact.
void test_public_http_session_id_requires_unified() {
  Harness h("plainsession");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  check(api.bind(), "plain server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  wait_http_ready(api.bound_port());

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "plain-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"}, {"content", "hello"}}})}};
  client::detail::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                "/v1/chat/completions";
  request.body = body.dump();
  request.headers.emplace("Content-Type", "application/json");
  request.timeout = client::Milliseconds{5000};
  bool rejected_with_400 = false;
  try {
    client::detail::perform(request);
  } catch (const client::Error& exc) {
    rejected_with_400 =
        std::string(exc.what()).find("HTTP 400") != std::string::npos &&
        std::string(exc.what()).find("requires unified mode") != std::string::npos;
  }
  check(rejected_with_400,
        "a non-unified request accepted session_id instead of rejecting it");
  check(h.no_submit(h.asr_recv),
        "a non-unified request with session_id reached the ASR stage");
  api.stop();
  server_thread.join();
}

// ---- Mixed output (image + tts) contract tests ---------------------------
//
// The image and tts "stages" are the same bound PULL/PUSH fakes as ASR/VLM.

}  // namespace
