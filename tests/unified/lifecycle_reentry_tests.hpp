// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "harness.hpp"

namespace {

void test_cancel_before_pumped_completion() {
  Harness h("stale", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "ASR stage received no submit");

  // A sentinel request whose terminal we can observe. Both completions go on
  // the one PUSH -> PULL completion socket, which is FIFO, so seeing the
  // sentinel terminal proves the stale completion ahead of it was consumed.
  const std::string sentinel = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "sentinel ASR stage received no submit");

  // Queue the ASR success for the cancelled request, then the sentinel's
  // failure (a failure terminates without starting a VLM leg). Nobody drains
  // either message until the pump starts.
  h.complete(id, "asr", true, text_result("transcript"));
  h.complete(sentinel, "asr", false, Value{}, std::string("sentinel failure"));
  check(engine.cancel(id), "cancel rejected an active request");
  check(log.terminal_for(id).has_value() &&
            log.terminal_for(id)->kind == UnifiedEvent::Kind::kCancelled,
        "cancel before pumping did not settle as kCancelled");

  // Drain the queue. The sentinel terminal is a deterministic processing
  // barrier: FIFO delivery means the stale completion was already dequeued.
  h.start_pump();
  wait_until([&] { return log.has_terminal(sentinel); },
             "the sentinel completion was not processed");
  check(log.terminal_for(sentinel) &&
            log.terminal_for(sentinel)->kind == UnifiedEvent::Kind::kError,
        "the sentinel did not terminate as expected");

  // The stale success reached a terminal record, so it neither resurrected the
  // request nor started the VLM leg, and produced no second terminal.
  check(log.terminal_count(id) == 1, "the stale completion added a terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after cancellation");
  engine.release(id);
  engine.release(sentinel);
}

// Both interleavings of the ASR->VLM handoff, repeated. Every request must end
// as exactly one kCancelled; any VLM leg that did start must be aborted.
void test_cancel_handoff_race_stress() {
  Harness h("racestress");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  constexpr int kIterations = 40;
  for (int i = 0; i < kIterations; ++i) {
    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    if ((i % 2) == 0) {
      // Completion queued first; cancel may win or lose the pump race.
      h.complete(id, "asr", true, text_result("t"));
      engine.cancel(id);
    } else {
      engine.cancel(id);
      h.complete(id, "asr", true, text_result("t"));
    }
    check(engine.wait(id, 10.0), "a stress iteration did not settle");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
          "a handoff race did not settle as kCancelled");
    check(log.terminal_count(id) == 1,
          "a handoff race produced a duplicate terminal");

    // Every VLM submit that escaped must have a matching abort.
    std::set<std::string> vlm_submits;
    while (true) {
      auto msg = h.vlm_recv.recv_timeout(20);
      if (!msg) break;
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        vlm_submits.insert(submit->request_id);
      } else if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        h.record_abort(abort->request_id);
      }
    }
    for (const auto& child : vlm_submits) {
      check(h.saw_abort(child), "a started VLM leg was not aborted");
    }
    engine.release(id);
  }
}

// Deterministic reproduction of the ASR->VLM handoff race:
//   1. the Coordinator's ASR record is already "completed", but the engine has
//      not observed that completion (its record is still in phase kAsr);
//   2. cancel() then aborts an already-terminal ASR id, which is a no-op;
//   3. the success completion is handed to the engine afterwards.
// The engine must settle as exactly one kCancelled and must not start the VLM
// leg. This fails on the pre-fix engine, which started the VLM leg instead.
void test_cancel_after_asr_completed_swallows_vlm() {
  Harness h("swallow", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  // Save the engine sink, then install a dropping sink that resolves a promise
  // when the ASR completion is consumed. The pump only ever reads this dropping
  // sink; it is installed before start_pump and never reassigned, so there is
  // no concurrent read/write of coordinator.on_complete. The saved engine sink
  // is a separate copy invoked only on this test thread.
  auto engine_sink = h.coordinator.on_complete;
  std::promise<void> dropped;
  std::future<void> dropped_future = dropped.get_future();
  h.coordinator.on_complete = [&dropped, id](const RequestInfo& info) {
    if (info.request_id == id) dropped.set_value();
  };
  h.start_pump();
  h.complete(id, "asr", true, text_result("transcript"));

  // The future synchronizes with the pump thread: once wait() returns, the
  // dropping callback has run past set_value, the Coordinator ASR record is
  // "completed", and the engine has provably never seen that completion.
  (void)wait_future_ready(dropped_future, "dropped_future timed out");
  {
    auto info = h.coordinator.info(id);
    check(info && info->state == "completed",
          "the ASR coordinator record did not settle as completed");
  }

  // ASR is terminal in the Coordinator: abort(id) is now a no-op, exactly the
  // reviewer's window. cancel must still record the intent.
  check(engine.cancel(id), "cancel rejected the handoff request");

  // Deliver the success the engine has not seen yet, on this thread.
  RequestInfo asr_done;
  asr_done.request_id = id;
  asr_done.state = "completed";
  asr_done.current_stage = "asr";
  asr_done.result = text_result("transcript");
  engine_sink(asr_done);

  const auto terminal = log.terminal_for(id);
  check(terminal.has_value() &&
            terminal->kind == UnifiedEvent::Kind::kCancelled,
        "a cancel whose abort lost to ASR completion did not yield kCancelled");
  check(log.terminal_count(id) == 1, "the handoff produced a duplicate terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after a handoff cancellation");
  engine.release(id);
}

// An out-of-order VLM chunk takes the forwarding path. The previous sink
// reenters info() on the pump thread, which deadlocks if forwarding holds mu_.
void test_previous_sink_reentry_does_not_deadlock() {
  Harness h("sinkreentry");
  UnifiedEngine* engine_ptr = nullptr;
  std::promise<std::optional<RequestInfo>> reentered;
  auto reentered_future = reentered.get_future();
  h.coordinator.on_stream = [&](const StreamMessage& msg) {
    reentered.set_value(engine_ptr->info(msg.request_id));
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  engine_ptr = &engine;
  h.start_pump();

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "vlm", text_chunk("early", false));
  const auto status = reentered_future.wait_for(std::chrono::seconds(1));
  check(status == std::future_status::ready,
        "previous sink reentry deadlocked against engine mu_");
  if (status != std::future_status::ready) std::abort();
  const auto info = reentered_future.get();
  check(info && info->request_id == id && info->state == "running" &&
            info->current_stage == kAsrStageId,
        "previous sink reentry returned the wrong unified state");
  engine.release(id);
}

// Hold the partial callback while cancel() runs on another thread. The terminal
// must queue behind the already-sequenced partial, including callback reentry.
void test_cancel_cannot_overtake_partial_dispatch() {
  Harness h("partialcancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  std::promise<void> partial_done;
  auto partial_done_future = partial_done.get_future();

  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      partial_entered.set_value();
      (void)wait_future_ready(release_partial_future, "release_partial_future timed out");
    }
    log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) partial_done.set_value();
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  (void)wait_future_ready(partial_entered_future, "partial_entered_future timed out");
  check(engine.cancel(id), "cancel rejected the partial-dispatch request");
  check(log.size() == 0, "cancel terminal overtook the blocked partial callback");
  release_partial.set_value();
  (void)wait_future_ready(partial_done_future, "partial_done_future timed out");
  check(engine.wait(id, 5.0), "cancel during partial dispatch did not settle");
  wait_until([&] { return log.terminal_count(id) == 1; },
             "cancel during partial dispatch did not emit its terminal");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "single-drainer dispatch did not preserve partial then cancelled order");
  engine.release(id);

  // Same-thread reentry: cancel from inside on_event must only enqueue the
  // terminal. The outer drainer emits it after the partial callback returns.
  EventLog reentrant_log;
  std::string reentrant_id;
  std::promise<void> reentrant_partial_done;
  auto reentrant_partial_future = reentrant_partial_done.get_future();
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      check(engine.cancel(reentrant_id), "reentrant cancel rejected the request");
    }
    reentrant_log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      reentrant_partial_done.set_value();
    }
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { reentrant_log.finish(event); };
  reentrant_id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no reentrant submit");
  h.stream(reentrant_id, "asr", text_chunk("reentrant", false));
  (void)wait_future_ready(reentrant_partial_future, "reentrant_partial_future timed out");
  check(engine.wait(reentrant_id, 5.0), "reentrant cancel did not settle");
  wait_until([&] { return reentrant_log.terminal_count(reentrant_id) == 1; },
             "reentrant cancel did not emit its terminal");
  const auto reentrant_events = reentrant_log.snapshot();
  check(reentrant_events.size() == 2 &&
            reentrant_events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            reentrant_events[1].kind == UnifiedEvent::Kind::kCancelled,
        "reentrant cancel recursively overtook the partial callback");
  engine.release(reentrant_id);
}

// Forwarded callbacks share the unified event order. Cancel either from another
// thread or from inside on_event; neither may complete the old sink before the
// earlier stream callback has returned.
void test_forwarded_sink_cancel_order(bool reentrant) {
  Harness h(reentrant ? "fwdreenter" : "fwdthread");
  std::mutex forwarded_mu;
  std::vector<std::string> forwarded;
  bool stream_active = false;
  bool complete_overlapped_stream = false;
  std::promise<void> stream_entered;
  auto stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  auto release_stream_future = release_stream.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    {
      std::lock_guard<std::mutex> lock(forwarded_mu);
      forwarded.push_back("stream");
      stream_active = true;
    }
    stream_entered.set_value();
    (void)wait_future_ready(release_stream_future, "release_stream_future timed out");
    std::lock_guard<std::mutex> lock(forwarded_mu);
    stream_active = false;
  };
  h.coordinator.on_complete = [&](const RequestInfo&) {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    complete_overlapped_stream |= stream_active;
    forwarded.push_back("complete");
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();

  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      if (reentrant) {
        check(engine.cancel(id), "reentrant cancel rejected the forwarding request");
      } else {
        partial_entered.set_value();
        (void)wait_future_ready(release_partial_future, "release_partial_future timed out");
      }
    }
    log.add(event);
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no forwarding submit");
  h.stream(id, "asr", text_chunk("partial", false));
  if (!reentrant) {
    (void)wait_future_ready(partial_entered_future, "partial_entered_future timed out");
    check(engine.cancel(id), "cross-thread cancel rejected the forwarding request");
    release_partial.set_value();
  }
  (void)wait_future_ready(stream_entered_future, "stream_entered_future timed out");
  release_stream.set_value();
  check(engine.wait(id, 5.0), "cancel during forwarding did not settle");
  wait_until([&] {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    return forwarded.size() == 2 && !stream_active &&
           log.terminal_count(id) == 1;
  }, "forwarded stream/completion or unified terminal did not arrive");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "forwarding changed unified partial/cancelled order");
  check(log.terminal_count(id) == 1, "forwarding produced a duplicate terminal");

  std::vector<std::string> order;
  bool overlapped = false;
  {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    order = forwarded;
    overlapped = complete_overlapped_stream;
  }
  check(!overlapped, "forwarded completion reentered the active stream sink");
  std::cerr << (reentrant ? "FORWARD_ORDER_REENTRANT=" : "FORWARD_ORDER_THREAD=");
  for (const auto& call : order) std::cerr << call << ",";
  std::cerr << "\n";
  check(order == std::vector<std::string>{"stream", "complete"},
        "forwarded completion overtook the earlier stream callback");
  engine.release(id);
}

// A previous sink blocked on the pump proves run() is active. Construction must
// reject the exchange without modifying that sink instead of racing it.
void test_construction_after_pump_start_rejected() {
  Harness h("lateconstruct");
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::promise<void> previous_done;
  auto previous_done_future = previous_done.get_future();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    entered.set_value();
    (void)wait_future_ready(release_future, "release_future timed out");
    previous_done.set_value();
  };
  h.start_pump();

  const std::string id = h.coordinator.submit(audio_request(), "asr");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  (void)wait_future_ready(entered_future, "entered_future timed out");

  bool rejected = false;
  try {
    UnifiedEngine late(h.coordinator, UnifiedConfig{});
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "construction exchanged sinks after the pump had started");
  release.set_value();
  (void)wait_future_ready(previous_done_future, "previous_done_future timed out");
}

// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
// The destructor cannot return while a pump callback still uses the engine.
// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
void test_destructor_synchronizes_inflight_callback() {
  Harness h("destruct");
  std::atomic<int> forwarded_calls{0};
  std::promise<void> entered;
  std::future<void> entered_future = entered.get_future();
  std::promise<void> release;
  std::shared_future<void> release_future = release.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    if (forwarded_calls.fetch_add(1) == 0) {
      entered.set_value();
      (void)wait_future_ready(release_future, "release_future timed out");
    }
  };

  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine->on_event = [&](const UnifiedEvent& event) { log.add(event); };
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  (void)wait_future_ready(entered_future, "entered_future timed out");
  const auto copied_sink = h.coordinator.on_stream;

  std::promise<void> destroy_started;
  std::future<void> destroy_started_future = destroy_started.get_future();
  std::promise<void> destroyed;
  std::future<void> destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    destroy_started.set_value();
    engine.reset();
    destroyed.set_value();
  });
  (void)wait_future_ready(destroy_started_future, "destroy_started_future timed out");
  const std::future_status early = destroyed_future.wait_for(std::chrono::seconds(1));
  check(early == std::future_status::timeout,
        "destructor returned while an inflight callback still used the engine");
  release.set_value();
  (void)wait_future_ready(destroyed_future, "destroyed_future timed out");
  destroyer.join();
  check(log.size() == 1, "the inflight callback did not finish before destruction");

  StreamMessage after;
  after.request_id = "unknown";
  after.from_stage = "asr";
  after.chunk = text_chunk("after", false);
  copied_sink(after);
  check(forwarded_calls.load() == 2,
        "a copied sink did not forward safely after engine destruction");
}

struct RetiredSinkProbe {
  std::mutex mu;
  std::vector<std::string> calls;
  bool stream_active = false;
  bool overlapped = false;
  bool nested = false;
  std::thread::id stream_thread;
  bool throw_after_stream = false;
  bool throw_after_complete = false;
  int stream_calls = 0;
  std::function<void()> stream_action;
  std::promise<void> stream_entered;
  std::future<void> stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  std::shared_future<void> release_stream_future = release_stream.get_future().share();

  void stream(const StreamMessage&) {
    bool first = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      calls.push_back("stream");
      stream_active = true;
      stream_thread = std::this_thread::get_id();
      first = ++stream_calls == 1;
    }
    if (first) stream_entered.set_value();
    if (stream_action) stream_action();
    (void)wait_future_ready(release_stream_future, "release_stream_future timed out");
    {
      std::lock_guard<std::mutex> lock(mu);
      stream_active = false;
    }
    if (throw_after_stream) throw std::runtime_error("previous stream failed");
  }

  void complete(const RequestInfo&) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (stream_active) {
        overlapped = true;
        if (std::this_thread::get_id() == stream_thread) nested = true;
      }
      calls.push_back("complete");
    }
    if (throw_after_complete) throw std::runtime_error("previous complete failed");
  }

  size_t call_count() {
    std::lock_guard<std::mutex> lock(mu);
    return calls.size();
  }

  void release() { release_stream.set_value(); }

  void check_order(const char* tag) {
    check_order_with_exceptions(tag, {"stream", "complete"});
  }

  void check_order_with_exceptions(
      const char* tag, const std::vector<std::string>& expected) {
    std::vector<std::string> order;
    bool saw_overlap = false;
    bool saw_nested = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      order = calls;
      saw_overlap = overlapped;
      saw_nested = nested;
    }
    std::cerr << tag << "=";
    for (const auto& call : order) std::cerr << call << ",";
    std::cerr << " OVERLAP=" << saw_overlap << " NESTED=" << saw_nested << "\n";
    check(order == expected, "forwarded callback order or drain is wrong");
    check(!saw_overlap, "forwarded completion overlapped the active stream callback");
    check(!saw_nested, "forwarded completion nested inside the stream callback");
  }
};

StreamMessage retired_stream_message() {
  StreamMessage msg;
  msg.request_id = "retired-stream";
  msg.from_stage = "asr";
  msg.chunk = text_chunk("after", false);
  return msg;
}

RequestInfo retired_complete_info() {
  RequestInfo info;
  info.request_id = "retired-complete";
  info.state = "aborted";
  return info;
}

// The copied completion arrives after retire() has nulled the engine but while
// the active previous_stream callback is still blocked.
void test_retire_wait_fallback_order() {
  RetiredSinkProbe probe;
  Harness h("retirewait");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  h.start_pump();
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "retire-wait ASR submit missing");
  h.stream(id, "asr", text_chunk("partial", false));
  (void)wait_future_ready(probe.stream_entered_future, "probe.stream_entered_future timed out");
  const auto copied_complete = h.coordinator.on_complete;
  UnifiedEngine* engine_ptr = engine.get();

  std::promise<void> destroyed;
  auto destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    engine.reset();
    destroyed.set_value();
  });
  bool retiring = false;
  wait_until([&] {
    retiring = unified_retirement_waiting_for_test(*engine_ptr);
    return retiring;
  }, "retire did not wait for the active forwarded stream");
  if (!retiring) {
    probe.release();
    (void)wait_future_ready(destroyed_future, "destroyed_future timed out");
    destroyer.join();
    return;
  }

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread fallback([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  (void)wait_future_ready(complete_returned_future, "complete_returned_future timed out");
  check(probe.call_count() == 1,
        "fallback completion ran before the active stream returned");
  probe.release();
  (void)wait_future_ready(destroyed_future, "destroyed_future timed out");
  fallback.join();
  destroyer.join();
  probe.check_order("RETIRE_WAIT_ORDER");
}

// Two copied sinks race after destruction. The completion caller may return
// after enqueueing, but its old sink must not run until stream is released.
void test_retired_cross_thread_order() {
  RetiredSinkProbe probe;
  Harness h("retiredthreads");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  (void)wait_future_ready(probe.stream_entered_future, "probe.stream_entered_future timed out");
  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  (void)wait_future_ready(complete_returned_future, "complete_returned_future timed out");
  check(probe.call_count() == 1,
        "post-retirement completion overlapped the blocked stream");
  probe.release();
  stream_thread.join();
  complete_thread.join();
  probe.check_order("RETIRED_THREAD_ORDER");
}

// A retired stream sink reenters completion on the same thread. Reentry must
// enqueue behind the active callback rather than recursively invoke the old sink.
void test_retired_same_thread_reentry() {
  RetiredSinkProbe probe;
  Harness h("retiredreenter");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.stream_action = [&] {
    copied_complete(retired_complete_info());
    check(probe.call_count() == 1,
          "retired completion recursively entered the old sink");
  };
  copied_stream(retired_stream_message());
  probe.check_order("RETIRED_REENTRANT_ORDER");
}

// A copied sink reports an exception only after every callback queued ahead of
// the error has drained. A later copied sink must still be usable.
void test_retired_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("retiredthrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "retired copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// The active engine drain has the same reentry contract before retirement.
void test_active_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("activethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "active copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// retire() is a destructor boundary: old sink exceptions are suppressed there,
// but the queued completion still runs and later copied sinks remain callable.
void test_destructor_suppresses_previous_sink_exception() {
  RetiredSinkProbe probe;
  Harness h("retirethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  (void)wait_future_ready(probe.stream_entered_future, "probe.stream_entered_future timed out");
  UnifiedEngine* engine_ptr = engine.get();
  std::thread destroyer([&] { engine.reset(); });
  wait_until([&] { return unified_retirement_waiting_for_test(*engine_ptr); },
             "retire did not wait for the blocked previous stream");

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  // The stream is still blocked, so this return proves the completion is in the
  // fallback queue and cannot run until retire() drains it.
  (void)wait_future_ready(complete_returned_future, "complete_returned_future timed out");
  probe.throw_after_complete = true;
  probe.release();
  stream_thread.join();
  complete_thread.join();
  destroyer.join();
  probe.check_order("DESTRUCTOR_EXCEPTION_DRAIN");

  probe.throw_after_complete = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("DESTRUCTOR_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}


}  // namespace
