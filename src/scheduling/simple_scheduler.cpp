// SPDX-License-Identifier: Apache-2.0
#include "umminfer/scheduling/simple_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "umminfer/obs/request_telemetry.hpp"

namespace umminfer {
namespace {

// Valid only while one compute function runs on this thread.
thread_local const StreamFn* g_stream_sink = nullptr;
thread_local bool g_in_compute = false;
thread_local std::string g_stream_request;
thread_local const Value::Map* g_request_params = nullptr;

}  // namespace

void emit_stage_stream(Value chunk) {
  if (g_stream_sink == nullptr || !*g_stream_sink) return;
  (*g_stream_sink)(g_stream_request, std::move(chunk));
}

StreamFn bound_stream_fn() {
  if (g_stream_sink == nullptr || !*g_stream_sink) return {};
  return *g_stream_sink;
}

std::string bound_request_id() {
  return g_in_compute ? g_stream_request : std::string();
}

const Value::Map& bound_request_params() {
  static const Value::Map kEmpty;
  return (g_in_compute && g_request_params != nullptr) ? *g_request_params
                                                       : kEmpty;
}

SimpleScheduler::SimpleScheduler(ComputeFn compute_fn,
                                 BatchComputeFn batch_compute_fn,
                                 int max_batch_size, int max_batch_wait_ms,
                                 bool batch_wait_when_idle,
                                 RequestCostFn request_cost_fn,
                                 std::optional<int> max_batch_cost, int max_concurrency,
                                 std::function<void(const std::string&)> abort_callback,
                                 std::function<void()> shutdown_callback,
                                 RequestArrivalHook request_arrival_hook,
                                 std::chrono::milliseconds abort_retention,
                                 EngineContract contract)
    : fn_(std::move(compute_fn)),
      batch_fn_(std::move(batch_compute_fn)),
      max_batch_size_(std::max(max_batch_size, 1)),
      max_batch_wait_s_(std::max(max_batch_wait_ms, 0) / 1000.0),
      batch_wait_when_idle_(batch_wait_when_idle),
      request_cost_fn_(std::move(request_cost_fn)),
      max_batch_cost_(max_batch_cost ? std::optional<int>(std::max(*max_batch_cost, 0))
                                     : std::nullopt),
      max_concurrency_(std::max(max_concurrency, 1)),
      abort_callback_(std::move(abort_callback)),
      shutdown_callback_(std::move(shutdown_callback)),
      request_arrival_hook_(std::move(request_arrival_hook)),
      abort_retention_(std::max(abort_retention, std::chrono::milliseconds{1})),
      contract_(contract) {
  if (max_concurrency_ > 1 && batch_fn_) {
    throw ProtocolError("max_concurrency > 1 and batch_compute_fn are mutually exclusive");
  }
  // Concurrency is only safe when the engine declares itself thread-safe.
  if (max_concurrency_ > 1 && contract_ != EngineContract::kThreadSafe) {
    throw ProtocolError(
        "max_concurrency > 1 requires EngineContract::kThreadSafe");
  }
}

void SimpleScheduler::enqueue(IncomingMessage msg) {
  // Runs on the stage loop, so both hooks must not block.
  if (msg.type == "new_request" && request_arrival_hook_) {
    request_arrival_hook_(msg.data);
  }
  const bool priority = msg.type == "new_request" && request_priority_fn_ &&
                        request_priority_fn_(msg);
  inbox_.put(std::move(msg), priority);
}

void SimpleScheduler::set_request_priority_fn(RequestPriorityFn fn) {
  if (running_ || stopped_) {
    throw ProtocolError("request priority must be configured before scheduler start");
  }
  if (fn && (batch_fn_ || max_concurrency_ != 1)) {
    throw ProtocolError("request priority requires serial, non-batched scheduling");
  }
  request_priority_fn_ = std::move(fn);
}

void SimpleScheduler::reap_aborted() {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(abort_mu_);
  for (auto it = aborted_.begin(); it != aborted_.end();) {
    if (it->second <= now) {
      it = aborted_.erase(it);
    } else {
      ++it;
    }
  }
}

bool SimpleScheduler::is_aborted(const std::string& request_id) {
  reap_aborted();
  std::lock_guard<std::mutex> lock(abort_mu_);
  return aborted_.count(request_id) != 0;
}

bool SimpleScheduler::consume_if_aborted(const std::string& request_id) {
  reap_aborted();
  std::lock_guard<std::mutex> lock(abort_mu_);
  return aborted_.erase(request_id) > 0;
}

void SimpleScheduler::abort(const std::string& request_id) {
  {
    std::lock_guard<std::mutex> lock(abort_mu_);
    // Keep the tombstone through abort-before-submit reordering. Stage's
    // reaper expires it after the normal request and transfer lifetime.
    aborted_[request_id] = std::chrono::steady_clock::now() + abort_retention_;
  }
  if (abort_callback_) {
    try {
      abort_callback_(request_id);
    } catch (const std::exception& e) {
      std::cerr << "SimpleScheduler: abort cleanup failed for " << request_id << ": "
                << e.what() << "\n";
    }
  }
}

int64_t SimpleScheduler::message_cost(const IncomingMessage& msg) const {
  if (!request_cost_fn_ || msg.type != "new_request") return 0;
  return std::max<int64_t>(request_cost_fn_(msg.data), 0);
}

void SimpleScheduler::collect_batch(std::vector<IncomingMessage>& batch) {
  if (!batch_fn_ || max_batch_size_ <= 1) return;

  int64_t batch_cost = message_cost(batch.front());
  std::optional<double> deadline =
      batch_wait_when_idle_
          ? std::optional<double>(std::chrono::steady_clock::now().time_since_epoch().count() /
                                  1e9 + max_batch_wait_s_)
          : std::nullopt;

  auto now_s = [] {
    return std::chrono::steady_clock::now().time_since_epoch().count() / 1e9;
  };

  while (running_ && static_cast<int>(batch.size()) < max_batch_size_) {
    std::optional<IncomingMessage> next = inbox_.get_nowait();
    if (!next) {
      if (!deadline) break;
      const double remaining = *deadline - now_s();
      if (remaining <= 0) break;
      next = inbox_.get(remaining);
      if (!next) break;
    }

    if (next->type == "new_request") {
      if (consume_if_aborted(next->request_id)) continue;
      if (max_batch_cost_) {
        int64_t cost;
        try {
          cost = message_cost(*next);
        } catch (const std::exception& e) {
          // The bad request has already left the inbox. Fail that request and
          // keep the valid prefix of the batch instead of failing both.
          if (!consume_if_aborted(next->request_id)) {
            emit_error(next->request_id, e.what());
          }
          continue;
        }
        // Costs are non-negative. This subtraction cannot overflow and avoids
        // signed overflow in batch_cost + cost.
        if (cost > *max_batch_cost_ - batch_cost) {
          std::lock_guard<std::mutex> lock(pending_mu_);
          pending_messages_.push_front(std::move(*next));
          break;
        }
        batch_cost += cost;
      }
      batch.push_back(std::move(*next));
      if (!deadline) deadline = now_s() + max_batch_wait_s_;
    } else {
      // A control message must not be reordered behind later arrivals.
      std::lock_guard<std::mutex> lock(pending_mu_);
      pending_messages_.push_front(std::move(*next));
      break;
    }
  }
}

void SimpleScheduler::emit_result(const std::string& request_id, Value result) {
  outbox_.put(OutgoingMessage{request_id, "result", std::move(result)});
}

void SimpleScheduler::emit_error(const std::string& request_id,
                                 const std::string& error) {
  outbox_.put(OutgoingMessage{request_id, "error", Value(error)});
}

void SimpleScheduler::set_cancellable_compute(CancellableComputeFn fn) {
  cancellable_fn_ = std::move(fn);
}

void SimpleScheduler::set_stream_fn(StreamFn fn) {
  stream_fn_ = std::move(fn);
}

void SimpleScheduler::run_single(const IncomingMessage& msg) {
  if (consume_if_aborted(msg.request_id)) return;
  // Bind emit_stage_stream() to this request for the duration of the call.
  // A compute function that streams never sees another request's id.
  struct StreamBinding {
    StreamBinding(const StreamFn* fn, const std::string& request_id,
                  const Value::Map* params) {
      g_stream_sink = fn;
      g_in_compute = true;
      g_stream_request = request_id;
      g_request_params = params;
    }
    ~StreamBinding() {
      g_stream_sink = nullptr;
      g_in_compute = false;
      g_stream_request.clear();
      g_request_params = nullptr;
    }
  } binding(stream_fn_ ? &stream_fn_ : nullptr, msg.request_id, &msg.params);
  Value result;
  try {
    if (cancellable_fn_) {
      // The check is bound to this request id: an abort for another id, or a
      // late abort for an id that already finished, cannot stop this work.
      const std::string request_id = msg.request_id;
      result = cancellable_fn_(msg.data, [this, request_id] {
        return is_aborted(request_id);
      });
    } else {
      result = fn_(msg.data);
    }
  } catch (RequestAborted& error) {
    // The request already has a terminal event from the abort path. Preserve
    // the historical no-output path unless this request explicitly opted into
    // a terminal native allocation snapshot.
    consume_if_aborted(msg.request_id);
    if (!error.telemetry_result.is_nil()) {
      obs::rewrite_result_telemetry_lifecycle(error.telemetry_result, "cancelled");
      outbox_.put(OutgoingMessage{msg.request_id, "cancelled",
                                  std::move(error.telemetry_result)});
    }
    return;
  } catch (const RequestComputeError& error) {
    Value result = error.telemetry_result;
    if (consume_if_aborted(msg.request_id)) {
      obs::rewrite_result_telemetry_lifecycle(result, "cancelled");
      outbox_.put(OutgoingMessage{msg.request_id, "cancelled", std::move(result)});
    } else {
      Value::Map failure = result.is_map() ? result.as_map() : Value::Map{};
      failure["error"] = Value(error.what());
      outbox_.put(OutgoingMessage{msg.request_id, "error",
                                  Value(std::move(failure))});
    }
    return;
  } catch (const std::exception& e) {
    if (!consume_if_aborted(msg.request_id)) emit_error(msg.request_id, e.what());
    return;
  }
  if (consume_if_aborted(msg.request_id)) {
    if (obs::rewrite_result_telemetry_lifecycle(result, "cancelled")) {
      outbox_.put(OutgoingMessage{msg.request_id, "cancelled", std::move(result)});
    }
    return;
  }
  emit_result(msg.request_id, std::move(result));
}

void SimpleScheduler::run_batch(std::vector<IncomingMessage> batch) {
  if (!batch_fn_ || batch.size() <= 1) {
    for (const auto& msg : batch) run_single(msg);
    return;
  }

  std::vector<IncomingMessage> active;
  active.reserve(batch.size());
  for (auto& msg : batch) {
    if (!consume_if_aborted(msg.request_id)) active.push_back(std::move(msg));
  }
  if (active.empty()) return;
  if (active.size() == 1) {
    run_single(active.front());
    return;
  }

  std::vector<Value> payloads;
  payloads.reserve(active.size());
  for (auto& msg : active) payloads.push_back(std::move(msg.data));

  std::vector<Value> results;
  try {
    results = batch_fn_(payloads);
  } catch (const std::exception& e) {
    for (const auto& failed : active) {
      if (!consume_if_aborted(failed.request_id)) emit_error(failed.request_id, e.what());
    }
    return;
  }
  if (results.size() != active.size()) {
    const std::string error =
        "batch_compute_fn returned " + std::to_string(results.size()) +
        " results for " + std::to_string(active.size()) + " requests";
    for (const auto& failed : active) {
      if (!consume_if_aborted(failed.request_id)) emit_error(failed.request_id, error);
    }
    return;
  }
  for (size_t i = 0; i < active.size(); ++i) {
    // A Value is always a legitimate result, a string included. Python only
    // treats a BaseException in a slot as an error, and a whole-batch failure
    // reaches the caller as a throw from batch_fn_.
    if (!consume_if_aborted(active[i].request_id)) {
      emit_result(active[i].request_id, std::move(results[i]));
    }
  }
}

void SimpleScheduler::start() {
  {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    if (running_ || stopped_) return;
    running_ = true;
  }
  if (max_concurrency_ > 1) {
    run_concurrent();
  } else {
    run_serial();
  }
}

void SimpleScheduler::run_serial() {
  while (running_) {
    IncomingMessage msg;
    bool have_msg = false;
    {
      std::lock_guard<std::mutex> lock(pending_mu_);
      if (!pending_messages_.empty()) {
        msg = std::move(pending_messages_.front());
        pending_messages_.pop_front();
        have_msg = true;
      }
    }
    if (!have_msg) {
      auto got = inbox_.get();
      if (!got) continue;
      msg = std::move(*got);
    }

    if (msg.type == "abort") {
      abort(msg.request_id);
      continue;
    }
    if (msg.type == "stop") {
      stop();
      continue;
    }
    if (msg.type != "new_request") continue;
    if (consume_if_aborted(msg.request_id)) continue;

    std::vector<IncomingMessage> batch;
    batch.push_back(std::move(msg));
    try {
      collect_batch(batch);
    } catch (const std::exception& e) {
      std::cerr << "SimpleScheduler: batch collection failed: " << e.what() << "\n";
      for (const auto& failed : batch) {
        if (consume_if_aborted(failed.request_id)) continue;
        emit_error(failed.request_id, e.what());
      }
      continue;
    }
    run_batch(std::move(batch));
  }
}

void SimpleScheduler::run_concurrent() {
  std::vector<std::thread> workers;
  workers.reserve(max_concurrency_);
  for (int i = 0; i < max_concurrency_; ++i) {
    workers.emplace_back([this] {
      while (running_) {
        auto got = inbox_.get(0.1);
        if (!got) continue;
        IncomingMessage msg = std::move(*got);
        if (msg.type == "abort") {
          abort(msg.request_id);
          continue;
        }
        if (msg.type == "stop") {
          stop();
          continue;
        }
        if (msg.type != "new_request") continue;
        if (consume_if_aborted(msg.request_id)) continue;
        // Compute runs on its worker so thread-local stage handles stay isolated.
        run_single(msg);
      }
    });
  }
  for (auto& worker : workers) worker.join();
}

void SimpleScheduler::stop() {
  {
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    stopped_ = true;
    running_ = false;
  }
  // Release a blocked get().
  inbox_.put(IncomingMessage{"", "wakeup", Value()});
  std::function<void()> callback;
  {
    std::lock_guard<std::mutex> lock(pending_mu_);
    callback = shutdown_callback_;
    shutdown_callback_ = nullptr;
  }
  if (callback) callback();
}

}  // namespace umminfer
