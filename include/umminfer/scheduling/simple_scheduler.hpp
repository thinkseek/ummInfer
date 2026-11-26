// SPDX-License-Identifier: Apache-2.0
// Serial and batched request scheduling.
//
// A non-autoregressive stage runs one callable per request. This scheduler owns
// the inbox/outbox pair the stage polls, so every scheduler backend presents the
// same shape to Stage.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer {

struct IncomingMessage {
  std::string request_id;
  std::string type;
  Value data;
  Value::Map params = {};
};

struct OutgoingMessage {
  std::string request_id;
  std::string type;  // result, error
  Value data;
};

// Bounded-free thread-safe queue; the C++ stand-in for queue.Queue.
template <typename T>
class MessageQueue {
 public:
  void put(T msg) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      items_.push_back(std::move(msg));
    }
    cv_.notify_one();
  }
  std::optional<T> get(double timeout_s = -1) {
    std::unique_lock<std::mutex> lock(mu_);
    if (timeout_s < 0) {
      cv_.wait(lock, [this] { return !items_.empty(); });
    } else if (!cv_.wait_for(lock, std::chrono::duration<double>(timeout_s),
                             [this] { return !items_.empty(); })) {
      return std::nullopt;
    }
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }
  std::optional<T> get_nowait() {
    std::lock_guard<std::mutex> lock(mu_);
    if (items_.empty()) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }
  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.size();
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> items_;
};

// Raised by a cancellable compute function when the request was aborted. The
// scheduler settles the request on its existing abort path instead of
// reporting the exception as a failure.
class RequestAborted : public std::runtime_error {
 public:
  explicit RequestAborted(std::string message = "request aborted",
                          Value telemetry_result = {})
      : std::runtime_error(std::move(message)),
        telemetry_result(std::move(telemetry_result)) {}
  // Nil on the historical path. Opted-in native requests retain their terminal
  // allocation snapshot here even though the public Coordinator already aborted.
  Value telemetry_result;
};

// Opt-in native telemetry companion for a failed compute call. The scheduler
// still emits the same error string; only the opted-in result carries metrics.
class RequestComputeError : public std::runtime_error {
 public:
  RequestComputeError(std::string message, Value telemetry_result)
      : std::runtime_error(std::move(message)),
        telemetry_result(std::move(telemetry_result)) {}
  Value telemetry_result;
};

// Declares how the scheduler may call the compute function. A serial engine
// owns one CUDA stream and one buffer set; a thread-safe engine may be called
// from several threads because it serialises work internally, either behind a
// lock or through its own single-threaded batch loop. The two are not
// interchangeable, so a concurrency above 1 requires kThreadSafe.
enum class EngineContract { kSerial, kThreadSafe };

// Bound to one request id: true once that id was aborted. A late abort for a
// different id never affects it.
using AbortCheck = std::function<bool()>;
using CancellableComputeFn = std::function<Value(const Value&, const AbortCheck&)>;

using ComputeFn = std::function<Value(const Value&)>;
using BatchComputeFn = std::function<std::vector<Value>(const std::vector<Value>&)>;
using RequestCostFn = std::function<int64_t(const Value&)>;
using RequestArrivalHook = std::function<void(const Value&)>;
// One incremental chunk of the request the scheduler is computing now.
using StreamFn = std::function<void(const std::string& request_id, Value chunk)>;

// Sends one incremental chunk for the current request. Callable only from
// inside a compute function; outside one it does nothing.
void emit_stage_stream(Value chunk);

// The sink and request id bound to the compute call on this thread. A compute
// function that hands work to another thread copies them, so the other thread
// can stream without relying on this thread's thread-local state. Empty
// outside a compute call.
StreamFn bound_stream_fn();
std::string bound_request_id();
const Value::Map& bound_request_params();

class SimpleScheduler {
 public:
  SimpleScheduler(ComputeFn compute_fn, BatchComputeFn batch_compute_fn = nullptr,
                  int max_batch_size = 1, int max_batch_wait_ms = 0,
                  bool batch_wait_when_idle = true,
                  RequestCostFn request_cost_fn = nullptr,
                  std::optional<int> max_batch_cost = std::nullopt,
                  int max_concurrency = 1,
                  std::function<void(const std::string&)> abort_callback = nullptr,
                  std::function<void()> shutdown_callback = nullptr,
                  RequestArrivalHook request_arrival_hook = nullptr,
                  std::chrono::milliseconds abort_retention = std::chrono::minutes(10),
                  EngineContract contract = EngineContract::kSerial);

  MessageQueue<IncomingMessage>& inbox() { return inbox_; }
  MessageQueue<OutgoingMessage>& outbox() { return outbox_; }

  void enqueue(IncomingMessage msg);
  // Installs the compute path that can observe its own cancellation. It takes
  // precedence over the plain compute function; an empty value restores it.
  void set_cancellable_compute(CancellableComputeFn fn);
  // Installs the sink for emit_stage_stream. Empty drops every chunk.
  void set_stream_fn(StreamFn fn);
  void start();  // blocks the calling thread
  void stop();
  void abort(const std::string& request_id);

  bool is_aborted(const std::string& request_id);
  bool consume_if_aborted(const std::string& request_id);
  void reap_aborted();

 private:
  void run_serial();
  void run_concurrent();
  void run_single(const IncomingMessage& msg);
  void run_batch(std::vector<IncomingMessage> batch);
  void collect_batch(std::vector<IncomingMessage>& batch);
  int64_t message_cost(const IncomingMessage& msg) const;
  void emit_result(const std::string& request_id, Value result);
  void emit_error(const std::string& request_id, const std::string& error);

  ComputeFn fn_;
  CancellableComputeFn cancellable_fn_;
  StreamFn stream_fn_;
  BatchComputeFn batch_fn_;
  int max_batch_size_;
  double max_batch_wait_s_;
  bool batch_wait_when_idle_;
  RequestCostFn request_cost_fn_;
  std::optional<int> max_batch_cost_;
  int max_concurrency_;
  EngineContract contract_ = EngineContract::kSerial;
  std::function<void(const std::string&)> abort_callback_;
  std::function<void()> shutdown_callback_;
  RequestArrivalHook request_arrival_hook_;

  MessageQueue<IncomingMessage> inbox_;
  MessageQueue<OutgoingMessage> outbox_;
  std::deque<IncomingMessage> pending_messages_;
  std::mutex pending_mu_;

  std::mutex abort_mu_;
  std::chrono::milliseconds abort_retention_;
  std::map<std::string, std::chrono::steady_clock::time_point> aborted_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stopped_{false};
  std::mutex lifecycle_mu_;
};

}  // namespace umminfer
