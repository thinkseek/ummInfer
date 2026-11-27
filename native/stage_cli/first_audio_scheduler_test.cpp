// SPDX-License-Identifier: Apache-2.0
// Host-only regression for serial realtime first-audio admission.
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "umminfer/scheduling/simple_scheduler.hpp"

namespace {
using namespace umminfer;
constexpr auto kWait = std::chrono::seconds(5);

void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool first_audio(const IncomingMessage& msg) {
  auto ends = [&](std::string_view suffix) {
    return msg.request_id.size() >= suffix.size() &&
           std::string_view(msg.request_id).substr(
               msg.request_id.size() - suffix.size()) == suffix;
  };
  return msg.type == "new_request" && (ends(":tts:0") || ends(":tts"));
}

void test_fair_priority_and_abort() {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  std::vector<std::string> order;
  auto compute = [&](const Value& input) {
    const std::string id = input.as_str();
    if (id == "block") {
      std::unique_lock<std::mutex> lock(mu);
      entered = true;
      cv.notify_all();
      cv.wait(lock, [&] { return release; });
    }
    std::lock_guard<std::mutex> lock(mu);
    order.push_back(id);
    return Value(id);
  };
  SimpleScheduler scheduler(compute);
  scheduler.set_request_priority_fn(first_audio);
  std::thread runner([&] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"block", "new_request", Value("block")});
  {
    std::unique_lock<std::mutex> lock(mu);
    expect(cv.wait_for(lock, kWait, [&] { return entered; }),
           "serial blocker did not start");
  }
  for (const char* id : {"req-1:tts:1", "req-2:tts:1", "req-3:tts:0",
                         "plain", "req-4:tts", "req-5:tts:2"}) {
    scheduler.enqueue(IncomingMessage{id, "new_request", Value(id)});
  }
  scheduler.abort("req-9:tts:0");
  scheduler.enqueue(IncomingMessage{"req-9:tts:0", "new_request",
                                    Value("req-9:tts:0")});
  {
    std::lock_guard<std::mutex> lock(mu);
    release = true;
  }
  cv.notify_all();
  std::set<std::string> settled;
  const auto deadline = std::chrono::steady_clock::now() + kWait;
  while (settled.size() < 7 && std::chrono::steady_clock::now() < deadline) {
    auto result = scheduler.outbox().get(0.1);
    if (result) settled.insert(result->request_id);
  }
  scheduler.stop();
  runner.join();
  expect(settled.size() == 7 && settled.count("req-9:tts:0") == 0,
         "lost work or executed an aborted first leg");
  const std::vector<std::string> expected{
      "block", "req-3:tts:0", "req-1:tts:1", "req-4:tts",
      "req-2:tts:1", "plain", "req-5:tts:2"};
  expect(order == expected, "priority, class FIFO or fair drain changed");
}

void test_default_fifo_and_rejections() {
  std::vector<std::string> order;
  SimpleScheduler fifo([&](const Value& input) {
    order.push_back(input.as_str());
    return input;
  });
  std::thread runner([&] { fifo.start(); });
  for (const char* id : {"req-1:tts:1", "req-3:tts:0", "req-4:tts"}) {
    fifo.enqueue(IncomingMessage{id, "new_request", Value(id)});
  }
  for (int count = 0; count < 3; ++count) {
    expect(fifo.outbox().get(5.0).has_value(), "FIFO request did not settle");
  }
  fifo.stop();
  runner.join();
  expect(order == std::vector<std::string>{
                      "req-1:tts:1", "req-3:tts:0", "req-4:tts"},
         "no-hook scheduler stopped being FIFO");

  SimpleScheduler batch(
      [](const Value& input) { return input; },
      [](const std::vector<Value>& input) { return input; }, 2);
  bool rejected = false;
  try {
    batch.set_request_priority_fn(first_audio);
  } catch (const ProtocolError&) {
    rejected = true;
  }
  expect(rejected, "priority was accepted for a batched scheduler");
}
}  // namespace

int main() {
  try {
    test_fair_priority_and_abort();
    test_default_fifo_and_rejections();
  } catch (const std::exception& error) {
    std::cerr << "FIRST_AUDIO_SCHEDULER FAIL " << error.what() << "\n";
    return 1;
  }
  std::cout << "FIRST_AUDIO_SCHEDULER PASS fair_drain=1 abort_tombstone=1 default_fifo=1\n";
  return 0;
}
