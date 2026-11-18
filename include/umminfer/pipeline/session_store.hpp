// SPDX-License-Identifier: Apache-2.0
// Bounded process-local multi-turn session history store for UnifiedEngine.
#pragma once

#include <cstddef>
#include <deque>
#include <map>
#include <string>

#include "umminfer/proto/wire.hpp"

namespace umminfer {

struct SessionHistoryPolicy {
  bool enabled = false;
  // Completed turns kept per session. Oldest turns are evicted first.
  size_t max_turns = 8;
  // Hard character budget per session across transcript and answer text.
  // Oldest turns leave first until the budget holds.
  size_t max_chars = 16384;
  // Hard cap on distinct sessions retained. The session whose last recorded
  // turn is oldest is evicted first.
  size_t max_sessions = 256;
};

// External synchronization is owned by the caller (UnifiedEngine::mu_).
class SessionHistoryStore {
 public:
  struct Turn {
    std::string user;
    std::string assistant;
  };

  explicit SessionHistoryStore(SessionHistoryPolicy policy = {})
      : policy_(policy) {}

  bool enabled() const { return policy_.enabled; }

  // Prepends retained user/assistant turns for `session_id` before
  // `current_messages`.
  Value::Array prepend_history(const std::string& session_id,
                               const Value::Array& current_messages) const;

  // Records a completed turn and enforces turn, character, and global LRU
  // session bounds.
  void record_turn(const std::string& session_id, std::string user,
                   std::string assistant);

  size_t session_count() const { return sessions_.size(); }

 private:
  void erase_from_order(const std::string& session_id);

  SessionHistoryPolicy policy_;
  std::map<std::string, std::deque<Turn>> sessions_;
  std::deque<std::string> order_;
};

}  // namespace umminfer
