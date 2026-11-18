// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/session_store.hpp"

#include <utility>

namespace umminfer {

void SessionHistoryStore::erase_from_order(const std::string& session_id) {
  for (auto it = order_.begin(); it != order_.end(); ++it) {
    if (*it == session_id) {
      order_.erase(it);
      return;
    }
  }
}

Value::Array SessionHistoryStore::prepend_history(
    const std::string& session_id,
    const Value::Array& current_messages) const {
  Value::Array messages;
  const auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    messages.reserve(it->second.size() * 2 + current_messages.size());
    for (const Turn& turn : it->second) {
      Value::Map user;
      user["role"] = Value("user");
      user["content"] = Value(turn.user);
      Value::Map assistant;
      assistant["role"] = Value("assistant");
      assistant["content"] = Value(turn.assistant);
      messages.push_back(Value(std::move(user)));
      messages.push_back(Value(std::move(assistant)));
    }
  }
  for (const Value& message : current_messages) messages.push_back(message);
  return messages;
}

void SessionHistoryStore::record_turn(const std::string& session_id,
                                      std::string user,
                                      std::string assistant) {
  const bool is_new_session = sessions_.find(session_id) == sessions_.end();
  std::deque<Turn>& turns = sessions_[session_id];
  turns.push_back(Turn{std::move(user), std::move(assistant)});

  size_t chars = 0;
  for (const Turn& turn : turns) {
    chars += turn.user.size() + turn.assistant.size();
  }
  // Oldest turns leave first until both bounds hold. A single turn larger than
  // the character budget is itself evicted: an unbounded turn must never stay.
  while ((turns.size() > policy_.max_turns || chars > policy_.max_chars) &&
         !turns.empty()) {
    chars -= turns.front().user.size() + turns.front().assistant.size();
    turns.pop_front();
  }
  if (turns.empty()) {
    sessions_.erase(session_id);
    if (!is_new_session) erase_from_order(session_id);
    return;
  }
  if (!is_new_session) erase_from_order(session_id);
  order_.push_back(session_id);

  // Global session cap: drop the session with the oldest last turn.
  while (sessions_.size() > policy_.max_sessions && !order_.empty()) {
    const std::string oldest = std::move(order_.front());
    order_.pop_front();
    sessions_.erase(oldest);
  }
}

}  // namespace umminfer
