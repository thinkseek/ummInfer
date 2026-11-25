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
  size_t first_turn = 0;
  while (first_turn < current_messages.size()) {
    const Value& msg = current_messages[first_turn];
    const Value* role = msg.is_map() ? msg.find("role") : nullptr;
    if (role != nullptr && role->is_str() && role->as_str() == "system") {
      ++first_turn;
    } else {
      break;
    }
  }
  // If the client already provided multi-turn conversation history (more than
  // one non-system message), preserve it directly instead of duplicating turns
  // from the server-side session store.
  if (current_messages.size() - first_turn > 1) {
    return current_messages;
  }

  Value::Array messages;
  const auto it = sessions_.find(session_id);
  const size_t history_msgs =
      it != sessions_.end() ? it->second.size() * 2 : 0;
  messages.reserve(history_msgs + current_messages.size());
  for (size_t i = 0; i < first_turn; ++i) {
    messages.push_back(current_messages[i]);
  }
  if (it != sessions_.end()) {
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
  for (size_t i = first_turn; i < current_messages.size(); ++i) {
    messages.push_back(current_messages[i]);
  }
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
