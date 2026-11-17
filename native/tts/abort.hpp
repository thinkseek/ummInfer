// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace umminfer::native {

// True once the caller's request was aborted. Empty means never abort.
using AbortCheck = std::function<bool()>;

// Raised at a cancellation boundary. The caller settles the request on its
// existing abort path.
class RequestAborted : public std::runtime_error {
 public:
  explicit RequestAborted(std::string message)
      : std::runtime_error(std::move(message)) {}
};

}  // namespace umminfer::native
