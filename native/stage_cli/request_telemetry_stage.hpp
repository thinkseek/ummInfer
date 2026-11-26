// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "stage_kind.hpp"
#include "umminfer/obs/request_telemetry.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

namespace umminfer::stage {

// Adds no wrapper work or result field unless the per-request parameter is true.
// instrumented is a compile-time backend property, not a user opt-in to claim
// coverage. Model loading stays outside every request epoch.
CancellableComputeFn with_native_request_telemetry(
    CancellableComputeFn compute, const StageOptions& options,
    std::string stage_name, bool instrumented);

}  // namespace umminfer::stage
