// SPDX-License-Identifier: Apache-2.0
// Post-load native Qwen process/model identity for online rollout attestation.
#pragma once

#include <cstdint>
#include <string>

#include "stage_kind.hpp"

namespace umminfer::stage {

struct WrittenNativeIdentity {
  std::string document_sha256;
  int64_t process_start_time_ticks = 0;
};

// Called only after make_compute has completed the real model/tokenizer load.
// The output is a new file; existing evidence is never overwritten. The exact
// document digest is carried in stage_ready so a gateway can bind the configured
// evidence to the process actually attached as its unified VLM stage.
WrittenNativeIdentity write_native_qwen_identity(const StageOptions& options,
                                                 const std::string& stage_name);

// Linux process-start identity, parsed from /proc/<pid>/stat field 22. This
// distinguishes a recycled pid and is also used by the independent server check.
int64_t native_process_start_time_ticks(int pid);

}  // namespace umminfer::stage
