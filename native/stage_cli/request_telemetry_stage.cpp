// SPDX-License-Identifier: Apache-2.0
#include "request_telemetry_stage.hpp"

#include "stage_common.hpp"

#include <exception>
#include <utility>

namespace umminfer::stage {
namespace {

Value telemetry_only_result(Value telemetry) {
  return Value(Value::Map{{obs::kRequestTelemetryResultKey,
                           std::move(telemetry)}});
}

}  // namespace

CancellableComputeFn with_native_request_telemetry(
    CancellableComputeFn compute, const StageOptions& options,
    std::string stage_name, bool instrumented) {
  return [compute = std::move(compute), options, stage_name = std::move(stage_name),
          instrumented](const Value& work, const AbortCheck& abort) mutable -> Value {
    if (!obs::native_allocator_telemetry_requested(bound_request_params())) {
      return compute(work, abort);
    }
    obs::NativeRequestTelemetryConfig config;
    config.request_id = bound_request_id();
    config.stage = stage_name;
    config.backend = compiled_kind();
    config.device_index = options.gpu;
    config.instrumented = instrumented;
    config.batch_capacity = options.batch_size > 1 ? options.batch_size : 1;
    config.execution = options.batch_size > 1
                           ? obs::NativeTelemetryExecution::kBatched
                           : obs::NativeTelemetryExecution::kSerial;
    if (std::string_view(compiled_kind()) == "image") {
      const bool image21 =
          parse_image_backend(options.image_backend) == ImageBackend::kQwen21;
      config.backend = image21 ? "qwen-image-2.1" : "qwen-image-2512";
      // Both backends share one executable, but only image2 routes every direct
      // device allocation/free through the request-owner adapter.
      config.instrumented = instrumented && image21;
    }
    if (std::string_view(compiled_kind()) == "video") {
      config.backend = options.video_backend.empty() ? "video-cuda"
                       : options.video_backend == "contract-stub"
                           ? "video-contract-stub"
                           : options.video_backend == "cuda-scaffold"
                               ? "video-cuda-scaffold"
                               : "video-cuda";
      if (options.video_backend == "contract-stub") {
        config.instrumented = false;
        config.unsupported_reason = "contract_stub_has_no_cuda_allocator";
      }
    }
    obs::NativeRequestTelemetry telemetry(std::move(config));
    try {
      Value result = compute(work, abort);
      if (!result.is_map()) {
        throw ProtocolError("native telemetry requires a map terminal result");
      }
      result[obs::kRequestTelemetryResultKey] = telemetry.finish("completed");
      return result;
    } catch (const RequestAborted& error) {
      Value result = telemetry_only_result(telemetry.finish("cancelled"));
      throw RequestAborted(error.what(), std::move(result));
    } catch (const std::exception& error) {
      Value result = telemetry_only_result(telemetry.finish("failed"));
      throw RequestComputeError(error.what(), std::move(result));
    } catch (...) {
      Value result = telemetry_only_result(telemetry.finish("failed"));
      throw RequestComputeError("native compute failed", std::move(result));
    }
  };
}

}  // namespace umminfer::stage
