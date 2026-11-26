# SPDX-License-Identifier: Apache-2.0
# Native stage executables: one accepted engine behind the existing C++ Stage
# protocol. Each kind is a separate executable because native/umm_asr_native
# and native/tts both define umminfer::native::SafeTensors/TensorInfo.
#
# Host build (parsing tests only):
#   cmake -S native/stage_cli -B build-host && cmake --build build-host
#   ctest --test-dir build-host
# CUDA build (GPU host):
#   cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON
#     -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.24)
project(umm_stage_native LANGUAGES CXX)

option(UMM_STAGE_CUDA "Build the native stage executables" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(UMM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(NATIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Homebrew installs icu4c keg-only; honor ICU_ROOT, then the opt path.
if(NOT DEFINED ICU_ROOT AND EXISTS "/opt/homebrew/opt/icu4c/include/unicode/unistr.h")
  set(ICU_ROOT "/opt/homebrew/opt/icu4c")
endif()
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp REQUIRED)
endif()
find_path(ZMQ_INCLUDE_DIR zmq.h REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq REQUIRED)
find_path(MSGPACK_INCLUDE_DIR msgpack.h REQUIRED)
find_library(MSGPACK_LIBRARY NAMES msgpackc REQUIRED)

# The Stage/Coordinator subset of the C++ runtime: no HTTP, no client library.
add_library(umm_stage_core STATIC
  ${UMM_ROOT}/src/util/codec.cpp
  ${UMM_ROOT}/src/proto/wire.cpp
  ${UMM_ROOT}/src/proto/messages.cpp
  ${UMM_ROOT}/src/obs/metrics.cpp
  ${UMM_ROOT}/src/obs/request_telemetry.cpp
  ${UMM_ROOT}/src/relay/relay.cpp
  ${UMM_ROOT}/src/comm/comm.cpp
  ${UMM_ROOT}/src/scheduling/simple_scheduler.cpp
  ${UMM_ROOT}/src/pipeline/pipeline.cpp
)
target_include_directories(umm_stage_core PUBLIC
  ${UMM_ROOT}/include ${UMM_ROOT}/src
  ${ZMQ_INCLUDE_DIR} ${MSGPACK_INCLUDE_DIR})
# The metrics module dlopen()s libnvidia-ml; CMAKE_DL_LIBS is empty on macOS,
# where dlopen lives in libSystem.
target_link_libraries(umm_stage_core PUBLIC ${ZMQ_LIBRARY} ${MSGPACK_LIBRARY}
  ${CMAKE_DL_LIBS})
if(nlohmann_json_FOUND)
  target_link_libraries(umm_stage_core PUBLIC nlohmann_json::nlohmann_json)
else()
  target_include_directories(umm_stage_core PUBLIC ${NLOHMANN_JSON_INCLUDE})
endif()

add_library(umm_stage_common STATIC
  stage_common.cpp
  request_telemetry_stage.cpp
  ${NATIVE_DIR}/umm_asr_native/wav.cpp
)
target_include_directories(umm_stage_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${NATIVE_DIR})
target_link_libraries(umm_stage_common PUBLIC umm_stage_core)

# Trailing-noise trim self-check. No CUDA, so it runs on any host.
enable_testing()
add_executable(tts_tail_trim_test
  ${NATIVE_DIR}/tts/tail_trim_test.cpp
  ${NATIVE_DIR}/tts/tail_trim.cpp
)
target_include_directories(tts_tail_trim_test PRIVATE ${NATIVE_DIR}/tts)
add_test(NAME tts_tail_trim_test COMMAND tts_tail_trim_test)

add_executable(stage_video_common_test
  video_stage_common_test.cpp
)
target_link_libraries(stage_video_common_test PRIVATE umm_stage_common)
add_test(NAME stage_video_common_test COMMAND stage_video_common_test)

add_executable(request_telemetry_test
  request_telemetry_test.cpp
)
target_link_libraries(request_telemetry_test PRIVATE umm_stage_common)
add_test(NAME request_telemetry_test COMMAND request_telemetry_test)

add_executable(stage_common_test
  stage_common_test.cpp
)
target_link_libraries(stage_common_test PRIVATE umm_stage_common)
add_test(NAME stage_common_test COMMAND stage_common_test)

# Stage CLI hooks. Links video_stage.cpp, so the checks run against the same
# parse_stage_flag the executable uses, on a host with no CUDA.
add_executable(stage_flag_test
  stage_flag_test.cpp
  video_stage.cpp
  ${NATIVE_DIR}/video/config.cpp
  ${NATIVE_DIR}/video/scheduler.cpp
  ${NATIVE_DIR}/video/mp4_encoder.cpp
  ${NATIVE_DIR}/video/engine.cpp
)
target_include_directories(stage_flag_test PRIVATE ${NATIVE_DIR})
target_link_libraries(stage_flag_test PRIVATE umm_stage_common)
add_test(NAME stage_flag_test COMMAND stage_flag_test)

# ---- Video (host contract-stub always available; CUDA kernels when UMM_STAGE_CUDA=ON) ----
add_executable(umm-native-stage-video
  main.cpp video_stage.cpp
  ${NATIVE_DIR}/video/config.cpp
  ${NATIVE_DIR}/video/scheduler.cpp
  ${NATIVE_DIR}/video/mp4_encoder.cpp
  ${NATIVE_DIR}/video/engine.cpp
)
target_include_directories(umm-native-stage-video PRIVATE ${NATIVE_DIR})
target_link_libraries(umm-native-stage-video PRIVATE umm_stage_common)

# Process-level checks: --help exits 0, and a flag of another stage exits
# non-zero before any model load.
add_test(NAME stage_video_help COMMAND umm-native-stage-video --help)
add_test(NAME stage_video_rejects_image_backend
  COMMAND umm-native-stage-video --kind video --image-backend 2512)
add_test(NAME stage_video_rejects_batch_size
  COMMAND umm-native-stage-video --kind video --batch-size 4)
set_tests_properties(stage_video_rejects_image_backend stage_video_rejects_batch_size
  PROPERTIES WILL_FAIL TRUE)


if(UMM_STAGE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(umm-native-stage-video PRIVATE
    ${NATIVE_DIR}/common/cuda_request_allocator.cpp
    ${NATIVE_DIR}/video/safetensors_mmap.cpp
    ${NATIVE_DIR}/video/umt5_tokenizer.cpp
    ${NATIVE_DIR}/video/umt5_encoder.cpp
    ${NATIVE_DIR}/video/wan_vae.cpp
    ${NATIVE_DIR}/video/wan_dit.cpp
    ${NATIVE_DIR}/video/cuda_kernels.cu
    ${NATIVE_DIR}/video/wan_vae_kernels.cu
    ${NATIVE_DIR}/video/umt5_kernels.cu
    ${NATIVE_DIR}/video/wan_dit_kernels.cu
  )
  target_compile_definitions(umm-native-stage-video PRIVATE UMM_VIDEO_CUDA=1 UMM_VIDEO_WAN_FULL=1 UMM_NATIVE_REQUEST_ALLOCATOR=1)
  target_link_libraries(umm-native-stage-video PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(umm-native-stage-video PROPERTIES CUDA_ARCHITECTURES 90)

  # ---- ASR ----
  add_executable(umm-native-stage-asr
    main.cpp asr_stage.cpp
    ${NATIVE_DIR}/umm_asr_native/feature_frontend.cpp
    ${NATIVE_DIR}/umm_asr_native/tokenizer.cpp
    ${NATIVE_DIR}/umm_asr_native/safetensors.cpp
    ${NATIVE_DIR}/umm_asr_native/engine.cpp
    ${NATIVE_DIR}/umm_asr_native/cuda_kernels.cu
  )
  target_link_libraries(umm-native-stage-asr PRIVATE umm_stage_common CUDA::cudart)
  set_target_properties(umm-native-stage-asr PROPERTIES CUDA_ARCHITECTURES 90)

  # ---- Omni ----
  add_executable(umm-native-stage-omni
    main.cpp omni_stage.cpp
    ${NATIVE_DIR}/omni/safetensors_mmap.cpp
    ${NATIVE_DIR}/omni/image_preprocess.cpp
    ${NATIVE_DIR}/omni/tokenizer.cpp
    ${NATIVE_DIR}/omni/mrope.cpp
    ${NATIVE_DIR}/omni/cuda_kernels_omni.cu
    ${NATIVE_DIR}/omni/omni_gemm.cu
    ${NATIVE_DIR}/omni/vision_forward.cu
    ${NATIVE_DIR}/omni/merger.cu
    ${NATIVE_DIR}/omni/text_decoder.cu
    ${NATIVE_DIR}/omni/omni_engine.cu
  )
  target_link_libraries(umm-native-stage-omni PRIVATE umm_stage_common CUDA::cudart)
  set_target_properties(umm-native-stage-omni PROPERTIES CUDA_ARCHITECTURES 90)

  # ---- Qwen3.8-27B ----
  add_executable(umm-native-stage-qwen38
    main.cpp qwen38_stage.cpp
    ${NATIVE_DIR}/common/cuda_request_allocator.cpp
    ${NATIVE_DIR}/qwen38/safetensors_mmap.cpp
    ${NATIVE_DIR}/qwen38/image_preprocess.cpp
    ${NATIVE_DIR}/qwen38/video_sampling.cpp
    ${NATIVE_DIR}/qwen38/video_input.cpp
    ${NATIVE_DIR}/qwen38/tokenizer.cpp
    ${NATIVE_DIR}/qwen38/prompt.cpp
    ${NATIVE_DIR}/qwen38/engine.cu
    ${NATIVE_DIR}/qwen38/qwen38_kernels.cu
    ${NATIVE_DIR}/qwen38/qwen38_gemm.cu
  )
  target_include_directories(umm-native-stage-qwen38 PRIVATE ${NATIVE_DIR})
  target_compile_definitions(umm-native-stage-qwen38 PRIVATE UMM_NATIVE_REQUEST_ALLOCATOR=1)
  target_link_libraries(umm-native-stage-qwen38 PRIVATE umm_stage_common CUDA::cudart)
  set_target_properties(umm-native-stage-qwen38 PROPERTIES CUDA_ARCHITECTURES 90)

  # ---- TTS ----
  find_package(ICU QUIET COMPONENTS uc i18n)
  if(NOT TARGET ICU::uc)
    find_path(ICU_INCLUDE_DIR unicode/unistr.h REQUIRED)
    find_library(ICU_UC_LIBRARY NAMES icuuc REQUIRED)
    find_library(ICU_I18N_LIBRARY NAMES icui18n REQUIRED)
    add_library(ICU::uc UNKNOWN IMPORTED)
    set_target_properties(ICU::uc PROPERTIES IMPORTED_LOCATION "${ICU_UC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ICU_INCLUDE_DIR}")
    add_library(ICU::i18n UNKNOWN IMPORTED)
    set_target_properties(ICU::i18n PROPERTIES IMPORTED_LOCATION "${ICU_I18N_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ICU_INCLUDE_DIR}")
  endif()

  add_executable(umm-native-stage-tts
    main.cpp tts_stage.cpp
    ${NATIVE_DIR}/tts/wav_io.cpp
    ${NATIVE_DIR}/tts/safetensors.cpp
    ${NATIVE_DIR}/tts/bpe_tokenizer.cpp
    ${NATIVE_DIR}/tts/prompt.cpp
    ${NATIVE_DIR}/tts/engine.cpp
    ${NATIVE_DIR}/tts/codec.cpp
    ${NATIVE_DIR}/tts/tail_trim.cpp
    ${NATIVE_DIR}/tts/cuda_kernels.cu
  )
  target_include_directories(umm-native-stage-tts PRIVATE ${NATIVE_DIR}/tts)
  target_link_libraries(umm-native-stage-tts PRIVATE
    umm_stage_common ICU::uc ICU::i18n CUDA::cudart)
  set_target_properties(umm-native-stage-tts PROPERTIES CUDA_ARCHITECTURES 90)

  # ---- Image (Qwen-Image-2512 + Qwen-Image-2.1) ----
  find_package(ZLIB REQUIRED)
  add_library(umm_image2512_backend OBJECT
    ${NATIVE_DIR}/image/scheduler.cpp
    ${NATIVE_DIR}/image/png.cpp
    ${NATIVE_DIR}/image/safetensors_shard.cpp
    ${NATIVE_DIR}/image/tokenizer.cpp
    ${NATIVE_DIR}/image/packing.cpp
    ${NATIVE_DIR}/image/tensor.cpp
    ${NATIVE_DIR}/image/weights.cpp
    ${NATIVE_DIR}/image/text_encoder.cpp
    ${NATIVE_DIR}/image/dit.cpp
    ${NATIVE_DIR}/image/vae.cpp
    ${NATIVE_DIR}/image/engine.cpp
    ${NATIVE_DIR}/image/cuda_kernels.cu
  )
  target_include_directories(umm_image2512_backend PRIVATE
    ${NATIVE_DIR}/image ${NATIVE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(umm_image2512_backend PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cublasLt ZLIB::ZLIB)
  if(nlohmann_json_FOUND)
    target_link_libraries(umm_image2512_backend PRIVATE nlohmann_json::nlohmann_json)
  else()
    target_include_directories(umm_image2512_backend PRIVATE ${NLOHMANN_JSON_INCLUDE})
  endif()
  set_target_properties(umm_image2512_backend PROPERTIES CUDA_ARCHITECTURES 90)

  add_library(umm_image21_backend OBJECT
    ${NATIVE_DIR}/image2/scheduler.cpp
    ${NATIVE_DIR}/image2/png.cpp
    ${NATIVE_DIR}/image2/safetensors_shard.cpp
    ${NATIVE_DIR}/image2/tokenizer.cpp
    ${NATIVE_DIR}/image2/tensor.cpp
    ${NATIVE_DIR}/image2/weights.cpp
    ${NATIVE_DIR}/image2/text_encoder.cpp
    ${NATIVE_DIR}/image2/dit.cpp
    ${NATIVE_DIR}/image2/vae.cpp
    ${NATIVE_DIR}/image2/engine.cpp
    ${NATIVE_DIR}/image2/cuda_kernels.cu
  )
  target_include_directories(umm_image21_backend PRIVATE
    ${NATIVE_DIR}/image2 ${NATIVE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(umm_image21_backend PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cublasLt ZLIB::ZLIB)
  if(nlohmann_json_FOUND)
    target_link_libraries(umm_image21_backend PRIVATE nlohmann_json::nlohmann_json)
  else()
    target_include_directories(umm_image21_backend PRIVATE ${NLOHMANN_JSON_INCLUDE})
  endif()
  set_target_properties(umm_image21_backend PROPERTIES CUDA_ARCHITECTURES 90)

  add_executable(umm-native-stage-image
    main.cpp image_stage.cpp
    $<TARGET_OBJECTS:umm_image2512_backend>
    $<TARGET_OBJECTS:umm_image21_backend>
  )
  target_include_directories(umm-native-stage-image PRIVATE ${NATIVE_DIR})
  target_link_libraries(umm-native-stage-image PRIVATE
    umm_stage_common CUDA::cudart CUDA::cublas CUDA::cublasLt ZLIB::ZLIB)
  set_target_properties(umm-native-stage-image PROPERTIES CUDA_ARCHITECTURES 90)
endif()
