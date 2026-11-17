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
  ${UMM_ROOT}/src/proto/wire.cpp
  ${UMM_ROOT}/src/proto/messages.cpp
  ${UMM_ROOT}/src/relay/relay.cpp
  ${UMM_ROOT}/src/comm/comm.cpp
  ${UMM_ROOT}/src/scheduling/simple_scheduler.cpp
  ${UMM_ROOT}/src/pipeline/pipeline.cpp
)
target_include_directories(umm_stage_core PUBLIC
  ${UMM_ROOT}/include ${UMM_ROOT}/src
  ${ZMQ_INCLUDE_DIR} ${MSGPACK_INCLUDE_DIR})
target_link_libraries(umm_stage_core PUBLIC ${ZMQ_LIBRARY} ${MSGPACK_LIBRARY})
if(nlohmann_json_FOUND)
  target_link_libraries(umm_stage_core PUBLIC nlohmann_json::nlohmann_json)
else()
  target_include_directories(umm_stage_core PUBLIC ${NLOHMANN_JSON_INCLUDE})
endif()

add_library(umm_stage_common STATIC
  stage_common.cpp
  ${NATIVE_DIR}/umm_asr_native/wav.cpp
)
target_include_directories(umm_stage_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${NATIVE_DIR})
target_link_libraries(umm_stage_common PUBLIC umm_stage_core)

enable_testing()
add_executable(stage_host_tests ${NATIVE_DIR}/stage_tests/test_host.cpp)
target_link_libraries(stage_host_tests PRIVATE umm_stage_common)
add_test(NAME stage_host_tests COMMAND stage_host_tests)

if(UMM_STAGE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

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
    ${NATIVE_DIR}/omni/cuda_kernels_omni.cu
    ${NATIVE_DIR}/omni/vision_forward.cu
    ${NATIVE_DIR}/omni/merger.cu
    ${NATIVE_DIR}/omni/text_decoder.cu
    ${NATIVE_DIR}/omni/omni_engine.cu
  )
  target_link_libraries(umm-native-stage-omni PRIVATE umm_stage_common CUDA::cudart)
  set_target_properties(umm-native-stage-omni PROPERTIES CUDA_ARCHITECTURES 90)

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
    ${NATIVE_DIR}/tts/cuda_kernels.cu
  )
  target_include_directories(umm-native-stage-tts PRIVATE ${NATIVE_DIR}/tts)
  target_link_libraries(umm-native-stage-tts PRIVATE
    umm_stage_common ICU::uc ICU::i18n CUDA::cudart)
  set_target_properties(umm-native-stage-tts PROPERTIES CUDA_ARCHITECTURES 90)
endif()
