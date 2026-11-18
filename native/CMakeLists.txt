# SPDX-License-Identifier: Apache-2.0
# Native Qwen3-ASR vertical slice. Host-side code builds without CUDA;
# the CUDA engine and the umm-asr-native CLI are added when UMM_ASR_CUDA=ON.
cmake_minimum_required(VERSION 3.24)
project(umm_asr_native LANGUAGES CXX)

option(UMM_ASR_CUDA "Build the CUDA inference engine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(umm_asr_host STATIC
  umm_asr_native/wav.cpp
  umm_asr_native/safetensors.cpp
  umm_asr_native/feature_frontend.cpp
  umm_asr_native/tokenizer.cpp
)
target_include_directories(umm_asr_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_asr_host PUBLIC nlohmann_json::nlohmann_json)
else()
  # Distro / homebrew header install without CMake package config.
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp)
  if(NLOHMANN_JSON_INCLUDE)
    target_include_directories(umm_asr_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
  else()
    message(FATAL_ERROR "nlohmann/json.hpp not found")
  endif()
endif()

enable_testing()
add_executable(asr_host_tests tests/test_host.cpp)
target_link_libraries(asr_host_tests PRIVATE umm_asr_host)
add_test(NAME asr_host_tests COMMAND asr_host_tests)

add_executable(mel_dump tests/mel_dump.cpp)
target_link_libraries(mel_dump PRIVATE umm_asr_host)

if(UMM_ASR_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(umm-asr-native
    cli/main.cpp
    umm_asr_native/feature_frontend.cpp
    umm_asr_native/tokenizer.cpp
    umm_asr_native/engine.cpp
    umm_asr_native/cuda_kernels.cu
  )
  target_link_libraries(umm-asr-native PRIVATE umm_asr_host CUDA::cudart)
  set_target_properties(umm-asr-native PROPERTIES CUDA_ARCHITECTURES 90)

  add_executable(gqa_tests tests/test_gqa.cu umm_asr_native/cuda_kernels.cu)
  target_include_directories(gqa_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(gqa_tests PRIVATE CUDA::cudart)
  set_target_properties(gqa_tests PROPERTIES CUDA_ARCHITECTURES 90)
  add_test(NAME gqa_tests COMMAND gqa_tests)
endif()

# ---- Native Qwen3.8-27B executor -------------------------------------------
option(UMM_QWEN38_CUDA "Build the native Qwen3.8-27B executor" OFF)

add_library(qwen38_host STATIC
  qwen38/safetensors_mmap.cpp
  qwen38/tokenizer.cpp
  qwen38/image_preprocess.cpp
  qwen38/prompt.cpp
)
target_include_directories(qwen38_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(nlohmann_json_FOUND)
  target_link_libraries(qwen38_host PUBLIC nlohmann_json::nlohmann_json)
elseif(NLOHMANN_JSON_INCLUDE)
  target_include_directories(qwen38_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
else()
  message(FATAL_ERROR "nlohmann/json.hpp not found")
endif()

add_executable(qwen38_host_tests qwen38_tests/test_host.cpp)
target_link_libraries(qwen38_host_tests PRIVATE qwen38_host)
add_test(NAME qwen38_host_tests COMMAND qwen38_host_tests)

add_executable(qwen38_tokenize qwen38_tests/tokenize_cli.cpp)
target_link_libraries(qwen38_tokenize PRIVATE qwen38_host)

add_executable(qwen38_preprocess qwen38_tests/preprocess_cli.cpp)
target_link_libraries(qwen38_preprocess PRIVATE qwen38_host)

if(UMM_QWEN38_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "UMM_QWEN38_CUDA=ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(umm-qwen38-native
    qwen38_cli/main.cpp
    qwen38/engine.cu
    qwen38/qwen38_kernels.cu
    qwen38/qwen38_gemm.cu
  )
  target_link_libraries(umm-qwen38-native PRIVATE qwen38_host CUDA::cudart)
  set_target_properties(umm-qwen38-native PROPERTIES CUDA_ARCHITECTURES 90)

  add_executable(qwen38_kernel_tests
    qwen38_tests/test_delta.cu
    qwen38/qwen38_kernels.cu
    qwen38/qwen38_gemm.cu
  )
  target_include_directories(qwen38_kernel_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qwen38_kernel_tests PRIVATE CUDA::cudart)
  set_target_properties(qwen38_kernel_tests PROPERTIES CUDA_ARCHITECTURES 90)
  add_test(NAME qwen38_kernel_tests COMMAND qwen38_kernel_tests)

  add_executable(qwen38_consistency_tests
    qwen38_tests/test_consistency.cu
    qwen38/qwen38_kernels.cu
    qwen38/qwen38_gemm.cu
  )
  target_include_directories(qwen38_consistency_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qwen38_consistency_tests PRIVATE CUDA::cudart)
  set_target_properties(qwen38_consistency_tests PROPERTIES CUDA_ARCHITECTURES 90)
  add_test(NAME qwen38_consistency_tests COMMAND qwen38_consistency_tests)

  # Times the batched decode GEMV against the single-row GEMV and checks that
  # every row is bitwise identical. Evidence for docs/batching.md.
  add_executable(qwen38_gemv_bench
    qwen38_tests/test_gemv_batch.cu
    qwen38/qwen38_kernels.cu
    qwen38/qwen38_gemm.cu
  )
  target_include_directories(qwen38_gemv_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qwen38_gemv_bench PRIVATE CUDA::cudart)
  set_target_properties(qwen38_gemv_bench PROPERTIES CUDA_ARCHITECTURES 90)
endif()
