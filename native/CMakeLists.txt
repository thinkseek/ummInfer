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
