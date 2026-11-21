# SPDX-License-Identifier: Apache-2.0
# Native Qwen-Image 2.1 vertical slice. Self-contained under native/image2 so
# it does not disturb the sibling builds. Host code (scheduler, PNG,
# safetensors reader, tokenizer) builds without CUDA; the engine and
# umm-image2-native need UMM_IMAGE2_CUDA=ON.
cmake_minimum_required(VERSION 3.24)
project(umm_image2_native LANGUAGES CXX)

option(UMM_IMAGE2_CUDA "Build the CUDA inference engine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)

add_library(umm_image2_host STATIC
  scheduler.cpp
  png.cpp
  safetensors_shard.cpp
  tokenizer.cpp
)
target_include_directories(umm_image2_host PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(umm_image2_host PUBLIC ZLIB::ZLIB)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_image2_host PUBLIC nlohmann_json::nlohmann_json)
else()
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp)
  if(NLOHMANN_JSON_INCLUDE)
    target_include_directories(umm_image2_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
  else()
    message(FATAL_ERROR "nlohmann/json.hpp not found")
  endif()
endif()

enable_testing()
add_executable(image2_host_tests ${CMAKE_CURRENT_SOURCE_DIR}/../image2_tests/test_host.cpp)
target_link_libraries(image2_host_tests PRIVATE umm_image2_host)
add_test(NAME image2_host_tests COMMAND image2_host_tests)

if(UMM_IMAGE2_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(umm-image2-native
    ${CMAKE_CURRENT_SOURCE_DIR}/../image2_cli/main.cpp
    engine.cpp
    text_encoder.cpp
    dit.cpp
    vae.cpp
    tensor.cpp
    weights.cpp
    cuda_kernels.cu
  )
  target_include_directories(umm-image2-native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(umm-image2-native PRIVATE umm_image2_host CUDA::cudart
                        CUDA::cublas CUDA::cublasLt)
  set_target_properties(umm-image2-native PROPERTIES CUDA_ARCHITECTURES 90)
endif()
