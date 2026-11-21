# SPDX-License-Identifier: Apache-2.0
# Native Qwen-Image-2512 vertical slice. This CMake project is self-contained
# under native/image and does not touch the sibling Qwen3-ASR build. Host-side
# code (scheduler, PNG, safetensors shard reader, tokenizer, packing) builds
# without CUDA; the CUDA engine and umm-image-native need UMM_IMAGE_CUDA=ON.
cmake_minimum_required(VERSION 3.24)
project(umm_image_native LANGUAGES CXX)

option(UMM_IMAGE_CUDA "Build the CUDA inference engine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)

# Sources are relative to this directory (native/image).
add_library(umm_image_host STATIC
  scheduler.cpp
  png.cpp
  safetensors_shard.cpp
  tokenizer.cpp
  packing.cpp
)
target_include_directories(umm_image_host PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}        # engine internal includes
  ${CMAKE_CURRENT_SOURCE_DIR}/..)    # lets tests include "image/png.hpp"
target_link_libraries(umm_image_host PUBLIC ZLIB::ZLIB)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_image_host PUBLIC nlohmann_json::nlohmann_json)
else()
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp)
  if(NLOHMANN_JSON_INCLUDE)
    target_include_directories(umm_image_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
  else()
    message(FATAL_ERROR "nlohmann/json.hpp not found")
  endif()
endif()

enable_testing()
add_executable(image_host_tests ${CMAKE_CURRENT_SOURCE_DIR}/../image_tests/test_host.cpp)
target_link_libraries(image_host_tests PRIVATE umm_image_host)
add_test(NAME image_host_tests COMMAND image_host_tests)

if(UMM_IMAGE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(umm-image-native
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_cli/main.cpp
    engine.cpp
    text_encoder.cpp
    dit.cpp
    vae.cpp
    tensor.cpp
    weights.cpp
    cuda_kernels.cu
  )
  target_include_directories(umm-image-native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(umm-image-native PRIVATE umm_image_host CUDA::cudart CUDA::cublas
                        CUDA::cublasLt)
  set_target_properties(umm-image-native PROPERTIES CUDA_ARCHITECTURES 90)
endif()
