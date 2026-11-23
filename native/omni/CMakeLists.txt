# SPDX-License-Identifier: Apache-2.0
# Self-contained build for the native Qwen3-Omni thinker/VLM slice. Lives
# entirely under native/omni*, independent of the ASR CMake project.
# Host build:  cmake -S native/omni -B native/omni/build && cmake --build ...
# CUDA build:  add -DUMM_OMNI_CUDA=ON on the GPU host.
cmake_minimum_required(VERSION 3.24)
project(umm_omni_native LANGUAGES CXX)

option(UMM_OMNI_CUDA "Build the CUDA inference engine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OMNI_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OMNI_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../omni_tests)
set(OMNI_CLI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../omni_cli)

add_library(umm_omni_host STATIC
  ${OMNI_DIR}/safetensors_mmap.cpp
  ${OMNI_DIR}/image_preprocess.cpp
  ${OMNI_DIR}/tokenizer.cpp
  ${OMNI_DIR}/mrope.cpp
)
target_include_directories(umm_omni_host PUBLIC ${OMNI_DIR}/..)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_omni_host PUBLIC nlohmann_json::nlohmann_json)
else()
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp)
  if(NLOHMANN_JSON_INCLUDE)
    target_include_directories(umm_omni_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
  else()
    message(FATAL_ERROR "nlohmann/json.hpp not found")
  endif()
endif()

enable_testing()
add_executable(omni_host_tests ${OMNI_TESTS_DIR}/test_omni_host.cpp)
target_link_libraries(omni_host_tests PRIVATE umm_omni_host)
add_test(NAME omni_host_tests
         COMMAND omni_host_tests ${OMNI_TESTS_DIR})

if(UMM_OMNI_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(${OMNI_CLI_DIR} ${CMAKE_BINARY_DIR}/omni_cli)
endif()
