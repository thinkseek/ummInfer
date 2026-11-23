# SPDX-License-Identifier: Apache-2.0
# Native C++/CUDA Wan2.2 video vertical slice. Self-contained under native/video
# so it does not disturb sibling builds. Host code (config, alignment, flow-match
# scheduler, POSIX ffmpeg MP4 encoder, contract-stub engine) builds without CUDA;
# CUDA kernels and cuda-scaffold engine enable when UMM_VIDEO_CUDA=ON.
cmake_minimum_required(VERSION 3.24)
project(umm_video_native LANGUAGES CXX)

option(UMM_VIDEO_CUDA "Build CUDA video kernels (euler/denorm/rgb24)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(umm_video_host STATIC
  config.cpp
  scheduler.cpp
  mp4_encoder.cpp
  engine.cpp
)
target_include_directories(umm_video_host PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_video_host PUBLIC nlohmann_json::nlohmann_json)
else()
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp)
  if(NLOHMANN_JSON_INCLUDE)
    target_include_directories(umm_video_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
  else()
    message(FATAL_ERROR "nlohmann/json.hpp not found")
  endif()
endif()

if(UMM_VIDEO_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(umm_video_host PRIVATE cuda_kernels.cu)
  target_compile_definitions(umm_video_host PUBLIC UMM_VIDEO_CUDA=1)
  target_link_libraries(umm_video_host PUBLIC CUDA::cudart)
  set_target_properties(umm_video_host PROPERTIES CUDA_ARCHITECTURES 90)
endif()

enable_testing()
add_executable(video_host_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/../video_tests/test_host.cpp)
target_link_libraries(video_host_tests PRIVATE umm_video_host)
add_test(NAME video_host_tests COMMAND video_host_tests)
