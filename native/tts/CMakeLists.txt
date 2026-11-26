# SPDX-License-Identifier: Apache-2.0
# Native Qwen3-TTS vertical slice. Standalone project: the ASR tree under
# ../umm_asr_native is intentionally untouched. Host code (WAV, safetensors,
# BPE tokenizer) builds without CUDA; the CUDA engine and umm-tts-native CLI
# are added when UMM_TTS_CUDA=ON.
cmake_minimum_required(VERSION 3.24)
project(umm_tts_native LANGUAGES CXX)

option(UMM_TTS_CUDA "Build the CUDA TTS engine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew installs icu4c keg-only; honor ICU_ROOT, then the opt path.
if(NOT DEFINED ICU_ROOT AND EXISTS "/opt/homebrew/opt/icu4c/include/unicode/unistr.h")
  set(ICU_ROOT "/opt/homebrew/opt/icu4c")
endif()
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
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp REQUIRED)
endif()

add_library(umm_tts_host STATIC
  wav_io.cpp
  safetensors.cpp
  bpe_tokenizer.cpp
  prompt.cpp
)
target_include_directories(umm_tts_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(umm_tts_host PUBLIC ICU::uc ICU::i18n)
if(nlohmann_json_FOUND)
  target_link_libraries(umm_tts_host PUBLIC nlohmann_json::nlohmann_json)
else()
  target_include_directories(umm_tts_host PUBLIC ${NLOHMANN_JSON_INCLUDE})
endif()

enable_testing()
add_executable(tts_host_tests ../tts_tests/test_host.cpp)
target_link_libraries(tts_host_tests PRIVATE umm_tts_host)
add_test(NAME tts_host_tests COMMAND tts_host_tests)

# Config parsing and weight-shape plan for both wired sizes. Header-only, no
# CUDA and no weights, so it runs on the development host.
add_executable(tts_config_test tts_config_test.cpp speaker_encoder.cpp)
target_link_libraries(tts_config_test PRIVATE umm_tts_host)
add_test(NAME tts_config_test COMMAND tts_config_test)

if(UMM_TTS_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(umm-tts-native
    ../tts_cli/main.cpp
    engine.cpp
    prompt.cpp
    codec.cpp
    tail_trim.cpp
    cuda_kernels.cu
  )
  target_include_directories(umm-tts-native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(umm-tts-native PRIVATE umm_tts_host CUDA::cudart)
  set_target_properties(umm-tts-native PROPERTIES CUDA_ARCHITECTURES 90)

  add_executable(tts_cuda_determinism_test
    ../tts_tests/test_cuda_determinism.cpp
    cuda_kernels.cu
  )
  target_include_directories(tts_cuda_determinism_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(tts_cuda_determinism_test PRIVATE CUDA::cudart)
  set_target_properties(tts_cuda_determinism_test PROPERTIES CUDA_ARCHITECTURES 90)
  add_test(NAME tts_cuda_determinism_test COMMAND tts_cuda_determinism_test)
endif()
