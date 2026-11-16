# SPDX-License-Identifier: Apache-2.0
set(OMNI_SOURCES
  ${OMNI_DIR}/safetensors_mmap.cpp
  ${OMNI_DIR}/image_preprocess.cpp
  ${OMNI_DIR}/tokenizer.cpp
  ${OMNI_DIR}/cuda_kernels_omni.cu
  ${OMNI_DIR}/vision_forward.cu
  ${OMNI_DIR}/merger.cu
  ${OMNI_DIR}/text_decoder.cu
  ${OMNI_DIR}/omni_engine.cu
)

add_executable(umm-omni-native
  ${OMNI_CLI_DIR}/main.cpp
  ${OMNI_SOURCES}
)
set_target_properties(umm-omni-native PROPERTIES CUDA_ARCHITECTURES 90)
target_link_libraries(umm-omni-native PRIVATE umm_omni_host CUDA::cudart)
target_include_directories(umm-omni-native PRIVATE ${OMNI_DIR}/..)
