# SPDX-License-Identifier: Apache-2.0
set(OMNI_SOURCES
  ${OMNI_DIR}/safetensors_mmap.cpp
  ${OMNI_DIR}/image_preprocess.cpp
  ${OMNI_DIR}/tokenizer.cpp
  ${OMNI_DIR}/cuda_kernels_omni.cu
  ${OMNI_DIR}/omni_gemm.cu
  ${OMNI_DIR}/vision_forward.cu
  ${OMNI_DIR}/merger.cu
  ${OMNI_DIR}/text_decoder.cu
  ${OMNI_DIR}/omni_engine.cu
)

# Shared CUDA engine static library: the CLI and the benchmark both link it, so
# a measurement always runs the same object code as the shipped binary.
add_library(umm_omni_cuda STATIC ${OMNI_SOURCES})
set_target_properties(umm_omni_cuda PROPERTIES CUDA_ARCHITECTURES 90)
target_link_libraries(umm_omni_cuda PUBLIC umm_omni_host CUDA::cudart)
target_include_directories(umm_omni_cuda PUBLIC ${OMNI_DIR}/..)

add_executable(umm-omni-native ${OMNI_CLI_DIR}/main.cpp)
target_link_libraries(umm-omni-native PRIVATE umm_omni_cuda)

# Fixed-argument benchmark: image prefill and decode throughput.
add_executable(umm-omni-bench ${OMNI_TESTS_DIR}/bench_omni.cpp)
target_link_libraries(umm-omni-bench PRIVATE umm_omni_cuda)

# GPU regression test: the MoE down projection must keep the activation in FP32.
add_executable(omni_moe_down_test ${OMNI_TESTS_DIR}/test_moe_down.cu)
target_link_libraries(omni_moe_down_test PRIVATE umm_omni_cuda)
add_test(NAME omni_moe_down_test COMMAND omni_moe_down_test)
