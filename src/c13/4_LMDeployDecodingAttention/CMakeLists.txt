# Copyright (c) OpenMMLab. All rights reserved.
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/c13/4_LMDeployDecodingAttention
)

add_library(decoding STATIC
            decoding.cu
            reduce.cu
            kv_cache_utils_v2.cu
            utils.cc
            unfused_attention_kernels.cu
            codegen/decoding_sm80_128_bf16_bf16.cu
            codegen/decoding_sm80_128_bf16_u4.cu
            codegen/decoding_sm80_128_bf16_u8.cu
            codegen/decoding_sm80_128_f16_f16.cu
            codegen/decoding_sm80_128_f16_u4.cu
            codegen/decoding_sm80_128_f16_u8.cu
            codegen/decoding_sm80_64_bf16_bf16.cu
            codegen/decoding_sm80_64_bf16_u4.cu
            codegen/decoding_sm80_64_bf16_u8.cu
            codegen/decoding_sm80_64_f16_f16.cu
            codegen/decoding_sm80_64_f16_u4.cu
            codegen/decoding_sm80_64_f16_u8.cu
            )
set_property(TARGET decoding PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET decoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(decoding PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr>)

target_compile_options(decoding PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info>)

add_executable(c134_lmdeploy_decoding_attention_test test_utils.cu test_attention.cu reference.cu)
target_compile_options(c134_lmdeploy_decoding_attention_test PRIVATE --generate-line-info -use_fast_math --expt-relaxed-constexpr)
target_link_libraries(c134_lmdeploy_decoding_attention_test PRIVATE decoding cublasMMWrapper logger tensor cublas)