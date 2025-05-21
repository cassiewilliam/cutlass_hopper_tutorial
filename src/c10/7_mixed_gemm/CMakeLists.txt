find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c1070_test_i2f_optimize i2f_optimize.cu)
target_include_directories(c1070_test_i2f_optimize PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1070_test_i2f_optimize PRIVATE nvidia::cutlass::cutlass)

add_executable(c1071_test_hopper_mixed_input_groupgemm hopper_int4_fp8_gemm.cu)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
        "hopper_int4_fp8_group_gehopper_int4_fp8_gemmmm.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "cutlass hopper_int4_fp8_gemm kernel required sm 90a")
endif()

target_include_directories(c1071_test_hopper_mixed_input_groupgemm PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1071_test_hopper_mixed_input_groupgemm PRIVATE nvidia::cutlass::cutlass)