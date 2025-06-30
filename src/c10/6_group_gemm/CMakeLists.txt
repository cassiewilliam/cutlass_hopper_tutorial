find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(c1060_test_hopper_grouped_gemm 0_hopper_grouped_gemm.cu)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
        "0_hopper_grouped_gemm.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "cutlass warp_specialization_gemm kernel required sm 90a")
endif()

target_include_directories(c1060_test_hopper_grouped_gemm PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1060_test_hopper_grouped_gemm PRIVATE nvidia::cutlass::cutlass CUDA::cudart CUDA::cublas)