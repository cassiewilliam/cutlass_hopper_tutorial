find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c108_test_deep_gemm deep_gemm.cu tiled_deep_gemm.cu)

if (DEFINED CMAKE_CUDA_COMPILER_VERSION AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(deep_gemm.cu tiled_deep_gemm.cu
        PROPERTIES COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "CUTLASS tiled_deep_gemm kernel requires CUDA 12+ and SM 90a")
endif()

target_include_directories(c108_test_deep_gemm PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c108_test_deep_gemm PRIVATE nvidia::cutlass::cutlass)