find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c100_handle_gemm_test gemm_test.cu)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
        "gemm_test.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "cutlass gemm_test kernel required sm 90a")
endif()

target_include_directories(c100_handle_gemm_test PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c100_handle_gemm_test PRIVATE nvidia::cutlass::cutlass -lcuda -lcudart)