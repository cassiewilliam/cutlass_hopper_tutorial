find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
  add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c1_test_gemm_kernel_raw gemm_kernel_raw.cu)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemm_kernel_raw.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass gemm_kernel_raw kernel required sm 90a")
endif()

target_include_directories(c1_test_gemm_kernel_raw PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1_test_gemm_kernel_raw PRIVATE nvidia::cutlass::cutlass)

add_executable(c1_test_gemm_kernel_swizzle gemm_kernel_swizzle.cu)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemm_kernel_swizzle.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass gemm_kernel_swizzle kernel required sm 90a")
endif()
target_include_directories(c1_test_gemm_kernel_swizzle PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1_test_gemm_kernel_swizzle PRIVATE nvidia::cutlass::cutlass)