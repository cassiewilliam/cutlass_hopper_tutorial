find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c1020_test_cutlass_warp_specialization_kernel 0_warp_specialization_kernel.cu)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
        "0_warp_specialization_kernel.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "cutlass 0_warp_specialization_kernel kernel required sm 90a")
endif()

target_include_directories(c1020_test_cutlass_warp_specialization_kernel PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1020_test_cutlass_warp_specialization_kernel PRIVATE nvidia::cutlass::cutlass)