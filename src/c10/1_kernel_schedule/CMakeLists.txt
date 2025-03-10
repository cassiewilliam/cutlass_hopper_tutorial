find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c101_test_cutlass_warp_specialization_kernel cutlass_warp_specialization_kernel.cu)
target_include_directories(c101_test_cutlass_warp_specialization_kernel PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c101_test_cutlass_warp_specialization_kernel PRIVATE nvidia::cutlass::cutlass)