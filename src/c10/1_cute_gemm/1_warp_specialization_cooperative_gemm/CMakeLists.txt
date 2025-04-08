find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c1011_warp_specialization_cooperative_gemm c1011_warp_specialization_cooperative_gemm.cu)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
        "c1011_warp_specialization_cooperative_gemm.cu"
        PROPERTIES
        COMPILE_FLAGS
        "-gencode arch=compute_90a,code=sm_90a")
else()
    message(FATAL_ERROR "cutlass c1011_warp_specialization_cooperative_gemm kernel required sm 90a")
endif()

target_include_directories(c1011_warp_specialization_cooperative_gemm PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c1011_warp_specialization_cooperative_gemm PRIVATE nvidia::cutlass::cutlass)