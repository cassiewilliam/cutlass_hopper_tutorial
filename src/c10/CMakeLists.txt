find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
  add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c10_test_i2f_optimize i2f_optimize.cu)
target_include_directories(c10_test_i2f_optimize PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c10_test_i2f_optimize PRIVATE nvidia::cutlass::cutlass)