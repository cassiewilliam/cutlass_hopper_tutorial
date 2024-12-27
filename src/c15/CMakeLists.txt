find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
  add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)

add_executable(c15_sync_host_device_example sync_host_device_example.cu)
target_include_directories(c15_sync_host_device_example PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${COMMON_HEADER_DIR})
target_link_libraries(c15_sync_host_device_example PRIVATE nvidia::cutlass::cutlass)