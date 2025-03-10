find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    add_definitions(-DTORCH_CUDA=1)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
)
remove_definitions(-DTORCH_CUDA=1)