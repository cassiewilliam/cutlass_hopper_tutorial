# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
project(TurboMind LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")
endif()

option(BUILD_FAST_MATH "Build in fast math mode" ON)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass)
set(CUTLASS_TOOLS_HEAD_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include)
set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include)
set(COMMON_HEADER_DIR ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)

set(CXX_STD "17" CACHE STRING "C++ standard")
# enable gold linker for binary and .so
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# profiling
option(USE_NVTX "Whether or not to use nvtx" ON)
if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl") # -Xptxas -v

# TODO: build for sm_72 & sm_87 on aarch64 platform (Jetson devices)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real)
  if (${CUDA_VERSION} VERSION_GREATER_EQUAL "11")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
  endif ()
  if (${CUDA_VERSION} VERSION_GREATER_EQUAL "11.1")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
  endif ()
  if (${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real 90-real)
  endif ()
endif ()

message(STATUS "Building with CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}        -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
set(CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE}        -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")

if(BUILD_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE}        --use_fast_math")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --use_fast_math")
    message("Release build CUDA flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# turn off warnings on windows
if (MSVC)
  foreach(
    flag_var
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_CUDA_FLAGS
    CMAKE_CUDA_FLAGS_DEBUG
    CMAKE_CUDA_FLAGS_RELEASE
    CMAKE_CUDA_FLAGS_MINSIZEREL
    CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "-Wall" " /W0 " ${flag_var} "${${flag_var}}")
  endforeach()
endif()

include_directories(${COMMON_HEADER_DIR})
add_subdirectory(src)