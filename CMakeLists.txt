cmake_minimum_required(VERSION 3.22)
project(attn_stress LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 90)   # A100 / H100

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS $ENV{NCCL_HOME}/include /usr/include /usr/local/include)
find_library(NCCL_LIBRARY nccl
  HINTS $ENV{NCCL_HOME}/lib $ENV{NCCL_HOME}/lib64
        /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_HOME=/path/to/nccl")
endif()

add_executable(attn_stress src/main.cu)

target_include_directories(attn_stress PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${NCCL_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(attn_stress PRIVATE
  MPI::MPI_CXX
  ${NCCL_LIBRARY}
  CUDA::cudart
  CUDA::cublas
)

target_compile_options(attn_stress PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--ptxas-options=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
