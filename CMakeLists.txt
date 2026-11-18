cmake_minimum_required(VERSION 3.18)
project(ese327_classifiers LANGUAGES CXX)

# Same targets as the Makefile: <name>_cpu, <name>_gpu_emu and, when a CUDA
# compiler is found, <name>_gpu, for name = dtree, nbayes, knn, logreg.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ml_common STATIC src/dataset.cpp src/eval.cpp src/runner.cpp src/util.cpp)
target_include_directories(ml_common PUBLIC include)
if(MSVC)
  target_compile_definitions(ml_common PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

add_executable(dataset_info src/dataset_info.cpp)
target_link_libraries(dataset_info PRIVATE ml_common)

set(dtree_SRC src/dtree.cpp)
set(nbayes_SRC)
set(knn_SRC src/encode.cpp)
set(logreg_SRC src/encode.cpp)
set(NAMES dtree nbayes knn logreg)

foreach(name ${NAMES})
  add_executable(${name}_cpu src/${name}_cpu.cpp ${${name}_SRC})
  target_link_libraries(${name}_cpu PRIVATE ml_common)

  # The CUDA source compiled as plain C++ (serial kernel emulation).
  add_executable(${name}_gpu_emu src/${name}_gpu_emu.cpp ${${name}_SRC})
  target_link_libraries(${name}_gpu_emu PRIVATE ml_common)
endforeach()

# Compile for the GPU of this machine (needs CMake 3.24; override with
# -DCMAKE_CUDA_ARCHITECTURES=86 or similar). A binary for another
# architecture is recompiled by the driver at startup, which fails when the
# driver is older than the CUDA Toolkit.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_VERSION VERSION_LESS 3.24)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  foreach(name ${NAMES})
    add_executable(${name}_gpu src/${name}_gpu.cu ${${name}_SRC})
    target_link_libraries(${name}_gpu PRIVATE ml_common)
    # No fused multiply-add: the kernels round exactly like the CPU code.
    target_compile_options(${name}_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    if(MSVC)
      # The CUB headers need the standard-conforming preprocessor.
      target_compile_options(${name}_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
  endforeach()
else()
  message(STATUS "No CUDA compiler found: the <name>_gpu targets are not built")
endif()
