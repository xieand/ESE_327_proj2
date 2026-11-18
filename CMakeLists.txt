cmake_minimum_required(VERSION 3.18)
project(ese327_classifiers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ml_common STATIC src/dataset.cpp src/eval.cpp src/util.cpp)
target_include_directories(ml_common PUBLIC include)

add_executable(dataset_info src/dataset_info.cpp)
target_link_libraries(dataset_info PRIVATE ml_common)

# The classifiers (CPU, CUDA and CUDA-emulation builds) are added here as
# they are implemented.
