cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)
project (CPP_workaround)
# fucking bullshit library that is necessary since python_pcl is not capeable of clusterin based on normal direction...
#... and shitty retared python is too slow to do the clustering without c++

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
message(STATUS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable (conditional_euclidean_clustering cec.cpp)
target_link_libraries (conditional_euclidean_clustering ${PCL_LIBRARIES})
