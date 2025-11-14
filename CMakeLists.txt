cmake_minimum_required(VERSION 3.10)
project(axis_sdk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Axis include and source directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/axis)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
option(USE_PCL "Enable PCL for LiDAR processing" OFF)
if(USE_PCL)
    find_package(PCL REQUIRED)
    add_definitions(-DWITH_PCL)
endif()

# axis_core library
file(GLOB AXIS_SOURCES src/*.cpp src/utils/*.cpp)
list(REMOVE_ITEM AXIS_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")
add_library(axis_core ${AXIS_SOURCES})
target_link_libraries(axis_core PRIVATE Eigen3::Eigen yaml-cpp)
if(USE_PCL)
    target_link_libraries(axis_core PRIVATE ${PCL_LIBRARIES})
    target_include_directories(axis_core PRIVATE ${PCL_INCLUDE_DIRS})
endif()

# axis_node executable (placeholder main)
add_executable(axis_node src/main.cpp)
target_link_libraries(axis_node PRIVATE axis_core Eigen3::Eigen yaml-cpp)

# Tests
enable_testing()
add_subdirectory(tests)


# Install targets
install(TARGETS axis_core axis_node
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
