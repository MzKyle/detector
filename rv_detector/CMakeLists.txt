cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

find_package(Ceres 2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/detector)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/component)