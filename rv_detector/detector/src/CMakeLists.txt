cmake_minimum_required(VERSION 3.12)

project(detector)

add_executable(${PROJECT_NAME} main.cpp detector.cpp number_classifier.cpp compensator.cpp ekf.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../device/controller
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    device_camera
    device_controller
)

add_dependencies(${PROJECT_NAME}
    device_camera
    device_controller
    module_component
)