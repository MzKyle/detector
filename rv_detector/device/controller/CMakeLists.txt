cmake_minimum_required(VERSION 3.12)

project(device_controller)

file(GLOB ${PROJECT_NAME}_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/crc16.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/serial.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/robot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/usb.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/behavior.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/node.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SRC})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    module_component
    spdlog::spdlog
    Threads::Threads
    usbp-1
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(${PROJECT_NAME} module_component)