cmake_minimum_required(VERSION 3.12)
project(device_camera)

# ---------------------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------------------
file(GLOB ${PROJECT_NAME}_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hik_camera.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SRC})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    module_component
    MvCameraControl
    spdlog::spdlog
    Threads::Threads
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(${PROJECT_NAME} PUBLIC /opt/MVS/lib/aarch64/)
else()
    target_link_directories(${PROJECT_NAME} PUBLIC /opt/MVS/lib/64/)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    /opt/MVS/include
)

add_dependencies(${PROJECT_NAME} module_component)
