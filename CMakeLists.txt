cmake_minimum_required(VERSION 3.22)

project(Zzz VERSION 1.0 LANGUAGES CXX)

include_directories(include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    apps/main.cpp
    )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})