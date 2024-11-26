cmake_minimum_required(VERSION 3.15...3.27)
project(example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Geogram REQUIRED)

add_executable(main minimal_example.cpp)
target_link_libraries(main PRIVATE Geogram::geogram)
