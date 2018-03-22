cmake_minimum_required(VERSION 2.8.3)
project(hungarian-algorithm-cpp)

add_compile_options(-std=c++11 -Wall -Wextra)

add_library(${PROJECT_NAME}
  Hungarian.cpp Hungarian.h
)

target_link_libraries(${PROJECT_NAME}
)
