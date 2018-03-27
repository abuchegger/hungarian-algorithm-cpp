cmake_minimum_required(VERSION 2.8.3)
project(hungarian-algorithm-cpp)

find_package(Boost REQUIRED COMPONENTS chrono)

add_compile_options(-std=c++98 -Wall -Wextra)

include_directories(Boost_INCLUDE_DIRS)

#add_library(${PROJECT_NAME}
#  hungarian_algorithm.cpp hungarian_algorithm.h
#)
#
#target_link_libraries(${PROJECT_NAME}
#)

enable_testing()

add_executable(${PROJECT_NAME}_test
  test.cpp hungarian_algorithm.h
)

target_link_libraries(${PROJECT_NAME}_test
  ${Boost_LIBRARIES}
)

add_test(COMMAND ${PROJECT_NAME}_test)
