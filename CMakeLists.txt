cmake_minimum_required(VERSION 2.8.3)
project(hungarian-algorithm-cpp)

find_package(Boost REQUIRED COMPONENTS chrono unit_test_framework)

add_compile_options(-std=c++98 -Wall -Wextra)

include_directories(Boost_INCLUDE_DIRS)

#add_library(${PROJECT_NAME}
#  hungarian_algorithm.cpp hungarian_algorithm.h
#)
#
#target_link_libraries(${PROJECT_NAME}
#)

enable_testing()

add_definitions(-DBOOST_TEST_DYN_LINK)

add_executable(${PROJECT_NAME}-test
  test.cpp hungarian_algorithm.h
)

target_link_libraries(${PROJECT_NAME}-test
  ${Boost_LIBRARIES}
)

add_test(NAME ${PROJECT_NAME}-test COMMAND ${PROJECT_NAME}-test)
