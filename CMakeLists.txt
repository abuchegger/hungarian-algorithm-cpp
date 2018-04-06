cmake_minimum_required(VERSION 2.8.3)
project(hungarian-algorithm-cpp)

add_compile_options(-std=c++98 -Wall -Wextra)

add_library(${PROJECT_NAME} SHARED
  hungarian_algorithm.cpp hungarian_algorithm.h
)

target_link_libraries(${PROJECT_NAME}
)

find_package(Boost COMPONENTS chrono unit_test_framework)
if(Boost_FOUND)
  enable_testing()

  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions(-DBOOST_TEST_DYN_LINK)

  add_executable(${PROJECT_NAME}-test
    test.cpp hungarian_algorithm.h
  )

  target_link_libraries(${PROJECT_NAME}-test
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
  )

  add_test(NAME ${PROJECT_NAME}-test COMMAND ${PROJECT_NAME}-test --log_level=test_suite)
endif()

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES hungarian_algorithm.h
  DESTINATION include
)

configure_file(${PROJECT_NAME}-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake DESTINATION share/${PROJECT_NAME}/cmake)

