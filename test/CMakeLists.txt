#add_subdirectory(lib/googletest)
#enable_testing()


# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${GTEST_SOURCE_DIR}/include")
endif()


# Now simply link your own targets against gtest, gmock,
add_executable(${PROJECT_NAME}-gtest utest.cpp)
target_link_libraries(${PROJECT_NAME}-gtest ${GTEST_LIBRARIES})
target_link_libraries(${PROJECT_NAME}-gtest ${PROJECT_NAME})

add_test(test_one  ${PROJECT_NAME}-gtest "/data/git/temp_ws/src/gpuimageproc/test")
