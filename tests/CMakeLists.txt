cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/axis)

set(TEST_COMMON_LIBS axis_core GTest::GTest GTest::Main Eigen3::Eigen)

add_executable(test_ekf test_ekf.cpp)
target_link_libraries(test_ekf PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME test_ekf COMMAND test_ekf)

add_executable(test_ml_weighting test_ml_weighting.cpp)
target_link_libraries(test_ml_weighting PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME test_ml_weighting COMMAND test_ml_weighting)

add_executable(test_map_lite test_map_lite.cpp)
target_link_libraries(test_map_lite PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME test_map_lite COMMAND test_map_lite)

add_executable(test_mode_transitions test_mode_transitions.cpp)
target_link_libraries(test_mode_transitions PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME test_mode_transitions COMMAND test_mode_transitions)

# Legacy aggregate test target (keeps existing dummy/node tests building)
file(GLOB LEGACY_TEST_SOURCES test_dummy.cpp test_node_manager.cpp)
if(LEGACY_TEST_SOURCES)
    add_executable(axis_tests ${LEGACY_TEST_SOURCES})
    target_link_libraries(axis_tests PRIVATE ${TEST_COMMON_LIBS})
    add_test(NAME axis_tests COMMAND axis_tests)
endif()
