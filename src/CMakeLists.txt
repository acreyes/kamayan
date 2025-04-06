set(_sources
    driver/kamayan_driver.cpp
    grid/grid.cpp
    kamayan/config.cpp
    kamayan/kamayan.cpp
    kamayan/runtime_parameters.cpp
    kamayan/unit.cpp
    physics/eos/eos.cpp
    physics/physics.cpp
    physics/hydro/hydro_time_step.cpp
    physics/hydro/hydro.cpp
    physics/hydro/hydro_add_flux_tasks.cpp
    physics/hydro/primconsflux.cpp
    utils/strings.cpp)

add_library(kamayan OBJECT ${_sources})
# target_sources(kamayan OBJECT ${_sources})

# maybe I should just glob for tests...
set(_test_sources
    utils/tests/test_strings.cpp
    utils/tests/test_type_list.cpp
    dispatcher/tests/test_dispatcher.cpp
    dispatcher/tests/test_options.cpp
    driver/tests/test_driver.cpp
    kamayan/tests/test_config.cpp
    kamayan/tests/test_runtime_parameters.cpp
    kamayan/tests/test_unit_collection.cpp
    physics/eos/tests/test_eos.cpp)

target_link_libraries(kamayan PUBLIC parthenon singularity-eos::singularity-eos)
target_include_directories(kamayan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(problems)

if(kamayan_ENABLE_TESTING)
  add_executable(kamayan_test ${_test_sources})
  target_link_libraries(kamayan_test GTest::gtest_main GTest::gmock_main
                        kamayan)
  target_include_directories(kamayan_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  include(GoogleTest)
  gtest_discover_tests(kamayan_test)
endif()
