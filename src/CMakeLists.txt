set(_sources
      utils/strings.cpp
      unit/runtime_parameters.cpp
)

add_executable( 
   kamayan
      main.cpp
      ${_sources}
)


set(_test_sources
      utils/tests/test_strings.cpp
      utils/tests/test_type_list.cpp
      dispatcher/tests/test_dispatcher.cpp
      dispatcher/tests/test_options.cpp
      unit/tests/test_config.cpp
      unit/tests/test_runtime_parameters.cpp
)

target_link_libraries(kamayan PRIVATE parthenon)
target_include_directories(kamayan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (kamayan_ENABLE_TESTING) 
   enable_testing()
   add_executable(
      kamayan_test
         ${_sources}
         ${_test_sources}
   )
   target_link_libraries(kamayan_test GTest::gtest_main parthenon) 
   target_include_directories(kamayan_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
   include(GoogleTest)
   gtest_discover_tests(kamayan_test)
endif()
