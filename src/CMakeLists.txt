set(_sources
      utils/strings.cpp
)

add_executable( 
   kamayan
      main.cpp
      ${_sources}
)


set(_test_sources
      utils/tests/test_strings.cpp
      utils/tests/test_type_list.cpp
      units/tests/test_dispatcher.cpp
)

target_link_libraries(kamayan PRIVATE parthenon)
if (kamayan_ENABLE_TESTING) 
   enable_testing()
   add_executable(
      kamayan_test
         ${_sources}
         ${_test_sources}
   )
   target_link_libraries(kamayan_test GTest::gtest_main parthenon) 
   include(GoogleTest)
   gtest_discover_tests(kamayan_test)
endif()
