set(_sources
    driver/kamayan_driver.cpp
    grid/grid.cpp
    grid/grid_refinement.cpp
    kamayan/config.cpp
    kamayan/kamayan.cpp
    kamayan/runtime_parameters.cpp
    kamayan/unit.cpp
    kamayan/unit_data.cpp
    physics/eos/eos.cpp
    physics/physics.cpp
    physics/hydro/hydro_time_step.cpp
    physics/hydro/hydro.cpp
    physics/hydro/hydro_add_flux_tasks.cpp
    physics/hydro/primconsflux.cpp
    utils/strings.cpp)

add_library(kamayan OBJECT ${_sources})
# target_sources(kamayan OBJECT ${_sources})

set(_pybind_sources grid/pybind/grid_py11.cpp kamayan/pybind/kamayan_py11.cpp
                    kamayan/pybind/kamayan_manager_py.cpp)

# maybe I should just glob for tests...
set(_test_sources
    grid/tests/test_grid.cpp
    grid/tests/test_scratch.cpp
    utils/tests/test_strings.cpp
    utils/tests/test_type_list.cpp
    utils/tests/test_type_list_array.cpp
    dispatcher/tests/test_dispatcher.cpp
    dispatcher/tests/test_options.cpp
    driver/tests/test_driver.cpp
    kamayan/tests/test_config.cpp
    kamayan/tests/test_runtime_parameters.cpp
    kamayan/tests/test_unit_collection.cpp
    kamayan/tests/test_unit_data.cpp
    physics/eos/tests/test_eos.cpp
    physics/hydro/tests/test_reconstruction.cpp)

target_link_libraries(kamayan PUBLIC parthenon singularity-eos::singularity-eos)
target_include_directories(kamayan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

nanobind_add_module(pyKamayan ${_pybind_sources})
target_link_libraries(pyKamayan PRIVATE kamayan)
set_target_properties(pyKamayan PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                           "${CMAKE_SOURCE_DIR}/kamayan")

set(_pyKamayan_modules
   Options
   Grid
)

function(NB_STUB module prefix)
   if(prefix)
      set(PREFIX "${prefix}.")
   endif()

   nanobind_add_stub("${module}_stub"
      MODULE "${PREFIX}${module}"
      OUTPUT "${CMAKE_SOURCE_DIR}/kamayan/${module}.pyi"
      PYTHON_PATH $<TARGET_FILE_DIR:pyKamayan>
      DEPENDS pyKamayan
   )
endfunction()

NB_STUB("pyKamayan" "")
foreach(module IN LISTS _pyKamayan_modules)
   NB_STUB("${module}" "pyKamayan")
endforeach()

option(kamayan_DEBUG_SCRATCH "Enable separate fields for each scratch variable")
if(kamayan_DEBUG_SCRATCH)
  add_compile_definitions(KAMAYAN_DEBUG_SCRATCH)
endif()

function(ADD_PROBLEM main problem_exe)
  add_executable(${problem_exe} ${main})
  target_link_libraries(${problem_exe} PRIVATE kamayan)
endfunction()

add_subdirectory(problems)

if(kamayan_ENABLE_TESTING)
  add_library(kamayan_test_lib OBJECT ${_test_sources})
  target_link_libraries(kamayan_test_lib PUBLIC GTest::gtest GTest::gmock
                                                kamayan)
  target_include_directories(kamayan_test_lib
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(kamayan_test tests/gtest_main.cpp)
  target_link_libraries(kamayan_test PRIVATE kamayan_test_lib kamayan)
  include(GoogleTest)
  gtest_discover_tests(kamayan_test PROPERTIES LABELS "unit")
endif()
