set(UNITS "eos" "grid" "physics" "hydro")
set(generated_ ${CMAKE_CURRENT_SOURCE_DIR}/assets/generated)
set(UNIT_DOCS "")

if(kamayan_BUILD_DOCS)
  if(NOT kamayan_ENABLE_TESTING)
    message(
      FATAL_ERROR
        "kamayan_ENABLE_TESTING must be set to on in order to build docs.")
  else()
    add_executable(kamayan_docs kamayan_docs.cpp)
    target_link_libraries(kamayan_docs PRIVATE kamayan kamayan_test_lib)

    find_program(GRAPHVIZ_DOT NAMES dot REQUIRED)
    set(parth_args -i ${CMAKE_SOURCE_DIR}/docs/kamayan_docs.in)
    add_custom_command(
      OUTPUT ${generated_}/driver_tasks.png
      COMMAND kamayan_docs ${parth_args} --tasks | ${GRAPHVIZ_DOT} -Tpng >
              ${generated_}/driver_tasks.png
      DEPENDS kamayan_docs)

    foreach(unit ${UNITS})
      set(cur_unit ${generated_}/${unit}.md)
      message(${unit})
      list(APPEND UNIT_DOCS ${cur_unit})
      add_custom_command(
        OUTPUT ${cur_unit}
        COMMAND kamayan_docs ${parth_args} --runtime_parameters --unit ${unit}
                --out ${cur_unit}
        DEPENDS kamayan_docs)
    endforeach()

    add_custom_target(
      docs
      COMMAND uv run mkdocs build
      DEPENDS ${generated_}/driver_tasks.png ${UNIT_DOCS}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  endif()
endif()
