include(${parthenon_SOURCE_DIR}/cmake/TestSetup.cmake)

function(setup_test nproc dir arg extra_labels)
  separate_arguments(arg)
  list(APPEND labels "regression;mpi")
  list(APPEND labels "${extra_labels}")

  if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
    set(PARTHENON_KOKKOS_TEST_ARGS "--kokkos-map-device-id-by=mpi_rank")
    list(APPEND labels "cuda")
  endif()
  if(Kokkos_ENABLE_OPENMP)
    set(PARTHENON_KOKKOS_TEST_ARGS
        "${PARTHENON_KOKKOS_TEST_ARGS} --kokkos-threads=${NUM_OMP_THREADS_PER_RANK}"
    )
  endif()
  process_mpi_args(${nproc})
  add_test(
    NAME regression_mpi_test:${dir}
    COMMAND
      uv run ${parthenon_SOURCE_DIR}/tst/regression/run_test.py ${MPIARGS}
      ${arg} --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${PROJECT_BINARY_DIR}/tst/regression/outputs/${dir}_mpi"
      --kokkos_args=${PARTHENON_KOKKOS_TEST_ARGS})

  # When targeting CUDA we don't have a great way of controlling how tests get
  # mapped to GPUs, so just enforce serial execution
  if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
    set(TEST_PROPERTIES RUN_SERIAL ON)
  else()
    set(TEST_PROPERTIES PROCESSOR_AFFINITY ON PROCESSORS ${nproc})
  endif()
  set_tests_properties(regression_mpi_test:${dir} PROPERTIES LABELS "${labels}"
                                                             ${TEST_PROPERTIES})
  record_driver("${arg}")
endfunction()

setup_test(
  4
  "isentropic_vortex"
  "--driver ${PROJECT_BINARY_DIR}/isentropic_vortex --driver_input ${PROJECT_SOURCE_DIR}/src/problems/isentropic_vortex.in --num_steps 4"
  "convergence")

setup_test(
  4
  "reconstruction"
  "--driver ${PROJECT_BINARY_DIR}/isentropic_vortex --driver_input ${PROJECT_SOURCE_DIR}/src/problems/isentropic_vortex.in --num_steps 7"
  "reconstruction")

