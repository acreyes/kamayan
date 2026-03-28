#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

int main(int argc, char *argv[]) {
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(&argc, &argv);
  }

  Kokkos::initialize(argc, argv);
  int results;
  {
    ::testing::InitGoogleTest(&argc, argv);
    results = RUN_ALL_TESTS();
  }
  Kokkos::finalize();

  if (!mpi_initialized) {
    MPI_Finalize();
  }
  return results;
}
