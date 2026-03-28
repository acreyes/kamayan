#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int results;
  {
    ::testing::InitGoogleTest(&argc, argv);
    results = RUN_ALL_TESTS();
  }
  Kokkos::finalize();
  return results;
}
