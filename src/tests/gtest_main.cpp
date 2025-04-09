#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// we need to have our own main so that we can initialize kokkos
// te be used in some tests
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int results;
  {
    ::testing::InitGoogleTest(&argc, argv);
    results = RUN_ALL_TESTS();
  }
  return results;
}
