#include "kamayan/kamayan.hpp"
#include "kamayan/unit.hpp"

int main(int argc, char *argv[]) {
  // initialize the environment
  // * mpi
  // * kokkos
  // * parthenon
  auto pman = kamayan::InitEnv(argc, argv);

  // put together all the kamayan units we want
  auto units = kamayan::ProcessUnits();

  // get the driver and we're ready to go!
  auto driver = kamayan::InitPackages(pman, units);
  auto driver_status = driver.Execute();

  pman->ParthenonFinalize();
}
