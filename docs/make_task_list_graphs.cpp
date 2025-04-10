#include <iostream>
#include <memory>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/tests/test_grid.hpp"
#include "kamayan/kamayan.hpp"

int main(int argc, char *argv[]) {
  auto pman = kamayan::InitEnv(argc, argv);
  auto units = kamayan::ProcessUnits();
  auto driver = kamayan::InitPackages(pman, units);
  {
    // generate all the graphs for the task collection/lists
    // for the driver as well as all the units
    auto pkg = std::make_shared<kamayan::StateDescriptor>("Test Package");
    auto block_list = kamayan::MakeTestBlockList(pkg, 1, 8, 3);
    auto tc = driver.MakeTaskCollection(block_list, 1);
    std::cout << tc;
  }
  pman->ParthenonFinalize();
  return 0;
}
