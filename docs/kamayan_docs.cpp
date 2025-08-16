#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/tests/test_grid.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "utils/error_checking.hpp"

struct ArgParse {
  ArgParse(int argc, char *argv[]) : taskgraph(false), rps(false) {
    unprocessed_args.push_back(argv[0]);
    // pop any args that we are going to use, so we can forward the rest down to parthenon
    for (int i = 1; i < argc; i++) {
      if (*argv[i] == '-' && *(argv[i] + 1) == '-' && *(argv[i] + 2) != '\0') {
        std::string opt = argv[i];
        if (opt == "--tasks") {
          taskgraph = true;
          continue;
        } else if (opt == "--runtime_parameters") {
          rps = true;
          continue;
        } else if (opt == "--unit") {
          unit_name = argv[++i];
          continue;
        } else if (opt == "--out") {
          out_file = argv[++i];
          continue;
        }
      }
      unprocessed_args.push_back(argv[i]);
    }
  }

  std::vector<std::string> unprocessed_args;
  bool taskgraph, rps;
  std::string out_file, unit_name;
};

int main(int argc_in, char *argv_in[]) {
  auto args = ArgParse(argc_in, argv_in);
  int argc = args.unprocessed_args.size();
  char **argv = new char *[argc + 1];
  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[args.unprocessed_args[i].size() + 1];
    strcpy(argv[i], args.unprocessed_args[i].c_str());
  }

  auto pman = kamayan::InitEnv(argc, argv);
  auto units = kamayan::ProcessUnits();
  auto driver = kamayan::InitPackages(pman, units);
  if (args.taskgraph) {
    // generate all the graphs for the task collection/lists
    // for the driver as well as all the units
    auto pkg = std::make_shared<kamayan::StateDescriptor>("Test Package");
    auto block_list = kamayan::MakeTestBlockList(pkg, 1, 8, 3);
    auto tc = driver.MakeTaskCollection(block_list, 1);
    std::cout << tc;
  }
  if (args.rps) {
    PARTHENON_REQUIRE_THROWS(args.unit_name.size() > 0 && args.out_file.size() > 0,
                             "[Error] Runtime Parameters requires --out filename (" +
                                 std::to_string(args.out_file.size() > 0) +
                                 ") and --unit name(" +
                                 std::to_string(args.unit_name.size() > 0) + ").")

    auto write_rps = [&](kamayan::KamayanUnit *unit) {
      auto ss = kamayan::RuntimeParameterDocs(unit, pman->pinput.get());
      std::ofstream out_file(args.out_file);
      if (out_file.is_open()) {
        out_file << ss.str();
      } else {
        PARTHENON_THROW("Couldn't open file for write: " + args.out_file)
      }
    };
    if (args.unit_name == "driver") {
      // there is some weird behavior when the driver does it ssetup
      // with everyone else, so we only include it in the unit here
      // for the sake of generating runtime parameter docs
      auto driver_unit = kamayan::driver::ProcessUnit(true);
      driver_unit->SetupParams(driver_unit->unit_data_collection);
      write_rps(driver_unit.get());
    } else {
      for (const auto &unit : units) {
        if (unit.first == args.unit_name) {
          write_rps(unit.second.get());
        }
      }
    }
  }
  pman->ParthenonFinalize();
  return 0;
}
