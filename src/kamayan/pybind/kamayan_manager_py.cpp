#include <pybind11/native_enum.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "driver/driver.hpp"
#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/pybind/kamayan_py11.hpp"
#include "kamayan/unit.hpp"
#include "parameter_input.hpp"
#include "parthenon_manager.hpp"

namespace kamayan {
void state_descrptor(pybind11::module_ &m) {
  pybind11::classh<Metadata> metadata(m, "Metadata");
  metadata.def(pybind11::init(
      [](std::vector<int> shape) { return Metadata(Metadata({}, shape)); }));

  // for python bindings we only really need exposore to:
  // * some callbacks, these are mostly TODO(acreyes): until they are needed for something
  // * registering new fields
  // * adding/getting/updating params, TODO(acreyes): I'm not sure I have a use case for
  //                                                  this yet
  //  * this point is a bit complicated since the params can be any type
  //  * maybe just let this be pybind11::objects, int, Real, string & bool?
  pybind11::classh<StateDescriptor> sd(m, "StateDescriptor");
  sd.def(pybind11::init<std::string>());
  sd.def("AddField", [](StateDescriptor &self, const std::string name,
                        const Metadata &m) { self.AddField(name, m); });
  sd.def("AddParam", [](StateDescriptor &self, const std::string &key,
                        pybind11::object obj) { self.AddParam(key, obj); });
  sd.def("GetParam", [](StateDescriptor &self, const std::string &key) {
    return self.Param<pybind11::object>(key);
  });
}

void driver_py(pybind11::module_ &m) {
  pybind11::native_enum<TaskStatus> task_status(m, "TaskStatus", "enum.Enum");
  task_status.value("fail", TaskStatus::fail);
  task_status.value("complete", TaskStatus::complete);
  task_status.value("incomplete", TaskStatus::incomplete);
  task_status.value("iterate", TaskStatus::iterate);
  task_status.finalize();

  // enum class DriverStatus { complete, timeout, failed };
  pybind11::native_enum<parthenon::DriverStatus> driver_status(m, "DriverStatus",
                                                               "enum.Enum");
  driver_status.value("complete", parthenon::DriverStatus::complete);
  driver_status.value("timeout", parthenon::DriverStatus::timeout);
  driver_status.value("failed", parthenon::DriverStatus::failed);
  driver_status.finalize();

  pybind11::classh<KamayanDriver> driver(m, "KamayanDriver");
  driver.def("Execute", &KamayanDriver::Execute);
}

void parthenon_manager(pybind11::module_ &m) {
  // TODO(acreyes): I'm not sure what exactly we would want to actually call from
  // a pman, besides just wanting to be able to pass it around during initialization
  pybind11::class_<parthenon::ParthenonManager,
                   std::shared_ptr<parthenon::ParthenonManager>>
      pman(m, "ParthenonManager");
  pman.def_property_readonly(
      "pinput", [](parthenon::ParthenonManager &self) { return self.pinput.get(); },
      pybind11::return_value_policy::reference);
  pman.def("ParthenonFinalize", &parthenon::ParthenonManager::ParthenonFinalize);

  pybind11::classh<parthenon::ParameterInput> pinput(m, "ParameterInput");
  pinput.def("GetReal", &parthenon::ParameterInput::GetReal);
  pinput.def("GetInt", &parthenon::ParameterInput::GetInteger);
  pinput.def("GetStr", [](parthenon::ParameterInput &self, const std::string &block,
                          const std::string &key) { return self.GetString(block, key); });
  pinput.def("GetBool", &parthenon::ParameterInput::GetBoolean);
  pinput.def("dump",
             [](parthenon::ParameterInput &self) { self.ParameterDump(std::cout); });

  pybind11::native_enum<parthenon::ParthenonStatus> parthenon_status(m, "ParthenonStatus",
                                                                     "enum.Enum");
  parthenon_status.value("ok", parthenon::ParthenonStatus::ok);
  parthenon_status.value("complete", parthenon::ParthenonStatus::complete);
  parthenon_status.value("error", parthenon::ParthenonStatus::error);
  parthenon_status.finalize();

  m.def("InitEnv", [](pybind11::list args) {
    // we need to initialize the parthenon/kamayan/kokkos environment by forwarding
    // the command line arguments. Ideally we should generate our own parameter
    // input by parsing all of our KamayanUnits' Setup callbacks
    // something like {"program_name", "-i", "dummy.in", ...}
    int argc = args.size();
    auto argv = std::make_unique<char *[]>(argc + 1);  // +1 for nullptr terminator

    for (int i = 0; i < argc; ++i) {
      // Allocate and copy string into a new char array
      auto arg = args[i].cast<std::string>();
      argv[i] = new char[arg.size() + 1];
      size_t len = arg.size() + 1;
      std::snprintf(argv[i], len, "%s", arg.c_str());
    }

    argv[argc] = nullptr;  // argv must be null-terminated
    return InitEnv(argc, argv.get());
  });

  m.def("InitPackages", &InitPackages);
  m.def("ProcessUnits", &ProcessUnits);

  driver_py(m);
}
}  // namespace kamayan
