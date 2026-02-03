#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/typing.h>

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
#include "kamayan/pybind/kamayan_bindings.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "parameter_input.hpp"
#include "parthenon_manager.hpp"

namespace kamayan {
void state_descrptor(nanobind::module_ &m) {
  nanobind::class_<Metadata> metadata(m, "Metadata");
  metadata.def("__init__", [](Metadata *self, std::vector<int> shape) {
    new (self) Metadata(Metadata({}, shape));
  });

  // for python bindings we only really need exposore to:
  // * some callbacks, these are mostly TODO(acreyes): until they are needed for something
  // * registering new fields
  // * adding/getting/updating params, TODO(acreyes): I'm not sure I have a use case for
  //                                                  this yet
  //  * this point is a bit complicated since the params can be any type
  //  * maybe just let this be pybind11::objects, int, Real, string & bool?
  nanobind::class_<StateDescriptor> sd(m, "StateDescriptor");
  sd.attr("T") = nanobind::type_var("T");
  sd.def("__init__", [](StateDescriptor *self, std::string &name) {
    new (self) StateDescriptor(name);
  });
  sd.def("AddField", [](StateDescriptor &self, const std::string name,
                        const Metadata &m) { self.AddField(name, m); });
  sd.def("AddParam", [](StateDescriptor &self, const std::string &key,
                        nanobind::object obj) { self.AddParam(key, obj); });

  sd.def(
      "GetParam",
      [](StateDescriptor &self, nanobind::object t, const std::string &key) {
        auto parm = self.Param<nanobind::object>(key);
        if (!nanobind::isinstance(nanobind::cast(parm), t)) {
          throw nanobind::type_error(
              "[StateDescriptor::GetParam] parameter is not of provided type.");
        }
        return parm;
      },
      nanobind::sig("def GetParam(self, t: typing.Type[T], key: str) -> T"));
}

void driver_py(nanobind::module_ &m) {
  nanobind::enum_<TaskStatus> task_status(m, "TaskStatus", "enum.Enum");
  task_status.value("fail", TaskStatus::fail);
  task_status.value("complete", TaskStatus::complete);
  task_status.value("incomplete", TaskStatus::incomplete);
  task_status.value("iterate", TaskStatus::iterate);

  // enum class DriverStatus { complete, timeout, failed };
  nanobind::enum_<parthenon::DriverStatus> driver_status(m, "DriverStatus", "enum.Enum");
  driver_status.value("complete", parthenon::DriverStatus::complete);
  driver_status.value("timeout", parthenon::DriverStatus::timeout);
  driver_status.value("failed", parthenon::DriverStatus::failed);

  nanobind::class_<KamayanDriver> driver(m, "KamayanDriver");
  driver.def("Execute", &KamayanDriver::Execute);
}

void parthenon_manager(nanobind::module_ &m) {
  // TODO(acreyes): I'm not sure what exactly we would want to actually call from
  // a pman, besides just wanting to be able to pass it around during initialization
  nanobind::class_<parthenon::ParthenonManager> pman(m, "ParthenonManager");
  pman.def_prop_ro(
      "pinput", [](parthenon::ParthenonManager &self) { return self.pinput.get(); },
      nanobind::rv_policy::reference);
  pman.def("ParthenonFinalize", &parthenon::ParthenonManager::ParthenonFinalize);

  nanobind::class_<parthenon::ParameterInput> pinput(m, "ParameterInput");
  pinput.def(nanobind::init<>());  // Default constructor
  pinput.def("GetReal", &parthenon::ParameterInput::GetReal);
  pinput.def("GetInt", &parthenon::ParameterInput::GetInteger);
  pinput.def("GetStr", [](parthenon::ParameterInput &self, const std::string &block,
                          const std::string &key) { return self.GetString(block, key); });
  pinput.def("GetBool", &parthenon::ParameterInput::GetBoolean);
  pinput.def("dump",
             [](parthenon::ParameterInput &self) { self.ParameterDump(std::cout); });

  auto pin_get = [](parthenon::ParameterInput &self, const std::string &key) {
    // Parse key in format "block/key"
    size_t slash_pos = key.find('/');
    if (slash_pos == std::string::npos) {
      throw std::invalid_argument("Key must be in format 'block/key'");
    }
    std::string block = key.substr(0, slash_pos);
    std::string param = key.substr(slash_pos + 1);

    // Try to determine the type by checking if parameter exists and
    // trying different types until one works
    if (self.DoesParameterExist(block, param)) {
      // Try Real first as it's most common
      try {
        Real val = self.GetReal(block, param);
        return nanobind::cast(val);
      } catch (...) {
        // Try int
        try {
          int val = self.GetInteger(block, param);
          return nanobind::cast(val);
        } catch (...) {
          // Try bool
          try {
            bool val = self.GetBoolean(block, param);
            return nanobind::cast(val);
          } catch (...) {
            // Fall back to string
            std::string val = self.GetString(block, param);
            return nanobind::cast(val);
          }
        }
      }
    } else {
      throw std::out_of_range("Parameter '" + key + "' not found");
    }
  };
  // Dictionary interface methods
  pinput.def("get", pin_get, nanobind::rv_policy::copy);

  pinput.def("__getitem__", pin_get, nanobind::rv_policy::copy);

  pinput.def("__contains__",
             [](parthenon::ParameterInput &self, const std::string &key) -> bool {
               // Parse key in format "block/key"
               size_t slash_pos = key.find('/');
               if (slash_pos == std::string::npos) {
                 return false;
               }
               std::string block = key.substr(0, slash_pos);
               std::string param = key.substr(slash_pos + 1);
               return self.DoesParameterExist(block, param);
             });

  pinput.def(
      "get",
      [](parthenon::ParameterInput &self, const std::string &key,
         nanobind::object default_val = nanobind::none()) {
        // Parse key in format "block/key"
        size_t slash_pos = key.find('/');
        if (slash_pos == std::string::npos) {
          throw std::invalid_argument("Key must be in format 'block/key'");
        }
        std::string block = key.substr(0, slash_pos);
        std::string param = key.substr(slash_pos + 1);

        if (self.DoesParameterExist(block, param)) {
          // Try Real first as it's most common
          try {
            Real val = self.GetReal(block, param);
            return nanobind::cast(val);
          } catch (...) {
            // Try int
            try {
              int val = self.GetInteger(block, param);
              return nanobind::cast(val);
            } catch (...) {
              // Try bool
              try {
                bool val = self.GetBoolean(block, param);
                return nanobind::cast(val);
              } catch (...) {
                // Fall back to string
                std::string val = self.GetString(block, param);
                return nanobind::cast(val);
              }
            }
          }
        } else {
          if (default_val.is_none()) {
            throw std::out_of_range("Parameter '" + key +
                                    "' not found and no default provided");
          }
          return default_val;
        }
      },
      nanobind::rv_policy::copy);

  nanobind::enum_<parthenon::ParthenonStatus> parthenon_status(m, "ParthenonStatus",
                                                               "enum.Enum");
  parthenon_status.value("ok", parthenon::ParthenonStatus::ok);
  parthenon_status.value("complete", parthenon::ParthenonStatus::complete);
  parthenon_status.value("error", parthenon::ParthenonStatus::error);

  m.def("InitEnv", [](std::vector<std::string> args) {
    int argc = args.size();
    auto argv = std::make_unique<char *[]>(argc + 1);

    for (int i = 0; i < argc; ++i) {
      auto arg = args[i];
      argv[i] = new char[arg.size() + 1];
      size_t len = arg.size() + 1;
      std::snprintf(argv[i], len, "%s", arg.c_str());
    }

    argv[argc] = nullptr;
    return InitEnv(argc, argv.get());
  });

  m.def("InitPackages",
        [](std::shared_ptr<parthenon::ParthenonManager> pman,
           std::shared_ptr<UnitCollection> units) { return InitPackages(pman, units); });
  m.def("ProcessUnits", &ProcessUnits);
  m.def("Finalize", &Finalize,
        "Properly cleanup Parthenon resources and break reference cycles");

  driver_py(m);
}

}  // namespace kamayan
