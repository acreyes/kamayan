#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

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
  sd.def("__init__", [](StateDescriptor *self, std::string &name) {
    new (self) StateDescriptor(name);
  });
  sd.def("AddField", [](StateDescriptor &self, const std::string name,
                        const Metadata &m) { self.AddField(name, m); });
  sd.def("AddParam", [](StateDescriptor &self, const std::string &key,
                        nanobind::object obj) { self.AddParam(key, obj); });
  sd.def("GetParam", [](StateDescriptor &self, const std::string &key) {
    return self.Param<nanobind::object>(key);
  });
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
  pinput.def("GetReal", &parthenon::ParameterInput::GetReal);
  pinput.def("GetInt", &parthenon::ParameterInput::GetInteger);
  pinput.def("GetStr", [](parthenon::ParameterInput &self, const std::string &block,
                          const std::string &key) { return self.GetString(block, key); });
  pinput.def("GetBool", &parthenon::ParameterInput::GetBoolean);
  pinput.def("dump",
             [](parthenon::ParameterInput &self) { self.ParameterDump(std::cout); });

  nanobind::enum_<parthenon::ParthenonStatus> parthenon_status(m, "ParthenonStatus",
                                                               "enum.Enum");
  parthenon_status.value("ok", parthenon::ParthenonStatus::ok);
  parthenon_status.value("complete", parthenon::ParthenonStatus::complete);
  parthenon_status.value("error", parthenon::ParthenonStatus::error);

  m.def("InitEnv", [](std::vector<std::string> args) {
    // we need to initialize the parthenon/kamayan/kokkos environment by forwarding
    // the command line arguments. Ideally we should generate our own parameter
    // input by parsing all of our KamayanUnits' Setup callbacks
    // something like {"program_name", "-i", "dummy.in", ...}
    int argc = args.size();
    auto argv = std::make_unique<char *[]>(argc + 1);  // +1 for nullptr terminator

    for (int i = 0; i < argc; ++i) {
      auto arg = args[i];
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

void unit_data_collection(nanobind::module_ &m) {
  nanobind::class_<UnitData::UnitParm> unit_parm(m, "UnitParm");
  unit_parm.def_prop_ro("key", &UnitData::UnitParm::Key);
  unit_parm.def_prop_ro("value", &UnitData::UnitParm::Get);
  unit_parm.def("Update", &UnitData::UnitParm::Update);

  nanobind::class_<UnitData> unit_data(m, "UnitData");
  unit_data.def(nanobind::init<const std::string &>());
  unit_data.def("AddReal", [](UnitData &self, const std::string &key, const Real &val,
                              const std::string &docstring) {
    self.AddParm<Real>(key, val, docstring);
  });
  unit_data.def("AddBool", [](UnitData &self, const std::string &key, const bool &val,
                              const std::string &docstring) {
    self.AddParm<bool>(key, val, docstring);
  });
  unit_data.def("AddInt", [](UnitData &self, const std::string &key, const int &val,
                             const std::string &docstring) {
    self.AddParm<int>(key, val, docstring);
  });
  unit_data.def("AddStr", [](UnitData &self, const std::string &key,
                             const std::string &val, const std::string &docstring) {
    self.AddParm<std::string>(key, val, docstring);
  });
  unit_data.def("AddParm",
                [](UnitData &self, const std::string &key,
                   const UnitData::DataType &value, const std::string &docstring) {
                  if (auto v = std::get_if<Real>(&value); v) {
                    self.AddParm<Real>(key, *v, docstring);
                  } else if (auto v = std::get_if<int>(&value); v) {
                    self.AddParm<int>(key, *v, docstring);
                  } else if (auto v = std::get_if<bool>(&value); v) {
                    self.AddParm<bool>(key, *v, docstring);
                  } else if (auto v = std::get_if<std::string>(&value); v) {
                    self.AddParm<std::string>(key, *v, docstring);
                  }
                });
  unit_data.def("UpdateParm", &UnitData::UpdateParm);
  unit_data.def_prop_ro("Block", &UnitData::Block);
  unit_data.def(
      "Get", [](UnitData &self, const std::string &key) { return self.Get(key); },
      nanobind::rv_policy::reference_internal);
  unit_data.def(
      "__getitem__", [](UnitData &self, const std::string &key) { return self.Get(key); },
      nanobind::rv_policy::reference_internal);
  unit_data.def("__setitem__",
                [](UnitData &self, const std::string &key,
                   const UnitData::DataType &value) { self.UpdateParm(key, value); });
  unit_data.def("__iter__", [](UnitData &self) {
    auto &parameters = self.Get();
    return nanobind::make_iterator(nanobind::type<UnitData>(), "UnitDataIterator",
                                   parameters.begin(), parameters.end());
  });

  nanobind::class_<UnitDataCollection> udc(m, "UnitDataCollection");
  udc.def("__init__", [](UnitDataCollection &self) {});
  udc.def("Package", &UnitDataCollection::Package);
  udc.def("Configuration", &UnitDataCollection::Configuration);
  udc.def("RuntimeParameters", &UnitDataCollection::RuntimeParameters);
  udc.def(
      "AddData",
      [](UnitDataCollection &self, const UnitData &data) { return self.AddData(data); },
      nanobind::rv_policy::reference_internal);
  udc.def(
      "AddData",
      [](UnitDataCollection &self, const std::string &block) {
        return &self.AddData(block);
      },
      nanobind::rv_policy::reference_internal);
  udc.def(
      "Data",
      [](UnitDataCollection &self, const std::string &block) {
        return &self.Data(block);
      },
      nanobind::rv_policy::reference_internal);
  udc.def("__iter__", [](UnitDataCollection &self) {
    auto &ud = self.GetUnitData();
    return nanobind::make_iterator(nanobind::type<UnitDataCollection>(),
                                   "UnitDataCollectionIterator", ud.begin(), ud.end());
  });
  udc.def(
      "__getitem__",
      [](UnitDataCollection &self, const std::string &key) { return self.Data(key); },
      nanobind::rv_policy::reference_internal);
}
}  // namespace kamayan
