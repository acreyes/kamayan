#include "kamayan/pybind/kamayan_bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <string>

#include "dispatcher/pybind/enum_options.hpp"
#include "grid/pybind/grid_bindings.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "nanobind/make_iterator.h"

// macro for defining getter/setter methods for std::function callbacks in both
// StateDescriptor & KamayanUnit
#define CALLBACK(pycls, cls, callback)                                                   \
  pycls.def("set_" #callback,                                                            \
            [](cls &self, decltype(cls::callback) fn) { self.callback = fn; });          \
  pycls.def("get_" #callback, [](cls &self) { return self.callback; });

namespace kamayan {

using RP = runtime_parameters::RuntimeParameters;
template <typename T>
void Add(RP &rps, const std::string &block, const std::string &key, const T &value,
         const std::string &docstring) {
  rps.Add(block, key, value, docstring);
}

template <typename T>
T Get(RP &rps, const std::string &block, const std::string &key) {
  return rps.Get<T>(block, key);
}

void RuntimeParameter_module(nanobind::module_ &m) {
  nanobind::class_<RP> runtime_parameters(m, "RuntimeParameters");
  runtime_parameters.def("AddBool",
                         [](RP &self, const std::string &block, const std::string &key,
                            const bool &value, const std::string &docstring) {
                           Add<bool>(self, block, key, value, docstring);
                         });
  runtime_parameters.def("AddStr",
                         [](RP &self, const std::string &block, const std::string &key,
                            const std::string &value, const std::string &docstring) {
                           Add<std::string>(self, block, key, value, docstring);
                         });
  runtime_parameters.def("AddReal",
                         [](RP &self, const std::string &block, const std::string &key,
                            const Real &value, const std::string &docstring) {
                           Add<Real>(self, block, key, value, docstring);
                         });
  runtime_parameters.def(
      "AddInt",
      [](RP &self, const std::string &block, const std::string &key, const int &value,
         const std::string &docstring) { Add<int>(self, block, key, value, docstring); });
  runtime_parameters.def("GetBool",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return Get<bool>(self, block, key);
                         });
  runtime_parameters.def("GetStr",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return Get<std::string>(self, block, key);
                         });
  runtime_parameters.def("GetReal",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return Get<Real>(self, block, key);
                         });
  runtime_parameters.def("GetInt",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return Get<int>(self, block, key);
                         });
}

NB_MODULE(pyKamayan, m) {
  m.doc() = "Main entrypoint for kamayan python bindings.";

  auto opts = m.def_submodule("Options", "Polymorphic Parameter options.");
  for (const auto &func : pybind::PybindOptions::pybind_options) {
    func(opts);
  }

  auto config = nanobind::class_<Config>(m, "Config");
  config.doc() = "Bindings to global kamayan Config type.";
  for (const auto &func : pybind::PybindOptions::pybind_config) {
    func(config);
  }

  RuntimeParameter_module(m);

  nanobind::class_<KamayanUnit, StateDescriptor> kamayan_unit(m, "KamayanUnit");
  kamayan_unit.def("__init__", [](KamayanUnit *self, std::string name) {
    new (self) KamayanUnit(name);
  });
  kamayan_unit.def_prop_ro("Name", &KamayanUnit::Name);
  kamayan_unit.def("Data", &KamayanUnit::Data, nanobind::rv_policy::reference);
  kamayan_unit.def("AddData", &KamayanUnit::AddData, nanobind::rv_policy::reference);
  kamayan_unit.def("HasData", &KamayanUnit::HasData);
  kamayan_unit.def("Configuration", &KamayanUnit::Configuration,
                   nanobind::rv_policy::reference);
  kamayan_unit.def("RuntimeParameters", &KamayanUnit::RuntimeParameters,
                   nanobind::rv_policy::reference);
  kamayan_unit.def("GetUnit", &KamayanUnit::GetUnit, nanobind::rv_policy::reference);
  kamayan_unit.def("GetUnitPtr", &KamayanUnit::GetUnitPtr);
  CALLBACK(kamayan_unit, KamayanUnit, SetupParams)
  CALLBACK(kamayan_unit, KamayanUnit, InitializeData)
  CALLBACK(kamayan_unit, KamayanUnit, ProblemGeneratorMeshBlock)
  CALLBACK(kamayan_unit, KamayanUnit, PrepareConserved)
  CALLBACK(kamayan_unit, KamayanUnit, PreparePrimitive)
  CALLBACK(kamayan_unit, KamayanUnit, AddFluxTasks)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksOneStep)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksSplit)

  nanobind::class_<UnitCollection> unit_collection(m, "UnitCollection");
  unit_collection.def("Get", &UnitCollection::Get, nanobind::rv_policy::reference);
  unit_collection.def("Add", &UnitCollection::Add);
  unit_collection.def("__contains__", [](UnitCollection &self, const std::string &key) {
    return self.GetMap()->count(key) > 0;
  });
  unit_collection.def("__iter__", [](UnitCollection &self) {
    auto &units = *self.GetMap();
    return nanobind::make_iterator(nanobind::type<UnitCollection>(),
                                   "UnitCollectionIterator", units.begin(), units.end());
  });

  state_descrptor(m);
  kamayan_unit(m);
  parthenon_manager(m);

  auto grid = m.def_submodule("Grid", "Bindings to grid structures.");
  grid_module(grid);
}
}  // namespace kamayan

// std::string name_;
