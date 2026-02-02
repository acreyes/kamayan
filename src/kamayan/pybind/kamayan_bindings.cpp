#include "kamayan/pybind/kamayan_bindings.hpp"

#include <Python.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <string>
#include <variant>

#include "dispatcher/pybind/enum_options.hpp"
#include "grid/pybind/grid_bindings.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "nanobind/make_iterator.h"

// include here all the headers that define POLYMORPHIC_PARMs so
// that they can be included in the python bindings
#include "grid/grid_refinement.hpp"
#include "physics/eos/eos_types.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/physics_types.hpp"

// macro for defining getter/setter methods for std::function callbacks in both
// StateDescriptor & KamayanUnit
#define CALLBACK(pycls, cls, callback)                                                   \
  pycls.def("set_" #callback,                                                            \
            [](cls &self, decltype(cls::callback) fn) { self.callback = fn; });          \
  pycls.def("get_" #callback, [](cls &self) { return self.callback; });

// Garbage collection support for KamayanUnit
int kamayan_unit_tp_traverse(PyObject *self, visitproc visit, void *arg) {
  // Traverse the implicit dependency of an object on its associated type object
  Py_VISIT(Py_TYPE(self));

  // The tp_traverse method may be called after __new__ but before or during
  // __init__, before the C++ constructor has been completed. We must not
  // inspect the C++ state if the constructor has not yet completed.
  if (!nanobind::inst_ready(self)) {
    return 0;
  }

  // Get the C++ object associated with 'self'
  kamayan::KamayanUnit *unit = nanobind::inst_ptr<kamayan::KamayanUnit>(self);

  // Traverse all Python function callbacks stored in std::function objects
  // nb::find() returns the Python object if the std::function wraps a Python callable
  nanobind::handle setup_params = nanobind::find(unit->SetupParams);
  Py_VISIT(setup_params.ptr());

  nanobind::handle init_data = nanobind::find(unit->InitializeData);
  Py_VISIT(init_data.ptr());

  nanobind::handle pgen = nanobind::find(unit->ProblemGeneratorMeshBlock);
  Py_VISIT(pgen.ptr());

  nanobind::handle prep_conserved = nanobind::find(unit->PrepareConserved);
  Py_VISIT(prep_conserved.ptr());

  nanobind::handle prep_primitive = nanobind::find(unit->PreparePrimitive);
  Py_VISIT(prep_primitive.ptr());

  nanobind::handle flux_tasks = nanobind::find(unit->AddFluxTasks);
  Py_VISIT(flux_tasks.ptr());

  nanobind::handle tasks_one_step = nanobind::find(unit->AddTasksOneStep);
  Py_VISIT(tasks_one_step.ptr());

  nanobind::handle tasks_split = nanobind::find(unit->AddTasksSplit);
  Py_VISIT(tasks_split.ptr());

  // Traverse shared_ptr members if they have Python wrappers
  nanobind::handle config = nanobind::find(unit->Configuration());
  Py_VISIT(config.ptr());

  nanobind::handle runtime_params = nanobind::find(unit->RuntimeParameters());
  Py_VISIT(runtime_params.ptr());

  // Traverse the units_ member which creates a cycle back to UnitCollection
  nanobind::handle units = nanobind::find(unit->units_);
  Py_VISIT(units.ptr());

  return 0;
}

int kamayan_unit_tp_clear(PyObject *self) {
  // Get the C++ object associated with 'self'
  kamayan::KamayanUnit *unit = nanobind::inst_ptr<kamayan::KamayanUnit>(self);

  // Debug: Uncomment to see when tp_clear is called
  // fprintf(stderr, "[DEBUG] kamayan_unit_tp_clear called for unit: %s\n",
  // unit->Name().c_str());

  // Break reference cycles by clearing all Python callbacks
  unit->SetupParams = nullptr;
  unit->InitializeData = nullptr;
  unit->ProblemGeneratorMeshBlock = nullptr;
  unit->PrepareConserved = nullptr;
  unit->PreparePrimitive = nullptr;
  unit->AddFluxTasks = nullptr;
  unit->AddTasksOneStep = nullptr;
  unit->AddTasksSplit = nullptr;

  // Clear the units_ shared_ptr to break the cycle with UnitCollection
  // This is safe because we're using const_cast on the const shared_ptr
  const_cast<std::shared_ptr<const kamayan::UnitCollection> &>(unit->units_).reset();

  return 0;
}

// Garbage collection support for UnitCollection
int unit_collection_tp_traverse(PyObject *self, visitproc visit, void *arg) {
  // Traverse the type object
  Py_VISIT(Py_TYPE(self));

  // Check if instance is ready
  if (!nanobind::inst_ready(self)) {
    return 0;
  }

  // Get the C++ object
  kamayan::UnitCollection *uc = nanobind::inst_ptr<kamayan::UnitCollection>(self);

  // Traverse all KamayanUnit shared_ptrs in the units map
  auto map = uc->GetMap();
  for (const auto &pair : *map) {
    nanobind::handle unit_handle = nanobind::find(pair.second);
    Py_VISIT(unit_handle.ptr());
  }

  return 0;
}

int unit_collection_tp_clear(PyObject *self) {
  // Get the C++ object
  kamayan::UnitCollection *uc = nanobind::inst_ptr<kamayan::UnitCollection>(self);

  // Access the private units map via friend function
  // Explicitly reset all shared_ptrs before clearing the map
  auto &units = uc->units;
  for (auto &pair : units) {
    pair.second.reset();
  }
  units.clear();

  return 0;
}

namespace kamayan {

// Type slots for KamayanUnit
PyType_Slot kamayan_unit_slots[] = {{Py_tp_traverse, (void *)kamayan_unit_tp_traverse},
                                    {Py_tp_clear, (void *)kamayan_unit_tp_clear},
                                    {0, 0}};

// Type slots for UnitCollection
PyType_Slot unit_collection_slots[] = {
    {Py_tp_traverse, (void *)unit_collection_tp_traverse},
    {Py_tp_clear, (void *)unit_collection_tp_clear},
    {0, 0}};

using RP = runtime_parameters::RuntimeParameters;
void RuntimeParameter_module(nanobind::module_ &m) {
  using Parameter = std::variant<bool, Real, int, std::string>;
  nanobind::class_<RP> runtime_parameters(m, "RuntimeParameters");
  // Factory functions for RuntimeParameters
  m.def(
      "make_runtime_parameters",
      [](parthenon::ParameterInput *pin) {
        auto rp = new RP(pin);
        return rp;
      },
      nanobind::rv_policy::take_ownership, nanobind::arg("pin"),
      nanobind::keep_alive<0, 1>(),  // Keep ParameterInput (arg 1) alive as long as
                                     // return value (arg 0) is alive
      "Create RuntimeParameters from ParameterInput");

  m.def(
      "make_runtime_parameters",
      []() {
        auto rp = new RP();
        return rp;
      },
      nanobind::rv_policy::take_ownership, "Create empty RuntimeParameters");

  // Add constructors
  runtime_parameters.def(nanobind::init<>());  // Default constructor
  runtime_parameters.def(
      nanobind::init<parthenon::ParameterInput *>(),
      nanobind::keep_alive<1, 2>());  // Keep ParameterInput (arg 2) alive as long as self
                                      // (arg 1) is alive
  runtime_parameters.def_prop_ro("pinput_ref", [](RP &self) { return self.GetPin(); });
  runtime_parameters.def("set", [](RP &self, const std::string &block,
                                   const std::string &key, const Parameter &value) {
    if (auto v = std::get_if<Real>(&value)) {
      self.Set(block, key, *v);
    } else if (auto v = std::get_if<int>(&value)) {
      self.Set(block, key, *v);
    } else if (auto v = std::get_if<bool>(&value)) {
      self.Set(block, key, *v);
    } else if (auto v = std::get_if<std::string>(&value)) {
      self.Set(block, key, *v);
    }
  });
  runtime_parameters.def("add",
                         [](RP &self, const std::string &block, const std::string &key,
                            const Parameter &value, const std::string &docstring) {
                           if (auto v = std::get_if<Real>(&value)) {
                             self.Add(block, key, *v, docstring);
                           } else if (auto v = std::get_if<int>(&value)) {
                             self.Add(block, key, *v, docstring);
                           } else if (auto v = std::get_if<bool>(&value)) {
                             self.Add(block, key, *v, docstring);
                           } else if (auto v = std::get_if<std::string>(&value)) {
                             self.Add(block, key, *v, docstring);
                           }
                         });
  runtime_parameters.def("get_bool",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return self.Get<bool>(block, key);
                         });
  runtime_parameters.def("get_str",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return self.Get<std::string>(block, key);
                         });
  runtime_parameters.def("get_real",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return self.Get<Real>(block, key);
                         });
  runtime_parameters.def("get_int",
                         [](RP &self, const std::string &block, const std::string &key) {
                           return self.Get<int>(block, key);
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

  state_descrptor(m);

  nanobind::class_<KamayanUnit, StateDescriptor> kamayan_unit(
      m, "KamayanUnit", nanobind::type_slots(kamayan_unit_slots));
  kamayan_unit.def("__init__", [](KamayanUnit *self, std::string name) {
    new (self) KamayanUnit(name);
  });
  kamayan_unit.def_prop_ro("Name", &KamayanUnit::Name);
  kamayan_unit.def("AllData", &KamayanUnit::AllData, nanobind::rv_policy::reference);
  kamayan_unit.def("Data", &KamayanUnit::Data, nanobind::rv_policy::reference);
  kamayan_unit.def("AddData", &KamayanUnit::AddData, nanobind::rv_policy::reference);
  kamayan_unit.def("HasData", &KamayanUnit::HasData);
  kamayan_unit.def("Configuration", &KamayanUnit::Configuration,
                   nanobind::rv_policy::reference);
  kamayan_unit.def("RuntimeParameters", &KamayanUnit::RuntimeParameters,
                   nanobind::rv_policy::reference);
  kamayan_unit.def("GetUnit", &KamayanUnit::GetUnit, nanobind::rv_policy::reference);
  kamayan_unit.def("GetUnitPtr", &KamayanUnit::GetUnitPtr);
  kamayan_unit.def_static("GetFromMesh", &KamayanUnit::GetFromMesh);
  kamayan_unit.def("init_resources", &KamayanUnit::InitResources);
  CALLBACK(kamayan_unit, KamayanUnit, SetupParams)
  CALLBACK(kamayan_unit, KamayanUnit, InitializeData)
  CALLBACK(kamayan_unit, KamayanUnit, ProblemGeneratorMeshBlock)
  CALLBACK(kamayan_unit, KamayanUnit, PrepareConserved)
  CALLBACK(kamayan_unit, KamayanUnit, PreparePrimitive)
  CALLBACK(kamayan_unit, KamayanUnit, AddFluxTasks)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksOneStep)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksSplit)

  nanobind::class_<UnitCollection> unit_collection(
      m, "UnitCollection", nanobind::type_slots(unit_collection_slots));
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

  nanobind::class_<UnitData::UnitParm> unit_parm(m, "UnitParm");
  unit_parm.def_prop_ro("key", &UnitData::UnitParm::Key);
  unit_parm.def_prop_ro("value", &UnitData::UnitParm::Get);
  unit_parm.def("Update", &UnitData::UnitParm::Update);

  nanobind::class_<UnitData> unit_data(m, "UnitData");
  unit_data.def(nanobind::init<const std::string &>());
  unit_data.def("Contains", &UnitData::Contains);
  unit_data.def_prop_ro("Block", &UnitData::Block);
  unit_data.def("AddParm",
                [](UnitData &self, const std::string &key,
                   const UnitData::DataType &value, const std::string &docstring) {
                  if (auto v = std::get_if<Real>(&value)) {
                    self.AddParm<Real>(key, *v, docstring);
                  } else if (auto v = std::get_if<int>(&value)) {
                    self.AddParm<int>(key, *v, docstring);
                  } else if (auto v = std::get_if<bool>(&value)) {
                    self.AddParm<bool>(key, *v, docstring);
                  } else if (auto v = std::get_if<std::string>(&value)) {
                    self.AddParm<std::string>(key, *v, docstring);
                  }
                });
  unit_data.def("UpdateParm", &UnitData::UpdateParm);
  unit_data.def(
      "Get",
      [](UnitData &self, nanobind::object t, const std::string &key) {
        auto &val = self.Get(key);
        if (!nanobind::isinstance(nanobind::cast(val), t)) {
          throw nanobind::type_error(
              "[UnitData::Get] parameter is not of provided type.");
        }
        return val;
      },
      nanobind::rv_policy::reference_internal,
      nanobind::sig("def Get(self, t: typing.Type[T], key: str) -> T"));
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

  parthenon_manager(m);

  auto grid = m.def_submodule("Grid", "Bindings to grid structures.");
  grid_module(grid);
}
}  // namespace kamayan

// std::string name_;
