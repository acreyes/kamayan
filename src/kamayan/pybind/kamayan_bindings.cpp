#include "kamayan/pybind/kamayan_bindings.hpp"

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
#include "kamayan/callback_registration.hpp"
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

// macro for defining getter/setter methods for CallbackRegistration in KamayanUnit
// Returns a reference to the CallbackRegistration so Python can call .Register() on it
#define CALLBACK(pycls, cls, cbname)                                                     \
  pycls.def(                                                                             \
      "set_" #cbname,                                                                    \
      [](cls &self, nanobind::object fn) {                                               \
        self.cbname = nanobind::cast<decltype(self.cbname.callback)>(fn);                \
      },                                                                                 \
      nanobind::rv_policy::reference);                                                   \
  pycls.def(                                                                             \
      "get_" #cbname, [](cls &self) -> auto & { return self.cbname; },                   \
      nanobind::rv_policy::reference);

namespace kamayan {

using RP = runtime_parameters::RuntimeParameters;
void RuntimeParameter_module(nanobind::module_ &m) {
  using Parameter = std::variant<bool, Real, int, std::string>;
  nanobind::class_<RP> runtime_parameters(m, "RuntimeParameters");
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

  // Create opaque bindings for each CallbackRegistration type
  // These are separate instantiations of the template so need individual bindings
  using SetupReg = CallbackRegistration<std::function<void(KamayanUnit *)>>;
  using InitReg = CallbackRegistration<std::function<void(KamayanUnit *)>>;
  using PgenReg = CallbackRegistration<std::function<void(MeshBlock *)>>;
  using PrepareReg = CallbackRegistration<std::function<TaskStatus(MeshData *)>>;
  using FluxReg =
      CallbackRegistration<std::function<TaskID(TaskID, TaskList &, MeshData *)>>;
  using OneStepReg = CallbackRegistration<
      std::function<TaskID(TaskID, TaskList &, MeshData *, MeshData *)>>;
  using SplitReg = CallbackRegistration<
      std::function<TaskID(TaskID, TaskList &, MeshData *, const Real &)>>;

  // SetupParams / InitializeData registration (same type)
  nanobind::class_<SetupReg>(m, "CallbackRegistration_Setup")
      .def("IsRegistered", &SetupReg::IsRegistered)
      .def("__call__",
           [](SetupReg &self, KamayanUnit *unit) {
             if (self.IsRegistered()) return self(unit);
           })
      .def("__bool__", [](SetupReg &self) { return self.IsRegistered(); });

  // ProblemGenerator registration
  nanobind::class_<PgenReg>(m, "CallbackRegistration_Pgen")
      .def("IsRegistered", &PgenReg::IsRegistered)
      .def("__call__",
           [](PgenReg &self, MeshBlock *mb) {
             if (self.IsRegistered()) return self(mb);
           })
      .def("__bool__", [](PgenReg &self) { return self.IsRegistered(); });

  // Prepare (Conserved/Primitive) registration
  nanobind::class_<PrepareReg>(m, "CallbackRegistration_Prepare")
      .def("IsRegistered", &PrepareReg::IsRegistered)
      .def("__call__",
           [](PrepareReg &self, MeshData *md) {
             if (self.IsRegistered()) return self(md);
             return TaskStatus::complete;
           })
      .def("__bool__", [](PrepareReg &self) { return self.IsRegistered(); });

  // Flux registration
  nanobind::class_<FluxReg>(m, "CallbackRegistration_Flux")
      .def("IsRegistered", &FluxReg::IsRegistered)
      .def("__call__",
           [](FluxReg &self, TaskID prev, TaskList &tl, MeshData *md) {
             if (self.IsRegistered()) return self(prev, tl, md);
             return prev;
           })
      .def("__bool__", [](FluxReg &self) { return self.IsRegistered(); });

  // OneStep registration
  nanobind::class_<OneStepReg>(m, "CallbackRegistration_OneStep")
      .def("IsRegistered", &OneStepReg::IsRegistered)
      .def("__call__",
           [](OneStepReg &self, TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt) {
             if (self.IsRegistered()) return self(prev, tl, md, dudt);
             return prev;
           })
      .def("__bool__", [](OneStepReg &self) { return self.IsRegistered(); });

  // Split registration
  nanobind::class_<SplitReg>(m, "CallbackRegistration_Split")
      .def("IsRegistered", &SplitReg::IsRegistered)
      .def("__call__",
           [](SplitReg &self, TaskID prev, TaskList &tl, MeshData *md, const Real &dt) {
             if (self.IsRegistered()) return self(prev, tl, md, dt);
             return prev;
           })
      .def("__bool__", [](SplitReg &self) { return self.IsRegistered(); });

  nanobind::class_<KamayanUnit, StateDescriptor> kamayan_unit(m, "KamayanUnit");
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
