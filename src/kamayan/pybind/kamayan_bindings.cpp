#include "kamayan/pybind/kamayan_bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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

template <typename R, typename... Args>
void AddCallbackregistration(CallbackRegistration<std::function<R(Args...)>>,
                             const std::string &name, nanobind::module_ &m) {
  auto class_name = "CallbackRegistration_" + name;
  using Registrar = CallbackRegistration<std::function<R(Args...)>>;
  using Function = typename Registrar::FunctionType;
  using ReturnType = std::conditional_t<std::is_void_v<R>, void, std::optional<R>>;
  nanobind::class_<Registrar>(m, class_name.c_str())
      .def("IsRegistered", &Registrar::IsRegistered)
      .def(
          "Register",
          [](Registrar &self, nanobind::object fn, std::vector<std::string> after,
             std::vector<std::string> before) -> Registrar & {
            return self.Register(nanobind::cast<Function>(fn), after, before);
          },
          nanobind::arg("fn"), nanobind::arg("after") = std::vector<std::string>(),
          nanobind::arg("before") = std::vector<std::string>(),
          nanobind::rv_policy::reference)
      .def_prop_ro("callback", [](Registrar &self) { return self.callback; })
      .def("__call__",
           [](Registrar &self, Args &&...args) -> ReturnType {
             if (self.IsRegistered()) return self(std::forward<Args>(args)...);
             if constexpr (!std::is_void_v<R>) return std::nullopt;
           })
      .def("__bool__", [](Registrar &self) { return self.IsRegistered(); })
      .def_rw("depends_on", &Registrar::depends_on)
      .def_rw("required_by", &Registrar::required_by);
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
  using SetupReg = decltype(KamayanUnit::SetupParams);
  using InitReg = decltype(KamayanUnit::InitializeData);
  using PgenReg = decltype(KamayanUnit::ProblemGeneratorMeshBlock);
  using PrepareReg = decltype(KamayanUnit::PreparePrimitive);
  using FluxReg = decltype(KamayanUnit::AddFluxTasks);
  using OneStepReg = decltype(KamayanUnit::AddTasksOneStep);
  using SplitReg = decltype(KamayanUnit::AddTasksSplit);
  // SetupParams / InitializeData registration (same type)
  AddCallbackregistration(SetupReg(), "Setup", m);
  // ProblemGenerator registration
  AddCallbackregistration(PgenReg(), "Pgen", m);
  // Prepare (Conserved/Primitive) registration
  AddCallbackregistration(PrepareReg(), "Prepare", m);
  // Flux registration
  AddCallbackregistration(FluxReg(), "Flux", m);
  // OneStep registration
  AddCallbackregistration(OneStepReg(), "OneStep", m);
  // Split registration
  AddCallbackregistration(SplitReg(), "Split", m);

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

  // Expose CallbackRegistration fields directly as properties
  kamayan_unit.def_prop_rw(
      "SetupParams", [](KamayanUnit &self) -> SetupReg & { return self.SetupParams; },
      [](KamayanUnit &self, SetupReg &val) { self.SetupParams = val; });
  kamayan_unit.def_prop_rw(
      "InitializeData",
      [](KamayanUnit &self) -> InitReg & { return self.InitializeData; },
      [](KamayanUnit &self, InitReg &val) { self.InitializeData = val; });
  kamayan_unit.def_prop_rw(
      "ProblemGeneratorMeshBlock",
      [](KamayanUnit &self) -> PgenReg & { return self.ProblemGeneratorMeshBlock; },
      [](KamayanUnit &self, PgenReg &val) { self.ProblemGeneratorMeshBlock = val; });
  kamayan_unit.def_prop_rw(
      "PrepareConserved",
      [](KamayanUnit &self) -> PrepareReg & { return self.PrepareConserved; },
      [](KamayanUnit &self, PrepareReg &val) { self.PrepareConserved = val; });
  kamayan_unit.def_prop_rw(
      "PreparePrimitive",
      [](KamayanUnit &self) -> PrepareReg & { return self.PreparePrimitive; },
      [](KamayanUnit &self, PrepareReg &val) { self.PreparePrimitive = val; });
  kamayan_unit.def_prop_rw(
      "AddFluxTasks", [](KamayanUnit &self) -> FluxReg & { return self.AddFluxTasks; },
      [](KamayanUnit &self, FluxReg &val) { self.AddFluxTasks = val; });
  kamayan_unit.def_prop_rw(
      "AddTasksOneStep",
      [](KamayanUnit &self) -> OneStepReg & { return self.AddTasksOneStep; },
      [](KamayanUnit &self, OneStepReg &val) { self.AddTasksOneStep = val; });
  kamayan_unit.def_prop_rw(
      "AddTasksSplit", [](KamayanUnit &self) -> SplitReg & { return self.AddTasksSplit; },
      [](KamayanUnit &self, SplitReg &val) { self.AddTasksSplit = val; });

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
