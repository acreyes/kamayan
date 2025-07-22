#include <pybind11/pybind11.h>

#include <string>

#include "dispatcher/pybind/enum_options.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"

namespace py = pybind11;

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

void RuntimeParameter_module(py::module_ &m) {
  py::classh<RP> runtime_parameters(m, "RuntimeParameters");
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
                           Get<bool>(self, block, key);
                         });
  runtime_parameters.def("GetStr",
                         [](RP &self, const std::string &block, const std::string &key) {
                           Get<std::string>(self, block, key);
                         });
  runtime_parameters.def("GetReal",
                         [](RP &self, const std::string &block, const std::string &key) {
                           Get<Real>(self, block, key);
                         });
  runtime_parameters.def("GetInt",
                         [](RP &self, const std::string &block, const std::string &key) {
                           Get<int>(self, block, key);
                         });
}

PYBIND11_MODULE(pyKamayan, m) {
  m.doc() = "Main entrypoint for kamayan python bindings.";

  auto opts = m.def_submodule("Options", "Polymorphic Parameter options.");
  for (const auto &func : pybind::PybindOptions::pybind_options) {
    func(opts);
  }

  auto config = py::classh<Config>(m, "Config");
  config.doc() = "Bindings to global kamayan Config type.";
  for (const auto &func : pybind::PybindOptions::pybind_config) {
    func(config);
  }

  auto rps =
      m.def_submodule("RuntimeParameters", "Bindings to kamayan RuntimeParameters.");
  RuntimeParameter_module(rps);

  py::classh<KamayanUnit> kamayan_unit(m, "KamayanUnit");
  kamayan_unit.def(py::init<std::string>());
  CALLBACK(kamayan_unit, KamayanUnit, Setup)
  CALLBACK(kamayan_unit, KamayanUnit, Initialize)
  CALLBACK(kamayan_unit, KamayanUnit, ProblemGeneratorMeshBlock)
  CALLBACK(kamayan_unit, KamayanUnit, PrepareConserved)
  CALLBACK(kamayan_unit, KamayanUnit, PreparePrimitive)
  CALLBACK(kamayan_unit, KamayanUnit, AddFluxTasks)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksOneStep)
  CALLBACK(kamayan_unit, KamayanUnit, AddTasksSplit)
}
}  // namespace kamayan

// std::string name_;
