#ifndef DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
#define DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
#ifdef kamayan_PYTHON
#include "kamayan/pybind/kamayan_nanobind.h"
#endif

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "dispatcher/option_types.hpp"
#include "kamayan/config.hpp"

// this is a trick to put these in different translation units
// depending on if we're compiling the python bindings or not
#ifdef kamayan_PYTHON
#define PYB_NAME pybind
#else
#define PYB_NAME nopy
#endif

namespace kamayan::PYB_NAME {
struct PybindOptions {
#ifdef kamayan_PYTHON
  using PyOptFunction = std::function<void(nanobind::module_ &)>;
  using PyConfigFunction = std::function<void(nanobind::class_<Config> &)>;
  inline static std::vector<PyOptFunction> pybind_options;
  inline static std::vector<PyConfigFunction> pybind_config;
  inline static std::unordered_set<std::string> options;
#endif

  // bind a POLYMORPHIC_PARM to a python enum.Enum
  // as well as all the template specializations for a Config
  template <typename T>
  requires(PolyOpt<T>)
  static void Register() {
#ifdef kamayan_PYTHON
    using opt_info = OptInfo<T>;
    const auto name = opt_info::key();
    if (!options.count(name)) {
      options.insert(name);

      pybind_options.push_back([=](nanobind::module_ &m) {
       // if (nanobind::type<T>().is_valid()) return;
        nanobind::enum_<T> enum_t(m, opt_info::key().c_str(), "enum.Enum");
        for (int i = static_cast<int>(opt_info::First()) + 1;
             i < static_cast<int>(opt_info::Last()); i++) {
          auto val = static_cast<T>(i);
          enum_t.value(opt_info::Label(val).c_str(), val);
        }
      });

      pybind_config.push_back([](nanobind::class_<Config> &cls) {
        // do we really want these exposed to python?
        // cls.def("Add", &Config::Add<T>);
        // cls.def("Update", &Config::Update<T>);
        cls.def(std::string("Get" + opt_info::key()).c_str(),
                [](Config &self) { return self.Get<T>(); });
      });
    }
#endif
  }
};

}  // namespace kamayan::PYB_NAME
#endif  // DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
