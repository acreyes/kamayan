#ifndef DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
#define DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
#include <map>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace kamayan::pybind {
struct PybindOptions {
  using PyOptFunction = std::function<void(pybind11::module_ &)>;
  inline static std::vector<PyOptFunction> pybind_options;
  inline static std::unordered_set<std::string> options;
  void static Register(PyOptFunction func, const std::string &name) {
    if (!options.count(name)) {
      options.insert(name);
      pybind_options.push_back(func);
    }
  }
};

}  // namespace kamayan::pybind
#endif  // DISPATCHER_PYBIND_ENUM_OPTIONS_HPP_
