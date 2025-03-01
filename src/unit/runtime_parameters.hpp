#ifndef UNIT_RUNTIME_PARAMETERS_HPP_
#define UNIT_RUNTIME_PARAMETERS_HPP_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "types.hpp"

namespace kamayan::runtime_parameters {
template <typename T>
concept rparm = std::is_same_v<T, int> || std::is_same_v<T, Real> ||
                std::is_same_v<T, std::string> || std::is_same_v<T, bool>;

template <typename T>
concept rparm_range = std::is_same_v<T, int> || std::is_same_v<T, Real>;

template <typename T>
concept rparm_str = std::is_same_v<T, std::string>;

template <typename T>
struct Rule {};

template <rparm_range T>
struct Rule<T> {
  T lower, upper;
  explicit Rule(T value) : lower(value), upper(value) {}
  explicit Rule(std::array<T, 2> values) : lower(values[0]), upper(values[1]) {}
  bool validate(T value) const { return value >= lower && value <= upper; }
};

template <rparm_str T>
struct Rule<T> {
  std::string val;
  explicit Rule(std::string value) : val(value) {}
  explicit Rule(const char *value) : val(value) {}
  bool validate(T value) const { return value == val; }
};

template <rparm T>
struct Parameter {
  Parameter() {}
  Parameter(const std::string &block_, const std::string key_,
            const std::string &docstring_, const T &value_)
      : block(block_), key(key_), docstring(docstring_), value(value_) {}
  std::string block, key, docstring;
  T value;
  std::vector<Rule<T>> rules;
};

class RuntimeParameters {
  // Add should check if key is already mapped. If it is then throw
  template <rparm T>
  void Add(const std::string &block, const std::string &key, const T &value,
           const std::string &docstring, std::vector<Rule<T>> rules = {});

  template <rparm T>
  T Get(const std::string &key);

  // organize all our keys in the map by the Parameter's block
  void write_docstrings();

  std::shared_ptr<parthenon::ParameterInput> pin;
  std::map<std::string, Parameter<bool>> bool_parms;
  std::map<std::string, Parameter<int>> int_parms;
  std::map<std::string, Parameter<Real>> Real_parms;
  std::map<std::string, Parameter<std::string>> string_parms;
};

}  // namespace kamayan::runtime_parameters

#endif  // UNIT_RUNTIME_PARAMETERS_HPP_
