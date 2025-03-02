#ifndef UNIT_RUNTIME_PARAMETERS_HPP_
#define UNIT_RUNTIME_PARAMETERS_HPP_

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "types.hpp"
#include "utils/strings.hpp"

namespace kamayan::runtime_parameters {
template <typename T>
concept rparm = std::is_same_v<T, int> || std::is_same_v<T, Real> ||
                std::is_same_v<T, std::string> || std::is_same_v<T, bool>;

template <typename T>
concept rparm_range = std::is_same_v<T, int> || std::is_same_v<T, Real>;

template <typename T>
concept rparm_single = std::is_same_v<T, std::string>;

template <typename T>
concept rparm_no_rule = std::is_same_v<T, bool>;

template <typename>
struct Rule {};

template <>
struct Rule<std::string> {
  std::string val;
  explicit(false) Rule(const std::string &value) : val(strings::lower(value)) {}
  template <std::size_t N>
  explicit(false) Rule(const char (&value)[N]) : val(strings::lower(value)) {}
  bool validate(std::string value) const { return value == val; }
};

template <rparm_no_rule T>
struct Rule<T> {
  bool validate(T value) const { return true; }
};

template <rparm_range T>
struct Rule<T> {
  T lower, upper;
  explicit(false) Rule(T value) : lower(value), upper(value) {}
  explicit(false) Rule(std::initializer_list<T> values)
      : lower(values.begin()[0]), upper(values.begin()[1]) {}
  bool validate(T value) const { return value >= lower && value <= upper; }
};

template <rparm_single T>
struct Rule<T> {
  std::string val;
  explicit(false) Rule(const T &value) : val(value) {}
  bool validate(T value) const { return value == val; }
};

namespace impl {
// no rules for this type so just give back the docstring
template <rparm T>
  requires(!rparm_range<T> && !rparm_single<T>)
std::string to_docstring(const std::string &docstring, std::vector<Rule<T>> rules) {
  return docstring;
}

template <rparm_range T>
std::string to_docstring(const std::string &docstring, std::vector<Rule<T>> rules) {
  std::stringstream rules_stream;
  rules_stream << " [";
  for (const auto &rule : rules) {
    rules_stream << rule.lower;
    if (rule.upper > rule.lower) {
      rules_stream << "..." << rule.upper << ", ";
    } else {
      rules_stream << ", ";
    }
  }
  rules_stream << "]\n";
  return rules_stream.str() + docstring;
}

template <rparm_single T>
std::string to_docstring(const std::string &docstring, std::vector<Rule<T>> rules) {
  std::stringstream rules_stream;
  rules_stream << " [";
  for (const auto &rule : rules) {
    rules_stream << " " << rule.val << ", ";
  }
  rules_stream << "]\n";
  return rules_stream.str() + docstring;
}

template <rparm T>
std::string type_str() {}
}  // namespace impl

template <rparm T>
struct Parameter {
  Parameter() {}
  Parameter(const std::string &block_, const std::string key_,
            const std::string &docstring_, const T &value_, std::vector<Rule<T>> rules)
      : block(block_), key(key_), docstring(impl::to_docstring(docstring_, rules)),
        value(value_) {
    if (rules.size() > 0) {
      // validate our parm against the rules
      bool valid_parm_value = false;
      for (const auto &rule : rules) {
        valid_parm_value = valid_parm_value || rule.validate(value);
      }
      std::stringstream err_msg;
      err_msg << "[Error] Invalid value for runtime parameter ";
      err_msg << "<" + block + ">/" << key << " = " << value;
      err_msg << docstring << "\n";
      PARTHENON_REQUIRE_THROWS(valid_parm_value, err_msg.str().c_str());
    }
  }
  std::string block, key, docstring;
  T value;
  std::vector<Rule<T>> rules;
};

class RuntimeParameters {
 public:
  RuntimeParameters() {}
  explicit RuntimeParameters(std::shared_ptr<parthenon::ParameterInput> pin_)
      : pin(pin_) {}

  // Add should check if key is already mapped. If it is then throw
  template <rparm T>
  void Add(const std::string &block, const std::string &key, const T &value,
           const std::string &docstring, std::initializer_list<Rule<T>> rules = {});
  // template <rparm T>
  //    requires (std::is_same_v<T, std::string>)
  // void Add(const std::string &block, const std::string &key, const T &value,
  //          const std::string &docstring, std::initializer_list<> rules = {});

  template <rparm T>
  T Get(const std::string &block, const std::string &key);

 private:
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
