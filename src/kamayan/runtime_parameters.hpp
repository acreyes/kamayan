#ifndef KAMAYAN_RUNTIME_PARAMETERS_HPP_
#define KAMAYAN_RUNTIME_PARAMETERS_HPP_

#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "utils/strings.hpp"

namespace kamayan {
struct KamayanUnit;
std::stringstream RuntimeParameterDocs(KamayanUnit *unit, ParameterInput *pin);
}  // namespace kamayan

namespace kamayan::runtime_parameters {
template <typename T>
concept Rparm = std::is_same_v<T, int> || std::is_same_v<T, Real> ||
                std::is_same_v<T, std::string> || std::is_same_v<T, bool>;

template <typename T>
concept RparmRange = std::is_same_v<T, int> || std::is_same_v<T, Real>;

template <typename T>
concept RparmSingle = std::is_same_v<T, std::string>;

template <typename T>
concept RparmNoRule = std::is_same_v<T, bool>;

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

template <typename T>
requires(RparmNoRule<T>)
struct Rule<T> {
  bool validate(T value) const { return true; }
};

template <typename T>
requires(RparmRange<T>)
struct Rule<T> {
  T lower, upper;
  explicit(false) Rule(T value) : lower(value), upper(value) {}
  explicit(false) Rule(std::initializer_list<T> values)
      : lower(values.begin()[0]), upper(values.begin()[1]) {}
  bool validate(T value) const { return value >= lower && value <= upper; }
};

template <typename T>
requires(RparmSingle<T>)
struct Rule<T> {
  std::string val;
  explicit(false) Rule(const T &value) : val(value) {}
  bool validate(T value) const { return value == val; }
};

namespace impl {
// no rules for this type so just give back the docstring
template <typename T>
requires(Rparm<T> && !RparmRange<T> && !RparmSingle<T>)
std::string ToDocString(const std::string &docstring, std::vector<Rule<T>> rules) {
  return " | " + docstring;
}

template <typename T>
requires(RparmRange<T>)
std::string ToDocString(const std::string &docstring, std::vector<Rule<T>> rules) {
  std::stringstream rules_stream;
  if (rules.size() > 0) {
    rules_stream << " [";
    for (const auto &rule : rules) {
      rules_stream << rule.lower;
      if (rule.upper > rule.lower) {
        rules_stream << "..." << rule.upper << ", ";
      } else {
        rules_stream << ", ";
      }
    }
    rules_stream << "]";
  }
  return rules_stream.str() + " | " + docstring;
}

template <typename T>
requires(RparmSingle<T>)
std::string ToDocString(const std::string &docstring, std::vector<Rule<T>> rules) {
  std::stringstream rules_stream;
  if (rules.size() > 0) {
    rules_stream << " [";
    for (const auto &rule : rules) {
      rules_stream << " " << rule.val << ", ";
    }
    rules_stream << "]";
  }
  return rules_stream.str() + " | " + docstring;
}

template <typename T>
requires(Rparm<T>)
std::string type_str();
}  // namespace impl

template <typename T>
requires(Rparm<T>)
struct Parameter {
  Parameter() {}
  Parameter(const std::string &block_, const std::string key_,
            const std::string &docstring_, const T &value_, std::vector<Rule<T>> rules,
            const T &def_val)
      : block(block_), key(key_), docstring(impl::ToDocString(docstring_, rules)),
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
    std::string default_str;
    if constexpr (std::is_same_v<T, std::string>) {
      default_str = def_val;
    } else if constexpr (std::is_same_v<T, bool>) {
      default_str = def_val ? "true" : "false";
    } else {
      default_str = std::to_string(def_val);
      if (default_str.size() > 5) {
        default_str = std::format("{:.5e}", static_cast<Real>(value_));
      }
    }
    docstring = default_str + " | " + docstring;
  }

  std::string DocString() {
    return " | " + key + " | " + Type() + " | " + docstring + "\n";
  }

  std::string Type() const { return impl::type_str<T>(); }
  std::string block, key, docstring;
  T value;
  std::vector<Rule<T>> rules;
};

class RuntimeParameters {
  friend std::stringstream kamayan::RuntimeParameterDocs(KamayanUnit *unit,
                                                         ParameterInput *pin);

 public:
  RuntimeParameters() {}
  explicit RuntimeParameters(parthenon::ParameterInput *pin_) : pin(pin_) {}

  // Add should check if key is already mapped. If it is then throw
  template <typename T>
  requires(Rparm<T>)
  void Add(const std::string &block, const std::string &key, const T &value,
           const std::string &docstring, std::vector<Rule<T>> rules = {});

  template <typename T>
  requires(Rparm<T>)
  void Add(const std::string &block, const std::string &key, const std::size_t &n,
           const T &value, const std::string &docstring,
           std::vector<Rule<T>> rules = {}) {
    for (int i = 0; i < n; i++) {
      Add<T>(block, key + std::to_string(i), value, docstring, rules);
    }
  }

  template <typename T>
  requires(Rparm<T>)
  void Set(const std::string &block, const std::string &key, const T &value);

  template <typename T>
  requires(Rparm<T>)
  T Get(const std::string &block, const std::string &key) const {
    require_exists_parm_throw(block + key);
    auto parm = parms.at(block + key);
    return std::get<Parameter<T>>(parm).value;
  }

  template <typename T>
  requires(Rparm<T>)
  T GetOrAdd(const std::string &block, const std::string &key, const T &value,
             const std::string &docstring, std::vector<Rule<T>> rules = {}) {
    if (!parms.contains(block + key)) Add<T>(block, key, value, docstring, rules);
    return Get<T>(block, key);
  }

  auto GetPin() const { return pin; }

 private:
  void require_exists_parm_throw(const std::string &key) const;

  void require_new_parm_throw(const std::string &key) const;

  parthenon::ParameterInput *pin;
  using Parm_t = std::variant<Parameter<bool>, Parameter<int>, Parameter<Real>,
                              Parameter<std::string>>;
  std::map<std::string, Parm_t> parms;
};

}  // namespace kamayan::runtime_parameters

#endif  // KAMAYAN_RUNTIME_PARAMETERS_HPP_
