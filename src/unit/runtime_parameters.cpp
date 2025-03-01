#include <sstream>

#include "unit/runtime_parameters.hpp"
#include "utils/error_checking.hpp"

namespace kamayan::runtime_parameters {

template <typename Map>
void require_new_parm_throw(const std::string &key, Map parms, const std::string &type) {
  if (!parms.contains(key)) return;

  auto parm = parms[key];
  std::string err_msg = "[Error] " + type + " Runtime Parameter " + key +
                        " already exists " + parm.block + "/" + parm.key;
  PARTHENON_THROW(err_msg.c_str());
}

template <rparm T>
Parameter<T> validate_parm(const Parameter<T> &parm, std::vector<Rule<T>> rules) {
  if (rules.size() > 0) {
    // validate our parm against the rules
    bool valid = false;
    for (const auto &rule : rules) {
      valid = valid || rule.validate(parm.value);
    }
    std::stringstream err_msg;
    err_msg << "[Error] Invalid value for runtime parameter ";
    err_msg << parm.block << " / " << parm.key << " = " << parm.value << "\n";
    err_msg << parm.docstring << "\n";
    PARTHENON_REQUIRE_THROWS(valid, err_msg.str().c_str());
  }
  return parm;
}

template <>
void RuntimeParameters::Add<Real>(const std::string &block, const std::string &key,
                                  const Real &value, const std::string &docstring,
                                  std::vector<Rule<Real>> rules) {
  require_new_parm_throw(key, Real_parms, "Real");
  const Real read_Real = pin->GetOrAddReal(block, key, value);
  Real_parms[key] = validate_parm(Parameter<Real>(block, key, docstring, value), rules);
}

template <>
void RuntimeParameters::Add<std::string>(const std::string &block, const std::string &key,
                                         const std::string &value,
                                         const std::string &docstring,
                                         std::vector<Rule<std::string>> rules) {
  require_new_parm_throw(key, string_parms, "String");
  const std::string read_string = pin->GetOrAddString(block, key, value);
  string_parms[key] =
      validate_parm(Parameter<std::string>(block, key, docstring, value), rules);
}

template <>
void RuntimeParameters::Add<int>(const std::string &block, const std::string &key,
                                 const int &value, const std::string &docstring,
                                 std::vector<Rule<int>> rules) {
  require_new_parm_throw(key, int_parms, "Integer");
  const int read_int = pin->GetOrAddInteger(block, key, value);
  int_parms[key] = validate_parm(Parameter<int>(block, key, docstring, value), rules);
}

template <>
void RuntimeParameters::Add<bool>(const std::string &block, const std::string &key,
                                  const bool &value, const std::string &docstring,
                                  std::vector<Rule<bool>> rules) {
  require_new_parm_throw(key, bool_parms, "Boolean");
  const bool read_bool = pin->GetOrAddBoolean(block, key, value);
  bool_parms[key] = Parameter<bool>(block, key, docstring, value);
}
}  // namespace kamayan::runtime_parameters
