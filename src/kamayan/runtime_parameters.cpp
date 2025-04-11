#include <map>
#include <string>

#include "kamayan/runtime_parameters.hpp"
#include "utils/error_checking.hpp"
#include "utils/strings.hpp"

namespace kamayan::runtime_parameters {

namespace impl {
template <>
std::string type_str<int>() {
  return "Integer";
}
template <>
std::string type_str<Real>() {
  return "Real";
}
template <>
std::string type_str<bool>() {
  return "Boolean";
}
template <>
std::string type_str<std::string>() {
  return "String";
}
}  // namespace impl

void RuntimeParameters::require_exists_parm_throw(const std::string &key) const {
  if (parms.contains(key)) return;

  std::string err_msg = "[Error] Runtime Parameter " + key + " doesn't exist ";
  PARTHENON_THROW(err_msg.c_str());
}

void RuntimeParameters::require_new_parm_throw(const std::string &key) const {
  if (!parms.contains(key)) return;

  auto p_type = std::visit([](auto &parm) { return parm.Type(); }, parms.at(key));
  auto p_block = std::visit([](auto &parm) { return parm.block; }, parms.at(key));
  auto p_key = std::visit([](auto &parm) { return parm.key; }, parms.at(key));
  std::string err_msg = "[Error] " + p_type + " Runtime Parameter " + key +
                        " already exists " + p_block + "/" + p_key;
  PARTHENON_THROW(err_msg.c_str());
}

template <>
void RuntimeParameters::Add<Real>(const std::string &block, const std::string &key,
                                  const Real &value, const std::string &docstring,
                                  std::initializer_list<Rule<Real>> rules) {
  require_new_parm_throw(block + key);
  const Real read_Real = pin->GetOrAddReal(block, key, value);
  parms[block + key] = Parameter<Real>(block, key, docstring, read_Real, rules);
}

template <>
void RuntimeParameters::Add<std::string>(const std::string &block, const std::string &key,
                                         const std::string &value,
                                         const std::string &docstring,
                                         std::initializer_list<Rule<std::string>> rules) {
  require_new_parm_throw(block + key);
  const std::string read_string = strings::lower(pin->GetOrAddString(block, key, value));
  parms[block + key] = Parameter<std::string>(block, key, docstring, read_string, rules);
}

template <>
void RuntimeParameters::Add<int>(const std::string &block, const std::string &key,
                                 const int &value, const std::string &docstring,
                                 std::initializer_list<Rule<int>> rules) {
  require_new_parm_throw(block + key);
  const int read_int = pin->GetOrAddInteger(block, key, value);
  parms[block + key] = Parameter<int>(block, key, docstring, read_int, rules);
}

template <>
void RuntimeParameters::Add<bool>(const std::string &block, const std::string &key,
                                  const bool &value, const std::string &docstring,
                                  std::initializer_list<Rule<bool>> rules) {
  require_new_parm_throw(block + key);
  const bool read_bool = pin->GetOrAddBoolean(block, key, value);
  parms[block + key] = Parameter<bool>(block, key, docstring, read_bool, rules);
}

}  // namespace kamayan::runtime_parameters
