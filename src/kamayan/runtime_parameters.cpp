#include <map>
#include <string>
#include <vector>

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
                                  std::vector<Rule<Real>> rules) {
  require_new_parm_throw(block + key);
  const Real read_Real = pin ? pin->GetOrAddReal(block, key, value) : value;
  parms[block + key] = Parameter<Real>(block, key, docstring, read_Real, rules, value);
}

template <>
void RuntimeParameters::Add<std::string>(const std::string &block, const std::string &key,
                                         const std::string &value,
                                         const std::string &docstring,
                                         std::vector<Rule<std::string>> rules) {
  require_new_parm_throw(block + key);
  const std::string read_string =
      pin ? strings::lower(pin->GetOrAddString(block, key, value))
          : strings::lower(value);
  parms[block + key] =
      Parameter<std::string>(block, key, docstring, read_string, rules, value);
}

template <>
void RuntimeParameters::Add<int>(const std::string &block, const std::string &key,
                                 const int &value, const std::string &docstring,
                                 std::vector<Rule<int>> rules) {
  require_new_parm_throw(block + key);
  const int read_int = pin ? pin->GetOrAddInteger(block, key, value) : value;
  parms[block + key] = Parameter<int>(block, key, docstring, read_int, rules, value);
}

template <>
void RuntimeParameters::Add<bool>(const std::string &block, const std::string &key,
                                  const bool &value, const std::string &docstring,
                                  std::vector<Rule<bool>> rules) {
  require_new_parm_throw(block + key);
  const bool read_bool = pin ? pin->GetOrAddBoolean(block, key, value) : value;
  parms[block + key] = Parameter<bool>(block, key, docstring, read_bool, rules, value);
}

template <>
void RuntimeParameters::Set<int>(const std::string &block, const std::string &key,
                                 const int &value) {
  std::get<Parameter<int>>(parms.at(block + key)) = value;
  if (pin) pin->SetInteger(block, key, value);
}

template <>
void RuntimeParameters::Set<std::string>(const std::string &block, const std::string &key,
                                         const std::string &value) {
  std::get<Parameter<std::string>>(parms.at(block + key)) = value;
  if (pin) pin->SetString(block, key, value);
}

template <>
void RuntimeParameters::Set<bool>(const std::string &block, const std::string &key,
                                  const bool &value) {
  std::get<Parameter<bool>>(parms.at(block + key)) = value;
  if (pin) pin->SetBoolean(block, key, value);
}

template <>
void RuntimeParameters::Set<Real>(const std::string &block, const std::string &key,
                                  const Real &value) {
  std::get<Parameter<Real>>(parms.at(block + key)) = value;
  if (pin) pin->SetReal(block, key, value);
}

}  // namespace kamayan::runtime_parameters
