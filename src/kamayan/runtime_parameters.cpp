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

template <Rparm T>
void require_exists_parm_throw(const std::string &key,
                               std::map<std::string, Parameter<T>> parms) {
  if (parms.contains(key)) return;

  auto parm = parms[key];
  std::string err_msg =
      "[Error] " + impl::type_str<T>() + " Runtime Parameter " + key + " doesn't exist ";
  PARTHENON_THROW(err_msg.c_str());
}

template <Rparm T>
void require_new_parm_throw(const std::string &key,
                            std::map<std::string, Parameter<T>> parms) {
  if (!parms.contains(key)) return;

  auto parm = parms[key];
  std::string err_msg = "[Error] " + impl::type_str<T>() + " Runtime Parameter " + key +
                        " already exists " + parm.block + "/" + parm.key;
  PARTHENON_THROW(err_msg.c_str());
}

template <>
void RuntimeParameters::Add<Real>(const std::string &block, const std::string &key,
                                  const Real &value, const std::string &docstring,
                                  std::initializer_list<Rule<Real>> rules) {
  require_new_parm_throw(block + key, Real_parms);
  const Real read_Real = pin->GetOrAddReal(block, key, value);
  Real_parms[block + key] = Parameter<Real>(block, key, docstring, read_Real, rules);
}

template <>
void RuntimeParameters::Add<std::string>(const std::string &block, const std::string &key,
                                         const std::string &value,
                                         const std::string &docstring,
                                         std::initializer_list<Rule<std::string>> rules) {
  require_new_parm_throw(block + key, string_parms);
  const std::string read_string = strings::lower(pin->GetOrAddString(block, key, value));
  string_parms[block + key] =
      Parameter<std::string>(block, key, docstring, read_string, rules);
}

template <>
void RuntimeParameters::Add<int>(const std::string &block, const std::string &key,
                                 const int &value, const std::string &docstring,
                                 std::initializer_list<Rule<int>> rules) {
  require_new_parm_throw(block + key, int_parms);
  const int read_int = pin->GetOrAddInteger(block, key, value);
  int_parms[block + key] = Parameter<int>(block, key, docstring, read_int, rules);
}

template <>
void RuntimeParameters::Add<bool>(const std::string &block, const std::string &key,
                                  const bool &value, const std::string &docstring,
                                  std::initializer_list<Rule<bool>> rules) {
  require_new_parm_throw(block + key, bool_parms);
  const bool read_bool = pin->GetOrAddBoolean(block, key, value);
  bool_parms[block + key] = Parameter<bool>(block, key, docstring, read_bool, rules);
}

template <>
bool RuntimeParameters::Get<bool>(const std::string &block, const std::string &key) {
  require_exists_parm_throw(block + key, bool_parms);
  auto parm = bool_parms[block + key];
  return parm.value;
}

template <>
std::string RuntimeParameters::Get<std::string>(const std::string &block,
                                                const std::string &key) {
  require_exists_parm_throw(block + key, string_parms);
  auto parm = string_parms[block + key];
  return parm.value;
}

template <>
Real RuntimeParameters::Get<Real>(const std::string &block, const std::string &key) {
  require_exists_parm_throw(block + key, Real_parms);
  auto parm = Real_parms[block + key];
  return parm.value;
}

template <>
int RuntimeParameters::Get<int>(const std::string &block, const std::string &key) {
  require_exists_parm_throw(block + key, int_parms);
  auto parm = int_parms[block + key];
  return parm.value;
}

template <>
bool RuntimeParameters::GetOrAdd<bool>(const std::string &block, const std::string &key,
                                       const bool &value, const std::string &docstring,
                                       std::initializer_list<Rule<bool>> rules) {
  if (bool_parms.contains(block + key)) return Get<bool>(block, key);
  Add<bool>(block, key, value, docstring, rules);

  return Get<bool>(block, key);
}

template <>
Real RuntimeParameters::GetOrAdd<Real>(const std::string &block, const std::string &key,
                                       const Real &value, const std::string &docstring,
                                       std::initializer_list<Rule<Real>> rules) {
  if (Real_parms.contains(block + key)) return Get<Real>(block, key);
  Add<Real>(block, key, value, docstring, rules);

  return Get<Real>(block, key);
}

template <>
std::string RuntimeParameters::GetOrAdd<std::string>(
    const std::string &block, const std::string &key, const std::string &value,
    const std::string &docstring, std::initializer_list<Rule<std::string>> rules) {
  if (string_parms.contains(block + key)) return Get<std::string>(block, key);
  Add<std::string>(block, key, value, docstring, rules);

  return Get<std::string>(block, key);
}

template <>
int RuntimeParameters::GetOrAdd<int>(const std::string &block, const std::string &key,
                                     const int &value, const std::string &docstring,
                                     std::initializer_list<Rule<int>> rules) {
  if (int_parms.contains(block + key)) return Get<int>(block, key);
  Add<int>(block, key, value, docstring, rules);

  return Get<int>(block, key);
}
}  // namespace kamayan::runtime_parameters
