#ifndef DISPATCHER_OPTIONS_HPP_
#define DISPATCHER_OPTIONS_HPP_

#include <array>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <parthenon/package.hpp>

#include "utils/strings.hpp"
#include "utils/type_list.hpp"

#include "dispatcher/pybind/enum_options.hpp"

namespace kamayan {

template <typename enum_opt>
struct PolyOpt_t : std::false_type {};

template <typename enum_opt>
struct OptInfo : std::false_type {};

template <typename enum_opt>
concept PolyOpt = PolyOpt_t<enum_opt>::value;

template <typename, auto...>
struct OptList {};

// Automagically have the POLYMORPHIC_PARM macro also define for us
// a function that will add the binding to the enum options
template <typename T, T first, T last>
requires(PolyOpt<T>)
void BindPolyOpt(pybind11::module_ &m) {
  using opt_info = OptInfo<T>;
  pybind11::native_enum<T> enum_t(m, opt_info::key().c_str(), "enum.Enum");
  for (int i = static_cast<int>(first) + 1; i < static_cast<int>(last); i++) {
    auto val = static_cast<T>(i);
    enum_t.value(opt_info::Label(val).c_str(), val);
  }
  enum_t.finalize();
}

// used to enumerate the allowed values of a PolyOpt in a given
// dispatch functor. By default this list is exactly what is written
// unless cmake has been configured with -DOPT_enum_opt="enum_v,..."
// in that case OptInf<enum_opt>::isdef == true and the ParmList()
// will just be those defined at configure time
template <typename enum_opt, auto enum_v, auto... enum_vs>
requires(PolyOpt<enum_opt>)
struct OptList<enum_opt, enum_v, enum_vs...> {
  using type = enum_opt;
  static constexpr auto value = OptInfo<enum_opt>::ParmList();
};

template <typename OL>
concept OptionsList = requires {
  typename OL::type;
  OL::value;
};

// derive from this to describe how to build
// a composite option inside a dispatch
struct OptionFactory {
  using factory = std::true_type;
};

template <typename T>
concept FactoryOption = requires {
  requires T::factory::value;
  typename T::options;
};

template <typename... Ts>
requires((OptionsList<Ts> || FactoryOption<Ts>) && ...)
struct OptTypeList {
  using type = TypeList<Ts...>;
  static constexpr std::size_t size = sizeof...(Ts);
};

template <typename opt, typename parm>
concept ParmOpt =
    requires { opt::value; } && std::is_same_v<base_dtype<opt::value>, parm>;

template <typename T>
struct StrPair_t : std::false_type {};

template <typename T, typename U>
requires(std::is_convertible_v<U, std::string>)
struct StrPair_t<std::pair<T, U>> : std::true_type {};

template <typename Enum, typename Pair>
concept EnumStrPair = requires {
  requires PolyOpt<Enum>;
  requires StrPair_t<Pair>::value;
};

// often we need to map a string runtime parameter onto an enum option
template <typename T, typename... Ts>
requires(EnumStrPair<T, Ts> && ...)
T MapStrToEnum(std::string parm, Ts... mappings) {
  T enum_out;
  bool found = false;
  (void)((
             [&] {
               if (parm == std::string(mappings.second)) {
                 found = true;
                 enum_out = mappings.first;
               }
             }(),
             !found) &&
         ...);
  if (found) return enum_out;

  std::ostringstream msg;
  msg << "String mapping for [" << parm << "] to " << OptInfo<T>::key();
  msg << " not handled.\n";
  msg << "Recognized values are: ";
  ([&] { msg << mappings.second << " "; }(), ...);
  msg << "\n";
  PARTHENON_THROW(msg.str());
  return enum_out;
}

// the following is some magic taken from
// https://stackoverflow.com/questions/18048039/c-constexpr-function-to-test-preprocessor-macros
// this will test the stringified-version of a requested macro against the evaluated macro
// as a constexpr check on whether it exists or not
constexpr bool _is_defined(const char s1[], const char s2[]) {
  return std::string_view(s1) != s2;
}

// here are some macros that help us expand out our compound macros taken from the
// POLYMORPHIC_PARM macro
#define _eval(...) #__VA_ARGS__
#define _getVal(x) _eval(x)
// get a comptime bool to determine if a macro is defined and can be used
// inside of a PP macro
#define is_defined(x) _is_defined(#x, _eval(x))
#define _parm_msg(x, y)                                                                  \
  "unrecognized option in " #x " = " _eval(x) " valid values are: " y

// We can use this macro to define an enum class to enumerate the
// allowed values for a runtime option that can be used in the DispatchFunctor.
// Will specialize the OptInfo<name> struct so we can have a type_trait that
// holds all the compile time info about our options
// The specialization of PolyOpt_t<name> enables the poly_opt concept
// that we can use to validate our OptLists
#define POLYMORPHIC_PARM(name, ...)                                                      \
  enum class name { _first, __VA_ARGS__, _last };                                        \
  template <>                                                                            \
  struct OptInfo<name> : std::true_type {                                                \
    static std::string Label(const name &_parm) {                                        \
      return strings::split({#__VA_ARGS__}, ',')[static_cast<int>(_parm) - 1];           \
    }                                                                                    \
    static constexpr bool isdef = is_defined(OPT_##name);                                \
    static std::string key() { return #name; }                                           \
    using type = name;                                                                   \
                                                                                         \
   private:                                                                              \
    template <std::size_t idx>                                                           \
    static constexpr name getVal_impl() {                                                \
      constexpr std::size_t n = strings::getLen(#__VA_ARGS__);                           \
      constexpr auto label = parm_list[idx];                                             \
      constexpr auto labels = strings::splitStrView<n>(#__VA_ARGS__);                    \
      constexpr bool inList = strings::strInList(label, labels);                         \
      static_assert(inList || !isdef, _parm_msg(OPT_##name, #__VA_ARGS__));              \
      for (int i = 0; i < n; i++) {                                                      \
        if (label == labels[i]) return static_cast<name>(i + 1);                         \
      }                                                                                  \
      return static_cast<name>(0);                                                       \
    }                                                                                    \
    template <std::size_t... Is>                                                         \
    static constexpr auto ParmList_impl(std::index_sequence<Is...>) {                    \
      return std::array<name, sizeof...(Is)>{getVal_impl<Is>()...};                      \
    }                                                                                    \
                                                                                         \
   public:                                                                               \
    static constexpr std::size_t nopts =                                                 \
        isdef ? strings::getLen(_getVal(OPT_##name)) : strings::getLen(#__VA_ARGS__);    \
    static constexpr std::string_view parm_list_str =                                    \
        isdef ? _getVal(OPT_##name) : #__VA_ARGS__;                                      \
    static constexpr auto parm_list = strings::splitStrView<nopts>(parm_list_str);       \
    static constexpr auto ParmList() {                                                   \
      return ParmList_impl(std::make_index_sequence<nopts>());                           \
    }                                                                                    \
  };                                                                                     \
  template <>                                                                            \
  struct PolyOpt_t<name> : std::true_type {                                              \
    template <name parm>                                                                 \
    struct opt {                                                                         \
      static constexpr name value = parm;                                                \
      static std::string Label() { return OptInfo<name>::Label(parm); }                  \
    };                                                                                   \
  };                                                                                     \
  namespace {                                                                            \
  struct PyEnumRegistrar_##name {                                                        \
    PyEnumRegistrar_##name() {                                                           \
      kamayan::pybind::PybindOptions::Register(                                          \
          BindPolyOpt<name, name::_first, name::_last>, OptInfo<name>::key());           \
    }                                                                                    \
  } py_enum_registrar_##name;                                                            \
  }

}  // namespace kamayan

#endif  // DISPATCHER_OPTIONS_HPP_
