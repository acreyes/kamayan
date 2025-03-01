#ifndef DISPATCHER_OPTIONS_HPP_
#define DISPATCHER_OPTIONS_HPP_

#include <array>
#include <type_traits>

#include "utils/strings.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

template <typename enum_opt>
struct PolyOpt : std::false_type {};

template <typename enum_opt>
struct OptInfo : std::false_type {};

template <typename enum_opt>
concept poly_opt = PolyOpt<enum_opt>::value;

template <typename enum_opt>
concept comptime_poly_opt = poly_opt<enum_opt> && OptInfo<enum_opt>::isdef;

template <typename enum_opt>
concept default_poly_opt = poly_opt<enum_opt> && !OptInfo<enum_opt>::isdef;

template <typename, auto...>
struct OptList {};

template <default_poly_opt enum_opt, auto enum_v, auto... enum_vs>
struct OptList<enum_opt, enum_v, enum_vs...> {
  using type = enum_opt;
  static constexpr auto value =
      std::array<enum_opt, 1 + sizeof...(enum_vs)>{enum_v, enum_vs...};
};

template <comptime_poly_opt enum_opt, auto enum_v, auto... enum_vs>
struct OptList<enum_opt, enum_v, enum_vs...> {
  using type = enum_opt;
  static constexpr auto value = OptInfo<enum_opt>::ParmList();
};

template <typename OL>
concept opt_list = requires {
  typename OL::type;
  OL::value;
};

template <opt_list... OLs>
struct OptTypeList {
  using type = TypeList<OLs...>;
};

template <typename opt, typename parm>
concept parm_opt =
    requires { opt::value; } && std::is_same_v<base_dtype<opt::value>, parm>;

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

#define POLYMORPHIC_PARM(name, ...)                                                      \
  enum class name { __VA_ARGS__ };                                                       \
  template <>                                                                            \
  struct OptInfo<name> : std::true_type {                                                \
    static std::string Label(const name &_parm) {                                        \
      return strings::split({#__VA_ARGS__}, ',')[static_cast<int>(_parm)];               \
    }                                                                                    \
    static constexpr bool isdef = is_defined(OPT_##name);                                \
    using type = name;                                                                   \
                                                                                         \
   private:                                                                              \
    template <std::size_t idx>                                                           \
    static constexpr name getVal_impl() {                                                \
      constexpr std::size_t n = strings::getLen(#__VA_ARGS__);                           \
      constexpr std::size_t nopts = strings::getLen(_getVal(OPT_##name));                \
      constexpr auto set_opts = strings::splitStrView<nopts>(_getVal(OPT_##name));       \
      constexpr auto label = set_opts[idx];                                              \
      constexpr auto labels = strings::splitStrView<n>(#__VA_ARGS__);                    \
      constexpr bool inList = strings::strInList(label, labels);                         \
      static_assert(inList || !isdef, _parm_msg(OPT_##name, #__VA_ARGS__));              \
      for (int i = 0; i < n; i++) {                                                      \
        if (label == labels[i]) return static_cast<name>(i);                             \
      }                                                                                  \
      return static_cast<name>(0);                                                       \
    }                                                                                    \
    template <std::size_t... Is>                                                         \
    static constexpr auto ParmList_impl(std::index_sequence<Is...>) {                    \
      return std::array<name, sizeof...(Is)>{getVal_impl<Is>()...};                      \
    }                                                                                    \
                                                                                         \
   public:                                                                               \
    static constexpr auto ParmList() {                                                   \
      return ParmList_impl(                                                              \
          std::make_index_sequence<strings::getLen(_getVal(OPT_##name))>());             \
    }                                                                                    \
  };                                                                                     \
  template <>                                                                            \
  struct PolyOpt<name> : std::true_type {                                                \
    template <name parm>                                                                 \
    struct opt {                                                                         \
      static constexpr name value = parm;                                                \
      static std::string Label() { return OptInfo<name>::Label(parm); }                  \
    };                                                                                   \
  };

} // namespace kamayan

#endif // DISPATCHER_OPTIONS_HPP_
