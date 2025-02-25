#ifndef UNITS_DISPATCHER_HPP_
#define UNITS_DISPATCHER_HPP_
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "utils/strings.hpp"
#include "utils/type_list.hpp"

// the following is some magic taken from
// https://stackoverflow.com/questions/18048039/c-constexpr-function-to-test-preprocessor-macros
// this will test the stringified-version of a requested macro against the evaluated macro
// as a constexpr check on whether it exists or not
constexpr bool _is_defined(const char s1[], const char s2[]) {
  return std::string_view(s1) != s2;
}

template <std::size_t N>
constexpr bool strInList(std::string_view s, std::array<std::string_view, N> sArr) {
  for (const auto &tst : sArr) {
    if (s == tst) return true;
  }
  return false;
}

// here are some macros that help us expand out our compound macros taken from the
// POLYMORPHIC_PARM macro
#define _eval(...) #__VA_ARGS__
#define _getVal(x) _eval(x)
#define is_defined(x) _is_defined(#x, _eval(x))
#define _parm_msg(x, y)                                                                  \
  "unrecognized option in " #x " = " _eval(x) " valid values are: " y

namespace kamayan {

// this lets us use a requires expression rather than std::enable_if_t<opt1 == opt2, void>
// for our SFINAE
template <auto opt1, auto opt2>
struct is_opt_impl {
  static_assert(std::is_same_v<decltype(opt1), decltype(opt2)>,
                "opts must be the same type");
  using value = std::conditional_t<opt1 == opt2, std::true_type, std::false_type>;
};

template <auto opt1, auto opt2>
using is_opt = is_opt_impl<opt1, opt2>::value;

// strategy here is that a user can declare a parameter "name" and enumerate the possible
// options for that parameter. These can then get used to define a TypeList with the
// OPT_LIST template below that used with the MAKE_POLYMORPHIC macro can instantiate all
// the possible template combinations for a given struct functor. Additionally if someone
// wishes to only instantiate a truncated list of options they can define at the level of
// cmake configuration "-DOPT_name=parm1,parm2". All this machinery gets baked into the
// opt_name<parm> struct specialized to the option that casts to int 0
#define POLYMORPHIC_PARM(name, ...)                                                      \
  enum class name { __VA_ARGS__ };                                                       \
  template <name parm>                                                                   \
  struct opt_##name {                                                                    \
    static constexpr name value = parm;                                                  \
    static std::string Label(const name &_parm) {                                        \
      return strings::split({#__VA_ARGS__}, ',')[static_cast<int>(_parm)];               \
    }                                                                                    \
    static std::string Label() { return Label(parm); }                                   \
  };                                                                                     \
  template <>                                                                            \
  struct opt_##name<static_cast<name>(0)> {                                              \
    static constexpr name value = static_cast<name>(0);                                  \
    static std::string Label(const name &_parm) {                                        \
      return strings::split({#__VA_ARGS__}, ',')[static_cast<int>(_parm)];               \
    }                                                                                    \
    static std::string Label() { return Label(value); }                                  \
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
      constexpr bool inList = strInList(label, labels);                                  \
      static_assert(inList || !isdef, _parm_msg(OPT_##name, #__VA_ARGS__));              \
      for (int i = 0; i < n; i++) {                                                      \
        if (label == labels[i]) return static_cast<name>(i);                             \
      }                                                                                  \
      return value;                                                                      \
    }                                                                                    \
    template <std::size_t... Is>                                                         \
    static constexpr auto ParmList_impl(std::index_sequence<Is...>) {                    \
      return TypeList<opt_##name<getVal_impl<Is>()>...>();                               \
    }                                                                                    \
                                                                                         \
   public:                                                                               \
    static constexpr auto ParmList() {                                                   \
      return ParmList_impl(                                                              \
          std::make_index_sequence<strings::getLen(_getVal(OPT_##name))>());             \
    }                                                                                    \
  };

// assert that a template parameter is an opt_parm
#define PARM_ASSERT(parm, opt) static_assert(std::is_same_v<base_dtype<parm::value>, opt>)

template <template <auto> typename t_opt, auto enum_t, auto... enum_ts>
using OPT_LIST =
    std::conditional_t<t_opt<static_cast<decltype(enum_t)>(0)>::isdef,
                       decltype(t_opt<static_cast<decltype(enum_t)>(0)>::ParmList()),
                       TypeList<t_opt<enum_t>, t_opt<enum_ts>...>>;

template <typename, typename, typename>
struct PolymorphicDispatch {};

template <typename FUNCTOR, typename... KnownParms, typename... PARM_ENUM_Ts,
          typename... PARM_ENUM_TLs>
struct PolymorphicDispatch<FUNCTOR, TypeList<KnownParms...>,
                           TypeList<TypeList<PARM_ENUM_Ts...>, PARM_ENUM_TLs...>> {
  explicit PolymorphicDispatch(const std::string &source_) : source(source_) {}

  template <typename OUT, typename PARM, typename... ARGS>
  inline OUT execute(PARM parm, ARGS &&...args) {
    bool found_parm = false;
    if constexpr (std::is_same_v<void, OUT>) {
      (void)((
                 [&] {
                   if (parm == PARM_ENUM_Ts::value) {
                     found_parm = true;
                     if constexpr (sizeof...(PARM_ENUM_TLs) == 0) {
                       return FUNCTOR().template dispatch<KnownParms..., PARM_ENUM_Ts>(
                           std::forward<ARGS>(args)...);
                     } else {
                       return PolymorphicDispatch<FUNCTOR,
                                                  TypeList<KnownParms..., PARM_ENUM_Ts>,
                                                  TypeList<PARM_ENUM_TLs...>>(source)
                           .template execute<OUT>(std::forward<ARGS>(args)...);
                     }
                   }
                   return;
                 }(),
                 !found_parm) &&
             ...);
      return;
    } else {
      OUT output;
      ((
           [&] {
             if (parm == PARM_ENUM_Ts::value) {
               found_parm = true;
               if constexpr (sizeof...(PARM_ENUM_TLs) == 0) {
                 output = FUNCTOR().template dispatch<KnownParms..., PARM_ENUM_Ts>(
                     std::forward<ARGS>(args)...);
               } else {
                 output =
                     PolymorphicDispatch<FUNCTOR, TypeList<KnownParms..., PARM_ENUM_Ts>,
                                         TypeList<PARM_ENUM_TLs...>>(source)
                         .template execute<OUT>(std::forward<ARGS>(args)...);
               }
             }
             return;
           }(),
           !found_parm) &&
       ...);
      if (found_parm) return output;
    }
    using OPT_PARM = typename TypeList<PARM_ENUM_Ts...>::template type<0>;
    std::ostringstream msg;
    msg << "dispatch parm [" << OPT_PARM::Label(parm) << "] not handled\n";
    msg << "Allowed options are: (";
    ([&] { msg << OPT_PARM::Label(PARM_ENUM_Ts::value) << " "; }(), ...);
    msg << ")\n";
    msg << "from: " << source << "\n";
    PARTHENON_REQUIRE_THROWS(found_parm, msg.str().c_str());
    return OUT();
  }

 private:
  std::string source;
};

template <typename Functor, typename... Ts>
struct Dispatcher_impl {
  using parm_list = Functor::options;
  using R_t = Functor::value;
  std::tuple<Ts...> runtime_values;
  const std::string source;

  explicit Dispatcher_impl(Ts... values)
      : runtime_values(std::make_tuple(std::forward<Ts>(values)...)) {}

  template <std::size_t... Is, typename... Args>
  R_t execute_impl(std::index_sequence<Is...>, Args &&...args) {
    return PolymorphicDispatch<Functor, TypeList<>, parm_list>("dummy source")
        .template execute<R_t>(std::get<Is>(runtime_values)...,
                               std::forward<Args>(args)...);
  }

  template <typename... Args>
  R_t execute(Args &&...args) {
    return execute_impl(std::make_index_sequence<sizeof...(Ts)>(),
                        std::forward<Args>(args)...);
  }
};

template <typename, typename>
struct Dispatcher_opts {};

template <typename Functor, typename... OptLists>
struct Dispatcher_opts<Functor, TypeList<OptLists...>> {
  using type = Dispatcher_impl<Functor, decltype(OptLists::template type<0>::value)...>;
};

template <typename Functor>
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options>::type;

} // namespace kamayan

#else
#endif
