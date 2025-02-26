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

template <typename enum_opt>
struct PolyOpt : std::false_type {};

template <typename enum_opt>
struct OptInfo : std::false_type {};

// strategy here is that a user can declare a parameter "name" and enumerate the possible
// options for that parameter. These can then get used to define a TypeList with the
// OPT_LIST template below that used with the MAKE_POLYMORPHIC macro can instantiate all
// the possible template combinations for a given struct functor. Additionally if someone
// wishes to only instantiate a truncated list of options they can define at the level of
// cmake configuration "-DOPT_name=parm1,parm2". All this machinery gets baked into the
// opt_name<parm> struct specialized to the option that casts to int 0
#define POLYMORPHIC_PARM(name, ...)                                                      \
  enum class name { __VA_ARGS__ };                                                       \
  template <>                                                                            \
  struct OptInfo<name> : std::true_type {                                                \
    static constexpr name value = static_cast<name>(0);                                  \
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
      constexpr bool inList = strInList(label, labels);                                  \
      static_assert(inList || !isdef, _parm_msg(OPT_##name, #__VA_ARGS__));              \
      for (int i = 0; i < n; i++) {                                                      \
        if (label == labels[i]) return static_cast<name>(i);                             \
      }                                                                                  \
      return value;                                                                      \
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
  static constexpr auto value = OptInfo<enum_opt>::ParmList();
};

template <auto enum_v>
struct Opt_t {
  static constexpr auto value = enum_v;
};

template <typename, typename, typename>
struct PolymorphicDispatch {};

template <typename FUNCTOR, typename... KnownParms, typename enum_opt,
          auto... PARM_ENUM_Ts, typename... PARM_ENUM_TLs>
struct PolymorphicDispatch<
    FUNCTOR, TypeList<KnownParms...>,
    TypeList<OptList<enum_opt, PARM_ENUM_Ts...>, PARM_ENUM_TLs...>> {
  explicit PolymorphicDispatch(const std::string &source_) : source(source_) {}

  template <typename OUT, typename PARM, typename... ARGS>
  inline OUT execute(PARM parm, ARGS &&...args) {
    bool found_parm = false;
    if constexpr (std::is_same_v<void, OUT>) {
      (void)((
                 [&] {
                   if (parm == PARM_ENUM_Ts) {
                     found_parm = true;
                     if constexpr (sizeof...(PARM_ENUM_TLs) == 0) {
                       return FUNCTOR()
                           .template dispatch<KnownParms::value..., PARM_ENUM_Ts>(
                               std::forward<ARGS>(args)...);
                     } else {
                       return PolymorphicDispatch<
                                  FUNCTOR, TypeList<KnownParms..., Opt_t<PARM_ENUM_Ts>>,
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
             if (parm == PARM_ENUM_Ts) {
               found_parm = true;
               if constexpr (sizeof...(PARM_ENUM_TLs) == 0) {
                 output = FUNCTOR().template dispatch<KnownParms..., PARM_ENUM_Ts>(
                     std::forward<ARGS>(args)...);
               } else {
                 output =
                     PolymorphicDispatch<FUNCTOR,
                                         TypeList<KnownParms..., Opt_t<PARM_ENUM_Ts>>,
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
    using opt_info = OptInfo<enum_opt>;
    std::ostringstream msg;
    msg << "dispatch parm [" << opt_info::Label(parm) << "] not handled\n";
    msg << "Allowed options are: (";
    ([&] { msg << opt_info::Label(PARM_ENUM_Ts) << " "; }(), ...);
    msg << ")\n";
    msg << "from: " << source << "\n";
    PARTHENON_REQUIRE_THROWS(found_parm, msg.str().c_str());
    return OUT();
  }

 private:
  std::string source;
};

template <typename opt, typename parm>
concept parm_opt =
    requires { opt::value; } && std::is_same_v<base_dtype<opt::value>, parm>;

template <typename Functor, typename... Ts>
concept DispatchFunctor = requires(Functor func) {
  typename Functor::options; // should be an OPT_LIST
  typename Functor::value;
} && is_specialization<typename Functor::options, TypeList>::value;

template <typename Functor, typename... Ts>
  requires DispatchFunctor<Functor, Ts...>
struct Dispatcher_impl {
  using parm_list = Functor::options;
  using R_t = Functor::value;
  std::tuple<Ts...> runtime_values;
  const std::string label_;

  explicit Dispatcher_impl(const std::string &label, Ts... values)
      : label_(label), runtime_values(std::make_tuple(std::forward<Ts>(values)...)) {}

  template <std::size_t... Is, typename... Args>
  R_t execute_impl(std::index_sequence<Is...>, Args &&...args) {
    return PolymorphicDispatch<Functor, TypeList<>, parm_list>(label_)
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
  using type = Dispatcher_impl<Functor, typename OptLists::type...>;
};

template <typename Functor>
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options>::type;

} // namespace kamayan

#else
#endif
