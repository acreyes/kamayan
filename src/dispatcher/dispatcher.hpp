#ifndef UNITS_DISPATCHER_HPP_
#define UNITS_DISPATCHER_HPP_
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"
#include "utils/strings.hpp"
#include "utils/type_abstractions.hpp"
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

// interface to describe all the types of dispatchable functors
template <typename Functor>
concept DispatchFunctor = requires {
  typename Functor::options; // enumerate all the possible template types that can be
                             // dispatched
  typename Functor::value;   // return type of functor
} && is_specialization<typename Functor::options, OptTypeList>::value;

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

// helper struct to keep track of all our discovered enum options
// inside of a TypeList
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

template <typename Functor, typename... Ts>
struct Dispatcher_impl {
  using parm_list = Functor::options::type;
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

template <DispatchFunctor Functor, typename... OptLists>
struct Dispatcher_opts<Functor, TypeList<OptLists...>> {
  using type = Dispatcher_impl<Functor, typename OptLists::type...>;
};

template <DispatchFunctor Functor>
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options::type>::type;

} // namespace kamayan

#else
#endif
