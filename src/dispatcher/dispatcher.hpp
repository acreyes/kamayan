#ifndef UNITS_DISPATCHER_HPP_
#define UNITS_DISPATCHER_HPP_
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"
#include "unit/config.hpp"
#include "utils/strings.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

// interface to describe all the types of dispatchable functors
template <typename Functor>
concept DispatchFunctor =
    requires {
      typename Functor::options; // enumerate all the possible template types that can be
                                 // dispatched
      typename Functor::value;   // return type of functor
    } && is_specialization<typename Functor::options, OptTypeList>::value &&
    (std::is_default_constructible_v<typename Functor::value> ||
     std::is_same_v<typename Functor::value, void>);

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
      (void)((
                 [&] {
                   if (parm == PARM_ENUM_Ts) {
                     found_parm = true;
                     if constexpr (sizeof...(PARM_ENUM_TLs) == 0) {
                       output =
                           FUNCTOR()
                               .template dispatch<KnownParms::value..., PARM_ENUM_Ts>(
                                   std::forward<ARGS>(args)...);
                     } else {
                       output = PolymorphicDispatch<
                                    FUNCTOR, TypeList<KnownParms..., Opt_t<PARM_ENUM_Ts>>,
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

  Dispatcher_impl(const std::string &label, Config *config)
      : label_(label), runtime_values(std::make_tuple(config->Get<Ts>()...)) {}

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
