#ifndef DISPATCHER_DISPATCHER_HPP_
#define DISPATCHER_DISPATCHER_HPP_
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"
#include "kamayan/config.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

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

  template <typename OUT, typename... ARGS>
  inline OUT execute(Config *config, ARGS &&...args) {
    bool found_parm = false;
    auto parm = config->Get<enum_opt>();
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
                           .template execute<OUT>(config, std::forward<ARGS>(args)...);
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
                                    .template execute<OUT>(config,
                                                           std::forward<ARGS>(args)...);
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

  explicit Dispatcher_impl(const std::string &label, Ts... values)
      : label_(label), runtime_values(std::make_tuple(std::forward<Ts>(values)...)) {
    config_ = std::make_shared<Config>();
    (void)([&] { config_->Add<Ts>(values); }(), ...);
  }

  // another way to flip this would be to use the config to get the runtime values
  // and in the other overload construct a config from all the runtime values to be passed
  // in to PolymorphicDispatch::Functor
  // * added benefit would be to have some template types that are composites
  //   of multiple runtime options
  // * Also wouldn't need to specify the runtime options in order when calling dispatch
  // * need to deal with how to do all the runtime checks in order to make the type, but
  //   this is basically what we're doing now...
  Dispatcher_impl(const std::string &label, std::shared_ptr<Config> config)
      : label_(label), config_(config),
        runtime_values(std::make_tuple(config->Get<Ts>()...)) {}

  template <std::size_t... Is, typename... Args>
  R_t execute_impl(std::index_sequence<Is...>, Args &&...args) {
    return PolymorphicDispatch<Functor, TypeList<>, parm_list>(label_)
        .template execute<R_t>(config_.get(), std::forward<Args>(args)...);
  }

  // for some reason these lines are tickling the iwyu checks for
  // system headers that are being included... kill for now
  template <typename... Args>
  R_t execute(Args &&...args) {
    return execute_impl(std::make_index_sequence<sizeof...(Ts)>(),
                        std::forward<Args>(args)...);  // NOLINT
  }

  std::tuple<Ts...> runtime_values;  // NOLINT
  std::shared_ptr<Config> config_;
  const std::string label_;  // NOLINT
};

template <typename, typename>
struct Dispatcher_opts {};

template <typename Functor, typename... OptLists>
struct Dispatcher_opts<Functor, TypeList<OptLists...>> {
  using type = Dispatcher_impl<Functor, typename OptLists::type...>;
};

// interface to describe all the types of dispatchable functors
template <typename Functor>
concept DispatchFunctor =
    requires {
      typename Functor::options;  // enumerate all the possible template types
                                  // that can be dispatched
      typename Functor::value;    // return type of functor
    } && is_specialization<typename Functor::options, OptTypeList>::value &&
    (std::is_default_constructible_v<typename Functor::value> ||
     std::is_same_v<typename Functor::value, void>);

template <typename Functor>
requires(DispatchFunctor<Functor>)
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options::type>::type;

}  // namespace kamayan

#else
#endif  // DISPATCHER_DISPATCHER_HPP_
