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
// inside of a TypeList. is_type lets us know if it is a composite type or not
template <auto enum_v>
struct Opt_t {
  static constexpr auto value = enum_v;
  static constexpr bool is_type = false;
};

template <typename T>
struct CompositeOpt_t {
  using value = T;
  static constexpr bool is_type = true;
};

template <typename... Opt_ts>
struct CountCompositeOpts {
  static constexpr std::size_t num() { return 0; }
};

template <typename T, typename... Opt_ts>
struct CountCompositeOpts<T, Opt_ts...> {
  static constexpr std::size_t num() {
    if constexpr (T::is_type) {
      return 1 + CountCompositeOpts<Opt_ts...>::num();
    }
    return 0;
  }
};

template <typename, typename, typename>
struct PolymorphicDispatch {};

// Once all the runtime values have been matched to compile time values
// we can call the actual functor
template <typename FUNCTOR, typename... KnownParms>
struct PolymorphicDispatch<FUNCTOR, TypeList<KnownParms...>, TypeList<>> {
  explicit PolymorphicDispatch(const std::string &source_) {}
  using SplitCompositeEnumOpts =
      SplitTypeList<CountCompositeOpts<KnownParms...>::num(), TypeList<KnownParms...>>;

  template <typename OUT, typename... ARGS>
  inline OUT execute(Config *config, ARGS &&...args) {
    return execute_impl<
        OUT, typename SplitCompositeEnumOpts::first,
        typename SplitCompositeEnumOpts::second>::execute(std::forward<ARGS>(args)...);
  }

 private:
  template <typename, typename, typename>
  struct execute_impl {};

  template <typename OUT, typename... CompositeOpts, typename... EnumOpts>
  struct execute_impl<OUT, TypeList<CompositeOpts...>, TypeList<EnumOpts...>> {
    template <typename... Args>
    static OUT execute(Args &&...args) {
      return FUNCTOR()
          .template dispatch<typename CompositeOpts::value..., EnumOpts::value...>(
              std::forward<Args>(args)...);
    }
  };
};

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
                     return PolymorphicDispatch<
                                FUNCTOR, TypeList<KnownParms..., Opt_t<PARM_ENUM_Ts>>,
                                TypeList<PARM_ENUM_TLs...>>(source)
                         .template execute<OUT>(config, std::forward<ARGS>(args)...);
                   }
                   return;
                 }(),
                 !found_parm) &&
             ...);
      if (found_parm) return;
    } else {
      OUT output;
      (void)((
                 [&] {
                   if (parm == PARM_ENUM_Ts) {
                     found_parm = true;
                     output =
                         PolymorphicDispatch<FUNCTOR,
                                             TypeList<KnownParms..., Opt_t<PARM_ENUM_Ts>>,
                                             TypeList<PARM_ENUM_TLs...>>(source)
                             .template execute<OUT>(config, std::forward<ARGS>(args)...);
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
template <typename FUNCTOR, typename... KnownParms, typename Factory,
          typename... PARM_ENUM_TLs>
requires(FactoryOption<Factory>)
struct PolymorphicDispatch<FUNCTOR, TypeList<KnownParms...>,
                           TypeList<Factory, PARM_ENUM_TLs...>> {
  explicit PolymorphicDispatch(const std::string &source_) : source(source_) {}

  template <typename OUT, typename... Args>
  inline OUT execute(Config *config, Args &&...args) {
    return BuildCompositeOption<TypeList<>, typename Factory::options::type>()
        .template execute<OUT>(config, source, std::forward<Args>(args)...);
  }

 private:
  template <typename, typename>
  struct BuildCompositeOption {};

  template <typename... KnownOpts>
  struct BuildCompositeOption<TypeList<KnownOpts...>, TypeList<>> {
    template <typename OUT, typename... Args>
    OUT execute(Config *config, const std::string &source, Args &&...args) {
      return PolymorphicDispatch<
                 FUNCTOR,
                 TypeList<KnownParms...,
                          CompositeOpt_t<
                              typename Factory::template composite<KnownOpts::value...>>>,
                 TypeList<PARM_ENUM_TLs...>>(source)
          .template execute<OUT>(config, std::forward<Args>(args)...);
    }
  };

  template <typename... KnownOpts, typename enum_opt, auto... runtime_values,
            typename... NextOptLists>
  struct BuildCompositeOption<
      TypeList<KnownOpts...>,
      TypeList<OptList<enum_opt, runtime_values...>, NextOptLists...>> {
    template <typename OUT, typename... Args>
    OUT execute(Config *config, const std::string &source, Args &&...args) {
      bool found_parm = false;
      auto parm = config->Get<enum_opt>();
      if constexpr (std::is_same_v<void, OUT>) {
        (void)((
                   [&] {
                     if (parm == runtime_values) {
                       found_parm = true;
                       return BuildCompositeOption<
                                  TypeList<KnownOpts..., Opt_t<runtime_values>>,
                                  TypeList<NextOptLists...>>()
                           .template execute<OUT>(config, source,
                                                  std::forward<Args>(args)...);
                     }
                   }(),
                   !found_parm) &&
               ...);
        if (found_parm) return;
      } else {
        OUT output;
        (void)((
                   [&] {
                     if (parm == runtime_values) {
                       found_parm = true;
                       output = BuildCompositeOption<
                                    TypeList<KnownOpts..., Opt_t<runtime_values>>,
                                    TypeList<NextOptLists...>>()
                                    .template execute<OUT>(config, source,
                                                           std::forward<Args>(args)...);
                     }
                   }(),
                   !found_parm) &&
               ...);
        if (found_parm) return output;
      }
      using opt_info = OptInfo<enum_opt>;
      std::ostringstream msg;
      msg << "dispatch parm [" << opt_info::Label(parm) << "] not handled\n";
      msg << "Allowed options are: (";
      ([&] { msg << opt_info::Label(runtime_values) << " "; }(), ...);
      msg << ")\n";
      msg << "from: " << source << "\n";
      PARTHENON_REQUIRE_THROWS(found_parm, msg.str().c_str());
      return OUT();
    }
  };

 private:
  std::string source;
};

template <typename Functor, typename... Ts>
struct Dispatcher_impl {
  using parm_list = Functor::options::type;
  using R_t = Functor::value;

  template <typename... Args>
  explicit Dispatcher_impl(const std::string &label, Args... values) : label_(label) {
    config_ = std::make_shared<Config>();
    (void)([&] { config_->Add<Args>(values); }(), ...);
  }

  // another way to flip this would be to use the config to get the runtime values
  // and in the other overload construct a config from all the runtime values to be
  // passed in to PolymorphicDispatch::Functor
  // * added benefit would be to have some template types that are composites
  //   of multiple runtime options
  // * Also wouldn't need to specify the runtime options in order when calling dispatch
  // * need to deal with how to do all the runtime checks in order to make the type, but
  //   this is basically what we're doing now...
  Dispatcher_impl(const std::string &label, std::shared_ptr<Config> config)
      : label_(label), config_(config) {}

  template <typename... Args>
  R_t execute_impl(Args &&...args) {
    return PolymorphicDispatch<Functor, TypeList<>, parm_list>(label_)
        .template execute<R_t>(config_.get(), std::forward<Args>(args)...);
  }

  // for some reason these lines are tickling the iwyu checks for
  // system headers that are being included... kill for now
  template <typename... Args>
  R_t execute(Args &&...args) {
    return execute_impl(std::forward<Args>(args)...);  // NOLINT
  }

  // DEV(acreyes): I have no idea why cpplint can't figure out
  // that I am #include'ing these...
  std::shared_ptr<Config> config_;  // NOLINT
  const std::string label_;         // NOLINT
};

template <typename, typename>
struct Dispatcher_opts {};

template <typename Functor, typename... OptLists>
struct Dispatcher_opts<Functor, TypeList<OptLists...>> {
  using type = Dispatcher_impl<Functor, typename OptLists::type...>;
};

// interface to describe all the types of dispatchable functors
template <typename Functor>
concept DispatchFunctor = requires {
  typename Functor::options;  // enumerate all the possible template types
                              // that can be dispatched
  typename Functor::value;    // return type of functor
  requires is_specialization<typename Functor::options, OptTypeList>::value;
  requires(std::is_default_constructible_v<typename Functor::value> ||
           std::is_same_v<typename Functor::value, void>);
};

template <typename Functor>
requires(DispatchFunctor<Functor>)
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options::type>::type;

}  // namespace kamayan

#else
#endif  // DISPATCHER_DISPATCHER_HPP_
