#ifndef DISPATCHER_DISPATCHER_HPP_
#define DISPATCHER_DISPATCHER_HPP_
#include <memory>
#include <string>
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
template <typename Functor, typename... KnownParms>
struct PolymorphicDispatch<Functor, TypeList<KnownParms...>, TypeList<>> {
  explicit PolymorphicDispatch(const std::string &source_) {}
  using SplitCompositeEnumOpts =
      SplitTypeList<CountCompositeOpts<KnownParms...>::num(), TypeList<KnownParms...>>;

  template <typename Out, typename... Args>
  inline Out execute(const Config *config, Args &&...args) {
    return execute_impl<
        Out, typename SplitCompositeEnumOpts::first,
        typename SplitCompositeEnumOpts::second>::execute(std::forward<Args>(args)...);
  }

 private:
  template <typename, typename, typename>
  struct execute_impl {};

  template <typename Out, typename... CompositeOpts, typename... EnumOpts>
  struct execute_impl<Out, TypeList<CompositeOpts...>, TypeList<EnumOpts...>> {
    template <typename... Args>
    static Out execute(Args &&...args) {
      return Functor()
          .template dispatch<typename CompositeOpts::value..., EnumOpts::value...>(
              std::forward<Args>(args)...);
    }
  };
};

template <typename Functor, typename... KnownParms, typename EnumOpt, auto... enum_values,
          typename... NextOptions>
struct PolymorphicDispatch<Functor, TypeList<KnownParms...>,
                           TypeList<OptList<EnumOpt, enum_values...>, NextOptions...>> {
  explicit PolymorphicDispatch(const std::string &source_) : source(source_) {}

  template <typename Out, typename... Args>
  inline Out execute(const Config *config, Args &&...args) {
    bool found_parm = false;
    auto parm = config->Get<EnumOpt>();
    if constexpr (std::is_same_v<void, Out>) {
      (void)((
                 [&] {
                   if (parm == enum_values) {
                     found_parm = true;
                     return PolymorphicDispatch<
                                Functor, TypeList<KnownParms..., Opt_t<enum_values>>,
                                TypeList<NextOptions...>>(source)
                         .template execute<Out>(config, std::forward<Args>(args)...);
                   }
                   return;
                 }(),
                 !found_parm) &&
             ...);
      if (found_parm) return;
    } else {
      Out output;
      (void)((
                 [&] {
                   if (parm == enum_values) {
                     found_parm = true;
                     output =
                         PolymorphicDispatch<Functor,
                                             TypeList<KnownParms..., Opt_t<enum_values>>,
                                             TypeList<NextOptions...>>(source)
                             .template execute<Out>(config, std::forward<Args>(args)...);
                   }
                   return;
                 }(),
                 !found_parm) &&
             ...);
      if (found_parm) return output;
    }
    using opt_info = OptInfo<EnumOpt>;
    std::ostringstream msg;
    msg << "dispatch parm [" << opt_info::Label(parm) << "] not handled\n";
    msg << "Allowed options are: (";
    ([&] { msg << opt_info::Label(enum_values) << " "; }(), ...);
    msg << ")\n";
    msg << "from: " << source << "\n";
    PARTHENON_REQUIRE_THROWS(found_parm, msg.str().c_str());
    return Out();
  }

 private:
  std::string source;
};
template <typename Functor, typename... KnownParms, typename Factory,
          typename... NextOptions>
requires(FactoryOption<Factory>)
struct PolymorphicDispatch<Functor, TypeList<KnownParms...>,
                           TypeList<Factory, NextOptions...>> {
  explicit PolymorphicDispatch(const std::string &source_) : source(source_) {}

  template <typename Out, typename... Args>
  inline Out execute(const Config *config, Args &&...args) {
    return BuildCompositeOption<TypeList<>, typename Factory::options::type>()
        .template execute<Out>(config, source, std::forward<Args>(args)...);
  }

 private:
  template <typename, typename>
  struct BuildCompositeOption {};

  template <typename... KnownOpts>
  struct BuildCompositeOption<TypeList<KnownOpts...>, TypeList<>> {
    template <typename Out, typename... Args>
    Out execute(const Config *config, const std::string &source, Args &&...args) {
      return PolymorphicDispatch<
                 Functor,
                 TypeList<KnownParms...,
                          CompositeOpt_t<
                              typename Factory::template composite<KnownOpts::value...>>>,
                 TypeList<NextOptions...>>(source)
          .template execute<Out>(config, std::forward<Args>(args)...);
    }
  };

  template <typename... KnownOpts, typename EnumOpt, auto... enum_values,
            typename... NextOptLists>
  struct BuildCompositeOption<
      TypeList<KnownOpts...>,
      TypeList<OptList<EnumOpt, enum_values...>, NextOptLists...>> {
    template <typename Out, typename... Args>
    Out execute(const Config *config, const std::string &source, Args &&...args) {
      bool found_parm = false;
      auto parm = config->Get<EnumOpt>();
      if constexpr (std::is_same_v<void, Out>) {
        (void)((
                   [&] {
                     if (parm == enum_values) {
                       found_parm = true;
                       return BuildCompositeOption<
                                  TypeList<KnownOpts..., Opt_t<enum_values>>,
                                  TypeList<NextOptLists...>>()
                           .template execute<Out>(config, source,
                                                  std::forward<Args>(args)...);
                     }
                   }(),
                   !found_parm) &&
               ...);
        if (found_parm) return;
      } else {
        Out output;
        (void)((
                   [&] {
                     if (parm == enum_values) {
                       found_parm = true;
                       output = BuildCompositeOption<
                                    TypeList<KnownOpts..., Opt_t<enum_values>>,
                                    TypeList<NextOptLists...>>()
                                    .template execute<Out>(config, source,
                                                           std::forward<Args>(args)...);
                     }
                   }(),
                   !found_parm) &&
               ...);
        if (found_parm) return output;
      }
      using opt_info = OptInfo<EnumOpt>;
      std::ostringstream msg;
      msg << "dispatch parm [" << opt_info::Label(parm) << "] not handled\n";
      msg << "Allowed options are: (";
      ([&] { msg << opt_info::Label(enum_values) << " "; }(), ...);
      msg << ")\n";
      msg << "from: " << source << "\n";
      PARTHENON_REQUIRE_THROWS(found_parm, msg.str().c_str());
      return Out();
    }
  };

 private:
  std::string source;
};

template <typename Functor, typename... Ts>
struct Dispatcher_impl {
  using parm_list = Functor::options::type;
  using Out = Functor::value;

  template <typename... Args>
  explicit Dispatcher_impl(const std::string &label, Args... values) : label_(label) {
    sconfig_ = std::make_shared<Config>();
    (void)([&] { sconfig_->Add<Args>(values); }(), ...);
    config_ = sconfig_.get();
  }

  Dispatcher_impl(const std::string &label, const Config *config)
      : label_(label), config_(config) {}

  Dispatcher_impl(const std::string &label, Config *config)
      : label_(label), config_(config) {}

  template <typename... Args>
  Out execute_impl(Args &&...args) {
    return PolymorphicDispatch<Functor, TypeList<>, parm_list>(label_)
        .template execute<Out>(config_, std::forward<Args>(args)...);
  }

  // for some reason these lines are tickling the iwyu checks for
  // system headers that are being included... kill for now
  template <typename... Args>
  Out execute(Args &&...args) {
    return execute_impl(std::forward<Args>(args)...);  // NOLINT
  }

  // DEV(acreyes): I have no idea why cpplint can't figure out
  // that I am #include'ing these...
  std::shared_ptr<Config> sconfig_;  // NOLINT
  const Config *config_;             // NOLINT
  const std::string label_;          // NOLINT
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
  requires(TemplateSpecialization<typename Functor::options, OptTypeList>);
  requires(std::is_default_constructible_v<typename Functor::value> ||
           std::is_same_v<typename Functor::value, void>);
};

template <typename Functor>
requires(DispatchFunctor<Functor>)
using Dispatcher = Dispatcher_opts<Functor, typename Functor::options::type>::type;

}  // namespace kamayan

#else
#endif  // DISPATCHER_DISPATCHER_HPP_
