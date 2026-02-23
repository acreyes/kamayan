#ifndef PHYSICS_MATERIAL_PROPERTIES_EOS_EQUATION_OF_STATE_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_EOS_EQUATION_OF_STATE_HPP_
#include <cmath>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>
#include <singularity-eos/eos/eos_ideal.hpp>

#include "kamayan/fields.hpp"
#include "kamayan/unit.hpp"
#include "physics/material_properties/eos/eos_singularity.hpp"
#include "physics/material_properties/eos/eos_types.hpp"
#include "physics/physics_types.hpp"
#include "ports-of-call/variant.hpp"
#include "singularity-eos/base/robust_utils.hpp"

namespace kamayan::eos {

template <typename T>
concept EquationOfStateImplementation = requires { T::modes; };

// idea is that we have one of these for each temperature component
// so in 1T you have just one EquationOfState saved to the eos
// package's Params
// in 3T you have one for both electrons & ions
template <EosModel>
struct EquationOfState {};

template <>
struct EquationOfState<EosModel::gamma> {
  static constexpr Kokkos::Array<EosMode, 3> modes{
      EosMode::ener,
      EosMode::pres,
      EosMode::temp,
  };

  EquationOfState() = default;

  EquationOfState(const Real &gamma, const Real &Abar) {
    // singularity uses Gruneisen parameter & heat capacity (gamma * Kt / abar)
    // TODO(acreyes) : some kind of physical constants...
    // probably should be a struct with static constexpr...
    constexpr Real kboltz = 1.380649e-16;
    const Real cv = gamma * kboltz / Abar;
    eos_ = singularity::IdealGas(gamma - 1.0, cv);
  }

  template <EosComponent component, EosMode mode, typename Container, typename... Ts,
            typename Lambda = NullIndexer>
  requires(AccessorLike<Lambda>)
  // requires(AccessorLike<Lambda>, IndexerLike<Container<Ts...>, Ts...>)
  KOKKOS_INLINE_FUNCTION Real Call(Container &indexer, Lambda lambda = Lambda()) const {
    constexpr auto output = SingularityEosFill<mode>::output;
    using vars = EosVars<component>;
    using eint = typename vars::eint;
    using temp = typename vars::temp;
    using pres = typename vars::pres;
    Real cv;
    eos_.FillEos(indexer(DENS()), indexer(typename vars::temp()),
                 indexer(typename vars::eint()), indexer(typename vars::pres()), cv,
                 indexer(BMOD()), output, lambda);
    return cv;
  }

  KOKKOS_INLINE_FUNCTION auto nlambda() const { return eos_.nlambda(); }

 private:
  singularity::IdealGas eos_;
};

// Fluid::oneT overload just calls eos and gets gamc/game
// Fluid::threeT would take two EOSs and call for ion/electrons separately
template <Fluid fluid, EosMode mode, typename EOS,
          template <typename...> typename Container, typename... Ts,
          typename Lambda = NullIndexer>
requires(EquationOfStateImplementation<EOS> && fluid == Fluid::oneT)
void EosCall(EOS eos, Container<Ts...> &indexer, Lambda lambda = Lambda()) {
  eos.template Call<EosComponent::oneT, mode>(indexer, lambda);
}

// need to use portable variant to work on GPU
using EosVariant = PortsOfCall::variant<EquationOfState<EosModel::gamma>>;

EosVariant MakeEosSingleSpecies(std::string spec, KamayanUnit *material);

class EOS_t {
 private:
  EosVariant eos_;

 public:
  template <typename EosChoice>
  explicit EOS_t(EosChoice eos) : eos_(eos) {}

  EOS_t() = default;

  template <typename EosChoice>
  EOS_t &operator=(EosChoice &&eos) {
    eos_ = std::move(std::forward<EosChoice>(eos));
    return *this;
  }

  template <typename T, EosComponent component, EosMode mode, typename Container,
            typename... Ts, typename Lambda = NullIndexer>
  KOKKOS_INLINE_FUNCTION Real CallAs(Container &indexer, Lambda lambda = Lambda()) const {
    return PortsOfCall::get<T>(eos_).template Call<component, mode>(indexer, lambda);
  }

  template <EosComponent component, EosMode mode, typename Container, typename... Ts,
            typename Lambda = NullIndexer>
  KOKKOS_INLINE_FUNCTION Real Call(Container &indexer, Lambda lambda = Lambda()) const {
    return PortsOfCall::visit(
        [&](auto &eos) { return eos.template Call<component, mode>(indexer, lambda); },
        eos_);
  }

  KOKKOS_INLINE_FUNCTION int nlambda() const {
    return PortsOfCall::visit([](const auto &eos) { return eos.nlambda(); }, eos_);
  }
};

}  // namespace kamayan::eos
#endif  // PHYSICS_MATERIAL_PROPERTIES_EOS_EQUATION_OF_STATE_HPP_

// EoS depends on multispecies and the model(s)
// singularity eos has you create a separate eos for the electrons & ions
// and calls them independently always with the ion mass density as the rho
// input parameter
