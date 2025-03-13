#ifndef PHYSICS_EOS_EQUATION_OF_STATE_HPP_
#define PHYSICS_EOS_EQUATION_OF_STATE_HPP_
#include <variant>

#include <singularity-eos/eos/eos.hpp>

#include "physics/eos/eos_singularity.hpp"
#include "physics/eos/eos_types.hpp"

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

  EquationOfState(const Real &gamma, const Real &Abar) {
    // singularity uses Gruneisen parameter & heat capacity (gamma * Kt / abar)
    // TODO(acreyes) : some kind of physical constants...
    // probably should be a struct with static constexpr...
    constexpr Real kboltz = 1.380649e-16;
    const Real cv = gamma * kboltz / Abar;
    eos_ = singularity::IdealGas(gamma - 1.0, cv);
  }

  template <EosComponent component, EosMode mode,
            template <typename...> typename Container, typename... Ts,
            typename Lambda = NullIndexer>
  requires(AccessorLike<Lambda>)
  KOKKOS_INLINE_FUNCTION Real Call(Container<Ts...> &indexer, Lambda lambda = Lambda()) {
    constexpr auto output = SingularityEosFill<mode>::output;
    using vars = EosVars<component>;
    Real cv;
    eos_.FillEos(indexer(DENS()), indexer(typename vars::temp()),
                 indexer(typename vars::eint()), indexer(typename vars::pres()), cv,
                 indexer(GAMC()), output, lambda);
    return cv;
  }

  auto nlambda() { return eos_.nlambda(); }

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

// export a std::variant of the possible allowed eos types that we can pull
// out of our Eos package params
using EOS_t = std::variant<std::monostate, EquationOfState<EosModel::gamma>>;

}  // namespace kamayan::eos
#endif  // PHYSICS_EOS_EQUATION_OF_STATE_HPP_

// EoS depends on multispecies and the model(s)
// singularity eos has you create a separate eos for the electrons & ions
// and calls them independently always with the ion mass density as the rho
// input parameter
