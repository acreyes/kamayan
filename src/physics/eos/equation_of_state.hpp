#ifndef PHYSICS_EOS_EQUATION_OF_STATE_HPP_
#define PHYSICS_EOS_EQUATION_OF_STATE_HPP_

#include <singularity-eos/eos/eos.hpp>

#include "physics/eos/eos_singularity.hpp"
#include "physics/eos/eos_types.hpp"

namespace kamayan::eos {
template <eosModel>
struct EquationOfState {};

template <>
struct EquationOfState<eosModel::gamma> {
  using vars = TypeList<DENS, TEMP, EINT, PRES, GAMC, GAME>;
  static constexpr Kokkos::Array<eosMode, 3> modes{
      eosMode::ener,
      eosMode::pres,
      eosMode::temp,
  };

  EquationOfState(const Real &gamma, const Real &Abar) {
    // singularity uses Gruneisen parameter & heat capacity (gamma * Kt / abar)
    constexpr Real kboltz = 1.380649e-16;
    const Real cv = gamma * kboltz / Abar;
    eos_ = singularity::IdealGas(gamma - 1.0, cv);
  }

  template <eosMode mode, template <typename...> typename Container, typename... Ts,
            typename Lambda = NullIndexer>
  requires(AccessorLike<Lambda>)
  void Call(Container<Ts...> &indexer, Lambda lambda = Lambda()) {
    constexpr auto output = SingularityEosFill<mode>::output;
    Real cv;
    eos_.FillEos(indexer(DENS()), indexer(TEMP()), indexer(EINT()), indexer(PRES()), cv,
                 indexer(GAMC()), output, lambda);
    // gamc here is the bulk modulus = Cs**2 * rho
    // Cs**2 = P * gamc / rho = B / rho
    // rho * eint = P / (game - 1)
    indexer(GAMC()) /= indexer(PRES());
    indexer(GAME()) = 1.0 + indexer(PRES()) / indexer(DENS());
  }

  auto nlambda() { return eos_.nlambda(); }

 private:
  singularity::IdealGas eos_;
};
}  // namespace kamayan::eos
#endif  // PHYSICS_EOS_EQUATION_OF_STATE_HPP_

// EoS depends on multispecies and the model(s)
// singularity eos has you create a separate eos for the electrons & ions
// and calls them independently always with the ion mass density as the rho
// input parameter
