#ifndef PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
#define PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
#include <limits>

#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "hydro_types.hpp"
#include "kamayan/fields.hpp"
#include "primconsflux.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::hydro {

template <TopologicalElement face, RiemannSolver riemann, typename hydro_traits,
          typename FluxIndexer, typename Scratch>
KOKKOS_INLINE_FUNCTION void RiemannFlux(FluxIndexer &pack, const Scratch &vL,
                                        const Scratch &vR) {}

template <TopologicalElement face, RiemannSolver riemann, typename hydro_traits,
          typename FluxIndexer, typename Scratch>
requires(riemann == RiemannSolver::hll)
KOKKOS_INLINE_FUNCTION void RiemannFlux(FluxIndexer &pack, const Scratch &vL,
                                        const Scratch &vR) {
  constexpr std::size_t dir1 = static_cast<std::size_t>(face) % 3;
  constexpr std::size_t dir2 = (dir1 + 1) % 3;
  constexpr std::size_t dir3 = (dir1 + 2) % 3;

  const Real cfL = FastSpeed<hydro_traits::MHD>(dir1, vL);
  const Real cfR = FastSpeed<hydro_traits::MHD>(dir1, vR);

  const Real tiny = std::numeric_limits<Real>::min();
  const Real sL =
      Kokkos::min(-tiny, Kokkos::min(vL(VELOCITY(dir1)) - cfL, vR(VELOCITY(dir1)) - cfR));
  const Real sR =
      Kokkos::max(tiny, Kokkos::max(vL(VELOCITY(dir1)) + cfL, vR(VELOCITY(dir1)) + cfR));
  const Real sRmsLi = 1.0 / (sR - sL);

  using Array_t = TypeListArray<typename hydro_traits::Conserved>;
  Array_t UL, UR, FL, FR;
  // --8<-- [start:tl-arr]
  Prim2Cons<hydro_traits>(vL, UL);
  Prim2Cons<hydro_traits>(vR, UR);
  Prim2Flux<dir1, hydro_traits>(vR, FR);
  Prim2Flux<dir1, hydro_traits>(vL, FL);
  // --8<-- [end:tl-arr]

  // --8<-- [start:type_for]
  type_for(typename hydro_traits::Conserved(), [&]<typename Vars>(const Vars &) {
    for (int comp = 0; comp < pack.GetSize(Vars()); comp++) {
      auto var = Vars(comp);
      pack.flux(face, var) =
          sRmsLi * (sR * FL(var) - sL * FR(var) + sR * sL * (UR(var) - UL(var)));
    }
  });
  // --8<-- [end:type_for]
}

template <TopologicalElement face, RiemannSolver riemann, typename hydro_traits,
          typename FluxIndexer, typename Scratch>
requires(riemann == RiemannSolver::hllc)
KOKKOS_INLINE_FUNCTION void RiemannFlux(FluxIndexer &pack, const Scratch &vL,
                                        const Scratch &vR) {
  constexpr std::size_t dir1 = static_cast<std::size_t>(face) % 3;
  constexpr std::size_t dir2 = (dir1 + 1) % 3;
  constexpr std::size_t dir3 = (dir1 + 2) % 3;

  const Real cfL = FastSpeed<hydro_traits::MHD>(dir1, vL);
  const Real cfR = FastSpeed<hydro_traits::MHD>(dir1, vR);

  const Real tiny = std::numeric_limits<Real>::min();
  const Real sL =
      Kokkos::min(-tiny, Kokkos::min(vL(VELOCITY(dir1)) - cfL, vR(VELOCITY(dir1)) - cfR));
  const Real sR =
      Kokkos::max(tiny, Kokkos::max(vL(VELOCITY(dir1)) + cfL, vR(VELOCITY(dir1)) + cfR));
  const Real sRmsLi = 1.0 / (sR - sL);

  using Conserved = typename hydro_traits::Conserved;
  using Array_t = TypeListArray<Conserved>;
  Array_t UL, UR, FL, FR;

  Prim2Cons<hydro_traits>(vL, UL);
  Prim2Cons<hydro_traits>(vR, UR);
  Prim2Flux<dir1, hydro_traits>(vR, FR);
  Prim2Flux<dir1, hydro_traits>(vL, FL);

  const Real total_presL = TotalPres<hydro_traits::MHD>(vL);
  const Real total_presR = TotalPres<hydro_traits::MHD>(vR);

  Real ustar = total_presR - total_presL +
               UL(MOMENTUM(dir1)) * (sL - vL(VELOCITY(dir1))) -
               UR(MOMENTUM(dir1)) * (sR - vR(VELOCITY(dir1)));
  ustar =
      ustar * 1. /
      (vL(DENS()) * (sL - vL(VELOCITY(dir1))) - vR(DENS()) * (sR - vR(VELOCITY(dir1))));

  Real pstar =
      0.5 * (total_presL + total_presR +
             vL(DENS()) * (sL - vL(VELOCITY(dir1))) * (ustar - vL(VELOCITY(dir1))) +
             vR(DENS()) * (sR - vR(VELOCITY(dir1))) * (ustar - vR(VELOCITY(dir1))));

  const auto hllc_state = [&](const Real &S, const Array_t &U, const Array_t &F) {
    Array_t Ustar;
    const Real susi = 1. / (S - ustar + tiny);
    Ustar(DENS()) = (S * U(DENS()) - F(DENS())) * susi;
    Ustar(MOMENTUM(dir1)) = (S * U(MOMENTUM(dir1)) - F(MOMENTUM(dir1)) + pstar) * susi;
    Ustar(MOMENTUM(dir2)) = (S * U(MOMENTUM(dir2)) - F(MOMENTUM(dir2))) * susi;
    Ustar(MOMENTUM(dir3)) = (S * U(MOMENTUM(dir3)) - F(MOMENTUM(dir3))) * susi;
    Ustar(ENER()) = (S * U(ENER()) - F(ENER()) + pstar * ustar) * susi;

    if constexpr (hydro_traits::MHD != Mhd::off) {
      const Real sRsLi = 1. / (sR - sL);
      const auto hll_state = [&]<typename Var>(const Var &var) {
        return sRsLi * (sR * UR(var) - sL * UL(var) + FL(var) - FR(var));
      };  // NOLINT
      Ustar(MAGC(dir1)) = hll_state(MAGC(dir1));
      Ustar(MAGC(dir2)) = hll_state(MAGC(dir2));
      Ustar(MAGC(dir3)) = hll_state(MAGC(dir3));
      Ustar(MOMENTUM(dir2)) -=
          (Ustar(MAGC(dir1)) * Ustar(MAGC(dir2))) - U(MAGC(dir1)) * U(MAGC(dir2)) * susi;
      Ustar(MOMENTUM(dir3)) -=
          (Ustar(MAGC(dir1)) * Ustar(MAGC(dir3))) - U(MAGC(dir1)) * U(MAGC(dir3)) * susi;
      Ustar(ENER()) -= susi * Ustar(MAGC(dir1)) *
                       (Ustar(MAGC(dir1)) * hll_state(MOMENTUM(dir1)) +
                        Ustar(MAGC(dir2)) * hll_state(MOMENTUM(dir2)) +
                        Ustar(MAGC(dir3)) * hll_state(MOMENTUM(dir3))) /
                       hll_state(DENS());
    }
    return Ustar;
  };

  const auto UstarL = hllc_state(sL, UL, FL);
  const auto UstarR = hllc_state(sR, UR, FR);
  const Real biasL = -Kokkos::min(-tiny, Kokkos::copysign(1., ustar));
  const Real biasR = Kokkos::max(tiny, Kokkos::copysign(1., ustar));

  type_for(Conserved(), [&]<typename Vars>(const Vars &) {
    for (int comp = 0; comp < pack.GetSize(Vars()); comp++) {
      const auto var = Vars(comp);
      pack.flux(face, var) = biasR * (FL(var) + sL * (UstarL(var) - UL(var))) +
                             biasL * (FR(var) + sR * (UstarR(var) - UR(var)));
    }
  });
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
