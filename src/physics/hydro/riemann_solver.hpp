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

  const Real ustarL = Kokkos::min(-tiny, ustar);
  const Real ustarR = Kokkos::max(tiny, ustar);

  auto Dstar = Array_t(0.);
  Dstar(MOMENTUM(dir1)) = 1.;
  Dstar(ENER()) = ustar;

  const Real sLusi = 1. / (sL - ustar);
  const Real sRusi = 1. / (sR - ustar);
  type_for(Conserved(), [&]<typename Vars>(const Vars &) {
    for (int comp = 0; comp < pack.GetSize(Vars()); comp++) {
      auto var = Vars(comp);
      pack.flux(face, var) = sLusi * (ustarR * (sL * UL(var) - FL(var))) +
                             sRusi * (ustarL * (sR * UR(var) - FR(var))) +
                             (sL * ustarR * sLusi + sR * ustarL * sRusi) * pstar *
                                 Dstar(var) / (ustar + tiny);
    }
  });
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
