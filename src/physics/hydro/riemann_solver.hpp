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

  const Real aL2 = vL(GAMC()) * vL(PRES()) / vL(DENS());
  const Real aR2 = vR(GAMC()) * vR(PRES()) / vR(DENS());

  Real cfL, cfR;
  if constexpr (hydro_traits::MHD == Mhd::off) {
    // sound speed
    cfL = Kokkos::sqrt(aL2);
    cfR = Kokkos::sqrt(aR2);
  } else {
    // fast magneto-sonic speed
  }

  const Real tiny = std::numeric_limits<Real>::min();
  const Real sL =
      Kokkos::min(-tiny, Kokkos::min(vL(VELOCITY(dir1)) - cfL, vR(VELOCITY(dir1)) - cfR));
  const Real sR =
      Kokkos::max(tiny, Kokkos::max(vL(VELOCITY(dir1)) + cfL, vR(VELOCITY(dir1)) + cfR));
  const Real sRmsLi = 1.0 / (sR - sL);

  using Array_t = TypeListArray<typename hydro_traits::Conserved>;
  Array_t UL, UR, FL, FR;
  Prim2Cons<hydro_traits>(vL, UL);
  Prim2Cons<hydro_traits>(vR, UR);
  Prim2Flux<dir1>(vR, FR);
  Prim2Flux<dir1>(vL, FL);

  pack.flux(face, DENS()) =
      sRmsLi * (sR * FL(DENS()) - sL * FR(DENS()) + sR * sL * (UR(DENS()) - UL(DENS())));
  pack.flux(face, MOMENTUM(0)) = sRmsLi * (sR * FL(MOMENTUM(0)) - sL * FR(MOMENTUM(0)) +
                                           sR * sL * (UR(MOMENTUM(0)) - UL(MOMENTUM(0))));
  pack.flux(face, MOMENTUM(1)) = sRmsLi * (sR * FL(MOMENTUM(1)) - sL * FR(MOMENTUM(1)) +
                                           sR * sL * (UR(MOMENTUM(1)) - UL(MOMENTUM(1))));
  pack.flux(face, MOMENTUM(2)) = sRmsLi * (sR * FL(MOMENTUM(2)) - sL * FR(MOMENTUM(2)) +
                                           sR * sL * (UR(MOMENTUM(2)) - UL(MOMENTUM(2))));
  pack.flux(face, ENER()) =
      sRmsLi * (sR * FL(ENER()) - sL * FR(ENER()) + sR * sL * (UR(ENER()) - UL(ENER())));
  // for whatever reason this is modifying sL/R variables some how... compiler bug? but
  // everyone does it
  // type_for(typename hydro_traits::Conserved(), [&]<typename
  // Vars>(const Vars &var) {
  //   for (int comp = 0; comp < pack.GetSize(Vars()); comp++) {
  //     pack.flux(face, Vars(comp)) = sR * FL(Vars(comp)) - sL * FR(Vars(comp));
  //     sRmsLi * (sR * FL(Vars(comp)) - sL * FR(Vars(comp)) +
  //               sR * sL * (UR(Vars(comp)) - UL(Vars(comp))));
  //   }
  // });
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
