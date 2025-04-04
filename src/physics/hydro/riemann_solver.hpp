#ifndef PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
#define PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
#include <Kokkos_Core.hpp>

#include "grid/indexer.hpp"
#include "hydro_types.hpp"
#include "primconsflux.hpp"

namespace kamayan::hydro {

template <std::size_t dir, RiemannSolver riemann, typename hydro_traits,
          typename FluxIndexer, typename Scratch>
KOKKOS_INLINE_FUNCTION void RiemannFlux(FluxIndexer &pack, const Scratch &vL,
                                        const Scratch &vR) {}

template <std::size_t dir, RiemannSolver riemann, typename hydro_traits,
          typename FluxIndexer, typename Scratch>
requires(riemann == RiemannSolver::hll)
KOKKOS_INLINE_FUNCTION void RiemannFlux(FluxIndexer &pack, const Scratch &vL,
                                        const Scratch &vR) {
  ConsArray<hydro_traits> UL, UR, FL, FR;
  Prim2Cons(vL, UL);
  Prim2Cons(vR, UR);
  Prim2Flux<dir>(vL, FL);
  Prim2Flux<dir>(vR, FR);
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_RIEMANN_SOLVER_HPP_
