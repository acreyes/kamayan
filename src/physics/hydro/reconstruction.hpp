#ifndef PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#define PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#include <Kokkos_Core.hpp>

#include "grid/indexer.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "utils/error_checking.hpp"

namespace kamayan::hydro {

template <typename T, typename Container>
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  PARTHENON_FAIL("Reconstruction not recognized");
}

template <typename reconstruct_traits, typename Container>
requires(reconstruct_traits::reconstruction == Reconstruction::fog &&
         Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  vM = stencil(0);
  vP = stencil(0);
}

template <SlopeLimiter limiter>
KOKKOS_INLINE_FUNCTION Real LimitedSlope(const Real a, const Real b) {
  if constexpr (limiter == SlopeLimiter::mc) {
    return (Kokkos::copysign(1., a) + Kokkos::copysign(1., b)) *
           Kokkos::min(Kokkos::abs(a),
                       Kokkos::min(.25 * Kokkos::abs(a + b), Kokkos::abs(b)));
  } else if constexpr (limiter == SlopeLimiter::van_leer) {
    return 2. * a * b / (a + b) * static_cast<Real>(a * b > 0);
  } else if constexpr (limiter == SlopeLimiter::minmod) {
    return 0.5 * (Kokkos::copysign(1.0, a) + Kokkos::copysign(1.0, b)) *
           Kokkos::min(Kokkos::abs(a), Kokkos::abs(b));
  }

  return 0.;
}

template <typename reconstruct_traits, typename Container>
requires(reconstruct_traits::reconstruction == Reconstruction::plm &&
         Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  // --8<-- [start:use-stncl]
  const Real dvL = stencil(0) - stencil(-1);
  const Real dvR = stencil(1) - stencil(0);
  // --8<-- [end:use-stncl]
  const Real del = LimitedSlope<reconstruct_traits::slope_limiter>(dvL, dvR);
  vM = stencil(0) - 0.5 * del;
  vP = stencil(0) + 0.5 * del;
}

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_RECONSTRUCTION_HPP_
