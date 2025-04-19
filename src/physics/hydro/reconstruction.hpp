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

template <SlopeLimiter limiter, typename Container>
KOKKOS_INLINE_FUNCTION Real Slope(const int &idx, Container stencil) {
  return LimitedSlope<limiter>(stencil(idx + 1) - stencil(idx),
                               stencil(idx) - stencil(idx - 1));
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

template <typename reconstruct_traits, typename Container>
requires(reconstruct_traits::reconstruction == Reconstruction::ppm &&
         Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  // the initial cubic reconstruction before monotonicity
  // a^\pm_0 = 0.5 * ( v_{i-1+s} + v_{i+s} ) - 1./6. (dv_{i+s} - dv_{i-1+s})
  // s^+/- = 1,0
  constexpr SlopeLimiter limiter = reconstruct_traits::slope_limiter;
  const Real dv_p = Slope<limiter>(1, stencil);
  const Real dv_0 = Slope<limiter>(0, stencil);
  const Real dv_m = Slope<limiter>(-1, stencil);

  vM = 0.5 * (stencil(-1) + stencil(0)) - 1. / 6. * (dv_0 - dv_m);
  vP = 0.5 * (stencil(0) + stencil(1)) - 1. / 6. * (dv_p - dv_0);

  if ((vP - stencil(0)) * (stencil(0) - vM) <= 0.) {
    vM = stencil(0);
    vP = stencil(0);
    return;
  }

  // enforce monotonicity of parabolic profile on cell from -1/2 to +1/2
  if (-(vP - vM) * (vP - vM) > 6. * (vP - vM) * (stencil(0) - 0.5 * (vP + vM))) {
    vP = 3.0 * stencil(0) - 2. * vM;
  }
  if ((vP - vM) * (vP - vM) < 6. * (vP - vM) * (stencil(0) - 0.5 * (vP + vM))) {
    vM = 3.0 * stencil(0) - 2. * vP;
  }
}

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_RECONSTRUCTION_HPP_
