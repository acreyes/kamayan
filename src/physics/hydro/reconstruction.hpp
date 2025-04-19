#ifndef PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#define PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#include <Kokkos_Core.hpp>

#include "Kokkos_MathematicalFunctions.hpp"
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

template <typename reconstruct_traits, typename Container>
requires(reconstruct_traits::reconstruction == Reconstruction::wenoz &&
         Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  const auto eno_reconstruction = [&](const int &pm) {
    // reconstruct the eno predictions at +/- 1/2 on the 3-point stencils
    // centered on +/- [1, 0, -1]
    Kokkos::Array<Real, 3> eno;
    eno[0] = 1. / 6. * (-stencil(pm * 2) + 5. * stencil(pm * 1) + 2. * stencil(0));
    eno[1] = 1. / 6. * (2. * stencil(pm * 1) + 5. * stencil(0) - stencil(-pm * 1));
    eno[2] = 1. / 6. * (11. * stencil(0) - 7. * stencil(-pm * 1) + 2. * stencil(-pm * 2));
    return eno;
  };

  const auto eno_plus = eno_reconstruction(1);
  const auto eno_minus = eno_reconstruction(-1);

  // smoothness indicators for the eno stencils
  const Kokkos::Array<Real, 3> smoothness_indicators = {
      13. / 12. * Kokkos::pow(stencil(-2) - 2. * stencil(-1) + stencil(0), 2) +
          0.25 * Kokkos::pow(stencil(-2) - 4. * stencil(-1) + 3. * stencil(0), 2),
      13. / 12. * Kokkos::pow(stencil(-1) - 2. * stencil(0) + stencil(1), 2) +
          0.25 * Kokkos::pow(stencil(-1) - stencil(1), 2),
      13. / 12. * Kokkos::pow(stencil(0) - 2. * stencil(1) + stencil(2), 2) +
          0.25 * Kokkos::pow(stencil(0) - 4. * stencil(1) + 3. * stencil(2), 2)};

  constexpr std::size_t m = 2;
  constexpr Real eps = 1.e-36;
  const auto weno_weighting = [&](const int &pm, const Kokkos::Array<Real, 3> eno) {
    // calculate the non-linear weights, normalize them and do the weno reconstruction
    Kokkos::Array<Real, 3> weights;
    weights[0] = 3. * (1.0 + Kokkos::pow(Kokkos::abs(smoothness_indicators[2] -
                                                     smoothness_indicators[0]) /
                                             (eps + smoothness_indicators[1 + pm]),
                                         m));
    weights[1] = 6. * (1.0 + Kokkos::pow(Kokkos::abs(smoothness_indicators[2] -
                                                     smoothness_indicators[0]) /
                                             (eps + smoothness_indicators[1]),
                                         m));
    weights[2] = 1. * (1.0 + Kokkos::pow(Kokkos::abs(smoothness_indicators[2] -
                                                     smoothness_indicators[0]) /
                                             (eps + smoothness_indicators[1 - pm]),
                                         m));

    const Real norm = weights[0] + weights[1] + weights[2];
    return weights[0] * eno[0] + weights[1] * eno[1] + weights[2] * eno[2];
  };
  vM = weno_weighting(-1, eno_minus);
  vP = weno_weighting(1, eno_plus);
}

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_RECONSTRUCTION_HPP_
