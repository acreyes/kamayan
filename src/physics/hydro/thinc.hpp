#ifndef PHYSICS_HYDRO_THINC_HPP_
#define PHYSICS_HYDRO_THINC_HPP_

#include <Kokkos_Core.hpp>

#include "grid/indexer.hpp"
#include "grid/subpack.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/reconstruction.hpp"

namespace kamayan::hydro {

// Mapping from ThincFallbackLimiter to SlopeLimiter for use in
// ReconstructTraits. The two enums have identical members but are
// separate types so that the dispatcher can resolve them independently.
template <ThincFallbackLimiter fb>
constexpr SlopeLimiter to_slope_limiter() {
  if constexpr (fb == ThincFallbackLimiter::minmod) return SlopeLimiter::minmod;
  else if constexpr (fb == ThincFallbackLimiter::van_leer)
    return SlopeLimiter::van_leer;
  else
    return SlopeLimiter::mc;
}

// THINC (Tangent of Hyperbola for INterface Capturing) reconstruction.
// Produces sharper interface states than standard slope limiters for
// monotone stencils. Non-monotone stencils fall back to first-order.
//
// Reference: FLASH hy_uhd_THINClim.F90, Wakimura et al. 2024.
template <typename Container>
requires(Stencil1D<Container>)
KOKKOS_INLINE_FUNCTION void THINCReconstruct(Container stencil, const Real beta,
                                             Real &vM, Real &vP) {
  const Real qm = stencil(-1);
  const Real qc = stencil(0);
  const Real qp = stencil(1);

  // Monotonicity check — degenerate to first-order if non-monotone
  if ((qp - qc) * (qc - qm) <= 0.0) {
    vM = qc;
    vP = qc;
    return;
  }

  const Real fplus = 0.5 * (qp + qm);
  const Real fminus = 0.5 * (qp - qm);
  const Real alpha = (qc - fplus) / fminus;

  const Real T1 = Kokkos::tanh(0.5 * beta);
  const Real T2 = Kokkos::tanh(0.5 * beta * alpha);

  vP = fplus + fminus * (T1 + T2 / T1) / (1.0 + T2);
  vM = fplus - fminus * (T1 - T2 / T1) / (1.0 - T2);
}

// BVD (Boundary Variation Diminishing) selection.
// Returns true if THINC states produce smaller total boundary variation
// than fallback states, comparing across neighboring faces.
//
// Parameters:
//   fb_L, fb_R   — fallback L/R states at current face
//   th_L, th_R   — THINC L/R states at current face
//   fb_Lm, th_Lm — fallback-R and THINC-R from face i-1
//   fb_Rp, th_Rp — fallback-L and THINC-L from face i+1
//
// Reference: Wakimura et al. 2024, Eq. 35.
KOKKOS_INLINE_FUNCTION
bool BVDSelect(const Real fb_L, const Real fb_R, const Real th_L, const Real th_R,
               const Real fb_Lm, const Real th_Lm, const Real fb_Rp,
               const Real th_Rp) {
  const Real tbv_fb = Kokkos::abs(th_Lm - fb_R) + Kokkos::abs(fb_L - th_Rp);
  const Real tbv_th = Kokkos::abs(fb_Lm - th_R) + Kokkos::abs(th_L - fb_Rp);
  return tbv_th <= tbv_fb;
}

// Two-pass reconstruction: fallback + THINC.
// For THINC-targeted variables, produces both fallback and THINC interface
// states. For non-targeted variables, THINC states equal fallback states.
// The caller uses BVDSelect to choose between them.
//
// Template parameters:
//   FBTraits — ReconstructTraits with the fallback slope limiter
//   axis     — reconstruction axis
template <ReconstructTrait FBTraits, Axis axis, typename PackType>
KOKKOS_INLINE_FUNCTION void ReconstructFace(const PackType &pack_recon, const int b,
                                            const int var, const int k, const int j,
                                            const int i, Real &fb_vM, Real &fb_vP,
                                            Real &th_vM, Real &th_vP,
                                            const Real beta_thinc,
                                            const bool is_thinc_var) {
  auto stencil = SubPack<axis>(pack_recon, b, var, k, j, i);
  Reconstruct<FBTraits>(stencil, fb_vM, fb_vP);
  if (is_thinc_var) {
    THINCReconstruct(stencil, beta_thinc, th_vM, th_vP);
  } else {
    th_vM = fb_vM;
    th_vP = fb_vP;
  }
}

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_THINC_HPP_
