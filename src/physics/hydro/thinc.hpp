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

// BVD (Boundary Variation Diminishing) — compute the best-case total
// boundary variation for a candidate reconstruction of a cell,
// minimizing over all 4 combinations of whether the left and right
// neighbor cells chose sharp (THINC) or linear (fallback) states.
//
// Matches FLASH's hy_uhd_BVD function. The TBV for cell i is the sum
// of Riemann jumps at both faces: |left_neighbor_R - cell_L| +
// |cell_R - right_neighbor_L|. The min over 4 combos handles
// uncertainty about what neighboring cells will choose.
//
// Parameters:
//   sLm  — sharp (THINC) R-face of left neighbor cell
//   xLc  — candidate R-face of current cell
//   xRc  — candidate L-face of current cell
//   sRp  — sharp (THINC) L-face of right neighbor cell
//   lLm  — linear (fallback) R-face of left neighbor cell
//   lRp  — linear (fallback) L-face of right neighbor cell
KOKKOS_INLINE_FUNCTION
Real BVD(const Real sLm, const Real xLc, const Real xRc, const Real sRp,
         const Real lLm, const Real lRp, const Real threshold) {
  const Real raw = Kokkos::min(
      Kokkos::min(Kokkos::abs(sLm - xRc) + Kokkos::abs(xLc - sRp),
                  Kokkos::abs(sLm - xRc) + Kokkos::abs(xLc - lRp)),
      Kokkos::min(Kokkos::abs(lLm - xRc) + Kokkos::abs(xLc - sRp),
                  Kokkos::abs(lLm - xRc) + Kokkos::abs(xLc - lRp)));
  return Kokkos::max(raw, threshold);
}

// BVD selection: returns true if THINC should replace fallback for a
// given cell. Uses the simplified BVD criterion (matching FLASH's default
// use_BVD=false mode): THINC is selected when the fallback reconstruction's
// TBV exceeds the threshold, indicating a genuine interface. This avoids
// the full two-candidate comparison which can activate THINC in smooth
// regions where both TBVs are tiny but THINC's is marginally smaller.
//
// Cell-based parameters:
//   fb_L, fb_R       — fallback R-face and L-face of current cell
//   th_L, th_R       — THINC R-face and L-face of current cell (unused)
//   fb_Lm, th_Lm     — fallback and THINC R-face of left neighbor cell
//   fb_Rp, th_Rp     — fallback and THINC L-face of right neighbor cell
KOKKOS_INLINE_FUNCTION
bool BVDSelect(const Real fb_L, const Real fb_R, const Real th_L, const Real th_R,
               const Real fb_Lm, const Real th_Lm, const Real fb_Rp,
               const Real th_Rp, const Real threshold) {
  const Real tbv_fb = BVD(th_Lm, fb_L, fb_R, th_Rp, fb_Lm, fb_Rp, threshold);
  // Simplified criterion: THINC wins only when fallback TBV exceeds
  // the threshold floor, meaning the raw TBV was genuinely large.
  // Since BVD returns max(raw, threshold), tbv_fb > threshold iff
  // raw > threshold.
  return tbv_fb > threshold;
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
