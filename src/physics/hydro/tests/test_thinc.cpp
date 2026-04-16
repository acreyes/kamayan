#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kamayan_utils/type_abstractions.hpp"
#include "physics/hydro/thinc.hpp"

namespace kamayan::hydro {

// Simple 3-point stencil helper for THINC tests
struct Stencil3 {
  Real data[3];  // data[-1], data[0], data[1]
  Real operator()(const int &idx) const { return data[idx + 1]; }
};

// --- THINCReconstruct tests ---

TEST(THINCReconstruct, UniformStencil) {
  // Uniform stencil: non-monotone guard triggers first-order fallback
  constexpr Real c = 3.14;
  Stencil3 s{c, c, c};
  Real vM, vP;
  THINCReconstruct(s, 1.6, vM, vP);
  EXPECT_DOUBLE_EQ(vM, c);
  EXPECT_DOUBLE_EQ(vP, c);
}

TEST(THINCReconstruct, MonotoneLinearProfile) {
  Stencil3 s{1.0, 2.0, 3.0};
  Real vM, vP;
  THINCReconstruct(s, 1.6, vM, vP);

  // THINC should produce sharper interface values
  EXPECT_LT(vM, 2.0);
  EXPECT_GT(vM, 1.0);
  EXPECT_GT(vP, 2.0);
  EXPECT_LT(vP, 3.0);

  // Conservation: 0.5*(vM+vP) == qc
  EXPECT_NEAR(0.5 * (vM + vP), 2.0, 1e-14);
}

TEST(THINCReconstruct, LeftRightSymmetry) {
  // Forward stencil (monotone increasing)
  Stencil3 fwd{0.0, 1.0, 2.0};
  Real vM_fwd, vP_fwd;
  THINCReconstruct(fwd, 1.6, vM_fwd, vP_fwd);

  // Reversed stencil (monotone decreasing)
  Stencil3 rev{2.0, 1.0, 0.0};
  Real vM_rev, vP_rev;
  THINCReconstruct(rev, 1.6, vM_rev, vP_rev);

  // Symmetry: vM(fwd) == vP(rev), vP(fwd) == vM(rev)
  EXPECT_DOUBLE_EQ(vM_fwd, vP_rev);
  EXPECT_DOUBLE_EQ(vP_fwd, vM_rev);
}

TEST(THINCReconstruct, NonMonotoneStencil) {
  // Non-monotone: (qp-qc)*(qc-qm) < 0 → first-order fallback
  Stencil3 s{2.0, 1.0, 3.0};
  Real vM, vP;
  THINCReconstruct(s, 1.6, vM, vP);
  EXPECT_DOUBLE_EQ(vM, 1.0);
  EXPECT_DOUBLE_EQ(vP, 1.0);
}

TEST(THINCReconstruct, BetaSensitivity) {
  Stencil3 s{1.0, 2.0, 3.0};
  Real vM_weak, vP_weak, vM_strong, vP_strong;
  THINCReconstruct(s, 0.1, vM_weak, vP_weak);
  THINCReconstruct(s, 10.0, vM_strong, vP_strong);

  // Larger beta → sharper jump → vP closer to stencil(1)=3.0
  EXPECT_LT(std::abs(vP_strong - 3.0), std::abs(vP_weak - 3.0));
  // And vM closer to stencil(-1)=1.0
  EXPECT_LT(std::abs(vM_strong - 1.0), std::abs(vM_weak - 1.0));
}

// --- BVDSelect tests ---

TEST(BVDSelect, THINCWins) {
  // Construct values where THINC has smaller boundary variation
  // fb states: large jumps at boundaries; th states: small jumps
  const Real fb_L = 1.0, fb_R = 2.0;
  const Real th_L = 1.4, th_R = 1.6;
  // Neighbor face values: fb_Lm (fb right at i-1), th_Lm (th right at i-1)
  const Real fb_Lm = 1.5, th_Lm = 1.55;
  // fb_Rp (fb left at i+1), th_Rp (th left at i+1)
  const Real fb_Rp = 2.5, th_Rp = 1.65;

  // tbv_fb = |th_Lm - fb_R| + |fb_L - th_Rp| = |1.55 - 2.0| + |1.0 - 1.65| = 0.45 + 0.65 = 1.1
  // tbv_th = |fb_Lm - th_R| + |th_L - fb_Rp| = |1.5 - 1.6| + |1.4 - 2.5| = 0.1 + 1.1 = 1.2
  // Hmm, that's tbv_th > tbv_fb. Let me recalculate with better values.

  // Actually let me just pick values directly.
  // We want tbv_th <= tbv_fb, i.e. BVDSelect returns true.
  // tbv_fb = |th_Lm - fb_R| + |fb_L - th_Rp|
  // tbv_th = |fb_Lm - th_R| + |th_L - fb_Rp|
  EXPECT_TRUE(BVDSelect(
      /*fb_L=*/1.0, /*fb_R=*/3.0,
      /*th_L=*/1.9, /*th_R=*/2.1,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // tbv_fb = |2.0 - 3.0| + |1.0 - 2.0| = 1 + 1 = 2
  // tbv_th = |2.0 - 2.1| + |1.9 - 2.0| = 0.1 + 0.1 = 0.2
  // 0.2 <= 2.0 → true ✓
}

TEST(BVDSelect, FallbackWins) {
  // Opposite: THINC states have larger boundary variation
  EXPECT_FALSE(BVDSelect(
      /*fb_L=*/1.9, /*fb_R=*/2.1,
      /*th_L=*/1.0, /*th_R=*/3.0,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // tbv_fb = |2.0 - 2.1| + |1.9 - 2.0| = 0.1 + 0.1 = 0.2
  // tbv_th = |2.0 - 3.0| + |1.0 - 2.0| = 1 + 1 = 2
  // 2 <= 0.2 → false ✓
}

TEST(BVDSelect, Tie) {
  // Equal boundary variation → returns true (per <= in design)
  EXPECT_TRUE(BVDSelect(
      /*fb_L=*/1.0, /*fb_R=*/2.0,
      /*th_L=*/1.0, /*th_R=*/2.0,
      /*fb_Lm=*/1.5, /*th_Lm=*/1.5,
      /*fb_Rp=*/1.5, /*th_Rp=*/1.5,
      /*threshold=*/1.0e-4));
  // tbv_fb = |1.5 - 2.0| + |1.0 - 1.5| = 0.5 + 0.5 = 1.0
  // tbv_th = |1.5 - 2.0| + |1.0 - 1.5| = 0.5 + 0.5 = 1.0
  // 1.0 <= 1.0 → true ✓
}

TEST(BVDSelect, ThresholdPreventsSelection) {
  // When both BVD values are below threshold, they both clamp to threshold,
  // so tbv_th == tbv_fb and BVDSelect returns true (tie).
  // Use values where raw BVD would be very small (< threshold).
  EXPECT_TRUE(BVDSelect(
      /*fb_L=*/2.0, /*fb_R=*/2.0,
      /*th_L=*/2.0, /*th_R=*/2.0,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // Both raw BVD = 0.0, both clamped to 1e-4, tie → true
}

}  // namespace kamayan::hydro
