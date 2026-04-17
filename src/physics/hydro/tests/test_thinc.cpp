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
  // THINC is selected when fallback's TBV exceeds the threshold,
  // indicating a genuine interface where fallback has large jumps.
  // tbv_fb = BVD(...) = max(raw, threshold); THINC wins iff tbv_fb > threshold.
  EXPECT_TRUE(BVDSelect(
      /*fb_L=*/1.0, /*fb_R=*/3.0,
      /*th_L=*/1.9, /*th_R=*/2.1,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // tbv_fb = |2.0 - 3.0| + |1.0 - 2.0| = 2.0, well above threshold → true ✓
}

TEST(BVDSelect, FallbackWins) {
  // Even though THINC has worse TBV, what matters is that fallback's TBV
  // is small (below threshold). Smooth fallback → no THINC activation.
  EXPECT_FALSE(BVDSelect(
      /*fb_L=*/2.0, /*fb_R=*/2.0,
      /*th_L=*/1.0, /*th_R=*/3.0,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // tbv_fb = |2.0 - 2.0| + |2.0 - 2.0| = 0.0, clamped to 1e-4 = threshold
  // threshold is not > threshold → false ✓
}

TEST(BVDSelect, ExactlyAtThreshold) {
  // When fallback TBV equals exactly the threshold, fallback wins (strict >).
  EXPECT_FALSE(BVDSelect(
      /*fb_L=*/1.0, /*fb_R=*/2.0,
      /*th_L=*/1.0, /*th_R=*/2.0,
      /*fb_Lm=*/1.5, /*th_Lm=*/1.5,
      /*fb_Rp=*/1.5, /*th_Rp=*/1.5,
      /*threshold=*/1.0));
  // tbv_fb raw = |1.5 - 2.0| + |1.0 - 1.5| = 1.0, clamped to max(1.0, 1.0) = 1.0
  // 1.0 > 1.0 is false → fallback wins ✓
}

TEST(BVDSelect, ThresholdPreventsSelection) {
  // In smooth regions (all values equal), fallback TBV is zero,
  // clamped to threshold. threshold is not > threshold → false.
  EXPECT_FALSE(BVDSelect(
      /*fb_L=*/2.0, /*fb_R=*/2.0,
      /*th_L=*/2.0, /*th_R=*/2.0,
      /*fb_Lm=*/2.0, /*th_Lm=*/2.0,
      /*fb_Rp=*/2.0, /*th_Rp=*/2.0,
      /*threshold=*/1.0e-4));
  // tbv_fb = 0.0, clamped to 1e-4 = threshold, not > threshold → false ✓
}

}  // namespace kamayan::hydro
