#ifndef GRID_REFINEMENT_OPERATIONS_HPP_
#define GRID_REFINEMENT_OPERATIONS_HPP_
#include <algorithm>
#include <array>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/grid_types.hpp"
#include "interface/variable_state.hpp"
#include <limits>
namespace kamayan::grid {
// Parthenon's implementation of non-cartesian coordinates assumes that
// it is determined at compile time for the entire library. In order
// to enable runtime geometry we re-implement the refinement operations upstream
// templated on the geometry.
//
namespace util {

// distances between element centroids, so that prolongation is always
// done in the volume coordinate
template <int DIM, TopologicalElement EL, Geometry geom>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const Coordinates<geom> &coords, const Coordinates<geom> &coarse_coords,
                const parthenon::IndexRange &cib, const parthenon::IndexRange &ib, int i,
                int fi, Real *dxm, Real *dxp, Real *dxfm, Real *dxfp) {
  // here "f" signifies the fine grid, not face locations.
  constexpr auto ax = AxisFromInt(DIM);
  const Real xm = coarse_coords.template X<ax, EL>(i - 1);
  const Real xc = coarse_coords.template X<ax, EL>(i);
  const Real xp = coarse_coords.template X<ax, EL>(i + 1);
  *dxm = xc - xm;
  *dxp = xp - xc;
  const Real fxm = coords.template X<ax, EL>(fi);
  const Real fxp = coords.template X<ax, EL>(fi + 1);
  *dxfm = xc - fxm;
  *dxfp = fxp - xc;
}

KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real fc, const Real fm, const Real fp, const Real dxm,
                const Real dxp, Real &gxm, Real &gxp) {
  gxm = (fc - fm) / dxm;
  gxp = (fp - fc) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}

}  // namespace util

template <Geometry geom>
struct RestrictAverage {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &pcoords,
     const parthenon::Coordinates_t &pcoarse_coords,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pcoarse,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pfine) {
    using TE = TopologicalElement;

    auto coords = Coordinates<geom>(pcoords);
    auto coarse_coords = Coordinates<geom>(pcoarse_coords);

    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);
    constexpr int element_idx = static_cast<int>(el) % 3;

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const int i = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
    const int j = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
    const int k = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;

    // JMM: If dimensionality is wrong, accesses are out of bounds. Only
    // access cells if dimensionality is correct.
    Real vol[2][2][2], terms[2][2][2];  // memset not available on all accelerators
    for (int ok = 0; ok < 2; ++ok) {
      for (int oj = 0; oj < 2; ++oj) {
        for (int oi = 0; oi < 2; ++oi) {
          vol[ok][oj][oi] = terms[ok][oj][oi] = 0;
        }
      }
    }

    for (int ok = 0; ok < 1 + INCLUDE_X3; ++ok) {
      for (int oj = 0; oj < 1 + INCLUDE_X2; ++oj) {
        for (int oi = 0; oi < 1 + INCLUDE_X1; ++oi) {
          vol[ok][oj][oi] = coords.template Volume<el>(k + ok, j + oj, i + oi);
          terms[ok][oj][oi] =
              vol[ok][oj][oi] * fine(element_idx, l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }
    // KGF: add the off-centered quantities first to preserve FP
    // symmetry
    const Real tvol = ((vol[0][0][0] + vol[0][1][0]) + (vol[0][0][1] + vol[0][1][1])) +
                      ((vol[1][0][0] + vol[1][1][0]) + (vol[1][0][1] + vol[1][1][1]));
    coarse(element_idx, l, m, n, ck, cj, ci) =
        tvol > 0.0
            ? (((terms[0][0][0] + terms[0][1][0]) + (terms[0][0][1] + terms[0][1][1])) +
               ((terms[1][0][0] + terms[1][1][0]) + (terms[1][0][1] + terms[1][1][1]))) /
                  tvol
            : 0.0;
  }
};

template <Geometry geom, bool use_minmod_slope, bool piecewise_constant = false>
struct ProlongateSharedGeneral {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &pcoords,
     const parthenon::Coordinates_t &pcoarse_coords,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pcoarse,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pfine) {
    using util::GetGridSpacings;
    using util::GradMinMod;
    using TE = TopologicalElement;

    auto coords = Coordinates<geom>(pcoords);
    auto coarse_coords = Coordinates<geom>(pcoarse_coords);

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;

    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);

    constexpr bool SCALE_BY_R =
        (geom == Geometry::cylindrical) && (el == TE::F1 || el == TE::F2);
    auto coarse_scaled = [&](int kk, int jj, int ii) {
      Real val = coarse(element_idx, l, m, n, kk, jj, ii);
      if constexpr (SCALE_BY_R) {
        val *= coarse_coords.template Xf<el, Axis::IAXIS>(ii);
      }
      return val;
    };

    const Real fc = coarse_scaled(k, j, i);

    Real dx1fm = 0;
    [[maybe_unused]] Real dx1fp = 0;
    [[maybe_unused]] Real gx1m = 0, gx1p = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      GetGridSpacings<1, el, geom>(coords, coarse_coords, cib, ib, i, fi, &dx1m, &dx1p,
                                   &dx1fm, &dx1fp);

      Real gx1c = GradMinMod(fc, coarse_scaled(k, j, i - 1), coarse_scaled(k, j, i + 1),
                             dx1m, dx1p, gx1m, gx1p);
      if constexpr (use_minmod_slope) {
        gx1m = gx1c;
        gx1p = gx1c;
      }
    }

    Real dx2fm = 0;
    [[maybe_unused]] Real dx2fp = 0;
    [[maybe_unused]] Real gx2m = 0, gx2p = 0;
    if constexpr (INCLUDE_X2) {
      Real dx2m, dx2p;
      GetGridSpacings<2, el, geom>(coords, coarse_coords, cjb, jb, j, fj, &dx2m, &dx2p,
                                   &dx2fm, &dx2fp);
      Real gx2c = GradMinMod(fc, coarse_scaled(k, j - 1, i), coarse_scaled(k, j + 1, i),
                             dx2m, dx2p, gx2m, gx2p);
      if constexpr (use_minmod_slope) {
        gx2m = gx2c;
        gx2p = gx2c;
      }
    }

    Real dx3fm = 0;
    [[maybe_unused]] Real dx3fp = 0;
    [[maybe_unused]] Real gx3m = 0, gx3p = 0;
    if constexpr (INCLUDE_X3) {
      Real dx3m, dx3p;
      GetGridSpacings<3, el, geom>(coords, coarse_coords, ckb, kb, k, fk, &dx3m, &dx3p,
                                   &dx3fm, &dx3fp);
      Real gx3c = GradMinMod(fc, coarse_scaled(k - 1, j, i), coarse_scaled(k + 1, j, i),
                             dx3m, dx3p, gx3m, gx3p);
      if constexpr (use_minmod_slope) {
        gx3m = gx3c;
        gx3p = gx3c;
      }
    }

    if constexpr (piecewise_constant) {
      gx1m = 0.0;
      gx1p = 0.0;
      gx2m = 0.0;
      gx2p = 0.0;
      gx3m = 0.0;
      gx3p = 0.0;
    }

    auto set_fine_scaled = [&](int kk, int jj, int ii, Real val) {
      if constexpr (SCALE_BY_R) {
        const Real r = coords.template Xf<el, Axis::IAXIS>(ii);
        val *= 1.0 / (r + std::numeric_limits<Real>::epsilon());
      }
      fine(element_idx, l, m, n, kk, jj, ii) = val;
    };

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    set_fine_scaled(fk, fj, fi, fc - (gx1m * dx1fm + gx2m * dx2fm + gx3m * dx3fm));
    if constexpr (INCLUDE_X1)
      set_fine_scaled(fk, fj, fi + 1, fc + (gx1p * dx1fp - gx2m * dx2fm - gx3m * dx3fm));
    if constexpr (INCLUDE_X2)
      set_fine_scaled(fk, fj + 1, fi, fc - (gx1m * dx1fm - gx2p * dx2fp + gx3m * dx3fm));
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      set_fine_scaled(fk, fj + 1, fi + 1,
                      fc + (gx1p * dx1fp + gx2p * dx2fp - gx3m * dx3fm));
    if constexpr (INCLUDE_X3)
      set_fine_scaled(fk + 1, fj, fi, fc - (gx1m * dx1fm + gx2m * dx2fm - gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      set_fine_scaled(fk + 1, fj, fi + 1,
                      fc + (gx1p * dx1fp - gx2m * dx2fm + gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      set_fine_scaled(fk + 1, fj + 1, fi,
                      fc - (gx1m * dx1fm - gx2p * dx2fp - gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      set_fine_scaled(fk + 1, fj + 1, fi + 1,
                      fc + (gx1p * dx1fp + gx2p * dx2fp + gx3p * dx3fp));
  }
};

template <Geometry geom>
using ProlongateSharedMinMod = ProlongateSharedGeneral<geom, true, false>;
template <Geometry geom>
using ProlongateSharedLinear = ProlongateSharedGeneral<geom, false, false>;
template <Geometry geom>
using ProlongatePiecewiseConstant = ProlongateSharedGeneral<geom, false, true>;

// Implements divergence-free prolongation to internal faces using the method
// described in Toth & Roe (2002). Any prolongation method for faces shared
// between the coarse and the fine grid can be used alongside this internal
// prolongation operation. Obviously, this prolongation operation is only
// defined for face fields.
//
// We modify it here for cylindrical coordinates where the divergence constraint
// is satisfied for r*B
template <Geometry geom>
struct ProlongateInternalTothAndRoe {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    using TE = TopologicalElement;
    return (cel == TE::CC) && (GetTopologicalType(fel) == TopologicalType::Face);
  }
  // Here, fel is the topological element on which the field is defined and
  // cel is the topological element on which we are filling the internal values
  // of the field. So, for instance, we could fill the fine cell values of an
  // x-face field within the volume of a coarse cell. This is assumes that the
  // values of the fine cells on the elements corresponding with the coarse cell
  // have been filled.
  template <int DIM, TopologicalElement fel = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &pcoords,
     const parthenon::Coordinates_t &pcoarse_coords,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pfine) {
    using TE = TopologicalElement;
    using util::GradMinMod;

    auto coords = Coordinates<geom>(pcoords);
    auto coarse_coords = Coordinates<geom>(pcoarse_coords);

    if constexpr (!IsSubmanifold(fel, cel)) {
      return;
    } else {
      const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

      // Here, we write the update for the x-component of the B-field and recover the
      // other components by cyclic permutation
      constexpr int element_idx = static_cast<int>(fel) % 3;
      auto get_fine_permuted_kji = [&](int ok, int oj, int oi) -> std::array<int, 3> {
        // Guard against offsetting in symmetry dimensions
        constexpr int g3 = (DIM > 2);
        constexpr int g2 = (DIM > 1);
        if constexpr (fel == TE::F1) {
          return {fk + ok * g3, fj + oj * g2, fi + oi};
        } else if constexpr (fel == TE::F2) {
          return {fk + oj * g3, fj + oi * g2, fi + ok};
        } else {
          return {fk + oi * g3, fj + ok * g2, fi + oj};
        }
      };
      auto get_fine_permuted = [&](int eidx, int ok, int oj, int oi) -> Real & {
        eidx = (element_idx + eidx) % 3;
        const auto [kk, jj, ii] = get_fine_permuted_kji(ok, oj, oi);
        return (*pfine)(eidx, l, m, n, kk, jj, ii);
      };
      auto safe_inverse = [&](Real radius) {
        if constexpr (geom == Geometry::cylindrical) {
          const Real denom = radius + std::numeric_limits<Real>::epsilon();
          return 1.0 / denom;
        }
        return 1.0;
      };
      auto get_radial_coord = [&]<int EIDX>(int ok, int oj, int oi) -> Real {
        if constexpr (geom == Geometry::cylindrical) {
          const auto [kk, jj, ii] = get_fine_permuted_kji(ok, oj, oi);
          constexpr TopologicalElement comp_el = static_cast<TopologicalElement>(
              static_cast<int>(TE::F1) + ((element_idx + EIDX) % 3));
          if constexpr (DIM <= 2 && comp_el == TE::F3) {
            return 1.0;
          }
          return coords.template Xf<comp_el, Axis::IAXIS>(ii);
        } else {
          return 1.0;
        }
      };

      using iarr2 = std::array<int, 2>;
      auto sg = [](int offset) -> Real { return offset == 0 ? -1.0 : 1.0; };
      Real Uxx{0.0};
      Real Vxyz{0.0};
      Real Wxyz{0.0};
      for (const int v : iarr2{0, 1}) {
        for (const int u : iarr2{0, 2}) {
          for (const int t : iarr2{0, 1}) {
            const auto fine2 = get_fine_permuted(1, v, u, t) *
                               get_radial_coord.template operator()<1>(v, u, t);
            const auto fine3 = get_fine_permuted(2, u, v, t) *
                               get_radial_coord.template operator()<2>(u, v, t);
            Uxx += sg(t) * sg(u) * (fine2 + fine3);
            Vxyz += sg(t) * sg(u) * sg(v) * fine2;
            Wxyz += sg(t) * sg(u) * sg(v) * fine3;
          }
        }
      }
      Uxx *= 0.125;
      const int dir1 = element_idx + 1;
      const int dir2 = (element_idx + 1) % 3 + 1;
      const int dir3 = (element_idx + 2) % 3 + 1;
      const auto dx2 = std::pow(coarse_coords.Dx(Axis::IAXIS), 2);
      const auto dy2 = std::pow(coarse_coords.Dx(Axis::JAXIS), 2);
      const auto dz2 = std::pow(coarse_coords.Dx(Axis::KAXIS), 2);
      Vxyz *= 0.125 * dz2 / (dx2 + dz2);
      Wxyz *= 0.125 * dy2 / (dx2 + dy2);

      for (int ok : iarr2{0, 1}) {
        for (int oj : iarr2{0, 1}) {
          get_fine_permuted(0, ok, oj, 1) =
              0.5 * (get_fine_permuted(0, ok, oj, 0) *
                         get_radial_coord.template operator()<0>(ok, oj, 0) +
                     get_fine_permuted(0, ok, oj, 2) *
                         get_radial_coord.template operator()<0>(ok, oj, 2)) +
              Uxx + sg(ok) * Vxyz + sg(oj) * Wxyz;
          get_fine_permuted(0, ok, oj, 1) *=
              safe_inverse(get_radial_coord.template operator()<0>(ok, oj, 1));
        }
      }
    }
  }
};

// TODO(acreyes): ProlongateInternalBalsara - broken implementation, needs fixing
// template <Geometry geom>
// struct ProlongateInternalBalsara { ... };

template <Geometry geom>
struct ProlongateInternalBalsara {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return IsSubmanifold(fel, cel);
  }

 private:
  // Helper: Returns offset {i,j,k} for a given direction
  // DIR: 0=x, 1=y, 2=z
  // DIM: dimensionality (2 or 3)
  // mult: multiplier for the offset (default 1)
  template <int DIR, int DIM, int mult = 1>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::array<int, 3> Offsets() {
    std::array<int, 3> o = {0, 0, 0};
    if (DIR == 0 && DIM > 0) {
      o[0] = mult;
    } else if (DIR == 1 && DIM > 1) {
      o[1] = mult;
    } else if (DIR == 2 && DIM > 2) {
      o[2] = mult;
    }
    return o;
  }

  // Get face value - uses r*B scaling for cylindrical (scale ALL components)
  template <int EIDX>
  KOKKOS_FORCEINLINE_FUNCTION static Real
  FaceField(const parthenon::ParArrayND<Real, parthenon::VariableState> &fine,
            const Coordinates<geom> &coords, int l, int m, int n, int k, int j, int i) {
    Real val = fine(EIDX, l, m, n, k, j, i);
    if constexpr (geom == Geometry::cylindrical) {
      // Scale ALL faces by r (following the proven TothAndRoe approach)
      val *= coords.Xf(Axis::IAXIS, i);
    }
    return val;
  }

  // Get coarse face average: 4-point in 3D, 2-point in 2D
  template <int EIDX, int DIR2, int DIR3, int DIM>
  KOKKOS_FORCEINLINE_FUNCTION static Real
  GetCoarseFaceVal(const parthenon::ParArrayND<Real, parthenon::VariableState> &fine,
                   const Coordinates<geom> &coords, int l, int m, int n, int fk, int fj,
                   int fi) {
    Real b = FaceField<EIDX>(fine, coords, l, m, n, fk, fj, fi);

    if constexpr (DIM == 3) {
      constexpr auto o2 = Offsets<DIR2, DIM>();
      constexpr auto o3 = Offsets<DIR3, DIM>();
      b += FaceField<EIDX>(fine, coords, l, m, n, fk + o2[2], fj + o2[1], fi + o2[0]);
      b += FaceField<EIDX>(fine, coords, l, m, n, fk + o3[2], fj + o3[1], fi + o3[0]);
      b += FaceField<EIDX>(fine, coords, l, m, n, fk + o2[2] + o3[2], fj + o2[1] + o3[1],
                           fi + o2[0] + o3[0]);
      return b * 0.25;
    } else {
      // 2D: use DIR2 if valid (I2 < 2 means dimension index less than 2), else DIR3
      constexpr auto N2 = (DIR2 < 2) ? Offsets<DIR2, DIM>() : Offsets<DIR3, DIM>();
      b += FaceField<EIDX>(fine, coords, l, m, n, fk + N2[2], fj + N2[1], fi + N2[0]);
      return b * 0.5;
    }
  }

  // Normal difference: d(b1)/d(x2) - d(b1)/d(x3) pattern
  template <int EIDX, int DIR2, int DIR3, int DIM>
  KOKKOS_FORCEINLINE_FUNCTION static Real
  GetFaceNormDif(const parthenon::ParArrayND<Real, parthenon::VariableState> &fine,
                 const Coordinates<geom> &coords, int l, int m, int n, int fk, int fj,
                 int fi) {
    constexpr auto o2 = Offsets<DIR2, DIM>();
    constexpr auto o3 = Offsets<DIR3, DIM>();
    return (FaceField<EIDX>(fine, coords, l, m, n, fk + o2[2] + o3[2], fj + o2[1] + o3[1],
                            fi + o2[0] + o3[0]) -
            FaceField<EIDX>(fine, coords, l, m, n, fk, fj, fi) +
            FaceField<EIDX>(fine, coords, l, m, n, fk + o2[2], fj + o2[1], fi + o2[0]) -
            FaceField<EIDX>(fine, coords, l, m, n, fk + o3[2], fj + o3[1], fi + o3[0]));
  }

 public:
  template <int DIM, TopologicalElement fel = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &pcoords,
     const parthenon::Coordinates_t &pcoarse_coords,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pfine) {
    auto coords = Coordinates<geom>(pcoords);
    auto coarse_coords = Coordinates<geom>(pcoarse_coords);

    if constexpr (!IsSubmanifold(fel, cel)) {
      return;
    } else {
      constexpr int I1 = static_cast<int>(fel) % 3;
      if constexpr (I1 + 1 > DIM) {
        return;
      } else {
        auto &fine = *pfine;

        const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
        const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
        const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

        constexpr int I2 = (I1 + 1) % 3;
        constexpr int I3 = (I1 + 2) % 3;

        const Real d1 = coarse_coords.Dx(AxisFromInt(I1 + 1));
        const Real d2 = coarse_coords.Dx(AxisFromInt(I2 + 1));
        const Real d3 = coarse_coords.Dx(AxisFromInt(I3 + 1));
        const Real id1 = 1.0 / d1;
        const Real id2 = 1.0 / d2;
        const Real id3 = 1.0 / d3;

        // Get coarse face values at minus/plus
        // Note: Offsets<I, DIM, 2> gives {2,0,0} or {0,2,0} or {0,0,2} based on I
        constexpr auto N1 = Offsets<I1, DIM, 2>();
        const Real b1m =
            GetCoarseFaceVal<I1, I2, I3, DIM>(fine, coords, l, m, n, fk, fj, fi);
        const Real b1p = GetCoarseFaceVal<I1, I2, I3, DIM>(
            fine, coords, l, m, n, fk + N1[2], fj + N1[1], fi + N1[0]);

        constexpr auto N2 = Offsets<I2, DIM, 2>();
        const Real b2m =
            GetCoarseFaceVal<I2, I3, I1, DIM>(fine, coords, l, m, n, fk, fj, fi);
        const Real b2p = GetCoarseFaceVal<I2, I3, I1, DIM>(
            fine, coords, l, m, n, fk + N2[2], fj + N2[1], fi + N2[0]);

        constexpr auto N3 = Offsets<I3, DIM, 2>();
        const Real b3m =
            GetCoarseFaceVal<I3, I1, I2, DIM>(fine, coords, l, m, n, fk, fj, fi);
        const Real b3p = GetCoarseFaceVal<I3, I1, I2, DIM>(
            fine, coords, l, m, n, fk + N3[2], fj + N3[1], fi + N3[0]);

        // Normal differences
        const Real d1b2p =
            GetFaceNormDif<I2, I1, I3, DIM>(fine, coords, l, m, n, fk + N2[2], fj + N2[1],
                                            fi + N2[0]) *
            id1;
        const Real d1b2m =
            GetFaceNormDif<I2, I1, I3, DIM>(fine, coords, l, m, n, fk, fj, fi) * id1;

        const Real d1b3p =
            GetFaceNormDif<I3, I1, I2, DIM>(fine, coords, l, m, n, fk + N3[2], fj + N3[1],
                                            fi + N3[0]) *
            id1;
        const Real d1b3m =
            GetFaceNormDif<I3, I1, I2, DIM>(fine, coords, l, m, n, fk, fj, fi) * id1;

        const Real d2b1p =
            GetFaceNormDif<I1, I2, I3, DIM>(fine, coords, l, m, n, fk + N1[2], fj + N1[1],
                                            fi + N1[0]) *
            id2;
        const Real d2b1m =
            GetFaceNormDif<I1, I2, I3, DIM>(fine, coords, l, m, n, fk, fj, fi) * id2;

        const Real d3b1p =
            GetFaceNormDif<I1, I3, I2, DIM>(fine, coords, l, m, n, fk + N1[2], fj + N1[1],
                                            fi + N1[0]) *
            id3;
        const Real d3b1m =
            GetFaceNormDif<I1, I3, I2, DIM>(fine, coords, l, m, n, fk, fj, fi) * id3;

        // Cross terms (only for 3D)
        Real a23 = 0.0;
        if constexpr (DIM == 3) {
          // Cross difference terms would go here
          // For 2D R-Z we skip this
        }

        const Real a2 = 0.5 * (d2b1p + d2b1m);
        const Real a3 = 0.5 * (d3b1p + d3b1m);
        const Real b12 = id2 * (d1b2p - d1b2m);
        const Real c13 = id3 * (d1b3p - d1b3m);

        const Real a11 = -0.5 * (b12 + c13);
        const Real a0 = 0.5 * (b1p + b1m) - 0.25 * a11 * d1 * d1;

        // n1 is offset by 1 in the normal direction
        constexpr auto n1 = Offsets<I1, DIM>();
        // n2 is offset by 1 in the transverse direction (I2 if valid, else I3)
        constexpr auto n2 = (I2 < DIM) ? Offsets<I2, DIM>() : Offsets<I3, DIM>();

        // Set the two internal face values
        if constexpr (I2 < DIM) {
          fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) = a0 - 0.25 * d2 * a2;
          fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) =
              a0 + 0.25 * d2 * a2;
        } else {
          fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) = a0 - 0.25 * d3 * a3;
          fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) =
              a0 + 0.25 * d3 * a3;
        }

        // Divide by r for cylindrical (divide ALL components, matching FaceField scaling)
        if constexpr (geom == Geometry::cylindrical) {
          const Real eps = std::numeric_limits<Real>::epsilon();
          fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) /=
              (coords.Xf(Axis::IAXIS, fi + n1[0]) + eps);
          fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) /=
              (coords.Xf(Axis::IAXIS, fi + n1[0] + n2[0]) + eps);
        }
      }
    }
  }
};

/// TODO(acreyes): ProlongateInternalAverage
// In principle I guess we should weight the averaging based on the generalized volume of
// the elements

}  // namespace kamayan::grid
#endif  // GRID_REFINEMENT_OPERATIONS_HPP_
