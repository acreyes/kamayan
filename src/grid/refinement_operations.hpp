#ifndef GRID_REFINEMENT_OPERATIONS_HPP_
#define GRID_REFINEMENT_OPERATIONS_HPP_
#include <algorithm>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/grid_types.hpp"
#include "interface/variable_state.hpp"
namespace kamayan::grid {

namespace util {
// compute distances from cell center to the nearest center in the + or -
// coordinate direction. Do so for both coarse and fine grids.
template <int DIM, TopologicalElement EL>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const parthenon::Coordinates_t &coords,
                const parthenon::Coordinates_t &coarse_coords,
                const parthenon::IndexRange &cib, const parthenon::IndexRange &ib, int i,
                int fi, Real *dxm, Real *dxp, Real *dxfm, Real *dxfp) {
  // here "f" signifies the fine grid, not face locations.
  const Real xm = coarse_coords.X<DIM, EL>(i - 1);
  const Real xc = coarse_coords.X<DIM, EL>(i);
  const Real xp = coarse_coords.X<DIM, EL>(i + 1);
  *dxm = xc - xm;
  *dxp = xp - xc;
  const Real fxm = coords.X<DIM, EL>(fi);
  const Real fxp = coords.X<DIM, EL>(fi + 1);
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

    const Real fc = coarse(element_idx, l, m, n, k, j, i);

    Real dx1fm = 0;
    [[maybe_unused]] Real dx1fp = 0;
    [[maybe_unused]] Real gx1m = 0, gx1p = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      GetGridSpacings<1, el>(pcoords, pcoarse_coords, cib, ib, i, fi, &dx1m, &dx1p,
                             &dx1fm, &dx1fp);

      Real gx1c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k, j, i - 1),
                     coarse(element_idx, l, m, n, k, j, i + 1), dx1m, dx1p, gx1m, gx1p);
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
      GetGridSpacings<2, el>(pcoords, pcoarse_coords, cjb, jb, j, fj, &dx2m, &dx2p,
                             &dx2fm, &dx2fp);
      Real gx2c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k, j - 1, i),
                     coarse(element_idx, l, m, n, k, j + 1, i), dx2m, dx2p, gx2m, gx2p);
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
      GetGridSpacings<3, el>(pcoords, pcoarse_coords, ckb, kb, k, fk, &dx3m, &dx3p,
                             &dx3fm, &dx3fp);
      Real gx3c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k - 1, j, i),
                     coarse(element_idx, l, m, n, k + 1, j, i), dx3m, dx3p, gx3m, gx3p);
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

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(element_idx, l, m, n, fk, fj, fi) =
        fc - (gx1m * dx1fm + gx2m * dx2fm + gx3m * dx3fm);
    if constexpr (INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm - gx3m * dx3fm);
    if constexpr (INCLUDE_X2)
      fine(element_idx, l, m, n, fk, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp + gx3m * dx3fm);
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp - gx3m * dx3fm);
    if constexpr (INCLUDE_X3)
      fine(element_idx, l, m, n, fk + 1, fj, fi) =
          fc - (gx1m * dx1fm + gx2m * dx2fm - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm + gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp + gx3p * dx3fp);
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
      auto get_geometric_factor = [&](int v, int u, int t) {
        if constexpr (geom == Geometry::cylindrical) {
          return coords.template Xf<fel, Axis::IAXIS>(fi + t);
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
        // Note step size of 2 for the direction normal to the eidx2/eidx3
        for (const int u : iarr2{0, 2}) {
          for (const int t : iarr2{0, 1}) {
            // we should multiply by the radial coordinate in cylindrical
            // coordinates for r-faces
            const auto fine2 =
                get_fine_permuted(1, v, u, t) * get_geometric_factor(v, u, t);
            const auto fine3 =
                get_fine_permuted(2, u, v, t) * get_geometric_factor(v, u, t);
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
      const auto dx2 = std::pow(coarse_coords.Dxc(dir1, k, j, i), 2);
      const auto dy2 = std::pow(coarse_coords.Dxc(dir2, k, j, i), 2);
      const auto dz2 = std::pow(coarse_coords.Dxc(dir3, k, j, i), 2);
      Vxyz *= 0.125 * dz2 / (dx2 + dz2);
      Wxyz *= 0.125 * dy2 / (dx2 + dy2);

      for (int ok : iarr2{0, 1}) {
        for (int oj : iarr2{0, 1}) {
          get_fine_permuted(0, ok, oj, 1) =
              0.5 * (get_fine_permuted(0, ok, oj, 0) + get_fine_permuted(0, ok, oj, 2)) +
              Uxx + sg(ok) * Vxyz + sg(oj) * Wxyz;
          get_fine_permuted(0, ok, oj, 1) *= 1. / get_geometric_factor(ok, oj, 1);
        }
      }
    }
  }
};
}  // namespace kamayan::grid
#endif  // GRID_REFINEMENT_OPERATIONS_HPP_
