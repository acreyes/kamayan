#ifndef GRID_GRID_UPDATE_HPP_
#define GRID_GRID_UPDATE_HPP_
#include <parthenon/parthenon.hpp>

#include "basic_types.hpp"
#include "grid.hpp"
#include "grid_types.hpp"
#include "kamayan/fields.hpp"
#include "utils/error_checking.hpp"
#include "utils/parallel.hpp"

namespace kamayan::grid {

template <TopologicalElement... faces>
void FluxDivergence(MeshData *md, MeshData *dudt_data) {
  static auto desc_cc =
      GetPackDescriptor(md, {Metadata::Cell, Metadata::WithFluxes}, {PDOpt::WithFluxes});
  auto u0 = desc_cc.GetPack(md);
  auto dudt = desc_cc.GetPack(dudt_data);

  if (u0.GetMaxNumberOfVars() == 0) return;

  const int nblocks = u0.GetNBlocks();
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        using TE = TopologicalElement;
        const auto coords = u0.GetCoordinates(b);
        const Real dxi[sizeof...(faces)]{
            1.0 / coords.Dxc<static_cast<int>(faces) % 3 + 1>()...};
        // we have to check the variable bounds for each block in case
        // that we are using sparse fields
        for (int var = u0.GetLowerBound(b); var <= u0.GetUpperBound(b); var++) {
          dudt(b, var, km, jm, im) = 0.;
          (
              [&]() {
                constexpr int dir = static_cast<int>(faces) % 3;
                const int kp = km + (dir == 2);
                const int jp = jm + (dir == 1);
                const int ip = im + (dir == 0);

                dudt(b, var, km, jm, im) -=
                    dxi[dir] * (u0.flux(b, faces, var, kp, jp, ip) -
                                u0.flux(b, faces, var, km, jm, im));
              }(),
              ...);
        }
      });
}

KOKKOS_INLINE_FUNCTION constexpr int AxisFromFaceEdge(const TopologicalElement &face,
                                                      const TopologicalElement &edge) {
  using TE = TopologicalElement;
  int axis;
  switch (face) {
  case (TE::F1):
    axis = edge == TE::E3 ? 1 : 2;
    break;
  case (TE::F2):
    axis = edge == TE::E3 ? 0 : 2;
    break;
  case (TE::F3):
    axis = edge == TE::E1 ? 1 : 0;
    break;
  default:
    axis = 0;
    break;
  }
  return axis;
}

template <TopologicalElement Face, TopologicalElement... edges>
void FluxStokes(MeshData *md, MeshData *dudt_data) {
  static_assert(Face >= TopologicalElement::F1 && Face <= TopologicalElement::F3,
                "Face must be F1-F3");
  static_assert(sizeof...(edges) <= 2, "Stokes only supported with up to two edges.");

  static auto desc_fc =
      GetPackDescriptor(md, {Metadata::Face, Metadata::WithFluxes}, {PDOpt::WithFluxes});
  auto u0 = desc_fc.GetPack(md);
  auto dudt = desc_fc.GetPack(dudt_data);

  if (u0.GetMaxNumberOfVars() == 0) return;

  const int nblocks = u0.GetNBlocks();
  auto ib = md->GetBoundsI(IndexDomain::interior, Face);
  auto jb = md->GetBoundsJ(IndexDomain::interior, Face);
  auto kb = md->GetBoundsK(IndexDomain::interior, Face);

  constexpr std::size_t nedges = sizeof...(edges);
  constexpr TopologicalElement face_edges[] = {edges...};
  par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        using TE = TopologicalElement;
        const auto coords = u0.GetCoordinates(b);
        // we have to check the variable bounds for each block in case
        // that we are using sparse fields
        for (int var = u0.GetLowerBound(b); var <= u0.GetUpperBound(b); var++) {
          dudt(b, Face, var, km, jm, im) = 0.;
          // loop over our edges and add their contribution to the line integral
          (
              [&]() {
                int ijk[] = {im, jm, km};
                constexpr auto axis = AxisFromFaceEdge(Face, edges);
                ijk[axis] += 1;
                // need to determine particular cyclic permutation of our axis and edge
                // from the curl
                // d_t v_i ~ eps_{ijk}d_j E_k
                // j -- axis
                // k -- edge
                constexpr Real sign =
                    ((axis + 1) % 3 == static_cast<int>(edges) % 3) ? 1.0 : -1.0;
                dudt(b, Face, var, km, jm, im) +=
                    sign * (coords.Volume(parthenon::CellLevel::same, edges, km, jm, im) *
                                u0.flux(b, edges, var, km, jm, im) -
                            coords.Volume(parthenon::CellLevel::same, edges, ijk[2],
                                          ijk[1], ijk[0]) *
                                u0.flux(b, edges, var, ijk[2], ijk[1], ijk[0]));
              }(),
              ...);
          dudt(b, Face, var, km, jm, im) *=
              1. / coords.Volume(parthenon::CellLevel::same, Face, km, jm, im);
        }
      });
}

}  // namespace kamayan::grid
#endif  // GRID_GRID_UPDATE_HPP_
