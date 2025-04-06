#ifndef GRID_GRID_UPDATE_HPP_
#define GRID_GRID_UPDATE_HPP_
#include <parthenon/parthenon.hpp>

#include "grid.hpp"
#include "grid_types.hpp"
#include "kamayan/fields.hpp"

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
  parthenon::par_for(
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
}  // namespace kamayan::grid
#endif  // GRID_GRID_UPDATE_HPP_
