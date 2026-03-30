#include "grid/coordinates.hpp"

#include "basic_types.hpp"
#include "grid/geometry.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan_utils/parallel.hpp"

namespace kamayan::grid {

// for a given variable T, need to:
//    * get index bounds
//    * launch par_for
//    * fill pack(T(), k, j, i) for the mb
//       * means calling the coords method
//       * pass a KOKKOS_LAMBDA(k,j,i) ?
template <Geometry geom, typename T, typename Functor>
void FillCoords(const Functor &functor, const parthenon::IndexShape cellbounds) {
  auto [kb, jb, ib] = CoordinateIndexRanges<geom, T>(cellbounds);
  par_for(PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, functor);
}

void CalculateCoordinates(MeshBlock *mb) {
  const auto geometry = GetConfig(mb)->Get<Geometry>();
  GeometryOptions::dispatch(
      [&]<Geometry geom>() {
        auto cellbounds = mb->cellbounds;

        auto pack = GetPack(CoordFields(), mb);
        auto coords = Coordinates<geom>(pack, 0);

        FillCoords<geom, Volume>(
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
              pack(0, Volume(), k, j, i) = coords.CellVolume(k, j, i);
            },
            cellbounds);

        [&]<Axis... axes>() {
          (FillCoords<geom, Dx<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, Dx<axes>(), k, j, i) = coords.template Dx<axes>();
               },
               cellbounds),
           ...);
          (FillCoords<geom, X<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, X<axes>(), k, j, i) = coords.template Xi<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, Xc<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, Xc<axes>(), k, j, i) = coords.template Xc<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, Xf<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, Xf<axes>(), k, j, i) = coords.template Xf<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, FaceArea<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, FaceArea<axes>(), k, j, i) =
                     coords.template FaceArea<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, EdgeLength<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, EdgeLength<axes>(), k, j, i) =
                     coords.template EdgeLength<axes>(k, j, i);
               },
               cellbounds),
           ...);
        }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();
      },
      geometry);
}
}  // namespace kamayan::grid
