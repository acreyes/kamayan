#include "grid/coordinates.hpp"

#include "basic_types.hpp"
#include "grid/geometry.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan_utils/parallel.hpp"

namespace kamayan::grid {

template <Geometry geom, typename T, typename Functor>
void FillCoords(const Functor &functor, const parthenon::IndexShape cellbounds) {
  auto [kb, jb, ib] = CoordinateIndexRanges<geom, T>(cellbounds);
  par_for(PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, functor);
}

template <Geometry geom>
void CalculateCoordinates(const Coordinates<geom> &coords, auto &pack,
                          const parthenon::IndexShape &cellbounds) {
  FillCoords<geom, coords::Volume>(
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::Volume(), k, j, i) = coords.CellVolume(k, j, i);
      },
      cellbounds);

  [&]<Axis... axes>() {
    (FillCoords<geom, coords::Dx<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::Dx<axes>(), k, j, i) = coords.template Dx<axes>();
         },
         cellbounds),
     ...);
    (FillCoords<geom, coords::X<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::X<axes>(), k, j, i) = coords.template Xi<axes>(k, j, i);
         },
         cellbounds),
     ...);
    (FillCoords<geom, coords::Xc<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::Xc<axes>(), k, j, i) = coords.template Xc<axes>(k, j, i);
         },
         cellbounds),
     ...);
    (FillCoords<geom, coords::Xf<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::Xf<axes>(), k, j, i) = coords.template Xf<axes>(k, j, i);
         },
         cellbounds),
     ...);
    (FillCoords<geom, coords::FaceArea<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::FaceArea<axes>(), k, j, i) =
               coords.template FaceArea<axes>(k, j, i);
         },
         cellbounds),
     ...);
    (FillCoords<geom, coords::EdgeLength<axes>>(
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
           pack(0, coords::EdgeLength<axes>(), k, j, i) =
               coords.template EdgeLength<axes>(k, j, i);
         },
         cellbounds),
     ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();
}

void CalculateCoordinates(MeshBlock *mb) {
  const auto geometry = GetConfig(mb)->Get<Geometry>();
  GeometryOptions::dispatch(
      [&]<Geometry geom>() {
        auto cellbounds = mb->cellbounds;

        auto pack = GetPack(CoordFields(), mb);
        auto coords = Coordinates<geom>(pack, 0);

        FillCoords<geom, coords::Volume>(
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
              pack(0, coords::Volume(), k, j, i) = coords.CellVolume(k, j, i);
            },
            cellbounds);

        [&]<Axis... axes>() {
          (FillCoords<geom, coords::Dx<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::Dx<axes>(), k, j, i) = coords.template Dx<axes>();
               },
               cellbounds),
           ...);
          (FillCoords<geom, coords::X<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::X<axes>(), k, j, i) = coords.template Xi<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, coords::Xc<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::Xc<axes>(), k, j, i) = coords.template Xc<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, coords::Xf<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::Xf<axes>(), k, j, i) = coords.template Xf<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, coords::FaceArea<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::FaceArea<axes>(), k, j, i) =
                     coords.template FaceArea<axes>(k, j, i);
               },
               cellbounds),
           ...);
          (FillCoords<geom, coords::EdgeLength<axes>>(
               KOKKOS_LAMBDA(const int k, const int j, const int i) {
                 pack(0, coords::EdgeLength<axes>(), k, j, i) =
                     coords.template EdgeLength<axes>(k, j, i);
               },
               cellbounds),
           ...);
        }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();
      },
      geometry);
}

void CalculateCoordinates(MeshBlock *mb, Geometry geom) {
  GeometryOptions::dispatch(
      [&]<Geometry g>() {
        auto cellbounds = mb->cellbounds;

        auto pack = GetPack(CoordFields(), mb);
        auto coords = Coordinates<g>(pack, 0);

        CalculateCoordinates(coords, pack, cellbounds);
      },
      geom);
}

}  // namespace kamayan::grid
