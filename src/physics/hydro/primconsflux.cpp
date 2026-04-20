#include "physics/hydro/primconsflux.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/indexer.hpp"
#include "kamayan/config.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/robust.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "physics/hydro/hydro_types.hpp"

namespace kamayan::hydro {

// --8<-- [start:impl]
struct ConvertToConserved_impl {
  using options = OptTypeList<HydroFactory, grid::GeometryOptions>;
  using value = TaskStatus;

  template <typename hydro_traits, Geometry geom>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using Fields = ConcatTypeLists_t<typename hydro_traits::ConsPrim, grid::CoordFields>;
    auto pack = grid::GetPack(Fields(), md);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    const auto ndim = md->GetNDim();

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          capture(ndim);
          // also need to average the face-fields if doing constrained transport
          auto coords = grid::CoordinatePack<geom, grid::CoordFields>(pack, b);
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using te = TopologicalElement;
            if (ndim > 1) {
              pack(b, MAGC(0), k, j, i) =
                  0.5 * coords.template Dx<Axis::IAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::IAXIS>(k, j, i + 1) *
                       pack(b, te::F1, MAG(), k, j, i + 1) +
                   coords.template FaceArea<Axis::IAXIS>(k, j, i) *
                       pack(b, te::F1, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
              pack(b, MAGC(1), k, j, i) =
                  0.5 * coords.template Dx<Axis::JAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::JAXIS>(k, j + 1, i) *
                       pack(b, te::F2, MAG(), k, j + 1, i) +
                   coords.template FaceArea<Axis::JAXIS>(k, j, i) *
                       pack(b, te::F2, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
            }
            if (ndim > 2) {
              pack(b, MAGC(2), k, j, i) =
                  0.5 * coords.template Dx<Axis::KAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::KAXIS>(k + 1, j, i) *
                       pack(b, te::F3, MAG(), k + 1, j, i) +
                   coords.template FaceArea<Axis::KAXIS>(k, j, i) *
                       pack(b, te::F3, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
            }
          }
          // --8<-- [start:make-idx]
          auto U = SubPack(pack, b, k, j, i);
          Prim2Cons<hydro_traits>(U, U);
          // --8<-- [end:make-idx]
          if constexpr (geom == Geometry::cylindrical) {
            // conserve angular momentum
            U(MOMENTUM(2)) *= coords.template Xc<Axis::IAXIS>(k, j, i);
          }
        });
    return TaskStatus::complete;
  }
};
// --8<-- [end:impl]

struct PreparePrimitive_impl {
  using options = OptTypeList<HydroFactory, grid::GeometryOptions>;
  using value = TaskStatus;

  template <typename hydro_traits, Geometry geom>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using Fields = ConcatTypeLists_t<typename hydro_traits::ConsPrim, grid::CoordFields>;
    auto pack = grid::GetPack(Fields(), md);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    const auto ndim = md->GetNDim();

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          capture(ndim);
          auto coords = grid::CoordinatePack<geom, grid::CoordFields>(pack, b);
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using TE = TopologicalElement;
            if (ndim > 1) {
              pack(b, MAGC(0), k, j, i) =
                  0.5 * coords.template Dx<Axis::IAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::IAXIS>(k, j, i + 1) *
                       pack(b, TE::F1, MAG(), k, j, i + 1) +
                   coords.template FaceArea<Axis::IAXIS>(k, j, i) *
                       pack(b, TE::F1, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
              pack(b, MAGC(1), k, j, i) =
                  0.5 * coords.template Dx<Axis::JAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::JAXIS>(k, j + 1, i) *
                       pack(b, TE::F2, MAG(), k, j + 1, i) +
                   coords.template FaceArea<Axis::JAXIS>(k, j, i) *
                       pack(b, TE::F2, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
            }
            if (ndim > 2) {
              pack(b, MAGC(2), k, j, i) =
                  0.5 * coords.template Dx<Axis::KAXIS>(k, j, i) *
                  (coords.template FaceArea<Axis::KAXIS>(k + 1, j, i) *
                       pack(b, TE::F3, MAG(), k + 1, j, i) +
                   coords.template FaceArea<Axis::KAXIS>(k, j, i) *
                       pack(b, TE::F3, MAG(), k, j, i)) /
                  coords.CellVolume(k, j, i);
            }
          }
          auto U = SubPack(pack, b, k, j, i);
          Cons2Prim<hydro_traits>(U, U);
          if constexpr (geom == Geometry::cylindrical) {
            // conserve angular momentum
            U(VELOCITY(2)) *= utils::Ratio(1.0, coords.template Xc<Axis::IAXIS>(k, j, i));
            if constexpr (hydro_traits::MHD != Mhd::off) {
              U(MAGC(2)) *= coords.template Xc<Axis::IAXIS>(k, j, i);
            }
          }
        });
    return TaskStatus::complete;
  }
};

// --8<-- [start:prepare-cons]
TaskStatus PostMeshInitialization(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<ConvertToConserved_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}
// --8<-- [end:prepare-cons]

// should this be a part of the eos_wrapped call?
TaskStatus PreparePrimitive(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<PreparePrimitive_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}

}  // namespace kamayan::hydro
