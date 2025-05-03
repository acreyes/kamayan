#include "physics/hydro/primconsflux.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/indexer.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "utils/parallel.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

// --8<-- [start:impl]
struct PrepareConserved_impl {
  using options = OptTypeList<HydroFactory>;
  using value = TaskStatus;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    auto pack = grid::GetPack(typename hydro_traits::ConsPrim(), md);
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
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using te = TopologicalElement;
            if (ndim > 1) {
              pack(b, MAGC(0), k, j, i) = 0.5 * (pack(b, te::F1, MAG(), k, j, i + 1) +
                                                 pack(b, te::F1, MAG(), k, j, i));
              pack(b, MAGC(1), k, j, i) = 0.5 * (pack(b, te::F2, MAG(), k, j + 1, i) +
                                                 pack(b, te::F2, MAG(), k, j, i));
            }
            if (ndim > 2) {
              pack(b, MAGC(2), k, j, i) = 0.5 * (pack(b, te::F3, MAG(), k + 1, j, i) +
                                                 pack(b, te::F3, MAG(), k, j, i));
            }
          }
          // --8<-- [start:make-idx]
          auto U = MakePackIndexer(pack, b, k, j, i);
          Prim2Cons<hydro_traits>(U, U);
          // --8<-- [end:make-idx]
        });
    return TaskStatus::complete;
  }
};
// --8<-- [end:impl]

struct PreparePrimitive_impl {
  using options = OptTypeList<HydroFactory>;
  using value = TaskStatus;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    auto pack = grid::GetPack(typename hydro_traits::ConsPrim(), md);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    const auto ndim = md->GetNDim();

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          capture(ndim);
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using te = TopologicalElement;
            if (ndim > 1) {
              pack(b, MAGC(0), k, j, i) = 0.5 * (pack(b, te::F1, MAG(), k, j, i + 1) +
                                                 pack(b, te::F1, MAG(), k, j, i));
              pack(b, MAGC(1), k, j, i) = 0.5 * (pack(b, te::F2, MAG(), k, j + 1, i) +
                                                 pack(b, te::F2, MAG(), k, j, i));
            }
            if (ndim > 2) {
              pack(b, MAGC(2), k, j, i) = 0.5 * (pack(b, te::F3, MAG(), k + 1, j, i) +
                                                 pack(b, te::F3, MAG(), k, j, i));
            }
          }
          auto U = MakePackIndexer(pack, b, k, j, i);
          Cons2Prim<hydro_traits>(U, U);
        });
    return TaskStatus::complete;
  }
};

// does this only need to happen during initialization?
// --8<-- [start:prepare-cons]
TaskStatus PrepareConserved(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<PrepareConserved_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}
// --8<-- [end:prepare-cons]

// should this be a part of the eos_wrapped call?
TaskStatus PreparePrimitive(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<PreparePrimitive_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}

}  // namespace kamayan::hydro
