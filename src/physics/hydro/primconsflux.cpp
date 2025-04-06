#include "physics/hydro/primconsflux.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/indexer.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

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

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto U = MakePackIndexer(pack, b, k, j, i);
          Prim2Cons<hydro_traits>(U, U);
        });
    return TaskStatus::complete;
  }
};

TaskStatus PrepareConserved(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<PrepareConserved_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}
}  // namespace kamayan::hydro
