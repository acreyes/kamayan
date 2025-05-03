#include <Kokkos_MinMax.hpp>

#include "dispatcher/options.hpp"
#include "grid/grid.hpp"
#include "grid/indexer.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/primconsflux.hpp"
#include "utils/parallel.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

struct EstimateTimeStep {
  using options = OptTypeList<HydroFactory>;
  using value = Real;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using vars = typename hydro_traits::ConsPrim;
    auto hydro = md->GetMeshPointer()->packages.Get("hydro");

    auto pack = grid::GetPack(vars(), md);
    const int ndim = md->GetNDim();
    const auto cfl = hydro->Param<Real>("cfl") / static_cast<Real>(ndim);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);

    Real dt_min;
    par_reduce(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &dt_local) {
          auto V = MakePackIndexer(pack, b, k, j, i);

          const auto &coords = pack.GetCoordinates(b);
          for (int dir = 0; dir < ndim; dir++) {
            const Real cfast = FastSpeed<hydro_traits::MHD>(dir, V);
            dt_local = Kokkos::min(dt_local, coords.Dx(dir + 1) /
                                                 (Kokkos::abs(V(VELOCITY(dir))) + cfast));
          }
        },
        Kokkos::Min<Real>(dt_min));
    return dt_min * cfl / ndim;
  }
};

Real EstimateTimeStepMesh(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<EstimateTimeStep>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}
}  // namespace kamayan::hydro
