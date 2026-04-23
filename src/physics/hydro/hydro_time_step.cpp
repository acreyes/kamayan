#include <Kokkos_MinMax.hpp>

#include "dispatcher/options.hpp"
#include "grid/coordinates.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/subpack.hpp"
#include "kamayan/config.hpp"
#include "kamayan/fields.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/primconsflux.hpp"

namespace kamayan::hydro {

struct EstimateTimeStep {
  using options = OptTypeList<HydroFactory>;
  using value = Real;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using vars = ConcatTypeLists_t<typename hydro_traits::ConsPrim, grid::CoordFields>;

    auto pack = grid::GetPack(vars(), md);
    const int ndim = md->GetNDim();
    // --8<-- [start:get_param]
    // pull out params from owning unit with full input parameter block + key
    auto hydro = md->GetMeshPointer()->packages.Get("hydro");
    const auto cfl = hydro->Param<Real>("hydro/cfl") / static_cast<Real>(ndim);
    // --8<-- [end:get_param]
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);

    const auto geometry = GetConfig(md)->Get<Geometry>();

    Real dt_min;
    par_reduce(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &dt_local) {
          auto V = SubPack(pack, b, k, j, i);

          const auto coords = grid::GenericCoordinatePack(geometry, pack, b);
          for (int dir = 0; dir < ndim; dir++) {
            const Real cfast = FastSpeed<hydro_traits::MHD>(dir, V);
            dt_local = Kokkos::min(dt_local, coords.Dx(AxisFromInt(dir + 1), k, j, i) /
                                                 (Kokkos::abs(V(VELOCITY(dir))) + cfast));
          }

          if (geometry == Geometry::cylindrical) {
            dt_local =
                Kokkos::min(dt_local, coords.template X<Axis::IAXIS>(k, j, i) /
                                          Kokkos::abs(pack(b, VELOCITY(2), k, j, i)));
          }
        },
        Kokkos::Min<Real>(dt_min));
    return dt_min * cfl;
  }
};

Real EstimateTimeStepMesh(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<EstimateTimeStep>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}
}  // namespace kamayan::hydro
