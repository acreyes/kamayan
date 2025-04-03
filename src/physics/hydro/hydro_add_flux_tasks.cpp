#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"

namespace kamayan::hydro {

struct CalculateFluxes {
  using options = OptTypeList<HydroFactory>;
  using value = TaskStatus;

  template <typename hydro_traits>
  requires(HydroTraits_t<hydro_traits>)
  value dispatch(MeshData *md) {
    using conserved_vars = hydro_traits::Conserved;
    using reconstruct_vars = hydro_traits::Reconstruct;
    auto pack_recon = grid::GetPack(reconstruct_vars(), md);
    auto pack_flux = grid::GetPack(conserved_vars(), md);
  }
};

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md) {
  // calculate fluxes -- CalculateFluxes

  // needs to return task id from last task
  // return tl.AddTask(
  //     prev,
  //     [](MeshData *md) {
  //       auto cfg = GetConfig(md);
  //       Dispatcher<CalculateFluxes>(PARTHENON_AUTO_LABEL, cfg).execute(md);
  //     },
  //     md);
  return prev;
}
}  // namespace kamayan::hydro
