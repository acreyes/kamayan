#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

struct CalculateFluxes {
  using options = OptTypeList<HydroFactory>;
  using value = TaskStatus;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using conserved_vars = typename hydro_traits::Conserved;
    using reconstruct_vars = typename hydro_traits::Reconstruct;
    auto pack_recon = grid::GetPack(reconstruct_vars(), md);
    auto pack_flux = grid::GetPack(conserved_vars(), md);

    return TaskStatus::complete;
  }
};

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md) {
  // calculate fluxes -- CalculateFluxes

  // needs to return task id from last task
  return tl.AddTask(
      prev,
      [](MeshData *md) {
        auto cfg = GetConfig(md);
        return Dispatcher<CalculateFluxes>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
      },
      md);
}
}  // namespace kamayan::hydro
