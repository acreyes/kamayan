#include "physics/hydro/hydro.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/fields.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/primconsflux.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto hydro = std::make_shared<KamayanUnit>();
  hydro->Setup = Setup;
  hydro->Initialize = Initialize;
  hydro->PrepareConserved = PrepareConserved;
  hydro->AddFluxTasks = AddFluxTasks;

  return hydro;
}

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps) {
  auto reconstruction_str = rps->GetOrAdd<std::string>(
      "hydro", "reconstruction", "fog",
      "reconstruction method used to get Riemann States", {"fog"});
  auto recon = MapStrToEnum<Reconstruction>(reconstruction_str,
                                            std::make_pair(Reconstruction::fog, "fog"));
  cfg->Add(recon);

  auto riemann_str = rps->GetOrAdd<std::string>(
      "hydro", "riemann", "hll", "Riemann solver used for high order upwinded fluxes.",
      {"hll"});
  auto riemann =
      MapStrToEnum<RiemannSolver>(riemann_str, std::make_pair(RiemannSolver::hll, "hll"));
  cfg->Add(riemann);

  auto recon_vars_str = rps->GetOrAdd<std::string>(
      "hydro", "ReconstructionVars", "primitive",
      "Choice of variables used for reconstruction.", {"primitive"});
  auto recon_vars = MapStrToEnum<ReconstructVars>(
      recon_vars_str, std::make_pair(ReconstructVars::primitive, "primitive"));
  cfg->Add(recon_vars);
}

struct InitializeHydro {
  using options = OptTypeList<HydroFactory>;
  using value = void;
  template <typename hydro_vars>
  requires(NonTypeTemplateSpecialization<hydro_vars, HydroTraits>)
  value dispatch(StateDescriptor *pkg) {
    AddFields(typename hydro_vars::Conserved(), pkg,
              {CENTER_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
    AddFields(typename hydro_vars::Primitive(), pkg, {CENTER_FLAGS()});
    if constexpr (hydro_vars::MHD == Mhd::ct) {
      AddField<MAG>(pkg, {FACE_FLAGS(Metadata::Independent)});
      AddField<MAGC>(pkg, {CENTER_FLAGS()});
    }
  }
};

std::shared_ptr<StateDescriptor>
Initialize(const Config *cfg, const runtime_parameters::RuntimeParameters *rps) {
  auto hydro_pkg = std::make_shared<StateDescriptor>("hydro");
  Dispatcher<InitializeHydro>(PARTHENON_AUTO_LABEL, cfg).execute(hydro_pkg.get());
  return hydro_pkg;
}

}  // namespace kamayan::hydro
