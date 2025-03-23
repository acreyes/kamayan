#include <memory>
#include <string>

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/indexer.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kokkos_abstraction.hpp"
#include "physics/eos/eos.hpp"
#include "physics/eos/eos_types.hpp"
#include "physics/eos/equation_of_state.hpp"
#include "utils/instrument.hpp"

namespace kamayan::eos {
namespace rp = runtime_parameters;
std::shared_ptr<KamayanUnit> ProcessUnit() {
  std::shared_ptr<KamayanUnit> eos_unit;
  eos_unit->Setup = Setup;
  eos_unit->Initialize = Initialize;
  return eos_unit;
}

void Setup(Config *cfg, rp::RuntimeParameters *rps) {
  // general eos configurations
  auto eos_model_str = rps->GetOrAdd<std::string>(
      "eos", "model", "single", "Type of Eos to use, single, tabulated or multitype.",
      {"single", "tabulated", "multitype"});
  EosModel type;
  if (eos_model_str == "single") {
    type = EosModel::gamma;
  } else if (eos_model_str == "tabulated") {
    type = EosModel::tabulated;
  } else if (eos_model_str == "multitype") {
    type = EosModel::multitype;
  }
  cfg->Add(type);
  // used in single fluid EoS
  rps->Add<Real>("eos/single", "Abar", 1.0, "Mean molecular weight in g/mol");

  // gamma law gas eos
  rps->Add<Real>("eos/gamma", "gamma", 1.4, "adiabatic index used in ideal gas EoS");

  // build the Eos Now
}

using supported_eos_options = OptTypeList<OptList<Fluid, Fluid::oneT>>;

struct AddEos {
  using options = supported_eos_options;
  using value = void;
  template <Fluid fluid>
  value dispatch(const EosModel model, StateDescriptor *pkg,
                 const runtime_parameters::RuntimeParameters *rps) {
    if (model == EosModel::gamma) {
      auto gamma = rps->Get<Real>("eos/gamma", "gamma");
      auto abar = rps->Get<Real>("eos/single", "abar");
      auto eos = EquationOfState<EosModel::gamma>(gamma, abar);
      pkg->AddParam("EoS", eos);
    } else {
      std::string msg =
          "EosModel " + rps->Get<std::string>("eos", "model") + "not implemented\n";
      PARTHENON_THROW(msg.c_str())
    }
  }
};

std::shared_ptr<StateDescriptor>
Initialize(const Config *cfg, const runtime_parameters::RuntimeParameters *rps) {
  auto eos_pkg = std::make_shared<StateDescriptor>("Eos");
  auto model = cfg->Get<EosModel>();
  auto fluid = cfg->Get<Fluid>();

  Dispatcher<AddEos>(PARTHENON_AUTO_LABEL, fluid).execute(model, eos_pkg.get(), rps);

  return eos_pkg;
}

struct EosWrappedImpl {
  using eos_vars = EosVars<EosComponent::oneT>;
  using options = OptTypeList<OptList<Fluid, Fluid::oneT>,
                              OptList<EosModel, EosModel::gamma>, eos_vars::modes>;
  using value = void;

  template <Fluid fluid, EosModel model, EosMode mode>
  value dispatch(MeshData *md) {
    auto eos_pkg = md->GetMeshPointer()->packages.Get("Eos");
    auto eos = eos_pkg->Param<EquationOfState<model>>("EoS");
    auto pack = grid::GetPack(eos_vars::types(), md);

    auto ib = md->GetBoundsI(parthenon::IndexDomain::interior);
    auto jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
    auto kb = md->GetBoundsK(parthenon::IndexDomain::interior);

    const int scratch_level = 0;
    std::size_t scratch_size_in_bytes = ScratchPad1D::shmem_size(eos.nlambda());

    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, (ib.e - ib.s) * scratch_size_in_bytes, scratch_level, 0,
        pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b, const int &k,
                      const int &j) {
          parthenon::par_for_inner(member, ib.s, ib.e, [&](const int &i) {
            ScratchPad1D lambda_view(member.team_scratch(scratch_level), eos.nlambda());
            auto lambda = ViewIndexer(lambda_view);
            auto indexer = MakePackIndexer(pack, b, k, j, i);
            eos.template Call<EosComponent::oneT, mode>(indexer, lambda);
          });
        });
  }
};

TaskStatus EosWrapped(MeshData *md, EosMode mode) {
  auto config = GetConfig(md);
  Dispatcher<EosWrappedImpl>(PARTHENON_AUTO_LABEL, config->Get<Fluid>(),
                             config->Get<EosModel>(), mode)
      .execute(md);
  return TaskStatus::complete;
}
}  // namespace kamayan::eos
