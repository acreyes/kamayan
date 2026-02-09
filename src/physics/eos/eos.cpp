#include <memory>
#include <string>
#include <utility>

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/subpack.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "kokkos_abstraction.hpp"
#include "physics/eos/eos.hpp"
#include "physics/eos/eos_types.hpp"
#include "physics/eos/equation_of_state.hpp"
#include "physics/physics_types.hpp"
#include "utils/instrument.hpp"

namespace kamayan::eos {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto eos_unit = std::make_shared<KamayanUnit>("Eos");
  eos_unit->SetupParams.Register(SetupParams);
  eos_unit->InitializeData.Register(InitializeData);
  // EOS should run AFTER hydro when preparing primitives
  eos_unit->PreparePrimitive.Register(PreparePrimitive, /*after=*/{"Hydro"});
  eos_unit->PrepareConserved.Register(PrepareConserved);
  return eos_unit;
}

void SetupParams(KamayanUnit *unit) {
  // general eos configurations
  auto &eos = unit->AddData("eos");
  eos.AddParm<EosModel>("model", "single",
                        "Type of Eos to use, single, tabulated or multitype.",
                        {{"single", EosModel::gamma},
                         {"tabulated", EosModel::tabulated},
                         {"multitype", EosModel::multitype}});

  auto &eos_single = unit->AddData("eos/single");
  // used in single fluid EoS
  eos_single.AddParm<Real>("Abar", 1.0, "Mean molecular weight in g/mol");

  // gamma law gas eos
  auto &eos_gamma = unit->AddData("eos/gamma");
  eos_gamma.AddParm<Real>("gamma", 1.4, "adiabatic index used in ideal gas EoS");

  // initialization
  eos.AddParm<std::string>("mode_init", "dens_pres",
                           "eos mode to call after initializing the grid.",
                           {"dens_pres", "dens_ener", "dens_temp"});

  // build the Eos Now
}

using supported_eos_options = OptTypeList<OptList<Fluid, Fluid::oneT>>;

struct AddEos {
  using options = supported_eos_options;
  using value = void;
  template <Fluid fluid>
  requires(fluid == Fluid::oneT)
  value dispatch(const EosModel model, StateDescriptor *pkg, KamayanUnit *unit) {
    EOS_t eos;
    if (model == EosModel::gamma) {
      auto gamma = unit->Data("eos/gamma").Get<Real>("gamma");
      auto abar = unit->Data("eos/single").Get<Real>("Abar");
      eos = EquationOfState<EosModel::gamma>(gamma, abar);
    } else {
      std::string msg =
          "EosModel " + unit->Data("eos").Get<std::string>("model") + "not implemented\n";
      PARTHENON_THROW(msg.c_str())
    }
    pkg->AddParam("EoS", eos);

    // declare vars we will need
    AddFields(EosVars<EosComponent::oneT>::types(), pkg,
              {Metadata::Cell, Metadata::Overridable});
  }
};

void InitializeData(KamayanUnit *unit) {
  auto cfg = unit->Configuration();
  auto model = cfg->Get<EosModel>();
  auto fluid = cfg->Get<Fluid>();

  auto mode_init_str = unit->Data("eos").Get<std::string>("mode_init");
  auto mode_init =
      MapStrToEnum<EosMode>(mode_init_str, std::make_pair(EosMode::pres, "dens_pres"),
                            std::make_pair(EosMode::ener, "dens_ener"),
                            std::make_pair(EosMode::temp, "dens_temp"));
  // unit IS the package (StateDescriptor)
  unit->AddParam("mode_init", mode_init);

  Dispatcher<AddEos>(PARTHENON_AUTO_LABEL, fluid).execute(model, unit, unit);
}

struct EosWrappedImpl {
  using eos_vars = EosVars<EosComponent::oneT>;
  using options = OptTypeList<OptList<Fluid, Fluid::oneT>,
                              OptList<EosModel, EosModel::gamma>, eos_vars::modes>;
  using value = void;

  template <Fluid fluid, EosModel model, EosMode mode>
  requires(fluid == Fluid::oneT)
  value dispatch(MeshData *md) {
    auto eos_pkg = md->GetMeshPointer()->packages.Get("Eos");
    auto eos = eos_pkg->Param<EOS_t>("EoS");
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
            auto indexer = SubPack(pack, b, k, j, i);
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

struct EosWrappedBlkImpl {
  using eos_vars = EosVars<EosComponent::oneT>;
  using options = OptTypeList<OptList<Fluid, Fluid::oneT>,
                              OptList<EosModel, EosModel::gamma>, eos_vars::modes>;
  using value = void;

  template <Fluid fluid, EosModel model, EosMode mode>
  requires(fluid == Fluid::oneT)
  value dispatch(MeshBlock *mb) {
    auto eos_pkg = mb->packages.Get("Eos");
    auto eos = eos_pkg->Param<EOS_t>("EoS");

    auto pack = grid::GetPack(eos_vars::types(), mb);

    auto cellbounds = mb->cellbounds;
    auto ib = cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    auto jb = cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    auto kb = cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

    const int scratch_level = 0;
    std::size_t scratch_size_in_bytes = ScratchPad1D::shmem_size(eos.nlambda());

    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, (ib.e - ib.s) * scratch_size_in_bytes, scratch_level, kb.s,
        kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &k, const int &j) {
          parthenon::par_for_inner(member, ib.s, ib.e, [&](const int &i) {
            ScratchPad1D lambda_view(member.team_scratch(scratch_level), eos.nlambda());
            auto lambda = ViewIndexer(lambda_view);
            auto indexer = SubPack(pack, 0, k, j, i);
            eos.template Call<EosComponent::oneT, mode>(indexer, lambda);
          });
        });
  }
};

TaskStatus EosWrapped(MeshBlock *mb, EosMode mode) {
  auto config = GetConfig(mb);
  Dispatcher<EosWrappedBlkImpl>(PARTHENON_AUTO_LABEL, config->Get<Fluid>(),
                                config->Get<EosModel>(), mode)
      .execute(mb);
  return TaskStatus::complete;
}

TaskStatus PreparePrimitive(MeshData *md) { return EosWrapped(md, EosMode::ener); }
TaskStatus PrepareConserved(MeshData *md) {
  auto eos_pkg = md->GetMeshPointer()->packages.Get("Eos");
  return EosWrapped(md, eos_pkg->Param<EosMode>("mode_init"));
}

}  // namespace kamayan::eos
