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
#include "physics/material_properties/eos/eos.hpp"
#include "physics/material_properties/eos/eos_types.hpp"
#include "physics/material_properties/eos/equation_of_state.hpp"
#include "physics/physics_types.hpp"
#include "utils/instrument.hpp"

namespace kamayan::eos {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto eos_unit = std::make_shared<KamayanUnit>("eos");
  eos_unit->SetupParams.Register(SetupParams);
  eos_unit->InitializeData.Register(InitializeData);
  eos_unit->PreparePrimitive.Register(PreparePrimitive);
  eos_unit->PostMeshInitialization.Register(PrepareConserved);
  return eos_unit;
}

void SetupSpeciesParams(UnitData &ud, std::string spec) {
  ud.AddParm<std::string>("eos_type", "gamma", "Equation of state for " + spec,
                          {"gamma"});

  // gamma law
  ud.AddParm<Real>("gamma", 5.0 / 3.0,
                   "Ratio of specific heats to use in gamma law for " + spec);
}

void SetupParams(KamayanUnit *unit) {
  // general eos configurations
  auto &eos = unit->AddData("eos");
  // initialization
  eos.AddParm<std::string>("mode_init", "dens_pres",
                           "eos mode to call after initializing the grid.",
                           {"dens_pres", "dens_ener", "dens_temp"});
}

using supported_eos_options = OptTypeList<OptList<Fluid, Fluid::oneT>>;

void InitializeData(KamayanUnit *unit) {
  auto cfg = unit->Configuration();

  auto mode_init_str = unit->Data("eos").Get<std::string>("mode_init");
  auto mode_init =
      MapStrToEnum<EosMode>(mode_init_str, std::make_pair(EosMode::pres, "dens_pres"),
                            std::make_pair(EosMode::ener, "dens_ener"),
                            std::make_pair(EosMode::temp, "dens_temp"));

  unit->AddParam("mode_init", mode_init);

  // declare vars we will need
  auto fluid = cfg->Get<Fluid>();
  if (fluid == Fluid::oneT) {
    AddFields(EosVars<EosComponent::oneT>::types(), unit,
              {Metadata::Cell, Metadata::Overridable});
  }
}

template <Fluid fluid>
struct EosWrappedImpl {
  using options = OptTypeList<EosModeOptions<fluid>>;
  using eos_vars = EosVariables<fluid>;
  using value = void;

  template <EosMode mode>
  value dispatch(MeshData *md) {
    auto material_pkg = md->GetMeshPointer()->packages.Get("material");
    auto eos = material_pkg->Param<EOS_t>("eos");
    auto pack = grid::GetPack(eos_vars(), md);

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
  auto fluid = config->Get<Fluid>();
  if (fluid == Fluid::oneT) {
    Dispatcher<EosWrappedImpl<Fluid::oneT>>(PARTHENON_AUTO_LABEL, mode).execute(md);
  } else {
    PARTHENON_FAIL("ThreeT eos not implemented")
  }
  return TaskStatus::complete;
}

template <Fluid fluid>
struct EosWrappedBlkImpl {
  using eos_vars = EosVariables<fluid>;
  using options = OptTypeList<EosModeOptions<fluid>>;
  using value = void;

  template <EosMode mode>
  value dispatch(MeshBlock *mb) {
    auto material_pkg = mb->packages.Get("material");
    auto eos = material_pkg->Param<EOS_t>("eos");

    auto pack = grid::GetPack(eos_vars(), mb);

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
  auto fluid = config->Get<Fluid>();
  if (fluid == Fluid::oneT) {
    Dispatcher<EosWrappedBlkImpl<Fluid::oneT>>(PARTHENON_AUTO_LABEL, config->Get<Fluid>(),
                                               config->Get<EosModel>(), mode)
        .execute(mb);
  } else {
    PARTHENON_FAIL("ThreeT eos not implemented")
  }
  return TaskStatus::complete;
}

TaskStatus PreparePrimitive(MeshData *md) { return EosWrapped(md, EosMode::ener); }
TaskStatus PrepareConserved(MeshData *md) {
  auto eos_pkg = md->GetMeshPointer()->packages.Get("eos");
  return EosWrapped(md, eos_pkg->Param<EosMode>("mode_init"));
}

EOS_t MakeEos(std::vector<std::string> species, KamayanUnit *material_unit) {
  auto config = material_unit->Configuration();
  auto fluid = config->Get<Fluid>();
  if (species.size() > 1) {
    // build a multispecies eos and return it
  }
  if (fluid == Fluid::threeT) {
  } else {
    return EOS_t(MakeEosSingleSpecies(species[0], material_unit));
  }
  return EOS_t();
}

}  // namespace kamayan::eos
