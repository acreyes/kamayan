#include "physics/hydro/hydro.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <prolong_restrict/pr_ops.hpp>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "hydro_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/unit_data.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/primconsflux.hpp"

namespace kamayan::hydro {

std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto hydro = std::make_shared<KamayanUnit>("hydro");
  // --8<-- [start:register]
  hydro->SetupParams.Register(SetupParams);
  hydro->InitializeData.Register(InitializeData);
  // need to convert to primitives before calling equation of state
  hydro->PreparePrimitive.Register(PreparePrimitive, /*after=*/{}, /*before=*/{"eos"});
  hydro->PostMeshInitialization.Register(PostMeshInitialization, {"eos"});
  hydro->AddFluxTasks.Register(AddFluxTasks);
  // --8<-- [end:register]
  return hydro;
}

void SetupParams(KamayanUnit *unit) {
  //
  auto &hydro_data = unit->AddData("hydro");

  hydro_data.AddParm<Reconstruction>("reconstruction", "fog",
                                     "reconstruction method used to get Riemann States",
                                     {{"fog", Reconstruction::fog},
                                      {"plm", Reconstruction::plm},
                                      {"ppm", Reconstruction::ppm},
                                      {"wenoz", Reconstruction::wenoz}});

  hydro_data.AddParm<SlopeLimiter>("slope_limiter", "minmod",
                                   "Slope limiter used in reconstruction.",
                                   {{"minmod", SlopeLimiter::minmod},
                                    {"van_leer", SlopeLimiter::van_leer},
                                    {"mc", SlopeLimiter::mc},
                                    {"thinc", SlopeLimiter::thinc}});

  hydro_data.AddParm<ThincFallbackLimiter>(
      "thinc_fallback", "mc",
      "Fallback slope limiter used inside THINC reconstruction.",
      {{"minmod", ThincFallbackLimiter::minmod},
       {"van_leer", ThincFallbackLimiter::van_leer},
       {"mc", ThincFallbackLimiter::mc}});
  hydro_data.AddParm<Real>("beta_thinc", 1.6, "THINC sharpness parameter.");
  hydro_data.AddParm<bool>("thinc_dens", true, "Apply THINC to density.");
  hydro_data.AddParm<bool>("thinc_eint", false, "Apply THINC to internal energy.");
  hydro_data.AddParm<Real>("thinc_threshold", 0.1,
                           "BVD threshold to avoid degenerate comparisons near zero.");

  hydro_data.AddParm<RiemannSolver>(
      "riemann", "hll", "Riemann solver used for high order upwinded fluxes.",
      {{"hll", RiemannSolver::hll}, {"hllc", RiemannSolver::hllc}});

  hydro_data.AddParm<ReconstructVars>("ReconstructionVars", "primitive",
                                      "Choice of variables used for reconstruction.",
                                      {{"primitive", ReconstructVars::primitive}});

  hydro_data.AddParm<ReconstructionStrategy>(
      "ReconstructionStrategy", "scratchpad",
      "Loop strategy for reconstruction and riemann solve.",
      {{"scratchpad", ReconstructionStrategy::scratchpad},
       {"scratchvar", ReconstructionStrategy::scratchvar}});

  // --8<-- [start:add_parm]
  // since EMFAveraging was declared with the POLYMORPHIC_PARM macro
  // this will get mapped to the Config
  hydro_data.AddParm<EMFAveraging>(
      "EMF_averaging", "arithmetic",
      "Method to use for averaging the Face fluxes to edge electric field",
      {{"arithmetic", EMFAveraging::arithmetic}});

  // runtime parameter type (Real, int, string, bool) get mapped to the Params
  hydro_data.AddParm<Real>("cfl", 0.8, "CFL stability number use in hydro");
  // --8<-- [end:add_parm]
}

struct InitializeHydro {
  using options = OptTypeList<HydroFactory>;
  using value = void;
  template <typename hydro_vars>
  requires(NonTypeTemplateSpecialization<hydro_vars, HydroTraits>)
  value dispatch(KamayanUnit *unit, Config *cfg) {
    // --8<-- [start:hydro_add_fields]
    // conserved variables are Independent in each multi-stage buffer
    AddFields(typename hydro_vars::WithFlux(), unit,
              {CENTER_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
    // primitive variables reference same data on each multi-stage buffer
    AddFields(typename hydro_vars::NonFlux(), unit, {CENTER_FLAGS()});
    // --8<-- [end:hydro_add_fields]
    // THINC sensor: diagnostic field showing where BVD activates THINC
    if (cfg->Get<SlopeLimiter>() == SlopeLimiter::thinc) {
      AddField<THINC_SENSOR>(unit, {Metadata::Cell, Metadata::OneCopy});
    }
    if constexpr (hydro_vars::MHD == Mhd::ct) {
      auto m = Metadata(std::vector<MetadataFlag>{
          FACE_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                              parthenon::refinement_ops::RestrictAverage,
                              parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
      unit->AddField<MAG>(m);
    }

    if (cfg->Get<ReconstructionStrategy>() == ReconstructionStrategy::scratchvar) {
      using reconstruct_vars = typename hydro_vars::Reconstruct;
      using RS = RiemannScratch;
      auto riemann_scratch = RS::type();
      // This really should include all mass scalars
      constexpr int nrecon = count_components(reconstruct_vars());

      auto nspecies =
          static_cast<int>(unit->GetUnit("material").Param<std::size_t>("nspecies"));
      nspecies = nspecies > 1 ? nspecies : 0;
      riemann_scratch.template RegisterShape<RS::Minus>({nrecon + nspecies});
      riemann_scratch.template RegisterShape<RS::Plus>({nrecon + nspecies});

      unit->AddParam("riemann_scratch", riemann_scratch);
      AddScratch(riemann_scratch, unit);
    }
  }
};

void InitializeData(KamayanUnit *unit) {
  auto cfg = unit->Configuration();
  // unit IS the package (StateDescriptor)
  Dispatcher<InitializeHydro>(PARTHENON_AUTO_LABEL, cfg.get()).execute(unit, cfg.get());

  unit->EstimateTimestepMesh = EstimateTimeStepMesh;
  unit->FillDerivedMesh = FillDerived;
}

struct FillDerived_impl {
  using options = OptTypeList<HydroFactory>;
  using value = TaskStatus;

  template <typename hydro_traits>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    auto pack = grid::GetPack(typename hydro_traits::All(), md);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    const auto ndim = md->GetNDim();

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          capture(ndim);
          const auto coords = pack.GetCoordinates(b);
          // also need to average the face-fields if doing constrained transport
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using te = TopologicalElement;
            if (ndim > 1) {
              pack(b, DIVB(), k, j, i) = 1. / coords.template Dxc<1>() *
                                             (pack(b, te::F1, MAG(), k, j, i + 1) -
                                              pack(b, te::F1, MAG(), k, j, i)) +
                                         1. / coords.template Dxc<2>() *
                                             (pack(b, te::F2, MAG(), k, j + 1, i) -
                                              pack(b, te::F2, MAG(), k, j, i));
            }
            // if (ndim > 2) {
            // }
          }
        });
    return TaskStatus::complete;
  }
};
TaskStatus FillDerived(MeshData *md) {
  auto cfg = GetConfig(md);
  return Dispatcher<FillDerived_impl>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
}

}  // namespace kamayan::hydro
