#include "physics/hydro/hydro.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/fields.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/primconsflux.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "utils/parallel.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto hydro = std::make_shared<KamayanUnit>("hydro");
  hydro->Setup = Setup;
  hydro->Initialize = Initialize;
  hydro->PreparePrimitive = PreparePrimitive;
  hydro->PrepareConserved = PrepareConserved;
  hydro->AddFluxTasks = AddFluxTasks;

  return hydro;
}

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps) {
  auto reconstruction_str = rps->GetOrAdd<std::string>(
      "hydro", "reconstruction", "fog",
      "reconstruction method used to get Riemann States", {"fog", "plm", "ppm", "wenoz"});
  auto recon = MapStrToEnum<Reconstruction>(
      reconstruction_str, std::make_pair(Reconstruction::fog, "fog"),
      std::make_pair(Reconstruction::plm, "plm"),
      std::make_pair(Reconstruction::ppm, "ppm"),
      std::make_pair(Reconstruction::wenoz, "wenoz"));
  cfg->Add(recon);

  auto slope_limiter_str = rps->GetOrAdd<std::string>(
      "hydro", "slope_limiter", "minmod", "Slope limiter used in reconstruction.",
      {"minmod", "van_leer", "mc"});
  auto slope_limiter = MapStrToEnum<SlopeLimiter>(
      slope_limiter_str, std::make_pair(SlopeLimiter::minmod, "minmod"),
      std::make_pair(SlopeLimiter::van_leer, "van_leer"),
      std::make_pair(SlopeLimiter::mc, "mc"));
  cfg->Add(slope_limiter);

  // --8<-- [start:getoradd]
  auto riemann_str = rps->GetOrAdd<std::string>(
      "hydro", "riemann", "hll", "Riemann solver used for high order upwinded fluxes.",
      {"hll", "hllc"});
  auto riemann =
      MapStrToEnum<RiemannSolver>(riemann_str, std::make_pair(RiemannSolver::hll, "hll"),
                                  std::make_pair(RiemannSolver::hllc, "hllc"));
  cfg->Add(riemann);
  // --8<-- [end:getoradd]

  auto recon_vars_str = rps->GetOrAdd<std::string>(
      "hydro", "ReconstructionVars", "primitive",
      "Choice of variables used for reconstruction.", {"primitive"});
  auto recon_vars = MapStrToEnum<ReconstructVars>(
      recon_vars_str, std::make_pair(ReconstructVars::primitive, "primitive"));
  cfg->Add(recon_vars);

  auto emf_avg_str = rps->GetOrAdd<std::string>(
      "hydro", "EMF_averaging", "arithmetic",
      "Method to use for averaging the Face fluxes to edge electric field",
      {"arithmetic"});
  auto emf_avg = MapStrToEnum<EMFAveraging>(
      emf_avg_str, std::make_pair(EMFAveraging::arithmetic, "arithmetic"));
  cfg->Add(emf_avg);

  rps->Add<Real>("hydro", "cfl", 0.8, "CFL stability number use in hydro");
}

struct InitializeHydro {
  using options = OptTypeList<HydroFactory>;
  using value = void;
  template <typename hydro_vars>
  requires(NonTypeTemplateSpecialization<hydro_vars, HydroTraits>)
  value dispatch(StateDescriptor *pkg) {
    // --8<-- [start:hydro_add_fields]
    // conserved variables are Independent in each multi-stage buffer
    AddFields(typename hydro_vars::WithFlux(), pkg,
              {CENTER_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
    // primitive variables reference same data on each multi-stage buffer
    AddFields(typename hydro_vars::NonFlux(), pkg, {CENTER_FLAGS()});
    // --8<-- [end:hydro_add_fields]
    if constexpr (hydro_vars::MHD == Mhd::ct) {
      auto m = Metadata(std::vector<MetadataFlag>{
          FACE_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                              parthenon::refinement_ops::RestrictAverage,
                              parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
      pkg->AddField<MAG>(m);
    }
  }
};

std::shared_ptr<StateDescriptor>
Initialize(const Config *cfg, const runtime_parameters::RuntimeParameters *rps) {
  auto hydro_pkg = std::make_shared<StateDescriptor>("hydro");

  hydro_pkg->AddParam("cfl", rps->Get<Real>("hydro", "cfl"));

  Dispatcher<InitializeHydro>(PARTHENON_AUTO_LABEL, cfg).execute(hydro_pkg.get());

  hydro_pkg->EstimateTimestepMesh = EstimateTimeStepMesh;
  hydro_pkg->FillDerivedMesh = FillDerived;

  return hydro_pkg;
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
