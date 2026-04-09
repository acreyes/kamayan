#include "physics/hydro/hydro.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <prolong_restrict/pr_ops.hpp>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/refinement_operations.hpp"
#include "grid/subpack.hpp"
#include "hydro_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/unit_data.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "kokkos_types.hpp"
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
  // for cylindrical geometry we also need to call prepareprimitive
  // for the azimuthal magnetic field
  hydro->PrepareConserved.Register(PrepareConserved, /*after=*/{}, /*before=*/{});
  hydro->PostMeshInitialization.Register(PostMeshInitialization, {"eos"});
  hydro->AddFluxTasks.Register(AddFluxTasks);
  hydro->AddTasksOneStep.Register(AddTasksOneStep);
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
                                    {"mc", SlopeLimiter::mc}});
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
    if constexpr (hydro_vars::MHD == Mhd::ct) {
      auto m = Metadata(std::vector<MetadataFlag>{
          FACE_FLAGS(Metadata::Independent, Metadata::WithFluxes)});
      auto register_ops = [&]<Geometry geom>() {
        m.RegisterRefinementOps<grid::ProlongateSharedMinMod<geom>,
                                grid::RestrictAverage<geom>,
                                grid::ProlongateInternalTothAndRoe<geom>>();
      };
      const auto geometry = unit->Configuration()->Get<Geometry>();
      const auto handled = grid::GeometryOptions::dispatch(register_ops, geometry);
      PARTHENON_REQUIRE_THROWS(handled,
                               "Geometry not handled for refinement operations.");
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
    const auto geometry = GetConfig(md)->Get<Geometry>();

    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          capture(ndim);
          const auto coords = grid::GenericCoordinate(geometry, pack.GetCoordinates(b));
          // also need to average the face-fields if doing constrained transport
          if constexpr (hydro_traits::MHD == Mhd::ct) {
            using te = TopologicalElement;
            if (ndim > 1) {
              pack(b, DIVB(), k, j, i) = 1. / coords.template Dx<Axis::IAXIS>() *
                                             (pack(b, te::F1, MAG(), k, j, i + 1) -
                                              pack(b, te::F1, MAG(), k, j, i)) +
                                         1. / coords.template Dx<Axis::JAXIS>() *
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

struct AddSourceTerms {
  using options = OptTypeList<HydroFactory, grid::GeometryOptions>;
  using value = TaskStatus;

  template <HydroTrait hydro_traits, Geometry geom>
  requires(geom == Geometry::cartesian)
  value dispatch(MeshData *md, MeshData *dudt) {
    return TaskStatus::complete;
  }
  template <HydroTrait hydro_traits, Geometry geom>
  value dispatch(MeshData *md, MeshData *dudt) {
    using primitives = hydro_traits::Primitive;
    using conserved = hydro_traits::Conserved;

    auto pack_dudt = grid::GetPack(conserved(), dudt);
    auto pack_prim = grid::GetPack(primitives(), md);

    const int nblocks = pack_prim.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);

    par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto prim = SubPack(pack_prim, b, k, j, i);
          auto du = SubPack(pack_dudt, b, k, j, i);
          AddGeometricSource<hydro_traits::MHD, geom>(
              prim.template GetCoordinates<geom>(), prim, du);
        });
    return TaskStatus::complete;
  }
};

TaskID AddTasksOneStep(TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt) {
  auto sources = tl.AddTask(
      prev, "hydro::AddSourceTerms",
      [](MeshData *md, MeshData *dudt) {
        auto cfg = GetConfig(md);
        return Dispatcher<AddSourceTerms>(PARTHENON_AUTO_LABEL, cfg.get())
            .execute(md, dudt);
      },
      md, dudt);

  auto final = sources;
  return final;
}

struct PrepareConserved_impl {
  using options = OptTypeList<HydroFactory, grid::GeometryOptions>;
  using value = TaskStatus;
  template <HydroTrait hydro_traits, Geometry geom>
  requires(hydro_traits::MHD == Mhd::off || geom == Geometry::cartesian)
  value dispatch(MeshData *md) {
    return TaskStatus::complete;
  }

  template <HydroTrait hydro_traits, Geometry geom>
  requires(hydro_traits::MHD != Mhd::off && geom != Geometry::cartesian)
  value dispatch(MeshData *md) {
    // for the most part our conserved variables are maintained in separate
    // fields, e.g., MOMENTUM, ENER, and so there isn't anything to prepare before
    // advancing the system
    //
    // The exception is the angular magnetic field in cylindrical r-z geometry
    // here we advance B_phi / r, and so need to perform the conversion before updating
    using Fields = TypeList<MAGC>;
    using PackVars = ConcatTypeLists_t<Fields, grid::Xcoord>;

    auto pack = grid::GetPack(PackVars(), md);
    const int nblocks = pack.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);

    par_for_outer(
        PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t team, const int b, const int k, const int j) {
          auto coords = grid::CoordinatePack<geom, grid::Xcoord>(pack, b);
          par_for_inner(team, ib.s, ib.e, [&](const int i) {
            if constexpr (geom == Geometry::cylindrical) {
              // in cylindrical advance Bphi / r
              pack(b, MAGC(2), k, j, i) *= 1.0 / coords.template Xc<Axis::IAXIS>(k, j, i);
            }
          });
        });
    return TaskStatus::complete;
  }
};

TaskStatus PrepareConserved(MeshData *md) {
  return Dispatcher<PrepareConserved_impl>(PARTHENON_AUTO_LABEL, GetConfig(md).get())
      .execute(md);
}

}  // namespace kamayan::hydro
