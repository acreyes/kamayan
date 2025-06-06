#include "isentropic_vortex.hpp"

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "outputs/outputs.hpp"
#include "physics/physics_types.hpp"
#include "utils/parallel.hpp"
#include "utils/type_list_array.hpp"

// --8<-- [start:isen_main]
int main(int argc, char *argv[]) {
  // initialize the environment
  // * mpi
  // * kokkos
  // * parthenon
  auto pman = kamayan::InitEnv(argc, argv);

  // put together all the kamayan units we want
  auto units = kamayan::ProcessUnits();

  // add a simulation unit that will set the initial conditions
  auto simulation = std::make_shared<kamayan::KamayanUnit>("isentropic_vortex");
  // configure any runtime parameters we will want to use
  simulation->Setup = kamayan::isentropic_vortex::Setup;
  // create a StateDescriptor instance for our simulation package to
  // hold persistent data that we will read from our runtime parameters
  simulation->Initialize = kamayan::isentropic_vortex::Initialize;
  // do the actual initialization of block data
  simulation->ProblemGeneratorMeshBlock = kamayan::isentropic_vortex::ProblemGenerator;
  // register the unit to our UnitCollection
  units.Add(simulation);

  // get the driver and we're ready to go!
  auto driver = kamayan::InitPackages(pman, units);
  // execute the evolution loop!
  auto driver_status = driver.Execute();

  pman->ParthenonFinalize();
}
// --8<-- [end:isen_main]

namespace kamayan::isentropic_vortex {

void Setup(Config *config, RuntimeParameters *rps) {
  // --8<-- [start:parms]
  rps->Add("isentropic_vortex", "density", 1.0, "Ambient density");
  rps->Add("isentropic_vortex", "pressure", 1.0, "Ambient pressure");
  rps->Add("isentropic_vortex", "velx", 1.0, "Ambient x-velcoty");
  rps->Add("isentropic_vortex", "vely", 1.0, "Ambient y-velcoty");
  rps->Add("isentropic_vortex", "strength", 5.0, "Vortex strength.");
  rps->Add("isentropic_vortex", "mhd_strength", 1.0, "Vortex mhd_strength.");
  // --8<-- [end:parms]
}

std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps) {
  auto pkg = std::make_shared<StateDescriptor>("isentropic_vortex");

  VortexData data;
  data.density = rps->Get<Real>("isentropic_vortex", "density");
  data.pressure = rps->Get<Real>("isentropic_vortex", "pressure");
  data.velx = rps->Get<Real>("isentropic_vortex", "velx");
  data.vely = rps->Get<Real>("isentropic_vortex", "vely");
  data.strength = rps->Get<Real>("isentropic_vortex", "strength");
  data.mhd_strength = rps->Get<Real>("isentropic_vortex", "mhd_strength");
  data.gamma = rps->Get<Real>("eos/gamma", "gamma");

  pkg->AddParam("data", data);

  const auto mhd = config->Get<Mhd>();

  parthenon::HstVar_list history_vars = {};
  history_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum,
      [=](MeshData *md) { return ErrorHistory<DENS>(md, mhd, 0); }, "density error"));
  history_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum,
      [=](MeshData *md) { return ErrorHistory<VELOCITY>(md, mhd, 0); }, "velx error"));
  history_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum,
      [=](MeshData *md) { return ErrorHistory<VELOCITY>(md, mhd, 1); }, "vely error"));
  history_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum,
      [=](MeshData *md) { return ErrorHistory<PRES>(md, mhd, 0); }, "pressure error"));
  if (mhd != Mhd::off) {
    history_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum,
        [=](MeshData *md) { return ErrorHistory<MAGC>(md, mhd, 0); }, "magx error"));
    history_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum,
        [=](MeshData *md) { return ErrorHistory<MAGC>(md, mhd, 1); }, "magy error"));
  }

  pkg->AddParam<>(parthenon::hist_param_key, history_vars);

  return pkg;
}

void ProblemGenerator(MeshBlock *mb) {
  auto &data = mb->meshblock_data.Get();
  auto pkg = mb->packages.Get("isentropic_vortex");
  auto vortex_data = pkg->Param<VortexData>("data");
  auto config = GetConfig(mb);

  auto cellbounds = mb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = mb->coords;

  auto mhd = config->Get<Mhd>();
  const Real entropy =
      vortex_data.pressure / Kokkos::pow(vortex_data.density, vortex_data.gamma);

  if (mhd == Mhd::off) {
    // get our pack
    // --8<-- [start:pack]
    auto pack = grid::GetPack<DENS, VELOCITY, PRES>(mb);
    // --8<-- [end:pack]
    par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real r2 =
              coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j);
          auto state = vortex_data.State(coords.Xc<1>(i), coords.Xc<2>(j));

          // --8<-- [start:index]
          pack(0, DENS(), k, j, i) = state(DENS());
          pack(0, PRES(), k, j, i) = state(PRES());
          pack(0, VELOCITY(0), k, j, i) = state(VELOCITY(0));
          pack(0, VELOCITY(1), k, j, i) = state(VELOCITY(1));
          // --8<-- [end:index]
        });
  } else {
    auto pack = grid::GetPack<DENS, VELOCITY, PRES, MAGC>(mb);
    par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real r2 =
              coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j);
          auto state = vortex_data.StateMHD(coords.Xc<1>(i), coords.Xc<2>(j));

          pack(0, DENS(), k, j, i) = state(DENS());
          pack(0, PRES(), k, j, i) = state(PRES());
          pack(0, VELOCITY(0), k, j, i) = state(VELOCITY(0));
          pack(0, VELOCITY(1), k, j, i) = state(VELOCITY(1));
          pack(0, MAGC(0), k, j, i) = state(MAGC(0));
          pack(0, MAGC(1), k, j, i) = state(MAGC(1));
        });
  }
  if (mhd == Mhd::ct && jb.e > jb.s) {  // CT and at least 2D
    // need to initialize div-free face fields from vector potential
    auto pack = grid::GetPack<MAG>(mb);
    auto k3d = kb.e > kb.s ? 1 : 0;
    par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e + k3d, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          using te = TopologicalElement;
          const Real xf_x = coords.Xf<1, 1>(k, j, i);
          const Real xf_y = coords.Xf<2, 1>(k, j, i);
          const Real xf_dy = coords.Dxf<2>(j);
          pack(0, te::F1, MAG(), k, j, i) = 1. / xf_dy *
                                            (vortex_data.Az(xf_x, xf_y + 0.5 * xf_dy) -
                                             vortex_data.Az(xf_x, xf_y - 0.5 * xf_dy));

          const Real yf_x = coords.Xf<1, 2>(k, j, i);
          const Real yf_y = coords.Xf<2, 2>(k, j, i);
          const Real yf_dx = coords.Dxf<1>(i);
          pack(0, te::F2, MAG(), k, j, i) = -1. / yf_dx *
                                            (vortex_data.Az(yf_x + 0.5 * yf_dx, yf_y) -
                                             vortex_data.Az(yf_x - 0.5 * yf_dx, yf_y));
        });
  }
}
}  // namespace kamayan::isentropic_vortex
