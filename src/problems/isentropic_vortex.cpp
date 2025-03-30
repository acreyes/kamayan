#include <memory>
#include <vector>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "utils/instrument.hpp"

namespace kamayan::isentropic_vortex {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(Config *config, RuntimeParameters *rps);
std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps);
void ProblemGenerator(MeshBlock *mb);

}  // namespace kamayan::isentropic_vortex

int main(int argc, char *argv[]) {
  // initialize the environment
  // * mpi
  // * kokkos
  // * parthenon
  auto pman = kamayan::InitEnv(argc, argv);

  // put together all the kamayan units we want
  auto units = kamayan::ProcessUnits();
  auto simulation = std::make_shared<kamayan::KamayanUnit>();
  simulation->Setup = kamayan::isentropic_vortex::Setup;
  simulation->Initialize = kamayan::isentropic_vortex::Initialize;
  simulation->ProblemGeneratorMeshBlock = kamayan::isentropic_vortex::ProblemGenerator;
  units["isentropic_vortex"] = simulation;

  // get the driver and we're ready to go!
  auto driver = kamayan::InitPackages(pman, units);
  auto driver_status = driver.Execute();

  pman->ParthenonFinalize();
}

namespace kamayan::isentropic_vortex {

void Setup(Config *config, RuntimeParameters *rps) {
  rps->Add("isentropic_vortex", "density", 1.0, "Ambient density");
  rps->Add("isentropic_vortex", "pressure", 1.0, "Ambient pressure");
  rps->Add("isentropic_vortex", "velx", 1.0, "Ambient x-velcoty");
  rps->Add("isentropic_vortex", "vely", 1.0, "Ambient y-velcoty");
  rps->Add("isentropic_vortex", "strength", 5.0, "Vortex strength.");
}

struct VortexData {
  Real density, pressure, velx, vely, strength, gamma;
};

std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps) {
  auto pkg = std::make_shared<StateDescriptor>("isentropic_vortex");

  VortexData data;
  data.density = rps->Get<Real>("isentropic_vortex", "density");
  data.pressure = rps->Get<Real>("isentropic_vortex", "pressure");
  data.velx = rps->Get<Real>("isentropic_vortex", "velx");
  data.vely = rps->Get<Real>("isentropic_vortex", "vely");
  data.strength = rps->Get<Real>("isentropic_vortex", "strength");
  data.gamma = rps->Get<Real>("eos/gamma", "gamma");

  pkg->AddParam("data", data);

  // we need these to exist in order to initialize
  pkg->AddField<DENS>(Metadata({Metadata::Cell, Metadata::Restart}));
  pkg->AddField<PRES>(Metadata({Metadata::Cell, Metadata::Restart}));
  pkg->AddField<VELOCITY>(
      Metadata({Metadata::Cell, Metadata::Restart}, std::vector<int>{3}));

  return pkg;
}

void ProblemGenerator(MeshBlock *mb) {
  auto &data = mb->meshblock_data.Get();
  auto pkg = mb->packages.Get("isentropic_vortex");
  auto vortex_data = pkg->Param<VortexData>("data");

  auto cellbounds = mb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = mb->coords;

  // get our pack
  auto pack = grid::GetPack<DENS, VELOCITY, PRES>(mb);

  const Real entropy =
      vortex_data.pressure / Kokkos::pow(vortex_data.density, vortex_data.gamma);

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real r2 =
            coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j);
        const Real dv =
            vortex_data.strength * Kokkos::exp(-0.5 * (1.0 - r2)) / (2.0 * M_PI);
        const Real T = vortex_data.pressure / vortex_data.density -
                       (vortex_data.gamma - 1.0) * vortex_data.strength *
                           vortex_data.strength * Kokkos::exp(1.0 - r2) /
                           (8.0 * vortex_data.gamma * M_PI * M_PI);
        // T = P / rho
        // entropy = constant = P / rho^gamma = T / rho^(gamma - 1)
        // density = (T)^-(gamma - 1)

        pack(0, DENS(), k, j, i) = Kokkos::pow(T, -(vortex_data.gamma - 1.0));
        pack(0, PRES(), k, j, i) = T * pack(0, DENS(), k, j, i);
        // velocity = v_ambient + r * dv * \hat{\phi}
        pack(0, VELOCITY(1), k, j, i) = vortex_data.velx - coords.Xc<2>(j) * dv;
        pack(0, VELOCITY(2), k, j, i) = vortex_data.vely + coords.Xc<1>(i) * dv;
      });
}
}  // namespace kamayan::isentropic_vortex
