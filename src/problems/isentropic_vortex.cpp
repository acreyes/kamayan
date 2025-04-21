#include <cmath>
#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::isentropic_vortex {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(Config *config, RuntimeParameters *rps);
void UserWorkBeforeOutput(Mesh *mesh, parthenon::ParameterInput *pin,
                          const parthenon::SimTime &time);
std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps);
void ProblemGenerator(MeshBlock *mb);

VARIABLE(DENS_ERR);

}  // namespace kamayan::isentropic_vortex

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
  // --8<-- [end:parms]
}

struct VortexData {
  using variables = TypeList<DENS, VELOCITY, PRES>;
  using Arr_t = TypeListArray<variables>;

  KOKKOS_INLINE_FUNCTION Arr_t State(const Real &x, const Real &y) const {
    Arr_t state;
    const Real r2 = x * x + y * y;
    const Real dv = strength * Kokkos::exp(0.5 * (1.0 - r2)) / (2.0 * M_PI);
    const Real T = pressure / density - (gamma - 1.0) * strength * strength *
                                            Kokkos::exp(1.0 - r2) /
                                            (8.0 * gamma * M_PI * M_PI);
    // T = P / rho
    // entropy = constant = P / rho^gamma = T / rho^(gamma - 1)
    // density = (T)^-(gamma - 1)
    state(DENS()) = Kokkos::pow(T, 1.0 / (gamma - 1.0));
    state(PRES()) = T * state(DENS());
    // velocity = v_+ r * dv * \hat{\phi}
    state(VELOCITY(0)) = velx - y * dv;
    state(VELOCITY(1)) = vely + x * dv;

    return state;
  }

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

  AddField<DENS_ERR>(pkg.get(), {Metadata::Cell, Metadata::Restart});
  pkg->UserWorkBeforeOutputMesh = UserWorkBeforeOutput;

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
  // --8<-- [start:pack]
  auto pack = grid::GetPack<DENS, VELOCITY, PRES>(mb);
  // --8<-- [end:pack]

  const Real entropy =
      vortex_data.pressure / Kokkos::pow(vortex_data.density, vortex_data.gamma);

  parthenon::par_for(
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
}

// use this callback to calculate the pointwise error everywhere on the mesh
void UserWorkBeforeOutput(Mesh *mesh, parthenon::ParameterInput *pin,
                          const parthenon::SimTime &sim_time_) {
  static auto pkg = mesh->packages.Get("isentropic_vortex");
  static auto driver_pkg = mesh->packages.Get("driver");
  auto vortex_data = pkg->Param<VortexData>("data");
  auto mesh_size = mesh->mesh_size;

  auto sim_time = driver_pkg->Param<SimTime>("sim_time");
  const Real time = sim_time.time;

  using CD = parthenon::CoordinateDirection;
  const Real x1_min = mesh_size.xmin(CD::X1DIR);
  const Real x1_max = mesh_size.xmax(CD::X1DIR);
  const Real x2_min = mesh_size.xmin(CD::X2DIR);
  const Real x2_max = mesh_size.xmax(CD::X2DIR);
  const Real size_x1 = x1_max - x1_min;
  const Real size_x2 = x2_max - x2_min;

  const Real xc = vortex_data.velx * time;
  const Real yc = vortex_data.vely * time;

  auto &md = mesh->mesh_data.Get("base");

  auto pack = grid::GetPack<DENS_ERR, DENS>(md.get());

  auto ib = md->GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md->GetBoundsK(parthenon::IndexDomain::interior);

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto coords = pack.GetCoordinates(b);
        const Real x0 = coords.Xc<1>(i) - xc;
        const Real y0 = coords.Xc<2>(j) - yc;
        // note this only works up to a single perdiod
        const Real x = x0 > x1_min ? x0 : x1_max + (x0 - x1_min);
        const Real y = y0 > x1_min ? y0 : x1_max + (y0 - x1_min);
        auto state = vortex_data.State(x, y);
        pack(b, DENS_ERR(), k, j, i) =
            Kokkos::abs(state(DENS()) - pack(b, DENS(), k, j, i)) / state(DENS());
      });
}
}  // namespace kamayan::isentropic_vortex
