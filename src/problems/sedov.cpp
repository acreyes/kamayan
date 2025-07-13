
#include <numbers>

#include <memory>

#include "Kokkos_Macros.hpp"

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/unit.hpp"
#include "utils/parallel.hpp"
#include "utils/type_list.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::sedov {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(Config *config, RuntimeParameters *rps);
std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps);
void ProblemGenerator(MeshBlock *mb);
}  // namespace kamayan::sedov

int main(int argc, char *argv[]) {
  auto pman = kamayan::InitEnv(argc, argv);
  auto units = kamayan::ProcessUnits();

  auto simulation = std::make_shared<kamayan::KamayanUnit>("sedov");
  simulation->Setup = kamayan::sedov::Setup;
  simulation->Initialize = kamayan::sedov::Initialize;
  simulation->ProblemGeneratorMeshBlock = kamayan::sedov::ProblemGenerator;
  units.Add(simulation);

  auto driver = kamayan::InitPackages(pman, units);
  auto driver_status = driver.Execute();

  pman->ParthenonFinalize();
}

namespace kamayan::sedov {

struct SedovData {
  using variables = TypeList<DENS, VELOCITY, PRES>;
  using Arr_t = TypeListArray<variables>;

  KOKKOS_INLINE_FUNCTION Arr_t State(const Real &r) const {
    Arr_t state;
    state(DENS()) = rho_ambient;
    state(VELOCITY(0)) = 0.;
    state(VELOCITY(1)) = 0.;
    state(VELOCITY(2)) = 0.;
    state(PRES()) = r <= radius ? p_explosion : p_ambient;

    return state;
  }

  Real radius, p_ambient, rho_ambient, p_explosion;
};

void Setup(Config *config, RuntimeParameters *rps) {
  rps->Add("sedov", "density", 1.0, "ambient density");
  rps->Add("sedov", "pressure", 1.0e-5, "ambient pressure");
  rps->Add("sedov", "energy", 1.0, "explosion energy");
}

std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps) {
  auto pkg = std::make_shared<StateDescriptor>("sedov");

  SedovData data;
  data.rho_ambient = rps->Get<Real>("sedov", "density");
  data.p_ambient = rps->Get<Real>("sedov", "pressure");

  const Real E = rps->Get<Real>("sedov", "energy");
  const Real gamma = rps->Get<Real>("eos/gamma", "gamma");
  const Real nu = 2.;  // 2 for cylindrical, 3 for spherical

  const int nlevels = rps->Get<int>("parthenon/mesh", "numlevel");
  const int nx = rps->Get<int>("parthenon/mesh", "nx1");
  const Real xmin = rps->Get<Real>("parthenon/mesh", "x1min");
  const Real xmax = rps->Get<Real>("parthenon/mesh", "x1max");
  const Real dx = (xmax - xmin) / (std::pow(2, nlevels - 1) * static_cast<Real>(nx));
  const Real radius = 3.5 * dx;

  const Real pres =
      3. * (gamma - 1) * E / ((nu + 1) * std::numbers::pi * std::pow(radius, nu));
  data.radius = radius;
  data.p_explosion = pres;
  pkg->AddParam("data", data);

  return pkg;
}
void ProblemGenerator(MeshBlock *mb) {
  auto &data = mb->meshblock_data.Get();
  auto pkg = mb->packages.Get("sedov");
  auto sedov_data = pkg->Param<SedovData>("data");
  auto config = GetConfig(mb);

  auto cellbounds = mb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = mb->coords;

  // get our pack
  auto pack = grid::GetPack<DENS, VELOCITY, PRES>(mb);
  par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real r2 =
            coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j);
        auto state = sedov_data.State(Kokkos::sqrt(r2));
        type_for(SedovData::variables(), [&]<typename Vars>(const Vars &) {
          for (int comp = 0; comp < Vars::n_comps; comp++) {
            pack(0, Vars(comp), k, j, i) = state(Vars(comp));
          }
        });
      });
}
}  // namespace kamayan::sedov
