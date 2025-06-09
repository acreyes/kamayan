#include <memory>

#include "Kokkos_Macros.hpp"

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/unit.hpp"
#include "physics/physics_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/parallel.hpp"
#include "utils/type_list.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::mhd_blast {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(Config *config, RuntimeParameters *rps);
std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps);
void ProblemGenerator(MeshBlock *mb);
}  // namespace kamayan::mhd_blast

int main(int argc, char *argv[]) {
  auto pman = kamayan::InitEnv(argc, argv);
  auto units = kamayan::ProcessUnits();

  auto simulation = std::make_shared<kamayan::KamayanUnit>("mhd_blast");
  simulation->Setup = kamayan::mhd_blast::Setup;
  simulation->Initialize = kamayan::mhd_blast::Initialize;
  simulation->ProblemGeneratorMeshBlock = kamayan::mhd_blast::ProblemGenerator;
  units.Add(simulation);

  auto driver = kamayan::InitPackages(pman, units);
  auto driver_status = driver.Execute();

  pman->ParthenonFinalize();
}

namespace kamayan::mhd_blast {

struct BlastData {
  using variables = TypeList<DENS, VELOCITY, PRES, MAGC>;
  using pack_variables = ConcatTypeLists_t<variables, TypeList<MAG>>;
  using Arr_t = TypeListArray<variables>;

  KOKKOS_INLINE_FUNCTION Arr_t State(const Real &r) const {
    Arr_t state;
    state(DENS()) = rho_ambient;
    state(VELOCITY(0)) = 0.;
    state(VELOCITY(1)) = 0.;
    state(VELOCITY(2)) = 0.;
    state(PRES()) = r <= radius ? p_explosion : p_ambient;
    state(MAGC(0)) = bx;
    state(MAGC(1)) = 0.;
    state(MAGC(2)) = 0.;

    return state;
  }

  // Bx(x,y) = Bx = dyAz
  KOKKOS_INLINE_FUNCTION Real Az(const Real &x, const Real &y) const { return bx * y; }

  Real radius, p_ambient, rho_ambient, p_explosion, bx;
};

void Setup(Config *config, RuntimeParameters *rps) {
  rps->Add("mhd_blast", "density", 1.0, "ambient density");
  rps->Add("mhd_blast", "pressure", 1.0e-1, "ambient pressure");
  rps->Add("mhd_blast", "explosion_pressure", 1.0e1, "explosion pressure");
  rps->Add("mhd_blast", "magx", 1., "uniform x-magnetic field");
  rps->Add("mhd_blast", "radius", 0.1, "initial radius of the blast");
}

std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps) {
  PARTHENON_REQUIRE_THROWS(config->Get<Mhd>() != Mhd::off,
                           "MHD Blast requires <physics/MHD> to not be off");

  auto pkg = std::make_shared<StateDescriptor>("mhd_blast");

  BlastData data;
  data.rho_ambient = rps->Get<Real>("mhd_blast", "density");
  data.p_ambient = rps->Get<Real>("mhd_blast", "pressure");
  data.p_explosion = rps->Get<Real>("mhd_blast", "explosion_pressure");
  data.bx = rps->Get<Real>("mhd_blast", "magx");
  data.radius = rps->Get<Real>("mhd_blast", "radius");

  pkg->AddParam("data", data);

  return pkg;
}
void ProblemGenerator(MeshBlock *mb) {
  auto &data = mb->meshblock_data.Get();
  auto pkg = mb->packages.Get("mhd_blast");
  auto mhd_blast_data = pkg->Param<BlastData>("data");
  auto config = GetConfig(mb);

  auto cellbounds = mb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = mb->coords;
  const int ndim = 1 + (jb.e > jb.s) + (kb.e > kb.s);
  const int k2d = (ndim > 1) ? 1 : 0;
  const int k3d = (ndim > 2) ? 1 : 0;

  // get our pack
  auto pack = grid::GetPack(BlastData::pack_variables(), mb);
  auto pack_mag = grid::GetPack<MAG>(mb);
  par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e + k3d, jb.s, jb.e + k2d, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real r2 =
            coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j);
        auto state = mhd_blast_data.State(Kokkos::sqrt(r2));
        type_for(BlastData::variables(), [&]<typename Vars>(const Vars &) {
          for (int comp = 0; comp < Vars::n_comps; comp++) {
            pack(0, Vars(comp), k, j, i) = state(Vars(comp));
          }

          using TE = TopologicalElement;
          if (ndim == 2) {
            if (j < jb.e + 1)
              pack_mag(0, TE::F1, MAG(), k, j, i) =
                  (mhd_blast_data.Az(coords.Xf<1>(i), coords.Xf<2>(j + 1)) -
                   mhd_blast_data.Az(coords.Xf<1>(i), coords.Xf<2>(j))) /
                  coords.Dxc<2>(j);
            if (i < ib.e + 1)
              pack_mag(0, TE::F2, MAG(), k, j, i) =
                  (mhd_blast_data.Az(coords.Xf<1>(i + 1), coords.Xf<2>(j)) -
                   mhd_blast_data.Az(coords.Xf<1>(i), coords.Xf<2>(j))) /
                  coords.Dxc<1>(i);
          }
        });
      });
}
}  // namespace kamayan::mhd_blast
