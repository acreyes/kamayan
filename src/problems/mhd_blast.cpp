#include <memory>

#include "Kokkos_Macros.hpp"

#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "physics/physics_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/parallel.hpp"
#include "utils/type_list.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::mhd_blast {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(KamayanUnit &unit);
void Initialize(KamayanUnit &unit);
void ProblemGenerator(MeshBlock *mb);
}  // namespace kamayan::mhd_blast

int main(int argc, char *argv[]) {
  auto pman = kamayan::InitEnv(argc, argv);
  auto units = std::make_shared<kamayan::UnitCollection>(kamayan::ProcessUnits());

  auto simulation = std::make_shared<kamayan::KamayanUnit>("mhd_blast");
  simulation->SetupParams = kamayan::mhd_blast::Setup;
  simulation->InitializeData = kamayan::mhd_blast::Initialize;
  simulation->ProblemGeneratorMeshBlock = kamayan::mhd_blast::ProblemGenerator;
  units->Add(simulation);

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

void Setup(KamayanUnit &unit) {
  auto &mhd_blast = unit.AddData("mhd_blast");
  mhd_blast.AddParm<Real>("density", 1.0, "ambient density");
  mhd_blast.AddParm<Real>("pressure", 1.0e-1, "ambient pressure");
  mhd_blast.AddParm<Real>("explosion_pressure", 1.0e1, "explosion pressure");
  mhd_blast.AddParm<Real>("magx", 1., "uniform x-magnetic field");
  mhd_blast.AddParm<Real>("radius", 0.1, "initial radius of the blast");
}

void Initialize(KamayanUnit &unit) {
  auto config = unit.Configuration();
  PARTHENON_REQUIRE_THROWS(config->Get<Mhd>() != Mhd::off,
                           "MHD Blast requires <physics/MHD> to not be off");

  auto mhd_blast = unit.Data("mhd_blast");
  BlastData data;
  data.rho_ambient = mhd_blast.Get<Real>("density");
  data.p_ambient = mhd_blast.Get<Real>("pressure");
  data.p_explosion = mhd_blast.Get<Real>("explosion_pressure");
  data.bx = mhd_blast.Get<Real>("magx");
  data.radius = mhd_blast.Get<Real>("radius");

  // unit IS the package (StateDescriptor)
  unit.AddParam("data", data);
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
