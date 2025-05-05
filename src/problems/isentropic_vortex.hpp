#ifndef PROBLEMS_ISENTROPIC_VORTEX_HPP_
#define PROBLEMS_ISENTROPIC_VORTEX_HPP_

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::isentropic_vortex {
using RuntimeParameters = runtime_parameters::RuntimeParameters;
void Setup(Config *config, RuntimeParameters *rps);
void UserWorkBeforeOutput(Mesh *mesh, parthenon::ParameterInput *pin,
                          const parthenon::SimTime &time);
std::shared_ptr<StateDescriptor> Initialize(const Config *config,
                                            const RuntimeParameters *rps);
void ProblemGenerator(MeshBlock *mb);

struct VortexData {
  using variables = TypeList<DENS, VELOCITY, PRES, MAGC>;
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

  KOKKOS_INLINE_FUNCTION Arr_t StateMHD(const Real &x, const Real &y) const {
    Arr_t state;
    const Real r2 = x * x + y * y;
    const Real dv = strength * Kokkos::exp(0.5 * (1.0 - r2)) / (2.0 * M_PI);
    const Real dB = mhd_strength * Kokkos::exp(0.5 * (1.0 - r2)) / (2.0 * M_PI);
    const Real exp = Kokkos::exp(1.0 - r2);
    // from Balsara 2004 we set density to unity
    // so only pressure (thermal + magnetic), magnetic tension & centrifugal force play
    // into the balance
    state(DENS()) = 1.0;
    state(PRES()) =
        pressure +
        (mhd_strength * mhd_strength / (8.0 * M_PI * M_PI)) * (1.0 - r2) * exp -
        (strength * strength / (8.0 * M_PI * M_PI)) * exp;
    // velocity = v_+ r * dv * \hat{\phi}
    state(VELOCITY(0)) = velx - y * dv;
    state(VELOCITY(1)) = vely + x * dv;
    state(MAGC(0)) = -y * dB;
    state(MAGC(1)) = +x * dB;

    return state;
  }

  KOKKOS_INLINE_FUNCTION Real Az(const Real &x, const Real &y) const {
    const Real r2 = x * x + y * y;
    const Real exp = Kokkos::exp(1.0 - r2);
    return mhd_strength / (2. * M_PI) * exp;
  }

  Real density, pressure, velx, vely, strength, gamma;
  Real mhd_strength;
};

template <typename Var>
Real ErrorHistory(MeshData *md, const int &component = 0) {
  auto mesh = md->GetMeshPointer();
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

  auto pack = grid::GetPack<Var>(md);

  auto ib = md->GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md->GetBoundsK(parthenon::IndexDomain::interior);

  const int comp = component;

  Real error = 0.;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL,
      parthenon::DevExecSpace(), 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lerr) {
        auto coords = pack.GetCoordinates(b);
        const Real x0 = coords.template Xc<1>(i);
        const Real y0 = coords.template Xc<2>(j);
        // note this only works up to a single perdiod
        // const Real x = x0 > x1_min ? x0 : x1_max + (x0 - x1_min);
        // const Real y = y0 > x1_min ? y0 : x1_max + (y0 - x1_min);
        auto state = vortex_data.State(x0, y0);
        lerr += Kokkos::abs(state(Var(comp)) - pack(b, Var(comp), k, j, i)) *
                coords.CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(error));
  return error / (size_x1 * size_x2);
}
}  // namespace kamayan::isentropic_vortex
#endif  // PROBLEMS_ISENTROPIC_VORTEX_HPP_
