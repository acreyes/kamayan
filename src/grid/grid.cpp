#include "grid.hpp"

#include <memory>
#include <string>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/grid_update.hpp"
#include "kamayan/runtime_parameters.hpp"

namespace kamayan::grid {

std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto grid_unit = std::make_shared<KamayanUnit>("grid");
  grid_unit->Setup = Setup;
  return grid_unit;
}

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps) {
  // most of what we're doing here is wrapping the parthenon mesh related
  // input parameters as runtime parameters for the docs!
  // <parthenon/mesh>
  rps->Add<std::string>("parthenon/mesh", "refinement", "adaptive",
                        "Mesh refinement startegy.", {"adaptive", "static", "none"});
  rps->Add<int>("parthenon/mesh", "numlevel", 1, "Number of refinement levels.");

  rps->Add<int>("parthenon/mesh", "nx1", 32,
                "Number of cells across the domain at level 0.");
  rps->Add<int>("parthenon/mesh", "nx2", 32,
                "Number of cells across the domain at level 0. Set to 1 for 1D.");
  rps->Add<int>("parthenon/mesh", "nx3", 32,
                "Number of cells across the domain at level 0. Set to 1 for 2D.");
  rps->Add<int>("parthenon/mesh", "nghost", 3,
                "Number of ghost zones to use on each block.");

  rps->Add<Real>("parthenon/mesh", "x1min", 0.0, "Minimum x1 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x2min", 0.0, "Minimum x2 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x3min", 0.0, "Minimum x3 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x1max", 1.0, "Maximum x1 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x2max", 1.0, "Maximum x2 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x3max", 1.0, "Maximum x3 value of domain.");

  rps->Add<std::string>("parthenon/mesh", "ix1_bc", "outflow",
                        "Inner boundary condition along x1.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ix2_bc", "outflow",
                        "Inner boundary condition along x2.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ix3_bc", "outflow",
                        "Inner boundary condition along x3.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox1_bc", "outflow",
                        "Outer boundary condition along x1.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox2_bc", "outflow",
                        "Outer boundary condition along x2.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox3_bc", "outflow",
                        "Outer boundary condition along x3.",
                        {"periodic", "outflow", "reflect", "user"});

  // <parthenon/meshblock>
  rps->Add<int>("parthenon/meshblock", "nx1", 16, "Size of meshblocks in x1.");
  rps->Add<int>("parthenon/meshblock", "nx2", 16, "Size of meshblocks in x2.");
  rps->Add<int>("parthenon/meshblock", "nx3", 16, "Size of meshblocks in x3.");
}

TaskStatus FluxesToDuDt(MeshData *md, MeshData *dudt) {
  static const int ndim = md->GetNDim();
  using TE = TopologicalElement;
  switch (ndim) {
  case 1:
    FluxDivergence<TE::F1>(md, dudt);
    break;
  case 2:
    FluxDivergence<TE::F1, TE::F2>(md, dudt);
    FluxStokes<TE::F1, TE::E3>(md, dudt);
    FluxStokes<TE::F2, TE::E3>(md, dudt);
    break;
  case 3:
    FluxDivergence<TE::F1, TE::F2, TE::F3>(md, dudt);
    FluxStokes<TE::F1, TE::E3, TE::E2>(md, dudt);
    FluxStokes<TE::F2, TE::E3, TE::E1>(md, dudt);
    FluxStokes<TE::F3, TE::E1, TE::E2>(md, dudt);
    break;
  }

  return TaskStatus::complete;
}

constexpr MetadataFlag TopologicalElementToMetaData(const TopologicalElement &te) {
  using TE = TopologicalElement;
  if (te < TE::F1) return Metadata::Cell;
  if (te < TE::E1) return Metadata::Face;

  return Metadata::Cell;
}

template <typename PackDesc_t>
TaskStatus ApplyDuDt_impl(PackDesc_t &desc, const TopologicalElement &te, MeshData *mbase,
                          MeshData *md0, MeshData *md1, MeshData *dudt_data,
                          const Real &beta, const Real &dt) {
  {
    auto pack_base = desc.GetPack(mbase);
    auto pack0 = desc.GetPack(md0);
    auto pack1 = desc.GetPack(md1);
    auto dudt = desc.GetPack(dudt_data);
    if (pack0.GetMaxNumberOfVars() == 0) return TaskStatus::complete;

    const int nblocks = pack0.GetNBlocks();
    auto ib = md0->GetBoundsI(IndexDomain::interior, te);
    auto jb = md0->GetBoundsJ(IndexDomain::interior, te);
    auto kb = md0->GetBoundsK(IndexDomain::interior, te);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          for (int var = pack0.GetLowerBound(b); var <= pack0.GetUpperBound(b); var++) {
            pack0(b, te, var, k, j, i) = beta * pack_base(b, te, var, k, j, i) +
                                         (1.0 - beta) * pack0(b, te, var, k, j, i);
            pack1(b, te, var, k, j, i) =
                pack0(b, te, var, k, j, i) + beta * dt * dudt(b, te, var, k, j, i);
          }
        });
  }

  return TaskStatus::complete;
}

TaskID ApplyDuDt(TaskID prev, TaskList &tl, MeshData *mbase, MeshData *md0, MeshData *md1,
                 MeshData *dudt_data, const Real &beta, const Real &dt) {
  using TE = TopologicalElement;
  const auto ndim = mbase->GetNDim();
  // cell-centered updates
  static auto desc_cc =
      GetPackDescriptor(md0, {Metadata::Cell, Metadata::WithFluxes}, {PDOpt::WithFluxes});
  auto cell_update = tl.AddTask(
      prev, "grid::ApplyDuDt_Cell",
      [&](MeshData *mbase, MeshData *md0, MeshData *md1, MeshData *dudt, const Real &beta,
          const Real &dt) {
        return ApplyDuDt_impl(desc_cc, TE::CC, mbase, md0, md1, dudt, beta, dt);
      },
      mbase, md0, md1, dudt_data, beta, dt);

  // all cell centers in 1D
  if (ndim < 2) return cell_update;

  // update face variables
  static auto desc_fc =
      GetPackDescriptor(md0, {Metadata::Face, Metadata::WithFluxes}, {PDOpt::WithFluxes});
  auto faces = ndim > 2 ? std::vector<TE>{TE::F1, TE::F2, TE::F3}
                        : std::vector<TE>{TE::F1, TE::F2};
  int nface = 0;
  TaskID face_update(0);
  for (auto face : faces) {
    nface++;
    const std::string label = "grid::ApplyDuDt_Face" + std::to_string(nface);
    face_update =
        face_update |
        tl.AddTask(
            prev, label,
            [=](MeshData *mbase, MeshData *md0, MeshData *md1, MeshData *dudt,
                const Real &beta, const Real &dt) {
              return ApplyDuDt_impl(desc_fc, face, mbase, md0, md1, dudt, beta, dt);
            },
            mbase, md0, md1, dudt_data, beta, dt);
  }
  return cell_update | face_update;
}

}  // namespace kamayan::grid
