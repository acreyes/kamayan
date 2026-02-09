#include "grid.hpp"

#include <memory>
#include <string>
#include <vector>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_refinement.hpp"
#include "grid/grid_types.hpp"
#include "grid/grid_update.hpp"
#include "grid/scratch_variables.hpp"
#include "kamayan/runtime_parameters.hpp"

namespace kamayan::grid {

std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto grid_unit = std::make_shared<KamayanUnit>("Grid");
  grid_unit->SetupParams.Register(SetupParams);
  grid_unit->InitializeData.Register(InitializeData);
  return grid_unit;
}

void SetupParams(KamayanUnit *unit) {
  // most of what we're doing here is wrapping the parthenon mesh related
  // input parameters as runtime parameters for the docs!
  // <parthenon/mesh>
  auto &parthenon_mesh = unit->AddData("parthenon/mesh");
  parthenon_mesh.AddParm<std::string>("refinement", "adaptive",
                                      "Mesh refinement strategy.",
                                      {"adaptive", "static", "none"});
  auto adaptive = parthenon_mesh.Get<std::string>("refinement");

  parthenon_mesh.AddParm<int>("numlevel", 1, "Number of refinement levels.");
  auto global_max_level = parthenon_mesh.Get<int>("numlevel");

  parthenon_mesh.AddParm<int>("nx1", 32, "Number of cells across the domain at level 0.");
  parthenon_mesh.AddParm<int>(
      "nx2", 32, "Number of cells across the domain at level 0. Set to 1 for 1D.");
  parthenon_mesh.AddParm<int>(
      "nx3", 32, "Number of cells across the domain at level 0. Set to 1 for 2D.");
  parthenon_mesh.AddParm<int>("nghost", 4, "Number of ghost zones to use on each block.");

  parthenon_mesh.AddParm<Real>("x1min", 0.0, "Minimum x1 value of domain.");
  parthenon_mesh.AddParm<Real>("x2min", 0.0, "Minimum x2 value of domain.");
  parthenon_mesh.AddParm<Real>("x3min", 0.0, "Minimum x3 value of domain.");
  parthenon_mesh.AddParm<Real>("x1max", 1.0, "Maximum x1 value of domain.");
  parthenon_mesh.AddParm<Real>("x2max", 1.0, "Maximum x2 value of domain.");
  parthenon_mesh.AddParm<Real>("x3max", 1.0, "Maximum x3 value of domain.");

  parthenon_mesh.AddParm<std::string>("ix1_bc", "outflow",
                                      "Inner boundary condition along x1.",
                                      {"periodic", "outflow", "reflect", "user"});
  parthenon_mesh.AddParm<std::string>("ix2_bc", "outflow",
                                      "Inner boundary condition along x2.",
                                      {"periodic", "outflow", "reflect", "user"});
  parthenon_mesh.AddParm<std::string>("ix3_bc", "outflow",
                                      "Inner boundary condition along x3.",
                                      {"periodic", "outflow", "reflect", "user"});
  parthenon_mesh.AddParm<std::string>("ox1_bc", "outflow",
                                      "Outer boundary condition along x1.",
                                      {"periodic", "outflow", "reflect", "user"});
  parthenon_mesh.AddParm<std::string>("ox2_bc", "outflow",
                                      "Outer boundary condition along x2.",
                                      {"periodic", "outflow", "reflect", "user"});
  parthenon_mesh.AddParm<std::string>("ox3_bc", "outflow",
                                      "Outer boundary condition along x3.",
                                      {"periodic", "outflow", "reflect", "user"});

  // <parthenon/meshblock>
  auto &parthenon_meshblock = unit->AddData("parthenon/meshblock");
  parthenon_meshblock.AddParm<int>("nx1", 16, "Size of meshblocks in x1.");
  parthenon_meshblock.AddParm<int>("nx2", 16, "Size of meshblocks in x2.");
  parthenon_meshblock.AddParm<int>("nx3", 16, "Size of meshblocks in x3.");

  // kamayan refinement
  // In the following refinement fields are only added if they are found in the
  // input parameters
  const std::string ref_block = "kamayan/refinement";
  int nref_vars = 0;
  auto rps = unit->RuntimeParameters();
  while (rps && adaptive == "adaptive") {
    const std::string ref_block_n = ref_block + std::to_string(nref_vars);
    if (!rps->GetPin()->DoesBlockExist(ref_block_n)) {
      break;
    }
    auto &kamayan_refinement = unit->AddData(ref_block_n);
    kamayan_refinement.AddParm<std::string>("field", "NO FIELD WAS SET",
                                            "Field to refine on.");
    kamayan_refinement.AddParm<std::string>(
        "method", "loehner", "Method to use for refinement",
        {"loehner", "derivative_order_1", "derivative_order_2"});
    kamayan_refinement.AddParm<Real>("refine_tol", 0.8, "Error threshold for refinement");
    kamayan_refinement.AddParm<Real>("derefine_tol", 0.2,
                                     "Error threshold for derefinement");
    kamayan_refinement.AddParm<Real>(
        "filter", 0.01, "Noise filtering strength used in Loehner estimator.");
    kamayan_refinement.AddParm<int>("max_level", global_max_level,
                                    "max refinement level for this field.");
    nref_vars += 1;
  }

  auto &kamayan_refinement = unit->AddData(ref_block);
  kamayan_refinement.AddParm<int>(
      "nref_vars", nref_vars,
      "Parameter determined at runtime for the number of registered refinement fields. "
      "Never any reason to be set.");
}

void InitializeData(KamayanUnit *unit) {
  auto rps = unit->RuntimeParameters();

  const std::string ref_block = "kamayan/refinement";
  auto adaptive = rps->Get<std::string>("parthenon/mesh", "refinement");
  int nref_vars = 0;
  while (true && adaptive == "adaptive") {
    std::string ref_block_n = ref_block + std::to_string(nref_vars);
    if (!rps->GetPin()->DoesBlockExist(ref_block_n)) {
      break;
    }
    const auto field = rps->Get<std::string>(ref_block_n, "field");
    if (field != "NO FIELD WAS SET") {
      // unit IS the package (StateDescriptor)
      unit->amr_criteria.push_back(MakeAMRCriteria(rps.get(), ref_block_n));
    }
    nref_vars += 1;
  }
  // --8<-- [start:addscratch]
  if (nref_vars > 0) AddScratch<RefinementScratch>(unit);
  // --8<-- [end:addscratch]
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
  if (mbase->NumBlocks() == 0) return prev;  // we don't have any blocks, just return
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
