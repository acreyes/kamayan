#include "bvals/boundary_conditions.hpp"
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include <defs.hpp>

#include "basic_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "pack/pack_utils.hpp"

namespace kamayan::grid {
using TE = TopologicalElement;
// use this as an interface to write boundary conditions per variable topological element
// types that can be registered through parthenon's appliciation input interface at the
// bottom of this file
using BoundaryConditionInterface =
    std::function<void(const TE, std::shared_ptr<MeshBlockData> &, bool)>;

void Axisymmetric(const TE el, std::shared_ptr<MeshBlockData> &mbd, bool coarse) {
  const auto ttFlag = [el] {
    const auto tt = GetTopologicalType(el);
    switch (tt) {
    case (TopologicalType::Cell):
      return Metadata::Cell;
    case (TopologicalType::Face):
      return Metadata::Face;
    case (TopologicalType::Edge):
      return Metadata::Edge;
    case (TopologicalType::Node):
      return Metadata::Node;
    default:
      PARTHENON_FAIL("Unknown topological type")
    }
  }();

  std::vector<MetadataFlag> flags{Metadata::FillGhost, ttFlag};
  std::set<PDOpt> opts = coarse ? std::set<PDOpt>{PDOpt::Coarse} : std::set<PDOpt>{};
  const auto desc =
      MakePackDescriptor<parthenon::variable_names::any>(mbd.get(), flags, opts);

  auto q = desc.GetPack(mbd.get());
  const int b = 0;
  const int lstart = q.GetLowerBoundHost(b);
  const int lend = q.GetUpperBoundHost(b);
  // early return if we don't get anything
  if (lend < lstart) return;
  auto nb = parthenon::IndexRange{lstart, lend};

  MeshBlock *pmb = mbd->GetBlockPointer();
  const auto &bounds = (coarse ? pmb->c_cellbounds : pmb->cellbounds);

  const auto &range = bounds.GetBoundsI(IndexDomain::interior, el);
  const int ref = range.s;

  constexpr IndexDomain domain = IndexDomain::inner_x1;
  // used for reflections
  const int offset = 2 * ref - 1;

  pmb->par_for_bndry(
      PARTHENON_AUTO_LABEL, nb, domain, el, coarse, false,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const auto vec_component = q(b, el, l).vector_component;
        const Real factor =
            (vec_component == 1 || vec_component == 3 || el == TE::F1) ? -1.0 : 1.0;
        q(b, el, l, k, j, i) = factor * q(b, el, l, k, j, offset - i);
      });
}

parthenon::BValFunc GenericBC(BoundaryConditionInterface bndry_func) {
  return [=](std::shared_ptr<MeshBlockData> &mbd, bool coarse) {
    for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
      bndry_func(el, mbd, coarse);
  };
}

void RegisterBoundaryConditions(parthenon::ApplicationInput *app) {
  app->RegisterBoundaryCondition(parthenon::BoundaryFace::inner_x1, "axisymmetric",
                                 GenericBC(Axisymmetric));
}
}  // namespace kamayan::grid
