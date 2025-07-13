#include "grid_refinement.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "grid/grid.hpp"
#include "grid/grid_types.hpp"

namespace kamayan::grid {
std::shared_ptr<parthenon::AMRCriteria>
MakeAMRCriteria(const runtime_parameters::RuntimeParameters *rps,
                std::string block_name) {
  auto method_str = rps->Get<std::string>(block_name, "method");
  auto method = MapStrToEnum<RefinementCriteria>(
      method_str, std::make_pair(RefinementCriteria::loehner, "loehner"),
      std::make_pair(RefinementCriteria::first, "derivative_order_1"),
      std::make_pair(RefinementCriteria::second, "derivative_order_2"));

  if (method == RefinementCriteria::first || method == RefinementCriteria::second) {
    // parthenon built in
    return parthenon::AMRCriteria::MakeAMRCriteria(method_str, rps->GetPin(), block_name);
  }
  return std::make_shared<AMRLoehner>(rps, block_name);
}

void AMRLoehner::operator()(MeshData *md,
                            parthenon::ParArray1D<AmrTag> &delta_level) const {
  const auto desc =
      MakePackDescriptor(md->GetMeshPointer()->resolved_packages.get(), {field});
  auto pack = desc.GetPack(md);

  const auto pack_der = GetPack<FirstDer>(md);

  const int ndim = md->GetMeshPointer()->ndim;
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);

  // this was just copied from
  auto dims = md->GetMeshPointer()->resolved_packages->FieldMetadata(field).Shape();
  const int k2d = ndim > 1 ? 1 : 0;
  const int k3d = ndim > 2 ? 1 : 0;
  int n5(0), n4(0);
  if (dims.size() > 2) {
    n5 = dims[1];
    n4 = dims[2];
  } else if (dims.size() > 1) {
    n5 = dims[0];
    n4 = dims[1];
  }
  const int var = comp4 + n4 * (comp5 + n5 * comp6);

  // calculate derivatives
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s - k3d, kb.e + k3d, jb.s - k2d,
      jb.e + k2d, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto coords = pack.GetCoordinates(b);
        using TE = TopologicalElement;
        // --8<-- [start:FirstDer]
        pack_der(b, FirstDer(0), k, j, i) =
            0.5 * (pack(b, var, k, j, i + 1) - pack(b, var, k, j, i - 1)) /
            coords.Dxc<1>();
        // --8<-- [end:FirstDer]

        if (ndim > 1)
          pack_der(b, FirstDer(1), k, j, i) =
              0.5 * (pack(b, var, k, j + 1, i) - pack(b, var, k, j - 1, i)) /
              coords.Dxc<2>();

        if (ndim > 2)
          pack_der(b, FirstDer(2), k, j, i) =
              0.5 * (pack(b, var, k + 1, j, i) - pack(b, var, k - 1, j, i)) /
              coords.Dxc<3>();
      });

  auto scatter_tags = delta_level.ToScatterView<Kokkos::Experimental::ScatterMax>();
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_CLASS_LAMBDA(parthenon::team_mbr_t team_member, const int b, const int k,
                          const int j) {
        const auto coords = pack.GetCoordinates(b);
        Real max_err_2 = 0.;
        parthenon::par_reduce_inner(
            parthenon::inner_loop_pattern_ttr_tag, team_member, ib.s, ib.e,
            [&](const int i, Real &loc_max_err_2) {
              // numerator = sum ( d2u_dpdq * dxpdxq )^2
              // denominator = sum [ dxp*(d|u|_dp_q+ + d|u|_dp_q-)
              //                    +eps*d2|u|_dpdq * dxpdxq]^2
              using TE = TopologicalElement;
              Real numerator = 0.0;
              Real denominator = 1.e-12;
              for (int q = 0; q < ndim; q++) {
                auto fq = static_cast<TE>(q + static_cast<int>(TE::F1));
                Kokkos::Array<int, 3> kji_q{(fq == TE::F3), (fq == TE::F2),
                                            (fq == TE::F1)};
                for (int p = 0; p < ndim; p++) {
                  auto fp = static_cast<TE>(p + static_cast<int>(TE::F1));
                  Kokkos::Array<int, 3> kji_p{(fp == TE::F3), (fp == TE::F2),
                                              (fp == TE::F1)};

                  const Real num = 0.5 *
                                   (pack_der(b, FirstDer(p), k + kji_q[0], j + kji_q[1],
                                             i + kji_q[2]) -
                                    pack_der(b, FirstDer(p), k - kji_q[0], j - kji_q[1],
                                             i - kji_q[2])) /
                                   coords.Dx(q + 1);
                  numerator += std::pow(num, 2);

                  const Real denom =
                      0.5 *
                          (std::abs(pack_der(b, FirstDer(p), k + kji_q[0], j + kji_q[1],
                                             i + kji_q[2])) +
                           std::abs(pack_der(b, FirstDer(p), k - kji_q[0], j - kji_q[1],
                                             i - kji_q[2]))) /
                          coords.Dx(p + 1) +
                      filter *
                          (std::abs(
                               pack(b, var, k + kji_q[0], j + kji_q[1], i + kji_q[2])) +
                           std::abs(
                               pack(b, var, k - kji_q[0], j - kji_q[1], i - kji_q[2])) +
                           std::abs(
                               pack(b, var, k - kji_p[0], j - kji_p[1], i - kji_p[2])) +
                           std::abs(
                               pack(b, var, k + kji_p[0], j + kji_p[1], i + kji_p[2]))) /
                          (coords.Dx(q + 1) * coords.Dx(p + 1));
                  denominator += std::pow(denom, 2);
                }
              }
              loc_max_err_2 = denominator == 0.0
                                  ? loc_max_err_2
                                  : Kokkos::max(loc_max_err_2, numerator / denominator);
            },
            Kokkos::Max<Real>(max_err_2));
        auto tags_access = scatter_tags.access();
        auto flag = AmrTag::same;
        const Real max_err = Kokkos::sqrt(max_err_2);
        if (max_err > refine_criteria && pack.GetLevel(b, 0, 0, 0) < max_level)
          flag = AmrTag::refine;
        if (max_err < derefine_criteria) flag = AmrTag::derefine;
        tags_access(b).update(flag);
      });
  delta_level.ContributeScatter(scatter_tags);
}
}  // namespace kamayan::grid
