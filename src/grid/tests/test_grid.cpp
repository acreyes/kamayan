#include "test_grid.hpp"

#include <gtest/gtest.h>

#include <memory>

#include <mesh/meshblock.hpp>

#include "basic_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/scratch_variables.hpp"
#include "grid/subpack.hpp"
#include "kamayan/fields.hpp"
#include "kokkos_abstraction.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "utils/instrument.hpp"
#include "utils/parallel.hpp"

using parthenon::BlockList_t;

namespace kamayan {
BlockList_t MakeTestBlockList(const std::shared_ptr<StateDescriptor> pkg,
                              const int NBLOCKS, const int NXB, const int NDIM) {
  BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<MeshBlock>(NXB, NDIM);
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}

MeshData MakeTestMeshData(parthenon::BlockList_t block_list) {
  MeshData mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);
  return mesh_data;
}

using Fields = TypeList<DENS, MOMENTUM, ENER>;

TEST(grid, PackIndexer) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 9;
  auto pkg = std::make_shared<StateDescriptor>("Test Package");
  AddFields(Fields(), pkg.get(), {CENTER_FLAGS(Metadata::WithFluxes)});

  // now build our test grid
  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  // because we are not actually inside parthenon we need
  // to build the pack/descriptor directly form our single package
  auto pack = grid::GetPack(Fields(), pkg.get(), &md);
  const Real di = 10.0;
  const Real dj = 25.0;
  const Real dk = 44.0;
  {
    auto ib = md.GetBoundsI(IndexDomain::entire);
    auto jb = md.GetBoundsJ(IndexDomain::entire);
    auto kb = md.GetBoundsK(IndexDomain::entire);
    // initialze our data using a pack
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          for (int var = pack.GetLowerBound(b); var <= pack.GetUpperBound(b); var++) {
            pack(b, var, k, j, i) =
                (b + NBLOCKS * var) + 10.0 * i * j + 25.0 * j * k + 44.0 * k * i;
          }
        });

    // check that our indexer is working
    int n_not_matching;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL,
        parthenon::DevExecSpace(), 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &ntot) {
          auto idxer = SubPack(pack, b, k, j, i);
          if (pack(b, DENS(), k, j, i) != idxer(DENS())) ntot += 1;
          if (pack(b, MOMENTUM(0), k, j, i) != idxer(MOMENTUM(0))) ntot += 1;
          if (pack(b, MOMENTUM(1), k, j, i) != idxer(MOMENTUM(1))) ntot += 1;
          if (pack(b, MOMENTUM(2), k, j, i) != idxer(MOMENTUM(2))) ntot += 1;
          if (pack(b, ENER(), k, j, i) != idxer(ENER())) ntot += 1;
        },
        Kokkos::Sum<int>(n_not_matching));
    EXPECT_EQ(n_not_matching, 0);
  }

  // check our stencils
  {
    auto ib = md.GetBoundsI(IndexDomain::entire);
    auto jb = md.GetBoundsJ(IndexDomain::entire);
    auto kb = md.GetBoundsK(IndexDomain::entire);
    using Arr_t = Real;
    Arr_t err;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL,
        parthenon::DevExecSpace(), 0, NBLOCKS - 1, kb.s + 1, kb.e - 1, jb.s + 1, jb.e - 1,
        ib.s + 1, ib.e - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Arr_t &err_loc) {
          for (int var = pack.GetLowerBound(b); var <= pack.GetUpperBound(b); var++) {
            auto stencilx = SubPack<Axis::IAXIS>(pack, b, var, k, j, i);
            auto stencily = SubPack<Axis::JAXIS>(pack, b, var, k, j, i);
            auto stencilz = SubPack<Axis::KAXIS>(pack, b, var, k, j, i);
            err_loc += Kokkos::abs(0.5 * (stencilx(1) - stencilx(-1)) - j * di - k * dk) /
                       (i * di + k * dk);
            err_loc += Kokkos::abs(0.5 * (stencily(1) - stencily(-1)) - i * di - k * dj) /
                       (i * di + k * dj);
            err_loc += Kokkos::abs(0.5 * (stencilz(1) - stencilz(-1)) - i * dk - dj * j) /
                       (i * dk + dj * j);
          }
        },
        Kokkos::Sum<Arr_t>(err));
    auto nvars = pack.GetMaxNumberOfVars();
    err = err / (nvars * NBLOCKS * NXB * NXB * NXB);
    EXPECT_LT(err, 1.e-12);
  }
}

}  // namespace kamayan

namespace kamayan {

struct Scratch {
  // we'll allocate scratch space for a 1 index vector & 2 index tensor
  static constexpr auto TT = TopologicalType::Cell;
  using Vector = RuntimeScratchVariable<"vector", TT>;
  using Tensor = RuntimeScratchVariable<"tensor", TT>;
  using type = RuntimeScratchVariableList<Vector, Tensor>;
};

KOKKOS_INLINE_FUNCTION Real ScratchValue(const int &b, const int &var, const int &k,
                                         const int &j, const int &i) {
  return 53. + b * 8.0 + var * (var - 45.0) + k * j - var * i + i * b;
}

TEST(ScratchVarTest, SubPack) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 9;
  constexpr int nvec = 3;
  constexpr int ntj = 5;
  constexpr int nti = 2;

  auto pkg = std::make_shared<StateDescriptor>("Test Package");

  Scratch::type scratch;
  scratch.template RegisterShape<Scratch::Vector>({nvec});
  scratch.template RegisterShape<Scratch::Tensor>({ntj, nti});
  AddScratch(scratch, pkg.get());

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = std::make_shared<MeshData>(MakeTestMeshData(block_list));
  auto ib = md->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBoundsK(IndexDomain::entire);
  // if we flatten our indices we get
  // 0 -- vector(0)
  // 1 -- vector(1)
  // 2 -- vector(2)
  //
  // 3 -- tensor(0,0) , 4 -- tensor(0,1)
  // 5 -- tensor(1,0) , 6 -- tensor(1,1)
  // 7 -- tensor(2,0) , 8 -- tensor(2,1)
  // 9 -- tensor(3,0) , 10 -- tensor(3,1)
  // 11 -- tensor(4,0), 12 -- tensor(4,1)

  auto pack = grid::GetPack(Scratch::type::list(), pkg.get(), md.get());
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        // note that parthenon will double count if the same type
        // appears twice. TODO(acreyes): our wrapper to get packs could
        // filter out types that appear twice
        for (int var = 0; var <= pack.GetUpperBound(b); var++) {
          pack(b, var, k, j, i) = ScratchValue(b, var, k, j, i);
        }
      });

  auto scratch_pack = ScratchPack(pkg.get(), md.get(), scratch);
  int nwrong = 0;
  par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &nwr) {
        const auto vs = scratch_pack.GetLowerBound(Scratch::Vector());
        const auto ve = scratch_pack.GetUpperBound(Scratch::Vector());

        auto subpack = scratch_pack.SubPack(b, k, j, i);
        for (int var = vs; var <= ve; var++) {
          const auto answer = ScratchValue(b, var, k, j, i);
          const auto idx = var - vs;
          const auto sub_answer = subpack(Scratch::Vector(idx));
          const auto pack_answer = scratch_pack(b, Scratch::Vector(idx), k, j, i);

          nwr += answer == sub_answer ? 0 : 1;
          nwr += answer == pack_answer ? 0 : 1;
          nwr += sub_answer == pack_answer ? 0 : 1;
        }
      },
      Kokkos::Sum<int>(nwrong));

  EXPECT_EQ(nwrong, 0);
}

}  // namespace kamayan
