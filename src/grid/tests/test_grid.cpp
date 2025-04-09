#include "test_grid.hpp"

#include <gtest/gtest.h>

#include <memory>

#include <mesh/meshblock.hpp>

#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "interface/make_pack_descriptor.hpp"
#include "kamayan/fields.hpp"

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
  auto desc = [&]<typename... Ts>(TypeList<Ts...>) {
    return parthenon::MakePackDescriptor<Ts...>(pkg.get());
  }(Fields());
  auto pack = desc.GetPack(&md);
  {
    auto ib = md.GetBoundsI(IndexDomain::entire);
    auto jb = md.GetBoundsJ(IndexDomain::entire);
    auto kb = md.GetBoundsK(IndexDomain::entire);
    // initialze our data using a pack
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          for (int var = pack.GetLowerBound(b); var <= pack.GetUpperBound(b); var++) {
            pack(b, var, k, j, i) = i + NXB * (j + NXB * (k + NXB * (b + NBLOCKS * var)));
          }
        });
  }
}

}  // namespace kamayan
