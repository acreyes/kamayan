#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include <mesh/meshblock.hpp>

#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/kamayan.hpp"
#include "kamayan/unit.hpp"

using namespace kamayan;
using namespace grid;

parthenon::BlockList_t MakeTestBlockList(const std::shared_ptr<KamayanUnit> pkg,
                                         const int NBLOCKS, const int NXB,
                                         const int NDIM) {
  parthenon::BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<parthenon::MeshBlock>(NXB, NDIM);
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}

parthenon::MeshData<Real> MakeTestMeshData(parthenon::BlockList_t block_list) {
  parthenon::MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);
  return mesh_data;
}

void AddCoordFields(KamayanUnit *pkg, Geometry geom) {
  using parthenon::Metadata;
  GeometryOptions::dispatch(
      [&]<Geometry g>() {
        auto md = Metadata({Metadata::Cell, Metadata::Derived});
        pkg->AddField<coords::Volume>(md);
        auto axis_md = Metadata({Metadata::Cell, Metadata::Derived});
        [&]<Axis... axes>() {
          (pkg->AddField<coords::Dx<axes>>(axis_md), ...);
          (pkg->AddField<coords::X<axes>>(axis_md), ...);
          (pkg->AddField<coords::Xc<axes>>(axis_md), ...);
          (pkg->AddField<coords::Xf<axes>>(axis_md), ...);
          (pkg->AddField<coords::FaceArea<axes>>(axis_md), ...);
          (pkg->AddField<coords::EdgeLength<axes>>(axis_md), ...);
        }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();
      },
      geom);
}

TEST(CoordinatePackTest, SimpleKernelConstruction) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cartesian);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), Geometry::cartesian);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  const Real dx = 1.0 / static_cast<Real>(NXB);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::Dx<Axis::IAXIS>(), k, j, i) = dx;
        pack(0, coords::Dx<Axis::JAXIS>(), k, j, i) = dx;
        pack(0, coords::Dx<Axis::KAXIS>(), k, j, i) = dx;
        pack(0, coords::Volume(), k, j, i) = dx * dx * dx;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.template Dx<Axis::IAXIS>(k, j, i) - 0.125) > 1e-10) nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}
