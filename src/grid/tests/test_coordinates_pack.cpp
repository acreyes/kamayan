#include <gtest/gtest.h>

#include <memory>

#include <mesh/meshblock.hpp>

#include "grid/coordinates.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/unit.hpp"

namespace kamayan::grid {
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

TEST(CoordinatePackTest, CartesianDx) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;
  const Real expected_dx = 0.125;

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
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.template Dx<Axis::IAXIS>(k, j, i) - expected_dx) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::JAXIS>(k, j, i) - expected_dx) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::KAXIS>(k, j, i) - expected_dx) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianVolume) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;
  const Real expected_vol = 0.125 * 0.125 * 0.125;

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
        pack(0, coords::Volume(), k, j, i) = dx * dx * dx;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.CellVolume(k, j, i) - expected_vol) > 1e-10) nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianXc) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;
  const Real dx = 1.0 / static_cast<Real>(NXB);

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

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::Xc<Axis::IAXIS>(), k, j, i) = (static_cast<Real>(i) + 0.5) * dx;
        pack(0, coords::Xc<Axis::JAXIS>(), k, j, i) = (static_cast<Real>(j) + 0.5) * dx;
        pack(0, coords::Xc<Axis::KAXIS>(), k, j, i) = (static_cast<Real>(k) + 0.5) * dx;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        auto expected_xc_i = (static_cast<Real>(i) + 0.5) * dx;
        auto expected_xc_j = (static_cast<Real>(j) + 0.5) * dx;
        auto expected_xc_k = (static_cast<Real>(k) + 0.5) * dx;
        if (Kokkos::abs(cpack.template Xc<Axis::IAXIS>(k, j, i) - expected_xc_i) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xc<Axis::JAXIS>(k, j, i) - expected_xc_j) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xc<Axis::KAXIS>(k, j, i) - expected_xc_k) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianFaceArea) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;
  const Real dx = 1.0 / static_cast<Real>(NXB);
  const Real expected_fa = dx * dx;

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

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::FaceArea<Axis::IAXIS>(), k, j, i) = dx * dx;
        pack(0, coords::FaceArea<Axis::JAXIS>(), k, j, i) = dx * dx;
        pack(0, coords::FaceArea<Axis::KAXIS>(), k, j, i) = dx * dx;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.template FaceArea<Axis::IAXIS>(k, j, i) - expected_fa) >
            1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template FaceArea<Axis::JAXIS>(k, j, i) - expected_fa) >
            1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template FaceArea<Axis::KAXIS>(k, j, i) - expected_fa) >
            1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianEdgeLength) {
  constexpr int NDIM = 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;
  const Real dx = 1.0 / static_cast<Real>(NXB);

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

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::EdgeLength<Axis::IAXIS>(), k, j, i) = dx;
        pack(0, coords::EdgeLength<Axis::JAXIS>(), k, j, i) = dx;
        pack(0, coords::EdgeLength<Axis::KAXIS>(), k, j, i) = dx;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cartesian, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.template EdgeLength<Axis::IAXIS>(k, j, i) - dx) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template EdgeLength<Axis::JAXIS>(k, j, i) - dx) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template EdgeLength<Axis::KAXIS>(k, j, i) - dx) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CylindricalDxAndVolume) {
  constexpr int NDIM = 3;
  constexpr int NXB = 4;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cylindrical);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), Geometry::cylindrical);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  const Real dr = 0.25;
  const Real dtheta = 0.7853981633974483;
  const Real dphi = 1.5707963267948966;

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        pack(0, coords::Dx<Axis::IAXIS>(), k, j, i) = dr;
        pack(0, coords::Dx<Axis::JAXIS>(), k, j, i) = dtheta;
        pack(0, coords::Dx<Axis::KAXIS>(), k, j, i) = dphi;
        pack(0, coords::Volume(), k, j, i) = dr * dtheta * dphi;
      });

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<Geometry::cylindrical, CoordFields>(pack, b);
        if (Kokkos::abs(cpack.template Dx<Axis::IAXIS>(k, j, i) - dr) > 1e-10) nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::JAXIS>(k, j, i) - dtheta) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::KAXIS>(k, j, i) - dphi) > 1e-10) nw += 1;
        if (Kokkos::abs(cpack.CellVolume(k, j, i) - dr * dtheta * dphi) > 1e-10) nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}
}  // namespace kamayan::grid
