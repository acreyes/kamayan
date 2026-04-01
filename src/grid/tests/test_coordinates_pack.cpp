#include "test_coordinates_pack.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include <coordinates/uniform_cartesian.hpp>
#include <mesh/meshblock.hpp>

#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {

struct TestRegion {
  std::array<Real, 3> xmin;
  std::array<Real, 3> xmax;
  std::array<Real, 3> xrat;
  std::array<int, 3> nx;
};

TestRegion Cartesian3D() {
  return {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {8, 8, 8}};
}

TestRegion Cylindrical2D() {
  return {
      {0.0, 0.0, 0.0}, {1.0, 2.0 * std::numbers::pi, 1.0}, {1.0, 1.0, 1.0}, {4, 4, 1}};
}

parthenon::RegionSize MakeRegionSize(const TestRegion &r) {
  return parthenon::RegionSize(r.xmin, r.xmax, r.xrat, r.nx);
}

std::shared_ptr<parthenon::MeshBlock> MakeTestMeshBlockCartesian3D() {
  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cartesian);
  pkg->InitResources(rps, cfg);

  auto pmb = std::make_shared<parthenon::MeshBlock>(8, 3);
  auto &pmbd = pmb->meshblock_data.Get();
  pmbd->Initialize(pkg, pmb);
  return pmb;
}

std::shared_ptr<parthenon::MeshBlock> MakeTestMeshBlockCylindrical2D() {
  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cylindrical);
  pkg->InitResources(rps, cfg);

  auto pmb = std::make_shared<parthenon::MeshBlock>(4, 2);
  auto &pmbd = pmb->meshblock_data.Get();
  pmbd->Initialize(pkg, pmb);
  return pmb;
}

parthenon::BlockList_t MakeTestBlockListCartesian3D() {
  auto pmb = MakeTestMeshBlockCartesian3D();
  parthenon::BlockList_t block_list;
  block_list.push_back(pmb);
  return block_list;
}

parthenon::BlockList_t MakeTestBlockListCylindrical2D() {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  parthenon::BlockList_t block_list;
  block_list.push_back(pmb);
  return block_list;
}

MeshData MakeTestMeshDataCartesian3D() {
  auto block_list = MakeTestBlockListCartesian3D();
  MeshData mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);
  return mesh_data;
}

MeshData MakeTestMeshDataCylindrical2D() {
  auto block_list = MakeTestBlockListCylindrical2D();
  MeshData mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);
  return mesh_data;
}

std::shared_ptr<KamayanUnit> MakeTestPackageCartesian3D() {
  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cartesian);
  pkg->InitResources(rps, cfg);
  return pkg;
}

std::shared_ptr<KamayanUnit> MakeTestPackageCylindrical2D() {
  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(Geometry::cylindrical);
  pkg->InitResources(rps, cfg);
  return pkg;
}

namespace {

constexpr Real REL_TOL = 1e-10;
constexpr Real ABS_TOL = 1e-12;

bool IsRelClose(Real a, Real b) {
  if (std::abs(b) < ABS_TOL) {
    return std::abs(a - b) < ABS_TOL;
  }
  return std::abs((a - b) / b) < REL_TOL;
}

}  // namespace

TEST(CoordinatePackTest, Cartesian3D_PackConstruction) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);
  (void)coord_pack;
}

TEST(CoordinatePackTest, Cartesian3D_Dx_Template) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::IAXIS>(0, 0, 0), 0.125);
  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::JAXIS>(0, 0, 0), 0.125);
  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::KAXIS>(0, 0, 0), 0.125);
}

TEST(CoordinatePackTest, Cartesian3D_Xc_Template) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  for (int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xc<Axis::IAXIS>(0, 0, i), 0.0625 + i * 0.125);
  }
  for (int j = 0; j < 8; j++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xc<Axis::JAXIS>(0, j, 0), 0.0625 + j * 0.125);
  }
  for (int k = 0; k < 8; k++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xc<Axis::KAXIS>(k, 0, 0), 0.0625 + k * 0.125);
  }
}

TEST(CoordinatePackTest, Cartesian3D_Xf_Template) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  for (int i = 0; i <= 8; i++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xf<Axis::IAXIS>(0, 0, i), i * 0.125);
  }
  for (int j = 0; j <= 8; j++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xf<Axis::JAXIS>(0, j, 0), j * 0.125);
  }
  for (int k = 0; k <= 8; k++) {
    EXPECT_FLOAT_EQ(coord_pack.template Xf<Axis::KAXIS>(k, 0, 0), k * 0.125);
  }
}

TEST(CoordinatePackTest, Cartesian3D_FaceArea_Template) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coord_pack.template FaceArea<Axis::IAXIS>(k, j, i),
                        0.125 * 0.125);
        EXPECT_FLOAT_EQ(coord_pack.template FaceArea<Axis::JAXIS>(k, j, i),
                        0.125 * 0.125);
        EXPECT_FLOAT_EQ(coord_pack.template FaceArea<Axis::KAXIS>(k, j, i),
                        0.125 * 0.125);
      }
    }
  }
}

TEST(CoordinatePackTest, Cartesian3D_EdgeLength_Template) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coord_pack.template EdgeLength<Axis::IAXIS>(k, j, i), 0.125);
        EXPECT_FLOAT_EQ(coord_pack.template EdgeLength<Axis::JAXIS>(k, j, i), 0.125);
        EXPECT_FLOAT_EQ(coord_pack.template EdgeLength<Axis::KAXIS>(k, j, i), 0.125);
      }
    }
  }
}

TEST(CoordinatePackTest, Cartesian3D_CellVolume) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coord_pack.CellVolume(k, j, i), 0.125 * 0.125 * 0.125);
      }
    }
  }
}

TEST(CoordinatePackTest, Cartesian3D_Volume_Topological) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  auto cell_vol = coord_pack.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::CC, 0, 0, 0), cell_vol);
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F1, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F2, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F3, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::E1, 0, 0, 0),
                  coord_pack.template EdgeLength<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::E2, 0, 0, 0),
                  coord_pack.template EdgeLength<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::E3, 0, 0, 0),
                  coord_pack.template EdgeLength<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::NN, 0, 0, 0), 1.0);
}

TEST(CoordinatePackTest, Cartesian3D_RuntimeAxisDispatch) {
  auto md = MakeTestMeshDataCartesian3D();
  auto pkg = MakeTestPackageCartesian3D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cartesian>(pack, 0);

  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::IAXIS>(0, 0, 0),
                  coord_pack.Dx(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template Xc<Axis::IAXIS>(0, 0, 0),
                  coord_pack.Xc(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template Xf<Axis::IAXIS>(0, 0, 0),
                  coord_pack.Xf(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  coord_pack.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  coord_pack.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

TEST(CoordinatePackTest, Cylindrical2D_PackConstruction) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);
  (void)coord_pack;
}

TEST(CoordinatePackTest, Cylindrical2D_Dx_Template) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::IAXIS>(0, 0, 0), 0.25);
  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::JAXIS>(0, 0, 0), std::numbers::pi / 2.0);
  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::KAXIS>(0, 0, 0), 2.0 * std::numbers::pi);
}

TEST(CoordinatePackTest, Cylindrical2D_Xc_Template) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  for (int i = 0; i < 4; i++) {
    auto r0 = coord_pack.template Xf<Axis::IAXIS>(0, 0, i);
    auto r1 = coord_pack.template Xf<Axis::IAXIS>(0, 0, i + 1);
    auto expected_xc = (2.0 / 3.0) * (r1 * r1 * r1 - r0 * r0 * r0) / (r1 * r1 - r0 * r0);
    auto actual_xc = coord_pack.template Xc<Axis::IAXIS>(0, 0, i);
    ASSERT_TRUE(IsRelClose(actual_xc, expected_xc));
  }

  for (int j = 0; j < 4; j++) {
    auto expected_xc_j = std::numbers::pi / 4.0 + j * (std::numbers::pi / 2.0);
    auto actual_xc_j = coord_pack.template Xc<Axis::JAXIS>(0, j, 0);
    ASSERT_TRUE(IsRelClose(actual_xc_j, expected_xc_j));
  }

  ASSERT_TRUE(IsRelClose(coord_pack.template Xc<Axis::KAXIS>(0, 0, 0), 0.5));
}

TEST(CoordinatePackTest, Cylindrical2D_FaceArea_Template) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto r = coord_pack.template Xf<Axis::IAXIS>(0, 0, i);
      auto dx_j = coord_pack.template Dx<Axis::JAXIS>(0, 0, 0);
      auto dx_k = coord_pack.template Dx<Axis::KAXIS>(0, 0, 0);
      auto expected_i = r * dx_j * dx_k;
      auto actual_i = coord_pack.template FaceArea<Axis::IAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_i, expected_i));

      auto rp = coord_pack.template Xf<Axis::IAXIS>(0, 0, i + 1);
      auto rm = coord_pack.template Xf<Axis::IAXIS>(0, 0, i);
      auto expected_j = 0.5 * (rp * rp - rm * rm) * dx_k;
      auto actual_j = coord_pack.template FaceArea<Axis::JAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_j, expected_j));

      auto dx_i = coord_pack.template Dx<Axis::IAXIS>(0, 0, 0);
      auto expected_k = dx_i * dx_j;
      auto actual_k = coord_pack.template FaceArea<Axis::KAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_k, expected_k));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_EdgeLength_Template) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto dx_i = coord_pack.template Dx<Axis::IAXIS>(0, 0, 0);
      auto el_i = coord_pack.template EdgeLength<Axis::IAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(el_i, dx_i));

      auto dx_j = coord_pack.template Dx<Axis::JAXIS>(0, 0, 0);
      auto el_j = coord_pack.template EdgeLength<Axis::JAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(el_j, dx_j));

      auto r = coord_pack.template Xf<Axis::IAXIS>(0, 0, i);
      auto dx_k = coord_pack.template Dx<Axis::KAXIS>(0, 0, 0);
      auto expected_k = std::abs(r) * dx_k;
      auto actual_k = coord_pack.template EdgeLength<Axis::KAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_k, expected_k));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_CellVolume) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto rp = coord_pack.template Xf<Axis::IAXIS>(0, 0, i + 1);
      auto rm = coord_pack.template Xf<Axis::IAXIS>(0, 0, i);
      auto dx_k = coord_pack.template Dx<Axis::KAXIS>(0, 0, 0);
      auto dx_j = coord_pack.template Dx<Axis::JAXIS>(0, 0, 0);
      auto expected = 0.5 * dx_k * (rp * rp - rm * rm) * dx_j;
      auto actual = coord_pack.CellVolume(0, j, i);
      ASSERT_TRUE(IsRelClose(actual, expected));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_Volume_Topological) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  auto cell_vol = coord_pack.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::CC, 0, 0, 0), cell_vol);
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F1, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F2, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::F3, 0, 0, 0),
                  coord_pack.template FaceArea<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.Volume(TopologicalElement::NN, 0, 0, 0), 1.0);
}

TEST(CoordinatePackTest, Cylindrical2D_RuntimeAxisDispatch) {
  auto md = MakeTestMeshDataCylindrical2D();
  auto pkg = MakeTestPackageCylindrical2D();

  auto pack = grid::GetPack(grid::CoordFields(), pkg.get(), &md);
  auto coord_pack = grid::CoordinatePack<Geometry::cylindrical>(pack, 0);

  EXPECT_FLOAT_EQ(coord_pack.template Dx<Axis::IAXIS>(0, 0, 0),
                  coord_pack.Dx(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template Xc<Axis::JAXIS>(0, 0, 0),
                  coord_pack.Xc(Axis::JAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template Xf<Axis::IAXIS>(0, 0, 0),
                  coord_pack.Xf(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  coord_pack.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coord_pack.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  coord_pack.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

}  // namespace kamayan
