#include "test_geometry.hpp"

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <coordinates/uniform_cartesian.hpp>
#include <mesh/meshblock.hpp>

#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
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
  return std::make_shared<parthenon::MeshBlock>(8, 2);
}

parthenon::UniformCartesian MakeCoordinatesCartesian3D() {
  return parthenon::UniformCartesian(MakeRegionSize(Cartesian3D()), nullptr);
}

parthenon::UniformCartesian MakeCoordinatesCylindrical2D() {
  return parthenon::UniformCartesian(MakeRegionSize(Cylindrical2D()), nullptr);
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

void VerifyCartesian3D_Dx(const grid::Coordinates<Geometry::cartesian> &coords) {
  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  EXPECT_FLOAT_EQ(dx_i, 0.125);
  EXPECT_FLOAT_EQ(dx_j, 0.125);
  EXPECT_FLOAT_EQ(dx_k, 0.125);
}

void VerifyCartesian3D_Xc_AllCells(const grid::Coordinates<Geometry::cartesian> &coords) {
  for (int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(coords.template Xc<Axis::IAXIS>(i), 0.0625 + i * 0.125);
  }
  for (int j = 0; j < 8; j++) {
    EXPECT_FLOAT_EQ(coords.template Xc<Axis::JAXIS>(j), 0.0625 + j * 0.125);
  }
  for (int k = 0; k < 8; k++) {
    EXPECT_FLOAT_EQ(coords.template Xc<Axis::KAXIS>(k), 0.0625 + k * 0.125);
  }
}

void VerifyCartesian3D_Xf_AllFaces(const grid::Coordinates<Geometry::cartesian> &coords) {
  for (int i = 0; i <= 8; i++) {
    EXPECT_FLOAT_EQ(coords.template Xf<Axis::IAXIS>(i), i * 0.125);
  }
  for (int j = 0; j <= 8; j++) {
    EXPECT_FLOAT_EQ(coords.template Xf<Axis::JAXIS>(j), j * 0.125);
  }
  for (int k = 0; k <= 8; k++) {
    EXPECT_FLOAT_EQ(coords.template Xf<Axis::KAXIS>(k), k * 0.125);
  }
}

void VerifyCartesian3D_FaceArea(const grid::Coordinates<Geometry::cartesian> &coords) {
  EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::IAXIS>(0, 0, 0), 0.125 * 0.125);
  EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::JAXIS>(0, 0, 0), 0.125 * 0.125);
  EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::KAXIS>(0, 0, 0), 0.125 * 0.125);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::IAXIS>(k, j, i), 0.125 * 0.125);
        EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::JAXIS>(k, j, i), 0.125 * 0.125);
        EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::KAXIS>(k, j, i), 0.125 * 0.125);
      }
    }
  }
}

void VerifyCartesian3D_EdgeLength(const grid::Coordinates<Geometry::cartesian> &coords) {
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(0, 0, 0), 0.125);
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::JAXIS>(0, 0, 0), 0.125);
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::KAXIS>(0, 0, 0), 0.125);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(k, j, i), 0.125);
        EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::JAXIS>(k, j, i), 0.125);
        EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::KAXIS>(k, j, i), 0.125);
      }
    }
  }
}

void VerifyCartesian3D_CellVolume(const grid::Coordinates<Geometry::cartesian> &coords) {
  EXPECT_FLOAT_EQ(coords.CellVolume(0, 0, 0), 0.125 * 0.125 * 0.125);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coords.CellVolume(k, j, i), 0.125 * 0.125 * 0.125);
      }
    }
  }
}

void VerifyCartesian3D_VolumeTopological(
    const grid::Coordinates<Geometry::cartesian> &coords) {
  auto cell_vol = coords.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::CC, 0, 0, 0), cell_vol);
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F1, 0, 0, 0),
                  coords.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F2, 0, 0, 0),
                  coords.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F3, 0, 0, 0),
                  coords.template FaceArea<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::E1, 0, 0, 0),
                  coords.template EdgeLength<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::E2, 0, 0, 0),
                  coords.template EdgeLength<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::E3, 0, 0, 0),
                  coords.template EdgeLength<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::NN, 0, 0, 0), 1.0);
}

void VerifyCylindrical2D_Dx(const grid::Coordinates<Geometry::cylindrical> &coords) {
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::IAXIS>(), 0.25);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::JAXIS>(), std::numbers::pi / 2.0);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::KAXIS>(), 2.0 * std::numbers::pi);
}

void VerifyCylindrical2D_Xc_AllCells(
    const grid::Coordinates<Geometry::cylindrical> &coords) {
  for (int i = 0; i < 4; i++) {
    auto r0 = coords.template Xf<Axis::IAXIS>(i);
    auto r1 = coords.template Xf<Axis::IAXIS>(i + 1);
    auto expected_xc = (2.0 / 3.0) * (r1 * r1 * r1 - r0 * r0 * r0) / (r1 * r1 - r0 * r0);
    auto actual_xc = coords.template Xc<Axis::IAXIS>(i);
    EXPECT_FLOAT_EQ(actual_xc, expected_xc);
  }

  for (int j = 0; j < 4; j++) {
    auto expected_xc_j = std::numbers::pi / 4.0 + j * (std::numbers::pi / 2.0);
    EXPECT_FLOAT_EQ(coords.template Xc<Axis::JAXIS>(j), expected_xc_j);
  }

  EXPECT_FLOAT_EQ(coords.template Xc<Axis::KAXIS>(0), 0.5);
}

void VerifyCylindrical2D_FaceArea_AllCells(
    const grid::Coordinates<Geometry::cylindrical> &coords) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto r = coords.template Xf<Axis::IAXIS>(i);
      auto dx_j = coords.template Dx<Axis::JAXIS>();
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto expected_i = r * dx_j * dx_k;
      EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::IAXIS>(0, j, i), expected_i);

      auto rp = coords.template Xf<Axis::IAXIS>(i + 1);
      auto rm = coords.template Xf<Axis::IAXIS>(i);
      auto expected_j = 0.5 * (rp * rp - rm * rm) * dx_k;
      EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::JAXIS>(0, j, i), expected_j);

      auto dx_i = coords.template Dx<Axis::IAXIS>();
      EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::KAXIS>(0, j, i), dx_i * dx_j);
    }
  }
}

void VerifyCylindrical2D_EdgeLength_AllCells(
    const grid::Coordinates<Geometry::cylindrical> &coords) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(0, j, i),
                      coords.template Dx<Axis::IAXIS>());
      EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::JAXIS>(0, j, i),
                      coords.template Dx<Axis::JAXIS>());

      auto r = coords.template Xf<Axis::IAXIS>(i);
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto expected_k = std::abs(r) * dx_k;
      EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::KAXIS>(0, j, i), expected_k);
    }
  }
}

void VerifyCylindrical2D_CellVolume_AllCells(
    const grid::Coordinates<Geometry::cylindrical> &coords) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto rp = coords.template Xf<Axis::IAXIS>(i + 1);
      auto rm = coords.template Xf<Axis::IAXIS>(i);
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto dx_j = coords.template Dx<Axis::JAXIS>();
      auto expected = 0.5 * dx_k * (rp * rp - rm * rm) * dx_j;
      EXPECT_FLOAT_EQ(coords.CellVolume(0, j, i), expected);
    }
  }
}

void VerifyCylindrical2D_VolumeTopological(
    const grid::Coordinates<Geometry::cylindrical> &coords) {
  auto cell_vol = coords.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::CC, 0, 0, 0), cell_vol);
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F1, 0, 0, 0),
                  coords.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F2, 0, 0, 0),
                  coords.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::F3, 0, 0, 0),
                  coords.template FaceArea<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(coords.Volume(TopologicalElement::NN, 0, 0, 0), 1.0);
}

}  // namespace

TEST(CoordinatesTest, Cartesian3D_Construction) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  (void)wrapped;
}

TEST(CoordinatesTest, Cartesian3D_Dx) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_Dx(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_Xc) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_Xc_AllCells(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_Xc_3DIndex) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);

  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::IAXIS>(0, 0, 0),
                  wrapped.template Xc<Axis::IAXIS>(0));
  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::JAXIS>(0, 0, 0),
                  wrapped.template Xc<Axis::JAXIS>(0));
  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::KAXIS>(0, 0, 0),
                  wrapped.template Xc<Axis::KAXIS>(0));
}

TEST(CoordinatesTest, Cartesian3D_Xf) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_Xf_AllFaces(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_FaceArea) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_FaceArea(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_EdgeLength) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_EdgeLength(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_CellVolume) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_CellVolume(wrapped);
}

TEST(CoordinatesTest, Cartesian3D_Volume_TopologicalElements) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);
  VerifyCartesian3D_VolumeTopological(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_Construction) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  (void)wrapped;
}

TEST(CoordinatesTest, Cylindrical2D_Dx) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_Dx(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_Xc) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_Xc_AllCells(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_Xc_3DIndex) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);

  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::JAXIS>(0, 0, 0),
                  wrapped.template Xc<Axis::JAXIS>(0));
  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::KAXIS>(0, 0, 0),
                  wrapped.template Xc<Axis::KAXIS>(0));
}

TEST(CoordinatesTest, Cylindrical2D_FaceArea) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_FaceArea_AllCells(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_EdgeLength) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_EdgeLength_AllCells(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_CellVolume) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_CellVolume_AllCells(wrapped);
}

TEST(CoordinatesTest, Cylindrical2D_Volume_TopologicalElements) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);
  VerifyCylindrical2D_VolumeTopological(wrapped);
}

TEST(CoordinatesTest, RuntimeAxisDispatch_Cartesian) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);

  EXPECT_FLOAT_EQ(wrapped.template Dx<Axis::IAXIS>(), wrapped.Dx(Axis::IAXIS));
  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::IAXIS>(0), wrapped.Xc(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(wrapped.template Xf<Axis::IAXIS>(0), wrapped.Xf(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(wrapped.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  wrapped.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(wrapped.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  wrapped.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

TEST(CoordinatesTest, RuntimeAxisDispatch_Cylindrical) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);

  EXPECT_FLOAT_EQ(wrapped.template Dx<Axis::IAXIS>(), wrapped.Dx(Axis::IAXIS));
  EXPECT_FLOAT_EQ(wrapped.template Xc<Axis::JAXIS>(0), wrapped.Xc(Axis::JAXIS, 0));
  EXPECT_FLOAT_EQ(wrapped.template Xf<Axis::IAXIS>(0), wrapped.Xf(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(wrapped.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  wrapped.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(wrapped.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  wrapped.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

TEST(CoordinatesTest, CoordinateIndexer_Cartesian) {
  auto coords = MakeCoordinatesCartesian3D();
  auto wrapped = grid::Coordinates<Geometry::cartesian>(coords);

  auto indexer = grid::CoordinateIndexer(wrapped, 0, 0, 0);

  EXPECT_FLOAT_EQ(indexer.template Dx<Axis::IAXIS>(), wrapped.template Dx<Axis::IAXIS>());
  EXPECT_FLOAT_EQ(indexer.template Xc<Axis::IAXIS>(),
                  wrapped.template Xc<Axis::IAXIS>(0));
  EXPECT_FLOAT_EQ(indexer.template Xf<Axis::IAXIS>(),
                  wrapped.template Xf<Axis::IAXIS>(0));
  EXPECT_FLOAT_EQ(indexer.template FaceArea<Axis::IAXIS>(),
                  wrapped.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(indexer.template EdgeLength<Axis::IAXIS>(),
                  wrapped.template EdgeLength<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(indexer.CellVolume(), wrapped.CellVolume(0, 0, 0));
}

TEST(CoordinatesTest, CoordinateIndexer_Cylindrical) {
  auto coords = MakeCoordinatesCylindrical2D();
  auto wrapped = grid::Coordinates<Geometry::cylindrical>(coords);

  auto indexer = grid::CoordinateIndexer(wrapped, 0, 0, 0);

  EXPECT_FLOAT_EQ(indexer.template Dx<Axis::IAXIS>(), wrapped.template Dx<Axis::IAXIS>());
  EXPECT_FLOAT_EQ(indexer.template Xc<Axis::JAXIS>(),
                  wrapped.template Xc<Axis::JAXIS>(0));
  EXPECT_FLOAT_EQ(indexer.template Xf<Axis::JAXIS>(),
                  wrapped.template Xf<Axis::JAXIS>(0));
  EXPECT_FLOAT_EQ(indexer.template FaceArea<Axis::JAXIS>(),
                  wrapped.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(indexer.template EdgeLength<Axis::JAXIS>(),
                  wrapped.template EdgeLength<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(indexer.CellVolume(), wrapped.CellVolume(0, 0, 0));
}

}  // namespace kamayan
