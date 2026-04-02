#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <coordinates/uniform_cartesian.hpp>
#include <mesh/meshblock.hpp>

#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid_types.hpp"

namespace kamayan {

struct TestRegion {
  std::array<Real, 3> xmin;
  std::array<Real, 3> xmax;
  std::array<Real, 3> xrat;
  std::array<int, 3> nx;
};

parthenon::UniformCartesian MakeCoordinatesCartesian3DForPack() {
  TestRegion region{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {8, 8, 8}};
  return parthenon::UniformCartesian(
      parthenon::RegionSize(region.xmin, region.xmax, region.xrat, region.nx), nullptr);
}

parthenon::UniformCartesian MakeCoordinatesCylindrical2DForPack() {
  TestRegion region{
      {0.0, 0.0, 0.0}, {1.0, 2.0 * std::numbers::pi, 1.0}, {1.0, 1.0, 1.0}, {4, 4, 1}};
  return parthenon::UniformCartesian(
      parthenon::RegionSize(region.xmin, region.xmax, region.xrat, region.nx), nullptr);
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

template <typename CoordPack, typename CoordObj>
void VerifyDx(CoordPack &pack, CoordObj &coords) {
  static_assert(
      std::is_same_v<CoordPack, grid::CoordinatePack<Geometry::cartesian>> ||
          std::is_same_v<CoordPack, grid::CoordinatePack<Geometry::cylindrical>>,
      "CoordPack must be CoordinatePack");
  EXPECT_FLOAT_EQ(pack.template Dx<Axis::IAXIS>(0, 0, 0),
                  coords.template Dx<Axis::IAXIS>());
  EXPECT_FLOAT_EQ(pack.template Dx<Axis::JAXIS>(0, 0, 0),
                  coords.template Dx<Axis::JAXIS>());
  EXPECT_FLOAT_EQ(pack.template Dx<Axis::KAXIS>(0, 0, 0),
                  coords.template Dx<Axis::KAXIS>());
}

template <typename CoordPack, typename CoordObj>
void VerifyXc(CoordPack &pack, CoordObj &coords) {
  for (int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(pack.template Xc<Axis::IAXIS>(0, 0, i),
                    coords.template Xc<Axis::IAXIS>(i));
  }
  for (int j = 0; j < 8; j++) {
    EXPECT_FLOAT_EQ(pack.template Xc<Axis::JAXIS>(0, j, 0),
                    coords.template Xc<Axis::JAXIS>(j));
  }
  for (int k = 0; k < 8; k++) {
    EXPECT_FLOAT_EQ(pack.template Xc<Axis::KAXIS>(k, 0, 0),
                    coords.template Xc<Axis::KAXIS>(k));
  }
}

template <typename CoordPack, typename CoordObj>
void VerifyXf(CoordPack &pack, CoordObj &coords) {
  for (int i = 0; i <= 8; i++) {
    EXPECT_FLOAT_EQ(pack.template Xf<Axis::IAXIS>(0, 0, i),
                    coords.template Xf<Axis::IAXIS>(i));
  }
  for (int j = 0; j <= 8; j++) {
    EXPECT_FLOAT_EQ(pack.template Xf<Axis::JAXIS>(0, j, 0),
                    coords.template Xf<Axis::JAXIS>(j));
  }
  for (int k = 0; k <= 8; k++) {
    EXPECT_FLOAT_EQ(pack.template Xf<Axis::KAXIS>(k, 0, 0),
                    coords.template Xf<Axis::KAXIS>(k));
  }
}

template <typename CoordPack, typename CoordObj>
void VerifyFaceArea(CoordPack &pack, CoordObj &coords) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(pack.template FaceArea<Axis::IAXIS>(k, j, i),
                        coords.template FaceArea<Axis::IAXIS>(k, j, i));
        EXPECT_FLOAT_EQ(pack.template FaceArea<Axis::JAXIS>(k, j, i),
                        coords.template FaceArea<Axis::JAXIS>(k, j, i));
        EXPECT_FLOAT_EQ(pack.template FaceArea<Axis::KAXIS>(k, j, i),
                        coords.template FaceArea<Axis::KAXIS>(k, j, i));
      }
    }
  }
}

template <typename CoordPack, typename CoordObj>
void VerifyEdgeLength(CoordPack &pack, CoordObj &coords) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(pack.template EdgeLength<Axis::IAXIS>(k, j, i),
                        coords.template EdgeLength<Axis::IAXIS>(k, j, i));
        EXPECT_FLOAT_EQ(pack.template EdgeLength<Axis::JAXIS>(k, j, i),
                        coords.template EdgeLength<Axis::JAXIS>(k, j, i));
        EXPECT_FLOAT_EQ(pack.template EdgeLength<Axis::KAXIS>(k, j, i),
                        coords.template EdgeLength<Axis::KAXIS>(k, j, i));
      }
    }
  }
}

template <typename CoordPack, typename CoordObj>
void VerifyCellVolume(CoordPack &pack, CoordObj &coords) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(pack.CellVolume(k, j, i), coords.CellVolume(k, j, i));
      }
    }
  }
}

template <typename CoordPack, typename CoordObj>
void VerifyVolumeTopological(CoordPack &pack, CoordObj &coords) {
  auto cell_vol = pack.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(pack.Volume(TopologicalElement::CC, 0, 0, 0), cell_vol);
  EXPECT_FLOAT_EQ(pack.Volume(TopologicalElement::F1, 0, 0, 0),
                  pack.template FaceArea<Axis::IAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(pack.Volume(TopologicalElement::F2, 0, 0, 0),
                  pack.template FaceArea<Axis::JAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(pack.Volume(TopologicalElement::F3, 0, 0, 0),
                  pack.template FaceArea<Axis::KAXIS>(0, 0, 0));
  EXPECT_FLOAT_EQ(pack.Volume(TopologicalElement::NN, 0, 0, 0), 1.0);
}

template <typename CoordPack, typename CoordObj>
void VerifyRuntimeAxisDispatch(CoordPack &pack, CoordObj &coords) {
  EXPECT_FLOAT_EQ(pack.template Dx<Axis::IAXIS>(0, 0, 0), pack.Dx(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(pack.template Xc<Axis::IAXIS>(0, 0, 0), pack.Xc(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(pack.template Xf<Axis::IAXIS>(0, 0, 0), pack.Xf(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(pack.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  pack.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(pack.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  pack.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

}  // namespace

TEST(CoordinatePackTest, Cartesian3D_PackConstruction) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);
  (void)coords;
}

TEST(CoordinatePackTest, Cartesian3D_Dx_Template) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

  EXPECT_FLOAT_EQ(coords.template Dx<Axis::IAXIS>(), 0.125);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::JAXIS>(), 0.125);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::KAXIS>(), 0.125);
}

TEST(CoordinatePackTest, Cartesian3D_Xc_Template) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

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

TEST(CoordinatePackTest, Cartesian3D_Xf_Template) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

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

TEST(CoordinatePackTest, Cartesian3D_FaceArea_Template) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

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

TEST(CoordinatePackTest, Cartesian3D_EdgeLength_Template) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

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

TEST(CoordinatePackTest, Cartesian3D_CellVolume) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

  EXPECT_FLOAT_EQ(coords.CellVolume(0, 0, 0), 0.125 * 0.125 * 0.125);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        EXPECT_FLOAT_EQ(coords.CellVolume(k, j, i), 0.125 * 0.125 * 0.125);
      }
    }
  }
}

TEST(CoordinatePackTest, Cartesian3D_Volume_Topological) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

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

TEST(CoordinatePackTest, Cartesian3D_RuntimeAxisDispatch) {
  auto raw_coords = MakeCoordinatesCartesian3DForPack();
  auto coords = grid::Coordinates<Geometry::cartesian>(raw_coords);

  EXPECT_FLOAT_EQ(coords.template Dx<Axis::IAXIS>(), coords.Dx(Axis::IAXIS));
  EXPECT_FLOAT_EQ(coords.template Xc<Axis::IAXIS>(0), coords.Xc(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(coords.template Xf<Axis::IAXIS>(0), coords.Xf(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  coords.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  coords.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

TEST(CoordinatePackTest, Cylindrical2D_PackConstruction) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);
  (void)coords;
}

TEST(CoordinatePackTest, Cylindrical2D_Dx_Template) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  EXPECT_FLOAT_EQ(coords.template Dx<Axis::IAXIS>(), 0.25);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::JAXIS>(), std::numbers::pi / 2.0);
  EXPECT_FLOAT_EQ(coords.template Dx<Axis::KAXIS>(), 2.0 * std::numbers::pi);
}

TEST(CoordinatePackTest, Cylindrical2D_Xc_Template) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  for (int i = 0; i < 4; i++) {
    auto r0 = coords.template Xf<Axis::IAXIS>(i);
    auto r1 = coords.template Xf<Axis::IAXIS>(i + 1);
    auto expected_xc = (2.0 / 3.0) * (r1 * r1 * r1 - r0 * r0 * r0) / (r1 * r1 - r0 * r0);
    auto actual_xc = coords.template Xc<Axis::IAXIS>(i);
    ASSERT_TRUE(IsRelClose(actual_xc, expected_xc));
  }

  for (int j = 0; j < 4; j++) {
    auto expected_xc_j = std::numbers::pi / 4.0 + j * (std::numbers::pi / 2.0);
    auto actual_xc_j = coords.template Xc<Axis::JAXIS>(j);
    ASSERT_TRUE(IsRelClose(actual_xc_j, expected_xc_j));
  }

  ASSERT_TRUE(IsRelClose(coords.template Xc<Axis::KAXIS>(0), 0.5));
}

TEST(CoordinatePackTest, Cylindrical2D_FaceArea_Template) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto r = coords.template Xf<Axis::IAXIS>(i);
      auto dx_j = coords.template Dx<Axis::JAXIS>();
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto expected_i = r * dx_j * dx_k;
      auto actual_i = coords.template FaceArea<Axis::IAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_i, expected_i));

      auto rp = coords.template Xf<Axis::IAXIS>(i + 1);
      auto rm = coords.template Xf<Axis::IAXIS>(i);
      auto expected_j = 0.5 * (rp * rp - rm * rm) * dx_k;
      auto actual_j = coords.template FaceArea<Axis::JAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_j, expected_j));

      auto dx_i = coords.template Dx<Axis::IAXIS>();
      auto expected_k = dx_i * dx_j;
      auto actual_k = coords.template FaceArea<Axis::KAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_k, expected_k));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_EdgeLength_Template) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto dx_i = coords.template Dx<Axis::IAXIS>();
      auto el_i = coords.template EdgeLength<Axis::IAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(el_i, dx_i));

      auto dx_j = coords.template Dx<Axis::JAXIS>();
      auto el_j = coords.template EdgeLength<Axis::JAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(el_j, dx_j));

      auto r = coords.template Xf<Axis::IAXIS>(i);
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto expected_k = std::abs(r) * dx_k;
      auto actual_k = coords.template EdgeLength<Axis::KAXIS>(0, j, i);
      ASSERT_TRUE(IsRelClose(actual_k, expected_k));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_CellVolume) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto rp = coords.template Xf<Axis::IAXIS>(i + 1);
      auto rm = coords.template Xf<Axis::IAXIS>(i);
      auto dx_k = coords.template Dx<Axis::KAXIS>();
      auto dx_j = coords.template Dx<Axis::JAXIS>();
      auto expected = 0.5 * dx_k * (rp * rp - rm * rm) * dx_j;
      auto actual = coords.CellVolume(0, j, i);
      ASSERT_TRUE(IsRelClose(actual, expected));
    }
  }
}

TEST(CoordinatePackTest, Cylindrical2D_Volume_Topological) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

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

TEST(CoordinatePackTest, Cylindrical2D_RuntimeAxisDispatch) {
  auto raw_coords = MakeCoordinatesCylindrical2DForPack();
  auto coords = grid::Coordinates<Geometry::cylindrical>(raw_coords);

  EXPECT_FLOAT_EQ(coords.template Dx<Axis::IAXIS>(), coords.Dx(Axis::IAXIS));
  EXPECT_FLOAT_EQ(coords.template Xc<Axis::JAXIS>(0), coords.Xc(Axis::JAXIS, 0));
  EXPECT_FLOAT_EQ(coords.template Xf<Axis::IAXIS>(0), coords.Xf(Axis::IAXIS, 0));
  EXPECT_FLOAT_EQ(coords.template FaceArea<Axis::IAXIS>(0, 0, 0),
                  coords.FaceArea(Axis::IAXIS, 0, 0, 0));
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(0, 0, 0),
                  coords.EdgeLength(Axis::IAXIS, 0, 0, 0));
}

}  // namespace kamayan
