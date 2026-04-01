#include "test_geometry.hpp"

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <mesh/meshblock.hpp>

#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {

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

  auto pmb = std::make_shared<parthenon::MeshBlock>(8, 2);
  auto &pmbd = pmb->meshblock_data.Get();
  pmbd->Initialize(pkg, pmb);
  return pmb;
}

namespace {

constexpr Real REL_TOL = 1e-12;
constexpr Real ABS_TOL = 1e-14;

bool IsRelClose(Real a, Real b) {
  if (std::abs(b) < ABS_TOL) {
    return std::abs(a - b) < ABS_TOL;
  }
  return std::abs((a - b) / b) < REL_TOL;
}

}  // namespace

TEST(CoordinatesTest, Cartesian3D_Construction) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);
  (void)coords;
}

TEST(CoordinatesTest, Cartesian3D_Dx) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();

  EXPECT_FLOAT_EQ(dx_i, dx_j);
  EXPECT_FLOAT_EQ(dx_j, dx_k);
}

TEST(CoordinatesTest, Cartesian3D_Xc) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto xc_i = coords.template Xc<Axis::IAXIS>(0);
  auto xc_j = coords.template Xc<Axis::JAXIS>(0);
  auto xc_k = coords.template Xc<Axis::KAXIS>(0);

  EXPECT_GT(xc_i, 0.0);
  EXPECT_GT(xc_j, 0.0);
  EXPECT_GT(xc_k, 0.0);
}

TEST(CoordinatesTest, Cartesian3D_Xc_3DIndex) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto xc_i = coords.template Xc<Axis::IAXIS>(0, 0, 0);
  auto xc_j = coords.template Xc<Axis::JAXIS>(0, 0, 0);
  auto xc_k = coords.template Xc<Axis::KAXIS>(0, 0, 0);

  auto xc_i_single = coords.template Xc<Axis::IAXIS>(0);
  auto xc_j_single = coords.template Xc<Axis::JAXIS>(0);
  auto xc_k_single = coords.template Xc<Axis::KAXIS>(0);

  EXPECT_FLOAT_EQ(xc_i, xc_i_single);
  EXPECT_FLOAT_EQ(xc_j, xc_j_single);
  EXPECT_FLOAT_EQ(xc_k, xc_k_single);
}

TEST(CoordinatesTest, Cartesian3D_Xf) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto xf_i = coords.template Xf<Axis::IAXIS>(0);
  auto xf_j = coords.template Xf<Axis::JAXIS>(0);
  auto xf_k = coords.template Xf<Axis::KAXIS>(0);

  EXPECT_GE(xf_i, 0.0);
  EXPECT_GE(xf_j, 0.0);
  EXPECT_GE(xf_k, 0.0);

  auto xf_i_idx = coords.template Xf<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(xf_i, xf_i_idx);
}

TEST(CoordinatesTest, Cartesian3D_Xf_TopologicalElement) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto xf_cc = coords.template Xf<TopologicalElement::CC, Axis::IAXIS>(0);
  auto xc = coords.template Xc<Axis::IAXIS>(0);
  EXPECT_FLOAT_EQ(xf_cc, xc);

  auto xf_f1 = coords.template Xf<TopologicalElement::F1, Axis::IAXIS>(0);
  auto xf_face = coords.template Xf<Axis::IAXIS>(0);
  EXPECT_FLOAT_EQ(xf_f1, xf_face);
}

TEST(CoordinatesTest, Cartesian3D_FaceArea) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  auto expected_area = dx_j * dx_k;

  auto face_area_i = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(face_area_i, expected_area);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_k2 = coords.template Dx<Axis::KAXIS>();
  auto face_area_j = coords.template FaceArea<Axis::JAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(face_area_j, dx_i * dx_k2);

  auto dx_i2 = coords.template Dx<Axis::IAXIS>();
  auto dx_j2 = coords.template Dx<Axis::JAXIS>();
  auto face_area_k = coords.template FaceArea<Axis::KAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(face_area_k, dx_i2 * dx_j2);
}

TEST(CoordinatesTest, Cartesian3D_EdgeLength) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();

  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::IAXIS>(0, 0, 0), dx_i);
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::JAXIS>(0, 0, 0), dx_j);
  EXPECT_FLOAT_EQ(coords.template EdgeLength<Axis::KAXIS>(0, 0, 0), dx_k);
}

TEST(CoordinatesTest, Cartesian3D_CellVolume) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  auto expected_volume = dx_i * dx_j * dx_k;

  auto volume = coords.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(volume, expected_volume);
}

TEST(CoordinatesTest, Cartesian3D_Volume_TopologicalElements) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto volume_cc = coords.Volume(TopologicalElement::CC, 0, 0, 0);
  auto cell_vol = coords.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_cc, cell_vol);

  auto volume_f1 = coords.Volume(TopologicalElement::F1, 0, 0, 0);
  auto face_area_i = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_f1, face_area_i);

  auto volume_f2 = coords.Volume(TopologicalElement::F2, 0, 0, 0);
  auto face_area_j = coords.template FaceArea<Axis::JAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_f2, face_area_j);

  auto volume_f3 = coords.Volume(TopologicalElement::F3, 0, 0, 0);
  auto face_area_k = coords.template FaceArea<Axis::KAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_f3, face_area_k);

  auto volume_e1 = coords.Volume(TopologicalElement::E1, 0, 0, 0);
  auto edge_i = coords.template EdgeLength<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_e1, edge_i);

  auto volume_e2 = coords.Volume(TopologicalElement::E2, 0, 0, 0);
  auto edge_j = coords.template EdgeLength<Axis::JAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_e2, edge_j);

  auto volume_e3 = coords.Volume(TopologicalElement::E3, 0, 0, 0);
  auto edge_k = coords.template EdgeLength<Axis::KAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(volume_e3, edge_k);

  auto volume_nn = coords.Volume(TopologicalElement::NN, 0, 0, 0);
  EXPECT_FLOAT_EQ(volume_nn, 1.0);
}

TEST(CoordinatesTest, Cylindrical2D_Construction) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);
  (void)coords;
}

TEST(CoordinatesTest, Cylindrical2D_Dx) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();

  auto expected_dx_k = 2.0 * std::numbers::pi;
  EXPECT_FLOAT_EQ(dx_k, expected_dx_k);

  EXPECT_GT(dx_k, dx_i);
  EXPECT_GT(dx_k, dx_j);
}

TEST(CoordinatesTest, Cylindrical2D_Xc) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto xc_k = coords.template Xc<Axis::KAXIS>(0);
  auto xc_j = coords.template Xc<Axis::JAXIS>(0);
  EXPECT_GT(xc_k, 0.0);
  EXPECT_GT(xc_j, 0.0);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  EXPECT_GT(dx_i, 0.0);
}

TEST(CoordinatesTest, Cylindrical2D_Xc_3DIndex) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto xc_j = coords.template Xc<Axis::JAXIS>(0, 0, 0);
  auto xc_j_single = coords.template Xc<Axis::JAXIS>(0);
  EXPECT_FLOAT_EQ(xc_j, xc_j_single);

  auto xc_k = coords.template Xc<Axis::KAXIS>(0, 0, 0);
  auto xc_k_single = coords.template Xc<Axis::KAXIS>(0);
  EXPECT_FLOAT_EQ(xc_k, xc_k_single);
}

TEST(CoordinatesTest, Cylindrical2D_FaceArea) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto r = coords.template Xf<Axis::IAXIS>(0);
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  auto expected_area_i = r * dx_j * dx_k;

  auto face_area_i = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(face_area_i, expected_area_i));

  auto rp = coords.template Xf<Axis::IAXIS>(1);
  auto rm = coords.template Xf<Axis::IAXIS>(0);
  auto expected_area_j = 0.5 * (rp * rp - rm * rm) * dx_k;

  auto face_area_j = coords.template FaceArea<Axis::JAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(face_area_j, expected_area_j));
}

TEST(CoordinatesTest, Cylindrical2D_EdgeLength) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto dx_i = coords.template Dx<Axis::IAXIS>();
  auto edge_i = coords.template EdgeLength<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(edge_i, dx_i);

  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto edge_j = coords.template EdgeLength<Axis::JAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(edge_j, dx_j);

  auto r = coords.template Xf<Axis::IAXIS>(0);
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  auto expected_edge_k = std::abs(r) * dx_k;
  auto edge_k = coords.template EdgeLength<Axis::KAXIS>(0, 0, 0);

  EXPECT_TRUE(IsRelClose(edge_k, expected_edge_k));
}

TEST(CoordinatesTest, Cylindrical2D_CellVolume) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto rp = coords.template Xf<Axis::IAXIS>(1);
  auto rm = coords.template Xf<Axis::IAXIS>(0);
  auto dx_k = coords.template Dx<Axis::KAXIS>();
  auto dx_j = coords.template Dx<Axis::JAXIS>();
  auto expected_volume = 0.5 * dx_k * (rp * rp - rm * rm) * dx_j;

  auto volume = coords.CellVolume(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume, expected_volume));
}

TEST(CoordinatesTest, Cylindrical2D_Volume_TopologicalElements) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto volume_cc = coords.Volume(TopologicalElement::CC, 0, 0, 0);
  auto cell_vol = coords.CellVolume(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume_cc, cell_vol));

  auto volume_f1 = coords.Volume(TopologicalElement::F1, 0, 0, 0);
  auto face_area_i = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume_f1, face_area_i));

  auto volume_f2 = coords.Volume(TopologicalElement::F2, 0, 0, 0);
  auto face_area_j = coords.template FaceArea<Axis::JAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume_f2, face_area_j));

  auto volume_f3 = coords.Volume(TopologicalElement::F3, 0, 0, 0);
  auto face_area_k = coords.template FaceArea<Axis::KAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume_f3, face_area_k));

  auto volume_nn = coords.Volume(TopologicalElement::NN, 0, 0, 0);
  EXPECT_FLOAT_EQ(volume_nn, 1.0);
}

TEST(CoordinatesTest, RuntimeAxisDispatch_Cartesian) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto dx_i_template = coords.template Dx<Axis::IAXIS>();
  auto dx_i_runtime = coords.Dx(Axis::IAXIS);
  EXPECT_FLOAT_EQ(dx_i_template, dx_i_runtime);

  auto xc_i_template = coords.template Xc<Axis::IAXIS>(0);
  auto xc_i_runtime = coords.Xc(Axis::IAXIS, 0);
  EXPECT_FLOAT_EQ(xc_i_template, xc_i_runtime);

  auto xf_i_template = coords.template Xf<Axis::IAXIS>(0);
  auto xf_i_runtime = coords.Xf(Axis::IAXIS, 0);
  EXPECT_FLOAT_EQ(xf_i_template, xf_i_runtime);

  auto face_area_template = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  auto face_area_runtime = coords.FaceArea(Axis::IAXIS, 0, 0, 0);
  EXPECT_FLOAT_EQ(face_area_template, face_area_runtime);

  auto edge_length_template = coords.template EdgeLength<Axis::IAXIS>(0, 0, 0);
  auto edge_length_runtime = coords.EdgeLength(Axis::IAXIS, 0, 0, 0);
  EXPECT_FLOAT_EQ(edge_length_template, edge_length_runtime);
}

TEST(CoordinatesTest, RuntimeAxisDispatch_Cylindrical) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto dx_i_template = coords.template Dx<Axis::IAXIS>();
  auto dx_i_runtime = coords.Dx(Axis::IAXIS);
  EXPECT_FLOAT_EQ(dx_i_template, dx_i_runtime);

  auto xc_j_template = coords.template Xc<Axis::JAXIS>(0);
  auto xc_j_runtime = coords.Xc(Axis::JAXIS, 0);
  EXPECT_FLOAT_EQ(xc_j_template, xc_j_runtime);

  auto xf_i_template = coords.template Xf<Axis::IAXIS>(0);
  auto xf_i_runtime = coords.Xf(Axis::IAXIS, 0);
  EXPECT_FLOAT_EQ(xf_i_template, xf_i_runtime);

  auto face_area_template = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  auto face_area_runtime = coords.FaceArea(Axis::IAXIS, 0, 0, 0);
  EXPECT_TRUE(IsRelClose(face_area_template, face_area_runtime));

  auto edge_length_template = coords.template EdgeLength<Axis::IAXIS>(0, 0, 0);
  auto edge_length_runtime = coords.EdgeLength(Axis::IAXIS, 0, 0, 0);
  EXPECT_FLOAT_EQ(edge_length_template, edge_length_runtime);
}

TEST(CoordinatesTest, CoordinateIndexer_Cartesian) {
  auto pmb = MakeTestMeshBlockCartesian3D();
  auto coords = grid::Coordinates<Geometry::cartesian>(pmb->coords);

  auto indexer = grid::CoordinateIndexer(coords, 0, 0, 0);

  auto dx = indexer.template Dx<Axis::IAXIS>();
  auto expected_dx = coords.template Dx<Axis::IAXIS>();
  EXPECT_FLOAT_EQ(dx, expected_dx);

  auto xc = indexer.template Xc<Axis::IAXIS>();
  auto expected_xc = coords.template Xc<Axis::IAXIS>(0);
  EXPECT_FLOAT_EQ(xc, expected_xc);

  auto xf = indexer.template Xf<Axis::IAXIS>();
  auto expected_xf = coords.template Xf<Axis::IAXIS>(0);
  EXPECT_FLOAT_EQ(xf, expected_xf);

  auto face_area = indexer.template FaceArea<Axis::IAXIS>();
  auto expected_face_area = coords.template FaceArea<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(face_area, expected_face_area);

  auto edge_length = indexer.template EdgeLength<Axis::IAXIS>();
  auto expected_edge_length = coords.template EdgeLength<Axis::IAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(edge_length, expected_edge_length);

  auto volume = indexer.CellVolume();
  auto expected_volume = coords.CellVolume(0, 0, 0);
  EXPECT_FLOAT_EQ(volume, expected_volume);
}

TEST(CoordinatesTest, CoordinateIndexer_Cylindrical) {
  auto pmb = MakeTestMeshBlockCylindrical2D();
  auto coords = grid::Coordinates<Geometry::cylindrical>(pmb->coords);

  auto indexer = grid::CoordinateIndexer(coords, 0, 0, 0);

  auto dx = indexer.template Dx<Axis::IAXIS>();
  auto expected_dx = coords.template Dx<Axis::IAXIS>();
  EXPECT_FLOAT_EQ(dx, expected_dx);

  auto xc = indexer.template Xc<Axis::JAXIS>();
  auto expected_xc = coords.template Xc<Axis::JAXIS>(0);
  EXPECT_FLOAT_EQ(xc, expected_xc);

  auto xf = indexer.template Xf<Axis::JAXIS>();
  auto expected_xf = coords.template Xf<Axis::JAXIS>(0);
  EXPECT_FLOAT_EQ(xf, expected_xf);

  auto face_area = indexer.template FaceArea<Axis::JAXIS>();
  auto expected_face_area = coords.template FaceArea<Axis::JAXIS>(0, 0, 0);
  EXPECT_TRUE(IsRelClose(face_area, expected_face_area));

  auto edge_length = indexer.template EdgeLength<Axis::JAXIS>();
  auto expected_edge_length = coords.template EdgeLength<Axis::JAXIS>(0, 0, 0);
  EXPECT_FLOAT_EQ(edge_length, expected_edge_length);

  auto volume = indexer.CellVolume();
  auto expected_volume = coords.CellVolume(0, 0, 0);
  EXPECT_TRUE(IsRelClose(volume, expected_volume));

  auto vol_top = indexer.Volume(TopologicalElement::CC);
  EXPECT_TRUE(IsRelClose(vol_top, expected_volume));
}

}  // namespace kamayan