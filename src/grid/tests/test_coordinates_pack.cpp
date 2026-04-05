#include <format>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <mesh/meshblock.hpp>

#include "grid/coordinates.hpp"
#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/tests/test_geometry.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/unit.hpp"
#include "kamayan_utils/type_abstractions.hpp"

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

void AddCoordFields(KamayanUnit *pkg, Geometry geom, const int nx3, const int nx2,
                    const int nx1) {
  using parthenon::Metadata;
  GeometryOptions::dispatch(
      [&]<Geometry g>() {
        const int nghost = 0;
        auto md = Metadata({Metadata::Cell, Metadata::OneCopy},
                           CoordinateShape<g, coords::Volume>(nx3, nx2, nx1, nghost));
        pkg->AddField<coords::Volume>(md);
        std::vector<MetadataFlag> axis_md{Metadata::None, Metadata::OneCopy};
        [&]<Axis... axes>() {
          (pkg->AddField<coords::Dx<axes>>(Metadata(
               axis_md, CoordinateShape<g, coords::Dx<axes>>(nx3, nx2, nx1, nghost))),
           ...);
          (pkg->AddField<coords::X<axes>>(Metadata(
               axis_md, CoordinateShape<g, coords::X<axes>>(nx3, nx2, nx1, nghost))),
           ...);
          (pkg->AddField<coords::Xc<axes>>(Metadata(
               axis_md, CoordinateShape<g, coords::Xc<axes>>(nx3, nx2, nx1, nghost))),
           ...);
          (pkg->AddField<coords::Xf<axes>>(Metadata(
               axis_md, CoordinateShape<g, coords::Xf<axes>>(nx3, nx2, nx1, nghost))),
           ...);
          (pkg->AddField<coords::FaceArea<axes>>(
               Metadata(axis_md, CoordinateShape<g, coords::FaceArea<axes>>(nx3, nx2, nx1,
                                                                            nghost))),
           ...);
          (pkg->AddField<coords::EdgeLength<axes>>(Metadata(
               axis_md,
               CoordinateShape<g, coords::EdgeLength<axes>>(nx3, nx2, nx1, nghost))),
           ...);
        }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();
      },
      geom);
}

template <Geometry geom, typename T, typename Functor>
void FillCoords(const Functor &functor, const int nblocks, const parthenon::IndexRange jb,
                const parthenon::IndexRange ib, const parthenon::IndexRange kb) {
  par_for(PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          functor);
}

template <Geometry geom>
auto MakeCoords() {
  if constexpr (geom == Geometry::cylindrical) {
    return MakeCoordinatesCylindrical2D();
  } else if constexpr (geom == Geometry::cartesian) {
    return MakeCoordinatesCartesian3D();
  } else {
    static_assert(always_false<Coordinates<geom>>, "Can't make coordinates for geometry");
  }

  return MakeCoordinatesCartesian3D();
}

template <Geometry geom>
void TestCoordsPackDx(const int ndim) {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = (geom == Geometry::cylindrical) ? 4 : 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] = CoordinateIndexRanges<geom, coords::Dx<ax>>(cellbounds);
          FillCoords<geom, coords::Dx<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::Dx<ax>(), k, j, i) = coords.template Dx<ax>(k, j, i);
              },
              NBLOCKS, jb, ib, kb);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, Deltas>(pack, b);
        if (Kokkos::abs(cpack.template Dx<Axis::IAXIS>(k, j, i) -
                        coords.template Dx<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::JAXIS>(k, j, i) -
                        coords.template Dx<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::KAXIS>(k, j, i) -
                        coords.template Dx<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

template <Geometry geom>
void TestCoordsPackVolume() {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  auto [kb_vol, jb_vol, ib_vol] = CoordinateIndexRanges<geom, coords::Volume>(cellbounds);
  FillCoords<geom, coords::Volume>(
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, coords::Volume(), k, j, i) = coords.CellVolume(k, j, i);
      },
      NBLOCKS, jb_vol, ib_vol, kb_vol);

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom>(pack, b);
        if (Kokkos::abs(cpack.CellVolume(k, j, i) - coords.CellVolume(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

template <Geometry geom>
void TestCoordsPackXc(const int ndim) {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] = CoordinateIndexRanges<geom, coords::Xc<ax>>(cellbounds);
          FillCoords<geom, coords::Xc<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::Xc<ax>(), k, j, i) = coords.template Xc<ax>(k, j, i);
              },
              NBLOCKS, jb, ib, kb);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, Xcoord>(pack, b);
        if (Kokkos::abs(cpack.template Xc<Axis::IAXIS>(k, j, i) -
                        coords.template Xc<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xc<Axis::JAXIS>(k, j, i) -
                        coords.template Xc<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xc<Axis::KAXIS>(k, j, i) -
                        coords.template Xc<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

template <Geometry geom>
void TestCoordsPackFaceArea(const int ndim) {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = (geom == Geometry::cylindrical) ? 4 : 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] =
              CoordinateIndexRanges<geom, coords::FaceArea<ax>>(cellbounds);
          FillCoords<geom, coords::FaceArea<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::FaceArea<ax>(), k, j, i) =
                    coords.template FaceArea<ax>(k, j, i);
              },
              NBLOCKS, jb, ib, kb);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, FaceAreas>(pack, b);
        if (Kokkos::abs(cpack.template FaceArea<Axis::IAXIS>(k, j, i) -
                        coords.template FaceArea<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template FaceArea<Axis::JAXIS>(k, j, i) -
                        coords.template FaceArea<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template FaceArea<Axis::KAXIS>(k, j, i) -
                        coords.template FaceArea<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

template <Geometry geom>
void TestCoordsPackEdgeLength(const int ndim) {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = (geom == Geometry::cylindrical) ? 4 : 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] =
              CoordinateIndexRanges<geom, coords::EdgeLength<ax>>(cellbounds);
          FillCoords<geom, coords::EdgeLength<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::EdgeLength<ax>(), k, j, i) =
                    coords.template EdgeLength<ax>(k, j, i);
              },
              NBLOCKS, jb, ib, kb);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, AllAxes<coords::EdgeLength>>(pack, b);
        if (Kokkos::abs(cpack.template EdgeLength<Axis::IAXIS>(k, j, i) -
                        coords.template EdgeLength<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template EdgeLength<Axis::JAXIS>(k, j, i) -
                        coords.template EdgeLength<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template EdgeLength<Axis::KAXIS>(k, j, i) -
                        coords.template EdgeLength<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

template <Geometry geom>
void TestCoordsPackXf(const int ndim) {
  constexpr int NDIM = (geom == Geometry::cylindrical) ? 2 : 3;
  constexpr int NXB = 8;
  constexpr int NBLOCKS = 1;

  auto pkg = std::make_shared<KamayanUnit>("Test Package");
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>();
  auto cfg = std::make_shared<Config>();
  cfg->Add(geom);
  pkg->InitResources(rps, cfg);

  AddCoordFields(pkg.get(), geom, NXB, NXB, NXB);

  auto block_list = MakeTestBlockList(pkg, NBLOCKS, NXB, NDIM);
  auto md = MakeTestMeshData(block_list);

  auto pack = GetPack(CoordFields(), pkg.get(), &md);

  auto ib = md.GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md.GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md.GetBoundsK(parthenon::IndexDomain::interior);

  auto cellbounds = parthenon::IndexShape(NXB, NXB, NXB, 0);
  auto coords = Coordinates<geom>(MakeCoords<geom>());

  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] = CoordinateIndexRanges<geom, coords::Xf<ax>>(cellbounds);
          FillCoords<geom, coords::Xf<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::Xf<ax>(), k, j, i) = coords.template Xf<ax>(k, j, i);
              },
              NBLOCKS, jb, ib, kb);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, Xface>(pack, b);
        if (Kokkos::abs(cpack.template Xf<Axis::IAXIS>(k, j, i) -
                        coords.template Xf<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xf<Axis::JAXIS>(k, j, i) -
                        coords.template Xf<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Xf<Axis::KAXIS>(k, j, i) -
                        coords.template Xf<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianDx) { TestCoordsPackDx<Geometry::cartesian>(3); }
TEST(CoordinatePackTest, CylindricalDx) { TestCoordsPackDx<Geometry::cylindrical>(2); }

TEST(CoordinatePackTest, CartesianXc) { TestCoordsPackXc<Geometry::cartesian>(3); }
TEST(CoordinatePackTest, CylindricalXc) { TestCoordsPackXc<Geometry::cylindrical>(2); }

TEST(CoordinatePackTest, CartesianXf) { TestCoordsPackXf<Geometry::cartesian>(3); }
TEST(CoordinatePackTest, CylindricalXf) { TestCoordsPackXf<Geometry::cylindrical>(2); }

TEST(CoordinatePackTest, CartesianVolume) { TestCoordsPackVolume<Geometry::cartesian>(); }
TEST(CoordinatePackTest, CylindricalVolume) {
  TestCoordsPackVolume<Geometry::cylindrical>();
}

TEST(CoordinatePackTest, CartesianFaceArea) {
  TestCoordsPackFaceArea<Geometry::cartesian>(3);
}
TEST(CoordinatePackTest, CylindricalFaceArea) {
  TestCoordsPackFaceArea<Geometry::cylindrical>(2);
}

TEST(CoordinatePackTest, CartesianEdgeLength) {
  TestCoordsPackEdgeLength<Geometry::cartesian>(3);
}
TEST(CoordinatePackTest, CylindricalEdgeLength) {
  TestCoordsPackEdgeLength<Geometry::cylindrical>(2);
}

}  // namespace kamayan::grid
