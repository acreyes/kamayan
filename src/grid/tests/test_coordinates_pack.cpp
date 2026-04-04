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
    static_assert(false, "Can't make coordinates for geometry");
  }

  return MakeCoordinatesCartesian3D();
}

template <Geometry geom>
void TestCoordsPackDx(const int ndim) {
  constexpr int NDIM = 3;
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

  const Real dx = 1.0 / static_cast<Real>(NXB);
  [&]<Axis... axes>() {
    (
        [&]<Axis ax>() {
          auto [kb, jb, ib] = CoordinateIndexRanges<geom, coords::Dx<ax>>(cellbounds);
          FillCoords<geom, coords::Dx<ax>>(
              KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                pack(b, coords::Dx<ax>(), k, j, i) = coords.template Dx<ax>(k, j, i);
              },
              NBLOCKS, kb, jb, ib);
        }.template operator()<axes>(),
        ...);
  }.template operator()<Axis::KAXIS, Axis::JAXIS, Axis::IAXIS>();

  int n_wrong = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nw) {
        auto cpack = CoordinatePack<geom, Deltas>(pack, b);
        if (Kokkos::abs(cpack.template Dx<Axis::IAXIS>(k, i, i) -
                        coords.template Dx<Axis::IAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::JAXIS>(k, i, i) -
                        coords.template Dx<Axis::JAXIS>(k, j, i)) > 1e-10)
          nw += 1;
        if (Kokkos::abs(cpack.template Dx<Axis::KAXIS>(k, i, i) -
                        coords.template Dx<Axis::KAXIS>(k, j, i)) > 1e-10)
          nw += 1;
      },
      Kokkos::Sum<int>(n_wrong));

  EXPECT_EQ(n_wrong, 0);
}

TEST(CoordinatePackTest, CartesianDx) { TestCoordsPackDx<Geometry::cartesian>(3); }
TEST(CoordinatePackTest, CylindricalDx) { TestCoordsPackDx<Geometry::cylindrical>(2); }

#if 0
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

  AddCoordFields(pkg.get(), Geometry::cartesian, NXB, NXB, NXB);

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

  AddCoordFields(pkg.get(), Geometry::cartesian, NXB, NXB, NXB);

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

  AddCoordFields(pkg.get(), Geometry::cartesian, NXB, NXB, NXB);

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

  AddCoordFields(pkg.get(), Geometry::cartesian, NXB, NXB, NXB);

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

  AddCoordFields(pkg.get(), Geometry::cylindrical, NXB, NXB, NXB);

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
#endif
}  // namespace kamayan::grid
