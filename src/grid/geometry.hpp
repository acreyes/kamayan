#ifndef GRID_GEOMETRY_HPP_
#define GRID_GEOMETRY_HPP_
#include <numbers>
#include <utility>

#include <Kokkos_Macros.hpp>

#include <coordinates/coordinates.hpp>
#include <pack/sparse_pack.hpp>
#include <ports-of-call/variant.hpp>

#include "dispatcher/options.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid_types.hpp"

namespace kamayan::grid {

// This is a (incomplete) wrapper around parthenon's uniform_coordinate (cartesian)
// to calculate coordinate related items in a different geometry.
template <Geometry geom>
struct Coordinates {
  KOKKOS_INLINE_FUNCTION Coordinates(parthenon::Coordinates_t coords) : coords_(coords) {}

  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION Coordinates(const parthenon::SparsePack<Ts...> &pack,
                                     const int block)
      : coords_(pack.GetCoordinates(block)) {}

  template <Axis ax, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real X(const int idx) const {
    constexpr int dir = AxisToInt(ax);
    if constexpr (ax == Axis::IAXIS && TopologicalOffsetI(el)) {
      return coords_.X<1, el>(idx);
    } else if constexpr (ax == Axis::JAXIS && TopologicalOffsetJ(el)) {
      return coords_.X<2, el>(idx);
    } else if constexpr (ax == Axis::JAXIS && TopologicalOffsetK(el)) {
      return coords_.X<3, el>(idx);
    } else {
      return Xc<ax>(idx);
    }
    return 0;  // This should never be reached, but w/o it some compilers generate
               // warnings
  }

  // distance between cell centers
  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Dx() const {
    if constexpr (geom == Geometry::cylindrical && ax == Axis::KAXIS) {
      // I guess we're assuming this only is a useful geometry in NDIM < 3
      return 2.0 * std::numbers::pi;
    }
    return coords_.Dx<AxisToInt(ax)>();
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Dx(const Axis &ax) const {
    return AxisOverload([this]<Axis AX>() { return Dx<AX>(); }, ax);
  }

  // position at cell centroids
  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    constexpr auto dir = AxisToInt(ax);
    if constexpr (ax == Axis::IAXIS && geom == Geometry::cylindrical) {
      const Real r0 = coords_.Xf<dir>(idx);
      const Real r0sq = r0 * r0;
      const Real r1 = coords_.Xf<dir>(idx + 1);
      const Real r1sq = r1 * r1;
      return (2.0 / 3.0) * (r1sq * std::abs(r1) - r0sq * std::abs(r0)) / (r1sq - r0sq);
    }
    return coords_.Xc<dir>(idx);
  }

  // position at cell centers
  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xi(const int idx) const {
    return coords_.Xc<AxisToInt(ax)>(idx);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xi(const int k, const int j, const int i) const {
    int kji[]{k, j, i};
    return coords_.Xc<AxisToInt(ax)>(kji[static_cast<int>(ax)]);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    int kji[]{k, j, i};
    return Xc<ax>(kji[static_cast<int>(ax)]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const Axis &ax, const int idx) const {
    return AxisOverload([&, this]<Axis AX>() { return Xc<AX>(idx); }, ax);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const Axis &ax, const int k, const int j,
                                      const int i) const {
    return AxisOverload([&, this]<Axis AX>() { return Xc<AX>(k, j, i); }, ax);
  }

  // position at face centers
  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    return coords_.Xf<AxisToInt(ax)>(idx);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const Axis ax, const int idx) const {
    return AxisOverload([&, this]<Axis AX>() { return Xf<AX>(idx); }, ax);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    int kji[]{k, j, i};
    return Xf<ax>(kji[static_cast<int>(ax)]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const Axis &ax, const int k, const int j,
                                      const int i) const {
    return AxisOverload([&, this]<Axis AX>() { return Xf<AX>(k, j, i); }, ax);
  }

  // coordinate ax at center of element el
  template <TopologicalElement el, Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    if constexpr (static_cast<int>(el) % 3 + 1 == AxisToInt(ax)) {
      return Xf<ax>(idx);
    } else {
      return Xi<ax>(idx);
    }
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    constexpr auto dir = AxisToInt(ax);
    if constexpr (geom == Geometry::cylindrical) {
      if constexpr (ax == Axis::IAXIS) {
        return Xf<Axis::IAXIS>(i) * Dx<Axis::JAXIS>() * Dx<Axis::KAXIS>();
      } else if constexpr (ax == Axis::JAXIS) {
        const Real rp = Xf<Axis::IAXIS>(i + 1);
        const Real rm = Xf<Axis::IAXIS>(i);
        return 0.5 * (rp * rp - rm * rm) * Dx<Axis::KAXIS>();
      }
    }
    return coords_.FaceArea<dir>(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const Axis &ax, const int k, const int j,
                                            const int i) const {
    return AxisOverload([&, this]<Axis AX>() { return FaceArea<AX>(k, j, i); }, ax);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    if constexpr (geom == Geometry::cylindrical && ax == Axis::KAXIS) {
      return std::abs(Xf<Axis::IAXIS>(i)) * Dx<ax>();
    }
    return coords_.EdgeLength<AxisToInt(ax)>(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const Axis &ax, const int k, const int j,
                                              const int i) const {
    return AxisOverload([&, this]<Axis AX>() { return EdgeLength<AX>(k, j, i); }, ax);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    if constexpr (geom == Geometry::cylindrical) {
      const Real rp = Xf<Axis::IAXIS>(i + 1);
      const Real rm = Xf<Axis::IAXIS>(i);
      return 0.5 * Dx<Axis::KAXIS>() * (rp * rp - rm * rm) * Dx<Axis::JAXIS>();
    }
    return coords_.CellVolume(k, j, i);
  }

  template <TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Volume(const int k, const int j, const int i) const {
    using TE = TopologicalElement;
    if constexpr (el == TE::CC)
      return CellVolume(k, j, i);
    else if constexpr (el == TE::F1)
      return FaceArea<Axis::IAXIS>(k, j, i);
    else if constexpr (el == TE::F2)
      return FaceArea<Axis::JAXIS>(k, j, i);
    else if constexpr (el == TE::F3)
      return FaceArea<Axis::KAXIS>(k, j, i);
    else if constexpr (el == TE::E1)
      return EdgeLength<Axis::IAXIS>(k, j, i);
    else if constexpr (el == TE::E2)
      return EdgeLength<Axis::JAXIS>(k, j, i);
    else if constexpr (el == TE::E3)
      return EdgeLength<Axis::KAXIS>(k, j, i);
    else if constexpr (el == TE::NN)
      return 1.0;
    PARTHENON_FAIL("ifyou reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(TopologicalElement el, const int k, const int j, const int i) const {
    using TE = TopologicalElement;
    if (el == TE::CC)
      return CellVolume(k, j, i);
    else if (el == TE::F1)
      return FaceArea<Axis::IAXIS>(k, j, i);
    else if (el == TE::F2)
      return FaceArea<Axis::JAXIS>(k, j, i);
    else if (el == TE::F3)
      return FaceArea<Axis::KAXIS>(k, j, i);
    else if (el == TE::E1)
      return EdgeLength<Axis::IAXIS>(k, j, i);
    else if (el == TE::E2)
      return EdgeLength<Axis::JAXIS>(k, j, i);
    else if (el == TE::E3)
      return EdgeLength<Axis::KAXIS>(k, j, i);
    else if (el == TE::NN)
      return 1.0;
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

 private:
  parthenon::Coordinates_t coords_;

  template <typename Func, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real AxisOverload(Func function, Axis ax,
                                                Args &&...args) const {
    if (ax == Axis::IAXIS) {
      return function.template operator()<Axis::IAXIS>(std::forward<Args>(args)...);
    } else if (ax == Axis::JAXIS) {
      return function.template operator()<Axis::JAXIS>(std::forward<Args>(args)...);
    } else if (ax == Axis::KAXIS) {
      return function.template operator()<Axis::KAXIS>(std::forward<Args>(args)...);
    }

    PARTHENON_FAIL("Axis should only be one of [IJK]AXIS");
    return function.template operator()<Axis::IAXIS>(std::forward<Args>(args)...);
  }
};

template <typename T>
concept CoordinateSystem = requires(T coords) {
  // Templated methods (compile-time axis dispatch)
  { coords.template Dx<Axis::IAXIS>() } -> std::same_as<Real>;
  { coords.template Xc<Axis::IAXIS>(1) } -> std::same_as<Real>;
  { coords.template Xf<Axis::IAXIS>(1) } -> std::same_as<Real>;
  { coords.template FaceArea<Axis::IAXIS>(0, 0, 0) } -> std::same_as<Real>;
  { coords.template EdgeLength<Axis::IAXIS>(0, 0, 0) } -> std::same_as<Real>;
  // Runtime methods (runtime axis dispatch)
  { coords.Dx(Axis::IAXIS) } -> std::same_as<Real>;
  { coords.Xc(Axis::IAXIS, 1) } -> std::same_as<Real>;
  { coords.Xf(Axis::IAXIS, 1) } -> std::same_as<Real>;
  { coords.FaceArea(Axis::IAXIS, 0, 0, 0) } -> std::same_as<Real>;
  { coords.EdgeLength(Axis::IAXIS, 0, 0, 0) } -> std::same_as<Real>;
  { coords.CellVolume(0, 0, 0) } -> std::same_as<Real>;
  { coords.Volume(TopologicalElement::CC, 0, 0, 0) } -> std::same_as<Real>;
};

template <CoordinateSystem T>
struct CoordinateIndexer {
  KOKKOS_INLINE_FUNCTION CoordinateIndexer(const T &coords, const int k, const int j,
                                           const int i)
      : coords_(coords), kji_({k, j, i}) {}

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Dx() const {
    return coords_.template Dx<ax>();
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Dx(const Axis &ax) const { return coords_.Dx(ax); }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc() const {
    return coords_.template Xc<ax>(kji_[static_cast<int>(ax)]);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xi() const {
    return coords_.template Xi<ax>(kji_[static_cast<int>(ax)]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const Axis &ax) const {
    return coords_.Xc(ax, kji_[static_cast<int>(ax)]);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf() const {
    return coords_.template Xf<ax>(kji_[static_cast<int>(ax)]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const Axis &ax) const {
    return coords_.Xf(ax, kji_[static_cast<int>(ax)]);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea() const {
    return coords_.template FaceArea<ax>(kji_[2], kji_[1], kji_[0]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const Axis &ax) const {
    return coords_.FaceArea(ax, kji_[2], kji_[1], kji_[0]);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength() const {
    return coords_.template EdgeLength<ax>(kji_[2], kji_[1], kji_[0]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const Axis &ax) const {
    return coords_.EdgeLength(ax, kji_[2], kji_[1], kji_[0]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume() const {
    return coords_.CellVolume(kji_[2], kji_[1], kji_[0]);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Volume(TopologicalElement el) const {
    return coords_.Volume(el, kji_[2], kji_[1], kji_[0]);
  }

 private:
  const T coords_;
  const Kokkos::Array<int, 3> kji_;
};

namespace impl {
template <typename>
struct CoordinatesVariant {};

template <Geometry geom, Geometry... geoms>
struct CoordinatesVariant<OptList<Geometry, geom, geoms...>> {
  using type = PortsOfCall::variant<Coordinates<geom>, Coordinates<geoms>...>;

  static KOKKOS_INLINE_FUNCTION type Get(const Geometry geometry,
                                         const parthenon::Coordinates_t coords) {
    type coordinates = Coordinates<geom>(coords);
    (
        [&]() {
          if (geoms == geometry) coordinates = Coordinates<geoms>(coords);
        }(),
        ...);

    return coordinates;
  }
};

}  // namespace impl

using CoordinatesManager = impl::CoordinatesVariant<GeometryOptions>;
using CoordinatesVariant = CoordinatesManager::type;

namespace impl {
KOKKOS_INLINE_FUNCTION CoordinatesManager::type GetCoordinates(MeshBlock *mb) {
  const Geometry geometry = GetConfig(mb)->Get<Geometry>();
  return CoordinatesManager::Get(geometry, mb->coords);
}

KOKKOS_INLINE_FUNCTION CoordinatesManager::type
GetCoordinates(const Geometry geometry, const parthenon::Coordinates_t &coords) {
  return CoordinatesManager::Get(geometry, coords);
}
}  // namespace impl

struct GenericCoordinate {
  KOKKOS_INLINE_FUNCTION GenericCoordinate(const CoordinatesVariant coords)
      : coords_(coords) {}

  KOKKOS_INLINE_FUNCTION GenericCoordinate(MeshBlock *mb)
      : coords_(impl::GetCoordinates(mb)) {}
  KOKKOS_INLINE_FUNCTION GenericCoordinate(const Geometry geometry,
                                           const parthenon::Coordinates_t &coords)
      : coords_(impl::GetCoordinates(geometry, coords)) {}

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Dx() const {
    return PortsOfCall::visit([](const auto &coords) { return coords.template Dx<ax>(); },
                              coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Dx(const Axis &ax) const {
    return PortsOfCall::visit([&ax](const auto &coords) { return coords.Dx(ax); },
                              coords_);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    return PortsOfCall::visit(
        [idx](const auto &coords) { return coords.template Xc<ax>(idx); }, coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const Axis &ax, const int idx) const {
    return PortsOfCall::visit(
        [&ax, idx](const auto &coords) { return coords.Xc(ax, idx); }, coords_);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    return PortsOfCall::visit(
        [idx](const auto &coords) { return coords.template Xf<ax>(idx); }, coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const Axis &ax, const int idx) const {
    return PortsOfCall::visit(
        [&ax, idx](const auto &coords) { return coords.Xf(ax, idx); }, coords_);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template FaceArea<ax>(k, j, i); },
        coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const Axis &ax, const int k, const int j,
                                            const int i) const {
    return PortsOfCall::visit(
        [&ax, k, j, i](const auto &coords) { return coords.FaceArea(ax, k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template EdgeLength<ax>(k, j, i); },
        coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const Axis &ax, const int k, const int j,
                                              const int i) const {
    return PortsOfCall::visit(
        [&ax, k, j, i](const auto &coords) { return coords.EdgeLength(ax, k, j, i); },
        coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.CellVolume(k, j, i); }, coords_);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Volume(TopologicalElement el, const int k, const int j,
                                          const int i) const {
    return PortsOfCall::visit(
        [el, k, j, i](const auto &coords) { return coords.Volume(el, k, j, i); },
        coords_);
  }

 private:
  CoordinatesVariant coords_;
};

}  // namespace kamayan::grid
#endif  // GRID_GEOMETRY_HPP_
