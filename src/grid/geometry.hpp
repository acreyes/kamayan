#ifndef GRID_GEOMETRY_HPP_
#define GRID_GEOMETRY_HPP_

#include <numbers>
#include <utility>

#include "Kokkos_Macros.hpp"

#include "coordinates/coordinates.hpp"

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "grid/subpack.hpp"
#include "pack/sparse_pack.hpp"

namespace kamayan {
POLYMORPHIC_PARM(Geometry, cartesian, cylindrical);

namespace grid {

using GeometryOptions = OptList<Geometry, Geometry::cartesian, Geometry::cylindrical>;

// This is a (incomplete) wrapper around parthenon's uniform_coordinate (cartesian)
// to calculate coordinate related items in a different geometry.
template <Geometry geom>
struct Coordinates {
  KOKKOS_INLINE_FUNCTION Coordinates(parthenon::Coordinates_t coords) : coords_(coords) {}

  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION Coordinates(const parthenon::SparsePack<Ts...> &pack,
                                     const int block)
      : coords_(pack.GetCoordinates(block)) {}

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
      return (2.0 / 3.0) * (r1sq * r1 - r0sq * r0) / (r1sq - r0sq);
    }
    return coords_.Xc<dir>(idx);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const Axis &ax, const int idx) const {
    return AxisOverload([&, this]<Axis AX>() { return Xc<AX>(idx); }, ax);
  }

  // position at face centers
  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    return coords_.Xf<AxisToInt(ax)>(idx);
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const Axis &ax, const int idx) const {
    return AxisOverload([&, this]<Axis AX>() { return Xf<AX>(idx); }, ax);
  }

  template <Axis ax>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    constexpr auto dir = AxisToInt(ax);
    if constexpr (geom == Geometry::cylindrical) {
      if constexpr (ax == Axis::IAXIS) {
        // 2 * pi * r_i * dz
        return Xf<Axis::IAXIS>(i) * Dx<Axis::JAXIS>() * Dx<Axis::KAXIS>();
      } else if constexpr (ax == Axis::JAXIS) {
        // pi * (r_i+1/2**2 - r_i-1/2**2)
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
      // phi aligned edges -- dphi * r
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
      // dphi / 2 * (r_p^2 - r_m^2) * dz
      const Real rp = Xf<Axis::IAXIS>(i + 1);
      const Real rm = Xf<Axis::IAXIS>(i);
      return 0.5 * Dx<Axis::KAXIS>() * (rp * rp - rm * rm);
    }
    return coords_.CellVolume(k, j, i);
  }

  // Generalized volumes for a topological element
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
  const parthenon::Coordinates_t coords_;

  template <typename Func, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real AxisOverload(Func function, Axis ax, Args &&...args) {
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

}  // namespace grid
}  // namespace kamayan
#endif  // GRID_GEOMETRY_HPP_
