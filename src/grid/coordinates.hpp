#ifndef GRID_COORDINATES_HPP_
#define GRID_COORDINATES_HPP_
#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

#include <ports-of-call/variant.hpp>

#include "grid/geometry.hpp"
#include "grid/geometry_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "interface/variable_state.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/strings.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "kokkos_types.hpp"
#include "utils/concepts_lite.hpp"

namespace kamayan::grid {

template <strings::CompileTimeString var_name,
          TopologicalElement el = TopologicalElement::CC>
struct CoordVar : parthenon::variable_names::base_t<false> {
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION CoordVar(Args &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Args>(args)...) {}

  static std::string name() { return std::string(var_name.value); }
  static constexpr auto label = var_name;
  static constexpr auto element = el;
};

namespace impl {
template <std::size_t N>
constexpr auto AxisNamer(const strings::CompileTimeString<N> base, Axis ax) {
  if (ax == Axis::IAXIS) {
    return strings::concat_cts(base, strings::make_cts("1"));
  } else if (ax == Axis::JAXIS) {
    return strings::concat_cts(base, strings::make_cts("2"));
  } else if (ax == Axis::KAXIS) {
    return strings::concat_cts(base, strings::make_cts("3"));
  }
  return strings::concat_cts(base, strings::make_cts("1"));
}
}  // namespace impl

constexpr TopologicalElement AxisToTE(const TopologicalElement el, Axis ax) {
  return static_cast<TopologicalElement>(static_cast<int>(el) + AxisToInt(ax) - 1);
}

template <strings::CompileTimeString var_name, Axis ax,
          TopologicalElement el = TopologicalElement::CC>
using AxisCoord = CoordVar<impl::AxisNamer(var_name, ax)>;

namespace coords {
template <Axis ax>
using Dx = AxisCoord<"geom.Dx", ax>;

template <Axis ax>
using X = AxisCoord<"geom.X", ax>;

template <Axis ax>
using Xc = AxisCoord<"geom.Xc", ax>;

template <Axis ax>
using Xf = AxisCoord<"geom.Xf", ax, AxisToTE(TopologicalElement::F1, ax)>;

template <Axis ax>
using FaceArea = AxisCoord<"geom.FaceArea", ax, AxisToTE(TopologicalElement::F1, ax)>;

template <Axis ax>
using EdgeLength = AxisCoord<"geom.EdgeLength", ax, AxisToTE(TopologicalElement::E1, ax)>;

using Volume = CoordVar<"geom.Volume">;
}  // namespace coords

template <template <Axis> typename T, template <Axis> typename... Ts>
auto AxisTL() {
  if constexpr (sizeof...(Ts) == 0) {
    return TypeList<T<Axis::IAXIS>, T<Axis::JAXIS>, T<Axis::KAXIS>>();
  } else {
    return ConcatTypeLists_t<decltype(AxisTL<T>()), decltype(AxisTL<Ts...>())>();
  }
}

// compiler seems really unhappy if I attempt to alias this
using AxisCoords = decltype(AxisTL<coords::Dx, coords::X, coords::Xc, coords::Xf,
                                   coords::FaceArea, coords::EdgeLength>());
using ScalarCoords = TypeList<coords::Volume>;
using CoordFields = ConcatTypeLists_t<AxisCoords, ScalarCoords>;

// some common combinations
template <template <Axis> typename Coord>
using AllAxes = TypeList<Coord<Axis::IAXIS>, Coord<Axis::JAXIS>, Coord<Axis::KAXIS>>;

using Deltas = AllAxes<coords::Dx>;
using Xcenter = AllAxes<coords::X>;
using Xcoord = AllAxes<coords::Xc>;
using Xface = AllAxes<coords::Xf>;
using FaceAreas = AllAxes<coords::FaceArea>;

namespace impl {
template <Geometry>
struct CoordShapes {
  using Scalars = TypeList<>;
  using Icoord = TypeList<>;
  using Jcoord = TypeList<>;
  using Kcoord = TypeList<>;
};

template <>
struct CoordShapes<Geometry::cartesian> {
  using Scalars = ConcatTypeLists_t<
      TypeList<coords::Volume>,
      decltype(AxisTL<coords::Dx, coords::FaceArea, coords::EdgeLength>())>;
  using Icoord =
      TypeList<coords::X<Axis::IAXIS>, coords::Xc<Axis::IAXIS>, coords::Xf<Axis::IAXIS>>;
  using Jcoord =
      TypeList<coords::X<Axis::JAXIS>, coords::Xc<Axis::JAXIS>, coords::Xf<Axis::JAXIS>>;
  using Kcoord =
      TypeList<coords::X<Axis::KAXIS>, coords::Xc<Axis::KAXIS>, coords::Xf<Axis::KAXIS>>;
};

template <>
struct CoordShapes<Geometry::cylindrical> {
  using Scalars = decltype(AxisTL<coords::Dx>());
  using Icoord =
      ConcatTypeLists_t<TypeList<coords::Volume, coords::X<Axis::IAXIS>,
                                 coords::Xc<Axis::IAXIS>, coords::Xf<Axis::IAXIS>>,
                        decltype(AxisTL<coords::FaceArea, coords::EdgeLength>())>;
  using Jcoord =
      TypeList<coords::X<Axis::JAXIS>, coords::Xc<Axis::JAXIS>, coords::Xf<Axis::JAXIS>>;
  using Kcoord =
      TypeList<coords::X<Axis::KAXIS>, coords::Xc<Axis::KAXIS>, coords::Xf<Axis::KAXIS>>;
};
}  // namespace impl

template <Geometry geom, typename T>
requires(CoordFields::template Contains<T>())
std::vector<int> CoordinateShape(const int nx3, const int nx2, const int nx1,
                                 const int nghost, const bool reverse = false) {
  using TE = TopologicalElement;
  using shapes = impl::CoordShapes<geom>;

  auto shape = [&]() -> std::vector<int> {
    if constexpr (shapes::Scalars::template Contains<T>()) {
      return {1, 1, 1};
    } else if constexpr (shapes::Icoord::template Contains<T>()) {
      auto N = nx1 + 2 * nghost;
      N += (T::element == TE::F1 || T::element == TE::E2 || T::element == TE::E3) ? 1 : 0;
      return {1, 1, N};
    } else if constexpr (shapes::Jcoord::template Contains<T>()) {
      auto N = nx2 + 2 * nghost;
      N += (T::element == TE::F3 || T::element == TE::E2 || T::element == TE::E1) ? 1 : 0;
      return {1, N, 1};
    } else if constexpr (shapes::Kcoord::template Contains<T>()) {
      auto N = nx3 + 2 * nghost;
      N += (T::element == TE::F3 || T::element == TE::E2 || T::element == TE::E1) ? 1 : 0;
      return {N, 1, 1};
    }

    PARTHENON_FAIL(std::format("Coordinate Variable {} not handled for geometry {}",
                               T::name(), OptInfo<Geometry>::Label(geom))
                       .c_str());
  }();

  if (reverse) std::reverse(shape.begin(), shape.end());
  return shape;
}

template <Geometry geom, typename T>
requires(CoordFields::template Contains<T>())
std::tuple<parthenon::IndexRange, parthenon::IndexRange, parthenon::IndexRange>
CoordinateIndexRanges(parthenon::IndexShape cellbounds,
                      const IndexDomain domain = IndexDomain::entire) {
  auto shapes =
      CoordinateShape<geom, T>(cellbounds.ncellsk(domain), cellbounds.ncellsj(domain),
                               cellbounds.ncellsi(domain), 0);

  auto make_index_range = [](const parthenon::IndexRange bounds, const int n) {
    auto out = parthenon::IndexRange{bounds.s, bounds.s};
    out.e += std::max(n - 1, 0);
    return out;
  };
  return {
      make_index_range(cellbounds.GetBoundsK(domain, T::element), shapes[0]),
      make_index_range(cellbounds.GetBoundsJ(domain, T::element), shapes[1]),
      make_index_range(cellbounds.GetBoundsI(domain, T::element), shapes[2]),
  };
}

// fill meshblock with coordinate CoordFields
void CalculateCoordinates(MeshBlock *mb);
void CalculateCoordinates(MeshBlock *mb, Geometry geom);

template <Geometry geom>
void CalculateCoordinates(const Coordinates<geom> &coords, auto &pack,
                          const parthenon::IndexShape &cellbounds);

template <Geometry geom, typename... Fields>
struct CoordinatePack {
  using FieldList = TypeList<Fields...>;
  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION CoordinatePack(const SparsePack<Ts...> &pack, const int b) {
    static_assert((TypeList<Ts...>::template Contains<Fields>() && ...),
                  "Pack must contain all requested coordinate fields.");
    (
        [&]<typename Field>() {
          Get_(Field()) = pack(b, Field());
        }.template operator()<Fields>(),
        ...);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Dx(const int k, const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::Dx<ax>>(),
                  "Coordinate Pack must be constructed with required Dx coordinate");
    auto kji = Index_<coords::Dx<ax>>(k, j, i);
    return Get_(coords::Dx<ax>())(kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real X(const int k, const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::X<ax>>(),
                  "Coordinate Pack must be constructed with required X coordinate");
    auto kji = Index_<coords::X<ax>>(k, j, i);
    return Get_(coords::X<ax>())(kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::Xc<ax>>(),
                  "Coordinate Pack must be constructed with required Xc coordinate");
    auto kji = Index_<coords::Xc<ax>>(k, j, i);
    return Get_(coords::Xc<ax>())(kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::Xf<ax>>(),
                  "Coordinate Pack must be constructed with required Xf coordinate");
    auto kji = Index_<coords::Xf<ax>>(k, j, i);
    return Get_(coords::Xf<ax>())(kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(
        FieldList::template Contains<coords::FaceArea<ax>>(),
        "Coordinate Pack must be constructed with required FaceArea coordinate");
    auto kji = Index_<coords::FaceArea<ax>>(k, j, i);
    return Get_(coords::FaceArea<ax>())(kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real EdgeLength(const int k, const int j, const int i) const {
    static_assert(
        FieldList::template Contains<coords::EdgeLength<ax>>(),
        "Coordinate Pack must be constructed with required EdgeLength coordinate");
    auto kji = Index_<coords::EdgeLength<ax>>(k, j, i);
    return Get_(coords::EdgeLength<ax>())(kji[0], kji[1], kji[2]);
  }

  KOKKOS_INLINE_FUNCTION Real CellVolume(const int k, const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::Volume>(),
                  "Coordinate Pack must be constructed with required Volume");
    auto kji = Index_<coords::Volume>(k, j, i);
    return Volume_(kji[0], kji[1], kji[2]);
  }

  KOKKOS_INLINE_FUNCTION Real Dx(const Axis ax, const int k, const int j,
                                 const int i) const {
    static_assert(
        FieldList::Contains(TypeList<coords::Dx<Axis::KAXIS>, coords::Dx<Axis::JAXIS>,
                                     coords::Dx<Axis::IAXIS>>()),
        "Dx requires all 3 Dx<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return Dx<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real X(const Axis ax, const int k, const int j,
                                const int i) const {
    static_assert(
        FieldList::Contains(TypeList<coords::X<Axis::KAXIS>, coords::X<Axis::JAXIS>,
                                     coords::X<Axis::IAXIS>>()),
        "X requires all 3 X<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return X<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real Xc(const Axis ax, const int k, const int j,
                                 const int i) const {
    static_assert(
        FieldList::Contains(TypeList<coords::Xc<Axis::KAXIS>, coords::Xc<Axis::JAXIS>,
                                     coords::Xc<Axis::IAXIS>>()),
        "Xc requires all 3 Xc<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return Xc<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real Xf(const Axis ax, const int k, const int j,
                                 const int i) const {
    static_assert(
        FieldList::Contains(TypeList<coords::Xf<Axis::KAXIS>, coords::Xf<Axis::JAXIS>,
                                     coords::Xf<Axis::IAXIS>>()),
        "Xf requires all 3 Xf<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return Xf<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real FaceArea(const Axis ax, const int k, const int j,
                                       const int i) const {
    static_assert(
        FieldList::Contains(
            TypeList<coords::FaceArea<Axis::KAXIS>, coords::FaceArea<Axis::JAXIS>,
                     coords::FaceArea<Axis::IAXIS>>()),
        "FaceArea requires all 3 FaceArea<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return FaceArea<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real EdgeLength(const Axis ax, const int k, const int j,
                                         const int i) const {
    static_assert(
        FieldList::Contains(
            TypeList<coords::EdgeLength<Axis::KAXIS>, coords::EdgeLength<Axis::JAXIS>,
                     coords::EdgeLength<Axis::IAXIS>>()),
        "EdgeLength requires all 3 EdgeLength<axes> to be packed");
    return AxisOverload([&, this]<Axis AX>() { return EdgeLength<AX>(k, j, i); }, ax);
  }

  KOKKOS_INLINE_FUNCTION Real Volume(const TopologicalElement el, const int k,
                                     const int j, const int i) const {
    static_assert(FieldList::template Contains<coords::Volume>(),
                  "Volume requires Volume to be packed.");
    static_assert(
        FieldList::Contains(
            TypeList<coords::FaceArea<Axis::KAXIS>, coords::FaceArea<Axis::JAXIS>,
                     coords::FaceArea<Axis::IAXIS>>()),
        "Volume requires all 3 FaceArea<axes> to be packed");
    static_assert(
        FieldList::Contains(
            TypeList<coords::EdgeLength<Axis::KAXIS>, coords::EdgeLength<Axis::JAXIS>,
                     coords::EdgeLength<Axis::IAXIS>>()),
        "Volume requires all 3 EdgeLength<axes> to be packed");
    using TE = TopologicalElement;
    if (el == TE::CC)
      return CellVolume(k, j, i);
    else if (el == TE::F1)
      return FaceArea(Axis::IAXIS, k, j, i);
    else if (el == TE::F2)
      return FaceArea(Axis::JAXIS, k, j, i);
    else if (el == TE::F3)
      return FaceArea(Axis::KAXIS, k, j, i);
    else if (el == TE::E1)
      return EdgeLength(Axis::IAXIS, k, j, i);
    else if (el == TE::E2)
      return EdgeLength(Axis::JAXIS, k, j, i);
    else if (el == TE::E3)
      return EdgeLength(Axis::KAXIS, k, j, i);
    else if (el == TE::NN)
      return 1.0;
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

 private:
  // par array type returned by SparsePack<>(b, V())
  using par_array_t = parthenon::ParArray3D<Real, parthenon::VariableState>;

  par_array_t Dx1_, Dx2_, Dx3_, X1_, X2_, X3_, Xc1_, Xc2_, Xc3_, Xf1_, Xf2_, Xf3_,
      FaceArea1_, FaceArea2_, FaceArea3_, EdgeLength1_, EdgeLength2_, EdgeLength3_,
      Volume_;

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

  template <typename T>
  KOKKOS_INLINE_FUNCTION std::array<int, 3> Index_(const int k, const int j,
                                                   const int i) const {
    using shapes = impl::CoordShapes<geom>;
    if constexpr (shapes::Scalars::template Contains<T>()) {
      return {0, 0, 0};
    } else if constexpr (shapes::Icoord::template Contains<T>()) {
      return {0, 0, i};
    } else if constexpr (shapes::Jcoord::template Contains<T>()) {
      return {0, j, 0};
    } else if constexpr (shapes::Kcoord::template Contains<T>()) {
      return {k, 0, 0};
    } else {
      static_assert(always_false<T>, "Type not handled by CoordShapes");
    }
    return {k, j, i};
  }

  template <typename T>
  requires(AxisCoords::Contains<T>())
  KOKKOS_INLINE_FUNCTION par_array_t &Get_(const T &t) {
    if constexpr (std::is_same_v<T, coords::Dx<Axis::KAXIS>>) {
      return Dx1_;
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::JAXIS>>) {
      return Dx2_;
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::IAXIS>>) {
      return Dx3_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::KAXIS>>) {
      return X1_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::JAXIS>>) {
      return X2_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::IAXIS>>) {
      return X3_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::KAXIS>>) {
      return Xc1_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::JAXIS>>) {
      return Xc2_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::IAXIS>>) {
      return Xc3_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::KAXIS>>) {
      return Xf1_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::JAXIS>>) {
      return Xf2_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::IAXIS>>) {
      return Xf3_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::KAXIS>>) {
      return FaceArea1_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::JAXIS>>) {
      return FaceArea2_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::IAXIS>>) {
      return FaceArea3_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::KAXIS>>) {
      return EdgeLength1_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::JAXIS>>) {
      return EdgeLength2_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::IAXIS>>) {
      return EdgeLength3_;
    } else {
      static_assert(always_false<T>, "Type not mapped to a coordinate");
    }
  }
  template <typename T>
  requires(AxisCoords::Contains<T>())
  KOKKOS_INLINE_FUNCTION const par_array_t &Get_(const T &t) const {
    if constexpr (std::is_same_v<T, coords::Dx<Axis::KAXIS>>) {
      return Dx1_;
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::JAXIS>>) {
      return Dx2_;
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::IAXIS>>) {
      return Dx3_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::KAXIS>>) {
      return X1_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::JAXIS>>) {
      return X2_;
    } else if constexpr (std::is_same_v<T, coords::X<Axis::IAXIS>>) {
      return X3_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::KAXIS>>) {
      return Xc1_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::JAXIS>>) {
      return Xc2_;
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::IAXIS>>) {
      return Xc3_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::KAXIS>>) {
      return Xf1_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::JAXIS>>) {
      return Xf2_;
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::IAXIS>>) {
      return Xf3_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::KAXIS>>) {
      return FaceArea1_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::JAXIS>>) {
      return FaceArea2_;
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::IAXIS>>) {
      return FaceArea3_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::KAXIS>>) {
      return EdgeLength1_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::JAXIS>>) {
      return EdgeLength2_;
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::IAXIS>>) {
      return EdgeLength3_;
    } else {
      static_assert(always_false<T>, "Type not mapped to a coordinate");
    }
  }

  template <typename T>
  requires(ScalarCoords::Contains<T>())
  KOKKOS_INLINE_FUNCTION par_array_t &Get_(const T &t) {
    if constexpr (std::is_same_v<T, coords::Volume>) {
      return Volume_;
    } else {
      static_assert(always_false<T>, "Type not mapped to a coordinate");
    }
  }
};

template <Geometry geom, typename... Fields>
struct CoordinatePack<geom, TypeList<Fields...>> : CoordinatePack<geom, Fields...> {
  template <typename Pack>
  KOKKOS_INLINE_FUNCTION CoordinatePack(const Pack &pack, const int b)
      : CoordinatePack<geom, Fields...>(pack, b) {}
};

template <Geometry geom>
struct CoordinatePack<geom> : CoordinatePack<geom, CoordFields> {
  template <typename Pack>
  KOKKOS_INLINE_FUNCTION CoordinatePack(const Pack &pack, const int b)
      : CoordinatePack<geom, CoordFields>(pack, b) {}
};

namespace impl {
template <typename>
struct CoordinatePackVariant {};

template <Geometry geom, Geometry... geoms>
struct CoordinatePackVariant<OptList<Geometry, geom, geoms...>> {
  using type = PortsOfCall::variant<CoordinatePack<geom>, CoordinatePack<geoms>...>;
};
}  // namespace impl

using CoordinatePackVariant = impl::CoordinatePackVariant<GeometryOptions>::type;

struct GenericCoordinatePack {
  template <typename... Cs, typename... Ts>
  GenericCoordinatePack(TypeList<Cs...>, const Geometry geometry,
                        const parthenon::SparsePack<Ts...> &pack, const int b) {
    GeometryOptions::dispatch(
        [&]<Geometry geom>() {
          coords_ = CoordinatePack<geom>(TypeList<Cs...>(), pack, b);
        },
        geometry);
  }

  template <typename... Ts>
  GenericCoordinatePack(const Geometry geometry, const parthenon::SparsePack<Ts...> &pack,
                        const int b) {
    GeometryOptions::dispatch(
        [&]<Geometry geom>() { coords_ = CoordinatePack<geom>(pack, b); }, geometry);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Dx(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template Dx<ax>(k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real X(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template X<ax>(k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template Xc<ax>(k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template Xf<ax>(k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template FaceArea<ax>(k, j, i); },
        coords_);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real EdgeLength(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.template EdgeLength<ax>(k, j, i); },
        coords_);
  }

  KOKKOS_INLINE_FUNCTION Real Dx(const Axis ax, const int k, const int j,
                                 const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.Dx(ax, k, j, i); }, coords_);
  }

  KOKKOS_INLINE_FUNCTION Real X(const Axis ax, const int k, const int j,
                                const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.X(ax, k, j, i); }, coords_);
  }

  KOKKOS_INLINE_FUNCTION Real Xc(const Axis ax, const int k, const int j,
                                 const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.Xc(ax, k, j, i); }, coords_);
  }

  KOKKOS_INLINE_FUNCTION Real Xf(const Axis ax, const int k, const int j,
                                 const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.Xf(ax, k, j, i); }, coords_);
  }

  KOKKOS_INLINE_FUNCTION Real FaceArea(const Axis ax, const int k, const int j,
                                       const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.FaceArea(ax, k, j, i); },
        coords_);
  }

  KOKKOS_INLINE_FUNCTION Real EdgeLength(const Axis ax, const int k, const int j,
                                         const int i) const {
    return PortsOfCall::visit(
        [ax, k, j, i](const auto &coords) { return coords.EdgeLength(ax, k, j, i); },
        coords_);
  }

  KOKKOS_INLINE_FUNCTION Real CellVolume(const int k, const int j, const int i) const {
    return PortsOfCall::visit(
        [k, j, i](const auto &coords) { return coords.CellVolume(k, j, i); }, coords_);
  }

  KOKKOS_INLINE_FUNCTION Real Volume(const TopologicalElement el, const int k,
                                     const int j, const int i) const {
    return PortsOfCall::visit(
        [el, k, j, i](const auto &coords) { return coords.Volume(el, k, j, i); },
        coords_);
  }

 private:
  CoordinatePackVariant coords_;
};

}  // namespace kamayan::grid
#endif  // GRID_COORDINATES_HPP_
