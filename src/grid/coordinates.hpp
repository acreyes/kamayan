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

#include "grid/geometry_types.hpp"
#include "grid/grid_types.hpp"
#include "interface/variable_state.hpp"
#include "kamayan_utils/strings.hpp"
#include "kamayan_utils/type_list.hpp"
#include "kokkos_types.hpp"

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
                                 const int nghost) {
  using TE = TopologicalElement;
  using shapes = impl::CoordShapes<geom>;

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
    out.e += std::max(n - 2, 0);
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

template <Geometry geom>
struct CoordinatePack {
  template <Axis ax>
  struct AxisHelper {
    static constexpr Axis axis = ax;
  };

  template <typename... Fields, typename... Ts>
  KOKKOS_INLINE_FUNCTION CoordinatePack(TypeList<Fields...>,
                                        const SparsePack<Ts...> &pack, const int b) {
    (
        [&]<typename Field>() {
          Get_(Field()) = pack(b, Field());
        }.template operator()<Fields>(),
        ...);
  }

  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION CoordinatePack(const SparsePack<Ts...> &pack, const int b)
      : CoordinatePack(CoordFields(), pack, b) {}

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Dx(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return Dx_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real X(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return X_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xc(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return Xc_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real Xf(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return Xf_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return FaceArea_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  template <Axis ax>
  KOKKOS_INLINE_FUNCTION Real EdgeLength(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return EdgeLength_[static_cast<int>(ax)](kji[0], kji[1], kji[2]);
  }

  KOKKOS_INLINE_FUNCTION Real Volume(const int k, const int j, const int i) {
    auto kji = Index_(k, j, i);
    return Volume_(kji[0], kji[1], kji[2]);
  }

 private:
  // par array type returned by SparsePack<>(b, V())
  using par_array_t = parthenon::ParArray3D<Real, parthenon::VariableState>;

  par_array_t Dx_[3], X_[3], Xc_[3], Xf_[3], FaceArea_[3], EdgeLength_[3], Volume_;

  template <typename T>
  requires(CoordFields::Contains<T>())
  KOKKOS_INLINE_FUNCTION std::array<int, 3> Index_(const int k, const int j,
                                                   const int i) {
    using shapes = impl::CoordShapes<geom>;
    if constexpr (shapes::Scalars::template Contains<T>()) {
      return {0, 0, 0};
    } else if constexpr (shapes::Icoord::template Contains<T>()) {
      return {0, 0, i};
    } else if constexpr (shapes::Jcoord::template Contains<T>()) {
      return {0, j, 0};
    } else if constexpr (shapes::Kcoord::template Contains<T>()) {
      return {k, 0, 0};
    }
  }

  template <typename T>
  requires(AxisCoords::Contains<T>())
  KOKKOS_INLINE_FUNCTION par_array_t &Get_(const T &t) {
    if constexpr (std::is_same_v<T, coords::Dx<Axis::KAXIS>>) {
      return Dx_[0];
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::JAXIS>>) {
      return Dx_[1];
    } else if constexpr (std::is_same_v<T, coords::Dx<Axis::IAXIS>>) {
      return Dx_[2];
    } else if constexpr (std::is_same_v<T, coords::X<Axis::KAXIS>>) {
      return X_[0];
    } else if constexpr (std::is_same_v<T, coords::X<Axis::JAXIS>>) {
      return X_[1];
    } else if constexpr (std::is_same_v<T, coords::X<Axis::IAXIS>>) {
      return X_[2];
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::KAXIS>>) {
      return Xc_[0];
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::JAXIS>>) {
      return Xc_[1];
    } else if constexpr (std::is_same_v<T, coords::Xc<Axis::IAXIS>>) {
      return Xc_[2];
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::KAXIS>>) {
      return Xf_[0];
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::JAXIS>>) {
      return Xf_[1];
    } else if constexpr (std::is_same_v<T, coords::Xf<Axis::IAXIS>>) {
      return Xf_[2];
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::KAXIS>>) {
      return FaceArea_[0];
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::JAXIS>>) {
      return FaceArea_[1];
    } else if constexpr (std::is_same_v<T, coords::FaceArea<Axis::IAXIS>>) {
      return FaceArea_[2];
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::KAXIS>>) {
      return EdgeLength_[0];
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::JAXIS>>) {
      return EdgeLength_[1];
    } else if constexpr (std::is_same_v<T, coords::EdgeLength<Axis::IAXIS>>) {
      return EdgeLength_[2];
    } else {
      static_assert(false, "Type not mapped to a coordinate");
    }
  }

  template <typename T>
  requires(ScalarCoords::Contains<T>())
  KOKKOS_INLINE_FUNCTION par_array_t &Get_(const T &t) {
    if constexpr (std::is_same_v<T, coords::Volume>) {
      return Volume_;
    } else {
      static_assert(false, "Type not mapped to a coordinate");
    }
  }
};

}  // namespace kamayan::grid
#endif  // GRID_COORDINATES_HPP_
