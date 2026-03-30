#ifndef GRID_COORDINATES_HPP_
#define GRID_COORDINATES_HPP_
#include <string>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "grid/grid_types.hpp"
#include "kamayan_utils/strings.hpp"
#include "kamayan_utils/type_list.hpp"

namespace kamayan::grid {

// Idea for this is to have a variable that is stored per meshblock
// that is associated with a topological element, but might be only allocated for a
// particular kji dimension. For example radial face areas in cylindrical coordinates
// these face areas on a mesh block only depend on the radial index (i) and are
// located on the TE::F1 faces. In the case that there are no axes this is just a scalar
//
// using AreaF1 = CoordVar<"geom.area_f1">
//
// The raw parthenon way of accessing this would be to pack it, then access it
// with the i-index
// pack(b, AreaF1())(i);
//
// It is not necessarily the case that an arbitrary coordinate system would
// index with i, it might even be the full (k,j,i), so we need some kind of helper
// to forward that
//
// coords.Area<Axes::IAXIS>(pack, b, k, j, i);
//
// I think we define all the variables as CoordVars with just the var_name,
// we collect them into a TypeList, and then each coordinate system becomes
// responsible for exporting the shapes, so we can add as
//
// const auto shape = coords.shape<AreaF1>(nx3, nx2, nx1); // should include ghosts
// pkg->AddField<AreaF1>(Metadata({Metadata::None, Metadata::OneCopy}, shape));
//
// this could also be wrapped in a type_for
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

template <template <Axis> typename T, template <Axis> typename... Ts>
auto AxisTL() {
  if constexpr (sizeof...(Ts) == 0) {
    return TypeList<T<Axis::IAXIS>, T<Axis::JAXIS>, T<Axis::KAXIS>>();
  } else {
    return ConcatTypeLists_t<decltype(AxisTL<T>()), decltype(AxisTL<Ts...>())>();
  }
}

// compiler seems really unhappy if I attempt to alias this
using AxisCoords = decltype(AxisTL<Dx, X, Xc, Xf, FaceArea, EdgeLength>());
using CoordFields = ConcatTypeLists_t<AxisCoords, TypeList<Volume>>;

// fill meshblock with coordinate CoordFields
void CalculateCoordinates(MeshBlock *mb);
}  // namespace kamayan::grid
#endif  // GRID_COORDINATES_HPP_
