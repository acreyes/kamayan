#ifndef GRID_GRID_TYPES_HPP_
#define GRID_GRID_TYPES_HPP_

#include <parthenon/parthenon.hpp>

namespace kamayan {
using Real = parthenon::Real;
using TopologicalElement = parthenon::TopologicalElement;
using TopologicalType = parthenon::TopologicalType;
enum class Axis { KAXIS = 0, JAXIS = 1, IAXIS = 2 };

KOKKOS_INLINE_FUNCTION constexpr TopologicalElement
IncrementTE(const TopologicalElement &out_te, const TopologicalElement &in_te,
            const int &increment) {
  const auto offset = (static_cast<int>(in_te) + increment) % 3;
  const auto out = static_cast<int>(out_te) + offset;
  return static_cast<TopologicalElement>(out);
}

constexpr int AxisToInt(Axis ax) {
  return ax == Axis::IAXIS ? 1 : ax == Axis::JAXIS ? 2 : 3;
}

constexpr Axis AxisFromTE(const TopologicalElement el) {
  auto dir = static_cast<int>(el) % 3;
  return dir == 0 ? Axis::IAXIS : dir == 1 ? Axis::JAXIS : Axis::KAXIS;
}

constexpr Axis AxisFromInt(const int dir) {
  return dir == 1 ? Axis::IAXIS : dir == 2 ? Axis::JAXIS : Axis::KAXIS;
}

template <TopologicalElement edge>
concept EdgeElement = (edge >= TopologicalElement::E1 && edge <= TopologicalElement::E3);

// parthenon types
// packs
template <typename... Ts>
using SparsePack = parthenon::SparsePack<Ts...>;
using PDOpt = parthenon::PDOpt;

// mesh
using BlockList_t = parthenon::BlockList_t;
using Mesh = parthenon::Mesh;
using MeshData = parthenon::MeshData<Real>;
using MeshBlockData = parthenon::MeshBlockData<Real>;
using MeshBlock = parthenon::MeshBlock;

// scratchpad memory for hierarchical loops
using ScratchPad1D = parthenon::ScratchPad1D<Real>;
using ScratchPad2D = parthenon::ScratchPad2D<Real>;

// domain
using IndexDomain = parthenon::IndexDomain;

template <template <typename...> typename T, typename... Ts>
concept PackLike = requires(T<Ts...> pack) {
  (pack(int(), TopologicalElement(), Ts(), int(), int(), int()), ...);
};

}  // namespace kamayan

#endif  // GRID_GRID_TYPES_HPP_
