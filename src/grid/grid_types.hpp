#ifndef GRID_GRID_TYPES_HPP_
#define GRID_GRID_TYPES_HPP_

#include "basic_types.hpp"
#include <parthenon/parthenon.hpp>

namespace kamayan {
using Real = parthenon::Real;
using TopologicalElement = parthenon::TopologicalElement;
using TopologicalType = parthenon::TopologicalType;

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
