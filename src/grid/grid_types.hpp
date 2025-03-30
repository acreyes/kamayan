#ifndef GRID_GRID_TYPES_HPP_
#define GRID_GRID_TYPES_HPP_

#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include <parthenon/parthenon.hpp>

namespace kamayan {
using Real = parthenon::Real;
using TopologicalElement = parthenon::TopologicalElement;

// parthenon types
// packs
template <typename... Ts>
using SparsePack = parthenon::SparsePack<Ts...>;

// mesh
using BlockList_t = parthenon::BlockList_t;
using Mesh = parthenon::Mesh;
using MeshData = parthenon::MeshData<Real>;
using MeshBlockData = parthenon::MeshBlockData<Real>;
using MeshBlock = parthenon::MeshBlock;

// scratchpad memory for hierarchical loops
using ScratchPad1D = parthenon::ScratchPad1D<Real>;

// domain
using IndexDomain = parthenon::IndexDomain;

template <template <typename...> typename T, typename... Ts>
concept PackLike = requires(T<Ts...> pack) {
  (pack(int(), TopologicalElement(), Ts(), int(), int(), int()), ...);
};

}  // namespace kamayan

#endif  // GRID_GRID_TYPES_HPP_
