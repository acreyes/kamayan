#ifndef GRID_GRID_TYPES_HPP_
#define GRID_GRID_TYPES_HPP_

#include "kokkos_abstraction.hpp"
#include <parthenon/parthenon.hpp>

namespace kamayan {
using Real = parthenon::Real;
using TopologicalElement = parthenon::TopologicalElement;

template <typename... Ts>
using SparsePack = parthenon::SparsePack<Ts...>;

using BlockList_t = parthenon::BlockList_t;
using Mesh = parthenon::Mesh;
using MeshData = parthenon::MeshData<Real>;
using MeshBlockData = parthenon::MeshBlockData<Real>;
using MeshBlock = parthenon::MeshBlock;

using ScratchPad1D = parthenon::ScratchPad1D<Real>;

template <template <typename...> typename T, typename... Ts>
concept PackLike = requires(T<Ts...> pack) {
  (pack(int(), TopologicalElement(), Ts(), int(), int(), int()), ...);
};

}  // namespace kamayan

#endif  // GRID_GRID_TYPES_HPP_
