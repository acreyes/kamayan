#ifndef GRID_GRID_TYPES_HPP_
#define GRID_GRID_TYPES_HPP_

#include "basic_types.hpp"
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

template <template <typename...> typename T, typename... Ts>
concept PackLike = requires(T<Ts...> pack, int i) { (pack(Ts(), i, i, i, i), ...); };

// can we abstract a way for indexing into a pack?
template <typename>
struct SparsePackIndexer {};

// use this to index into a slice along the field component of a pack
template <template <typename...> typename Container, typename... Ts>
requires(PackLike<Container, Ts...>)
struct SparsePackIndexer<Container<Ts...>> {
  KOKKOS_INLINE_FUNCTION
  SparsePackIndexer(const Container<Ts...> &pack_, const int &b_, const int &k_,
                    const int &j_, const int &i_)
      : pack(pack_), b(b_), k(k_), j(j_), i(i_) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION Real &operator()(T) {
    return pack(T(), b, k, j, i);
  }

 private:
  const Container<Ts...> &pack;
  const int b, k, j, i;
};

}  // namespace kamayan

#endif  // GRID_GRID_TYPES_HPP_
