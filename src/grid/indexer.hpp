#ifndef GRID_INDEXER_HPP_
#define GRID_INDEXER_HPP_

#include "grid/grid_types.hpp"

namespace kamayan {
template <typename T, typename... Ts>
concept IndexerLike = requires(T indexer) { (indexer(Ts()), ...); };

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

#endif  // GRID_INDEXER_HPP_
