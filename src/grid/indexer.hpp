#ifndef GRID_INDEXER_HPP_
#define GRID_INDEXER_HPP_

#include <Kokkos_Macros.hpp>

#include "grid/grid_types.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan {
template <typename T, typename... Ts>
concept IndexerLike = requires(T indexer) { (indexer(Ts()), ...); };

template <typename T, typename... Ts>
concept IndexerLike1D = requires(T indexer) { (indexer(Ts(), int()), ...); };

template <typename T, typename... Ts>
concept IndexerLike2D = requires(T indexer) { (indexer(Ts(), int(), int()), ...); };

template <typename T, typename... Ts>
concept IndexerLike3D =
    requires(T indexer) { (indexer(Ts(), int(), int(), int()), ...); };

template <typename T>
concept Stencil1D = requires(T stencil) { stencil(int()); };

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
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, const T &t) const {
    return pack(b, te, t, k, j, i);
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION Real &operator()(const T &t) const {
    return pack(b, t, k, j, i);
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION Real &flux(const TopologicalElement &te, const T &t) const {
    return pack.flux(b, te, t, k, j, i);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetSize(const V &var) const {
    return pack.GetSize(b, var);
  }

 private:
  const Container<Ts...> &pack;
  const int b, k, j, i;
};

// use this to index into a scratch pad view that has the same
// number of variables as are in a pack
template <typename ScratchPad, typename... Ts>
struct ScratchIndexer {
  KOKKOS_INLINE_FUNCTION
  ScratchIndexer(const SparsePack<Ts...> &pack_, ScratchPad &scratch_, const int &b_,
                 const int &i_)
      : pack(pack_), scratch(scratch_), b(b_), i(i_) {}

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) const {
    return scratch(pack.GetIndex(b, var), i);
  }

 private:
  const SparsePack<Ts...> &pack;
  ScratchPad &scratch;
  const int i, b;
};

enum class Axis { IAXIS, JAXIS, KAXIS };

template <Axis axis, template <typename...> typename Container, typename... Ts>
requires(PackLike<Container, Ts...>)
struct SparsePackStencil1D {
  KOKKOS_INLINE_FUNCTION
  SparsePackStencil1D(const Container<Ts...> &pack_, const int &b_, const int &var_,
                      const int &k_, const int &j_, const int &i_)
      : pack(pack_), b(b_), var(var_), k(k_), j(j_), i(i_) {}

  KOKKOS_INLINE_FUNCTION Real &operator()(const int &idx) {
    if constexpr (axis == Axis::KAXIS) {
      return pack(b, var, k + idx, j, i + idx);
    } else if constexpr (axis == Axis::JAXIS) {
      return pack(b, var, k, j + idx, i + idx);
    }
    return pack(b, var, k, j, i + idx);
  }

 private:
  const Container<Ts...> &pack;
  const int b, var, k, j, i;
};

template <template <typename...> typename Container, typename... Ts>
KOKKOS_INLINE_FUNCTION auto MakePackIndexer(const Container<Ts...> &pack, const int &b,
                                            const int &k, const int &j, const int &i) {
  return SparsePackIndexer<Container<Ts...>>(pack, b, k, j, i);
}

template <Axis axis, template <typename...> typename Container, typename... Ts>
KOKKOS_INLINE_FUNCTION auto MakePackStencil1D(const Container<Ts...> &pack, const int &b,
                                              const int &var, const int &k, const int &j,
                                              const int &i) {
  return SparsePackStencil1D<axis, Container, Ts...>(pack, b, var, k, j, i);
}

template <typename ScratchPad, typename... Ts>
KOKKOS_INLINE_FUNCTION auto MakeScratchIndexer(const SparsePack<Ts...> &pack,
                                               ScratchPad &scratch, const int &b,
                                               const int &i) {
  return ScratchIndexer<ScratchPad, Ts...>(pack, scratch, b, i);
}
}  // namespace kamayan

#endif  // GRID_INDEXER_HPP_
