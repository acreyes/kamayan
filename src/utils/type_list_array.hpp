#ifndef UTILS_TYPE_LIST_ARRAY_HPP_
#define UTILS_TYPE_LIST_ARRAY_HPP_
#include <Kokkos_Core.hpp>

#include "Kokkos_Macros.hpp"
#include "grid/grid_types.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

// maps a dense type var in a typelist to an integer index
template <typename... Ts>
struct TypeVarIndexer {
  template <typename V>
  static KOKKOS_INLINE_FUNCTION std::size_t Idx(const V &var) {
    return GetIndex_(TypeList<Ts...>(), var);
  }

 private:
  template <typename V, typename... Vs>
  static KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<V, Vs...>, const V &var) {
    return var.idx;
  }
  template <typename V, typename U, typename... Us>
  static KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<U, Us...>, const V &var) {
    // I don't love that we depend on the VARIABLE macro declaring the size correctly
    // at compile time, whereas parthenon lets us decide the shape of an array
    return U::n_comps + GetIndex_(TypeList<Us...>(), var);
  }
};

template <typename>
struct TypeListArray {};

template <template <typename...> typename TL, typename... Ts>
struct TypeListArray<TL<Ts...>> {
  using indexer = TypeVarIndexer<Ts...>;
  using type = TypeList<Ts...>;
  static constexpr std::size_t n_vars = (0 + ... + Ts::n_comps);

  KOKKOS_INLINE_FUNCTION TypeListArray() = default;
  KOKKOS_INLINE_FUNCTION TypeListArray(const Real &value) {
    for (int idx = 0; idx < n_vars; idx++) {
      data[idx] = value;
    }
  }
  KOKKOS_INLINE_FUNCTION TypeListArray(Kokkos::Array<Real, n_vars> data_) : data(data_) {}

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) {
    return data[indexer::Idx(var)];
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real operator()(const V &var) const {
    return data[indexer::Idx(var)];
  }

  KOKKOS_INLINE_FUNCTION Real &operator[](const int &idx) { return data[idx]; }

  // private:
  Kokkos::Array<Real, n_vars> data;
};
}  // namespace kamayan

#endif  // UTILS_TYPE_LIST_ARRAY_HPP_
