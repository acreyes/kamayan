#ifndef KAMAYAN_UTILS_TYPE_LIST_ARRAY_HPP_
#define KAMAYAN_UTILS_TYPE_LIST_ARRAY_HPP_
#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "kamayan_utils/type_list.hpp"

namespace kamayan {

// maps a dense type var in a typelist to an integer index
template <DenseVar... Ts>
struct TypeVarIndexer {
  using TL = TypeList<Ts...>;
  template <typename V>
  static KOKKOS_INLINE_FUNCTION std::size_t Idx(const V &var) {
    static_assert(TL::template Contains<V>(), "Indexer doesn't containt variable");
    return GetIndex_(TL(), var);
  }

 private:
  template <DenseVar V, DenseVar... Vs>
  static KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<V, Vs...>, const V &var) {
    return var.idx;
  }
  template <DenseVar V, DenseVar U, DenseVar... Us>
  static KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<U, Us...>, const V &var) {
    return U::n_comps + GetIndex_(TypeList<Us...>(), var);
  }
};

template <typename>
struct TypeListArray {};

template <template <typename...> typename TL, DenseVar... Ts>
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

#endif  // KAMAYAN_UTILS_TYPE_LIST_ARRAY_HPP_
