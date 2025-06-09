#ifndef UTILS_TYPE_LIST_ARRAY_HPP_
#define UTILS_TYPE_LIST_ARRAY_HPP_
#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

template <typename>
struct TypeListArray {};

template <template <typename...> typename TL, typename... Ts>
struct TypeListArray<TL<Ts...>> {
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
    return data[GetIndex_(type(), var)];
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real operator()(const V &var) const {
    return data[GetIndex_(type(), var)];
  }

  KOKKOS_INLINE_FUNCTION Real &operator[](const int &idx) { return data[idx]; }

  // private:
  template <typename V, typename... Vs>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<V, Vs...>, const V &var) const {
    return var.idx;
  }
  template <typename V, typename U, typename... Us>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<U, Us...>, const V &var) const {
    // I don't love that we depend on the VARIABLE macro declaring the size correctly
    // at compile time, whereas parthenon lets us decide the shape of an array
    return U::n_comps + GetIndex_(TypeList<Us...>(), var);
  }
  Kokkos::Array<Real, n_vars> data;
};
}  // namespace kamayan

#endif  // UTILS_TYPE_LIST_ARRAY_HPP_
