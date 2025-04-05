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
  static constexpr std::size_t n_vars = sizeof...(Ts);

  KOKKOS_INLINE_FUNCTION TypeListArray() = default;
  KOKKOS_INLINE_FUNCTION TypeListArray(Kokkos::Array<Real, n_vars> data_) : data(data_) {}

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) {
    return data[GetIndex_(type(), var)];
  }

  KOKKOS_INLINE_FUNCTION Real &operator[](const int &idx) { return data[idx]; }

  KOKKOS_INLINE_FUNCTION TypeListArray operator*(const TypeListArray &other) const {
    for (int v = 0; v < n_vars; v++) {
      data[v] *= other[v];
    }
    return TypeListArray(data);
  }

  KOKKOS_INLINE_FUNCTION TypeListArray operator-(const TypeListArray &other) const {
    for (int v = 0; v < n_vars; v++) {
      data[v] -= other[v];
    }
    return TypeListArray(data);
  }

  KOKKOS_INLINE_FUNCTION TypeListArray operator+(const TypeListArray &other) const {
    for (int v = 0; v < n_vars; v++) {
      data[v] += other[v];
    }
    return TypeListArray(data);
  }

 private:
  template <typename V, typename... Vs>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<V, Vs...>, const V &var) {
    return var.idx;
  }
  template <typename V, typename U, typename... Us>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<U, Us...>, const V &var) {
    return U::size() + GetIndex_(TypeList<Us...>(), var);
  }
  Kokkos::Array<Real, n_vars> data;
};
}  // namespace kamayan

#endif  // UTILS_TYPE_LIST_ARRAY_HPP_
