#ifndef KAMAYAN_UTILS_ROBUST_HPP_
#define KAMAYAN_UTILS_ROBUST_HPP_

#include <limits>
#include <type_traits>

#include <Kokkos_Macros.hpp>

namespace kamayan::utils {
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T Tiny(const T &) {
  return 10.0 * std::numeric_limits<T>::min();
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T Eps(const T &) {
  return 10.0 * std::numeric_limits<T>::epsilon();
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION int Sgn(const T &val) {
  if constexpr (std::is_unsigned_v<T>) {
    return 1;
  } else {
    return (T(0) <= val) - (val < T(0));
  }
}

template <typename A, typename B>
KOKKOS_FORCEINLINE_FUNCTION auto Ratio(const A &a, const B &b) {
  return a / (b + Sgn(b) * Tiny(b));
}
}  // namespace kamayan::utils
#endif  // KAMAYAN_UTILS_ROBUST_HPP_
