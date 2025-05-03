#ifndef UTILS_PARALLEL_HPP_
#define UTILS_PARALLEL_HPP_

#include <Kokkos_Macros.hpp>
namespace kamayan {
// use this to work around cuda not letting you lambda capture first
// in a if constexpr block
template <typename... Args>
KOKKOS_INLINE_FUNCTION void capture(Args &&...args) {}
}  // namespace kamayan
#endif  // UTILS_PARALLEL_HPP_
