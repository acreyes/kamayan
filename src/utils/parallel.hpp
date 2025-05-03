#ifndef UTILS_PARALLEL_HPP_
#define UTILS_PARALLEL_HPP_

#include <string>
#include <utility>

#include <Kokkos_Macros.hpp>

#include <kokkos_abstraction.hpp>

namespace kamayan {
// use this to work around cuda not letting you lambda capture first
// in a if constexpr block
template <typename... Args>
KOKKOS_INLINE_FUNCTION void capture(Args &&...args) {}

template <typename... Args>
void par_for(Args &&...args) {
  parthenon::par_for(std::forward<Args>(args)...);
}

// parthenon only supports parallel reductions through the LoopPatternMDRange
// pattern. At least until this PR gets merged in:
// https://github.com/parthenon-hpc-lab/parthenon/pull/1142
template <typename... Args>
void par_reduce(const std::string &label, Args &&...args) {
  parthenon::par_reduce(parthenon::LoopPatternMDRange(), label, parthenon::DevExecSpace(),
                        std::forward<Args>(args)...);
}
}  // namespace kamayan
#endif  // UTILS_PARALLEL_HPP_
