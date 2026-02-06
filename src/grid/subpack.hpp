#ifndef GRID_SUBPACK_HPP_
#define GRID_SUBPACK_HPP_

#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan {
enum class Axis { KAXIS = 0, JAXIS = 1, IAXIS = 2 };

template <typename PackType, Axis... axes>
requires(TemplateSpecialization<PackType, SparsePack>)
struct SubPack_impl {
  KOKKOS_INLINE_FUNCTION SubPack_impl(PackType &pack, const int &b, const int &k,
                                      const int &j, const int &i)
      : pack_(pack), b_(b), k_(k), j_(j), i_(i) {}

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &operator()(const Var_t &var) const {
    return pack_(b_, var, k_, j_, i_);
  }

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, const Var_t &var) const {
    return pack_(b_, te, var, k_, j_, i_);
  }

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, const Var_t &var) const {
    return pack_.flux(b_, te, var, k_, j_, i_);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetSize(const V &var) const {
    return pack_.GetSize(b_, var);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION auto Indexer() const {
    return [=, this](const std::size_t idx) { return pack_(b_, V(idx), k_, j_, i_); };
  }

 private:
  PackType &pack_;
  const int b_, k_, j_, i_;
};

template <typename PackType, Axis... axes>
requires(TemplateSpecialization<PackType, SparsePack>)
struct StencilSubPack_impl {
  KOKKOS_INLINE_FUNCTION StencilSubPack_impl(PackType &pack, const int &b, const int &k,
                                             const int &j, const int &i)
      : pack_(pack), b_(b), kji_({k, j, i}) {}

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(const Var_t &var, Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_(b_, var, kji[0], kji[1], kji[2]);
  }

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, const Var_t &var,
                                          Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_(b_, te, var, kji[0], kji[1], kji[2]);
  }

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, const Var_t &var,
                                    Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_.flux(b_, te, var, kji[0], kji[1], kji[2]);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetSize(const V &var) const {
    return pack_.GetSize(b_, var);
  }

 private:
  const PackType &pack_;
  const Kokkos::Array<int, 3> kji_;
  const int b_;
};

template <typename Var_t, typename PackType, Axis... axes>
requires(TemplateSpecialization<PackType, SparsePack>)
struct VarStencilSubPack_impl {
  KOKKOS_INLINE_FUNCTION VarStencilSubPack_impl(PackType &pack, const int &b,
                                                const Var_t &var, const int &k,
                                                const int &j, const int &i)
      : pack_(pack), b_(b), var_(var), kji_({k, j, i}) {}

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_(b_, var_, kji[0], kji[1], kji[2]);
  }

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_(b_, te, var_, kji[0], kji[1], kji[2]);
  }

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, Is &&...idxs) {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[static_cast<int>(axes)] += idxs; }(), ...);
    return pack_.flux(b_, te, var_, kji[0], kji[1], kji[2]);
  }

 private:
  const PackType &pack_;
  const Kokkos::Array<int, 3> kji_;
  const Var_t var_;
  const int b_;
};

template <Axis axis, Axis... axes, typename Var_t, typename PackType>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const Var_t &var,
                                    const int &k, const int &j, const int &i) {
  return VarStencilSubPack_impl<Var_t, PackType, axis, axes...>(pack, b, var, k, j, i);
}

template <Axis axis, Axis... axes, typename PackType>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const int &k,
                                    const int &j, const int &i) {
  return StencilSubPack_impl<PackType, axis, axes...>(pack, b, k, j, i);
}

template <typename PackType>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const int &k,
                                    const int &j, const int &i) {
  return SubPack_impl<PackType>(pack, b, k, j, i);
}

}  // namespace kamayan
#endif  // GRID_SUBPACK_HPP_
