#ifndef UTILS_TYPE_LIST_HPP_
#define UTILS_TYPE_LIST_HPP_

#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "type_abstractions.hpp"

namespace kamayan {
template <std::size_t>
struct getTL_impl;

template <typename... Ts>
struct TypeList {
  using types = std::tuple<Ts...>;

  static constexpr std::size_t n_types{sizeof...(Ts)};

  template <std::size_t I>
  using type = typename std::tuple_element<I, types>::type;

  template <std::size_t i, typename T, typename T0, typename... Tn>
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t Idx_impl() {
    static_assert(i < n_types || always_false<T, TypeList<Ts...>>,
                  "type T not in TypeList");
    if constexpr (std::is_same_v<T, T0>) {
      return i;
    } else if constexpr (i + 1 < n_types) {
      return Idx_impl<i + 1, T, Tn...>();
    }
    return n_types;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t Idx() {
    return Idx_impl<0, T, Ts...>();
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t Idx(T) {
    return Idx<T>();
  }

  template <typename... Vs>
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<std::size_t, sizeof...(Vs)>
  GetIdxArr(TypeList<Vs...>) {
    return {Idx<Vs>()...};
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION static constexpr bool Contains() {
    return Idx<T>() < n_types;
  }

  // this is useful when you have a parameter pack of arguments that corresponds to the
  // types in a TypeList and you want to pluck out the argument that corresponds to a
  // specific type
  template <typename Var, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION static const auto &get(Var, Args &&...args) {
    return getTL_impl<Idx<Var>()>::getImpl(std::forward<Args>(args)...);
  }
};

template <std::size_t index>
struct getTL_impl {
  template <typename T, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION static const auto &getImpl(T &&arg, Args &&...args) {
    return getTL_impl<index - 1>::getImpl(std::forward<Args>(args)...);
  }
};

template <>
struct getTL_impl<0> {
  template <typename T, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION static const auto &getImpl(T &&arg, Args &&...args) {
    return std::forward<T>(arg);
  }
};

template <typename TL, typename T>
requires(TemplateSpecialization<TL, TypeList>)
KOKKOS_INLINE_FUNCTION constexpr std::size_t Idx() {
  constexpr std::size_t idx = TL::template Idx<T>();
  return idx;
}

template <typename... Ts>
constexpr int SizeOfList(TypeList<Ts...>) {
  return sizeof...(Ts);
}

template <typename...>
struct ConcatTypeLists {};

template <typename... Ts>
struct ConcatTypeLists<TypeList<Ts...>> {
  using type = TypeList<Ts...>;
};

template <typename... Ts, typename... Us>
struct ConcatTypeLists<TypeList<Ts...>, TypeList<Us...>> {
  using type = TypeList<Ts..., Us...>;
};

template <typename... Ts, typename... TLs>
struct ConcatTypeLists<TypeList<Ts...>, TLs...> {
  using next = ConcatTypeLists<TLs...>;
  using type = typename ConcatTypeLists<TypeList<Ts...>, typename next::type>::type;
};

template <typename... Ts>
using ConcatTypeLists_t = typename ConcatTypeLists<Ts...>::type;

template <std::size_t, typename>
struct SplitTypeList {};

template <typename... Ts>
struct SplitTypeList<0, TypeList<Ts...>> {
  using first = TypeList<>;
  using second = TypeList<Ts...>;
};

template <std::size_t idx, typename T, typename... Ts>
requires(idx > 0)
struct SplitTypeList<idx, TypeList<T, Ts...>> {
  using NextSplit = SplitTypeList<idx - 1, TypeList<Ts...>>;
  using first = ConcatTypeLists<TypeList<T>, typename NextSplit::first>::type;
  using second = NextSplit::second;
};

}  // namespace kamayan

#endif  // UTILS_TYPE_LIST_HPP_
