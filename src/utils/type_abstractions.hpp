#ifndef UTILS_TYPE_ABSTRACTIONS_HPP_
#define UTILS_TYPE_ABSTRACTIONS_HPP_
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace kamayan {
template <typename T>
using base_type = std::remove_cvref_t<T>;

template <auto V>
using base_dtype = base_type<decltype(V)>;

// check if a type is a specialization of a desired type
template <typename T, template <typename...> typename Template>
concept TemplateSpecialization = requires(base_type<T> t) {
  // check that a functor that is called with a "T" can
  // be matched to our Template type
  []<typename... Ts>(Template<Ts...> &) {}(t);
};

template <typename T, template <auto...> typename Template>
concept NonTypeTemplateSpecialization = requires(base_type<T> t) {
  // check that a functor that is called with a "T" can
  // be matched to our Template type
  []<auto... NTs>(Template<NTs...> &) {}(t);
};

// This is a class template that is required for doing something like static_assert(false)
// in constexpr if blocks. Actually writing static_assert(false) will always cause a
// compilation error, even if it is an unchosen constexpr if block. This is fixed in C++23
// I think.
template <class...>
constexpr std::false_type always_false{};

template <typename T, typename... Args>
requires(std::is_same_v<T, Args> && ...)
constexpr bool is_one_of(const T &val, Args &&...args) {
  return (... || (val == args));
}
template <typename T, std::size_t N>
constexpr bool is_one_of(const T &val, Kokkos::Array<T, N> values) {
  for (auto &v : values) {
    if (val == v) return true;
  }
  return false;
}

}  // namespace kamayan

#endif  // UTILS_TYPE_ABSTRACTIONS_HPP_
