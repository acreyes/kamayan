#ifndef UTILS_TYPE_ABSTRACTIONS_HPP_
#define UTILS_TYPE_ABSTRACTIONS_HPP_

#include <type_traits>

namespace kamayan {
// c++-20 has std:remove_cvref_t that does this same thing
template <typename T>
using base_type = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

template <auto V>
using base_dtype = base_type<decltype(V)>;

// check if a type is a specialization of a desired type
template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

// This is a class template that is required for doing something like static_assert(false)
// in constexpr if blocks. Actually writing static_assert(false) will always cause a
// compilation error, even if it is an unchosen constexpr if block. This is fixed in C++23
// I think.
template <class...>
constexpr std::false_type always_false{};

} // namespace kamayan

#endif // UTILS_TYPE_ABSTRACTIONS_HPP_
