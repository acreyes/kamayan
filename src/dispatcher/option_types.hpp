#ifndef DISPATCHER_OPTION_TYPES_HPP_
#define DISPATCHER_OPTION_TYPES_HPP_
#include <type_traits>
namespace kamayan {
template <typename enum_opt>
struct PolyOpt_t : std::false_type {};

template <typename enum_opt>
struct OptInfo : std::false_type {};

template <typename enum_opt>
concept PolyOpt = PolyOpt_t<enum_opt>::value;
}  // namespace kamayan
#endif  // DISPATCHER_OPTION_TYPES_HPP_
