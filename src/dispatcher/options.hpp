#ifndef DISPATCHER_OPTIONS_HPP_
#define DISPATCHER_OPTIONS_HPP_

#include <array>
#include <type_traits>

#include "utils/type_list.hpp"

namespace kamayan {
template <typename enum_opt>
struct PolyOpt : std::false_type {};

template <typename enum_opt>
struct OptInfo : std::false_type {};

template <typename enum_opt>
concept poly_opt = PolyOpt<enum_opt>::value;

template <typename enum_opt>
concept comptime_poly_opt = poly_opt<enum_opt> && OptInfo<enum_opt>::isdef;

template <typename enum_opt>
concept default_poly_opt = poly_opt<enum_opt> && !OptInfo<enum_opt>::isdef;

template <typename, auto...>
struct OptList {};

template <default_poly_opt enum_opt, auto enum_v, auto... enum_vs>
struct OptList<enum_opt, enum_v, enum_vs...> {
  using type = enum_opt;
  static constexpr auto value =
      std::array<enum_opt, 1 + sizeof...(enum_vs)>{enum_v, enum_vs...};
};

template <comptime_poly_opt enum_opt, auto enum_v, auto... enum_vs>
struct OptList<enum_opt, enum_v, enum_vs...> {
  using type = enum_opt;
  static constexpr auto value = OptInfo<enum_opt>::ParmList();
};

template <typename OL>
concept opt_list = requires {
  typename OL::type;
  OL::value;
};

template <opt_list... OLs>
struct OptTypeList {
  using type = TypeList<OLs...>;
};

template <typename opt, typename parm>
concept parm_opt =
    requires { opt::value; } && std::is_same_v<base_dtype<opt::value>, parm>;

} // namespace kamayan

#endif // DISPATCHER_OPTIONS_HPP_
