#ifndef PHYSICS_EOS_EOS_TYPES_HPP_
#define PHYSICS_EOS_EOS_TYPES_HPP_
#include <concepts>
#include <type_traits>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "physics/physics_types.hpp"

namespace kamayan {
// recognized eos options
POLYMORPHIC_PARM(EosMode, ener, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather,
                 pres, none);
POLYMORPHIC_PARM(EosType, Single, MultiType);
POLYMORPHIC_PARM(EosModel, gamma, tabulated, multitype);
POLYMORPHIC_PARM(EosComponent, oneT, ele, ion)

namespace eos {
template <typename T>
concept AccessorLike = requires(T obj) {
  { &obj[int()] } -> std::convertible_to<Real *>;
};

struct NullIndexer {
  Real *operator[](int i) { return nullptr; }
};

// probably a better pattern would be to compose these from individual traits
// to allow more extensibility in the model
// * maybe depends on
//    * 1T vs 3T,
//    * single species vs multi
//    * vof or some other representation
template <typename T, typename... Args>
requires(std::is_same_v<T, Args> && ...)
constexpr bool is_one_of(const T &val, Args &&...args) {
  return (... || (val == args));
}
template <typename T, std::size_t N>
constexpr bool is_onf_of(const T &val, Kokkos::Array<T, N> values) {
  for (auto &v : values) {
    if (val == v) return true;
  }
  return false;
}

template <EosMode mode>
concept oneT = is_one_of(mode, EosMode::ener, EosMode::temp, EosMode::pres);

template <EosMode mode>
concept threeT = is_one_of(mode, EosMode::temp_equi, EosMode::temp_gather, EosMode::ei,
                           EosMode::ei_scatter, EosMode::ei_gather);

template <EosComponent>
struct EosVars {};

template <>
struct EosVars<EosComponent::oneT> {
  using temp = TEMP;
  using eint = EINT;
  using pres = PRES;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
};

template <>
struct EosVars<EosComponent::ion> {
  using temp = TION;
  using eint = EION;
  using pres = PION;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
};

template <>
struct EosVars<EosComponent::ele> {
  using temp = TELE;
  using eint = EELE;
  using pres = PELE;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
};

}  // namespace eos
}  // namespace kamayan

#endif  // PHYSICS_EOS_EOS_TYPES_HPP_
