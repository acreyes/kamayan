#ifndef EOS_EOS_TYPES_HPP_
#define EOS_EOS_TYPES_HPP_
#include <concepts>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"

namespace kamayan {
// recognized eos options
POLYMORPHIC_PARM(eosMode, ener, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather,
                 pres, none);
POLYMORPHIC_PARM(eosType, Single, MultiType);

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
constexpr bool is_one_of(const T &val, Args &&...args) {
  return (... || (val == args));
}
template <eosMode mode>
concept oneT = is_one_of(mode, eosMode::ener, eosMode::temp, eosMode::pres);

template <eosMode mode>
concept threeT = is_one_of(mode, eosMode::temp_equi, eosMode::temp_gather, eosMode::ei,
                           eosMode::ei_scatter, eosMode::ei_gather);

template <eosMode>
struct EosVars {};

template <eosMode mode>
requires(oneT<mode>)
struct EosVars<mode> {
  using types = TypeList<DENS, TEMP, EINT, PRES, GAMC, GAME>;
};
}  // namespace eos
}  // namespace kamayan

#endif  // EOS_EOS_TYPES_HPP_
