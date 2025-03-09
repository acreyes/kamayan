#ifndef EOS_EOS_TYPES_HPP_
#define EOS_EOS_TYPES_HPP_

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"

namespace kamayan {
// recognized eos options
POLYMORPHIC_PARM(eosMode, ener, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather,
                 pres, none);
POLYMORPHIC_PARM(eosType, oneT, threeT, multiType);

namespace eos {
template <typename T>
concept AccessorLike = requires(T obj) {
  { obj.operator[](int()) } -> std::convertible_to<Real *>;
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
template <eosType>
struct EosVars {};

template <>
struct EosVars<eosType::oneT> {
  using types = TypeList<DENS, TEMP, EINT, PRES, GAMC, GAME>;
};
}  // namespace eos
}  // namespace kamayan

#endif  // EOS_EOS_TYPES_HPP_
