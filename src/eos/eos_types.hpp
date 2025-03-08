#ifndef EOS_EOS_TYPES_HPP_
#define EOS_EOS_TYPES_HPP_

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"

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

class NullIndexer {
  Real *operator[](int i) { return nullptr; }
};
}  // namespace eos
}  // namespace kamayan

#endif  // EOS_EOS_TYPES_HPP_
