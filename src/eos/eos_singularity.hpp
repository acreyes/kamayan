#ifndef EOS_EOS_SINGULARITY_HPP_
#define EOS_EOS_SINGULARITY_HPP_

#include <singularity-eos/eos/eos.hpp>

#include "eos/eos_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "singularity-eos/base/constants.hpp"

namespace kamayan::eos {

template <eosMode>
struct SingularityEosFill {};

template <>
struct SingularityEosFill<eosMode::ener> {
  static constexpr int64_t output =
      (singularity::thermalqs::temperature | singularity::thermalqs::pressure |
       singularity::thermalqs::bulk_modulus);
};

template <>
struct SingularityEosFill<eosMode::temp> {
  static constexpr int64_t output =
      (singularity::thermalqs::specific_internal_energy |
       singularity::thermalqs::pressure | singularity::thermalqs::bulk_modulus);
};

template <>
struct SingularityEosFill<eosMode::pres> {
  static constexpr int64_t output = (singularity::thermalqs::temperature |
                                     singularity::thermalqs::specific_internal_energy |
                                     singularity::thermalqs::bulk_modulus);
};

template <eosMode mode, template <typename...> typename Container, typename... Ts,
          typename Lambda = NullIndexer>
requires(AccessorLike<Lambda>)
void EosSingle(Container<Ts...> &indexer, singularity::EOS eos,
               Lambda lambda = Lambda()) {
  constexpr auto output = SingularityEosFill<mode>::output;
  Real cv;
  eos.FillEos(indexer(DENS()), indexer(TEMP()), indexer(EINT()), indexer(PRES()), cv,
              indexer(GAMC()), output, lambda);
}
// need a specialization for multitype to call on each component
// need a specialization for threeT to call on each temperature component

}  // namespace kamayan::eos

#endif  // EOS_EOS_SINGULARITY_HPP_
