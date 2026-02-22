#ifndef PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_SINGULARITY_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_SINGULARITY_HPP_

#include <singularity-eos/base/constants.hpp>

#include "physics/material_properties/eos/eos_types.hpp"

namespace kamayan::eos {

// need an eos type

// these map the EosMode enum onto the
// output bits used by singularityEoS' fillEoS methods
template <EosMode>
struct SingularityEosFill {};

template <>
struct SingularityEosFill<EosMode::ener> {
  static constexpr int64_t output =
      (singularity::thermalqs::temperature | singularity::thermalqs::pressure |
       singularity::thermalqs::bulk_modulus);
};

template <>
struct SingularityEosFill<EosMode::temp> {
  static constexpr int64_t output =
      (singularity::thermalqs::specific_internal_energy |
       singularity::thermalqs::pressure | singularity::thermalqs::bulk_modulus);
};

template <>
struct SingularityEosFill<EosMode::pres> {
  static constexpr int64_t output = (singularity::thermalqs::temperature |
                                     singularity::thermalqs::specific_internal_energy |
                                     singularity::thermalqs::bulk_modulus);
};
// need a specialization for multitype to call on each component
// need a specialization for threeT to call on each temperature component

}  // namespace kamayan::eos

#endif  // PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_SINGULARITY_HPP_
