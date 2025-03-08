#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_

#include <memory>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {
POLYMORPHIC_PARM(eosMode, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather, pres,
                 none);

namespace eos {
std::shared_ptr<KamayanUnit> ProcessUnit();

void Setup(runtime_parameters::RuntimeParameters *rps);

std::shared_ptr<StateDescriptor>
Initialize(const runtime_parameters::RuntimeParameters *rps);

void EosWrapped();
void EosWrappedBlk();
}  // namespace eos

}  // namespace kamayan

#endif  // EOS_EOS_HPP_
