#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"

namespace kamayan::eos {

std::shared_ptr<KamayanUnit> ProcessUnit();

void Setup(runtime_parameters::RuntimeParameters *rps);

std::shared_ptr<StateDescriptor>
Initialize(const runtime_parameters::RuntimeParameters *rps);

void EosWrapped();
void EosWrappedBlk();
}  // namespace kamayan::eos

#endif  // EOS_EOS_HPP_
