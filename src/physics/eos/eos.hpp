#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "physics/eos/eos_types.hpp"

namespace kamayan::eos {

std::shared_ptr<KamayanUnit> ProcessUnit();

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps);

std::shared_ptr<StateDescriptor>
Initialize(const runtime_parameters::RuntimeParameters *rps);

TaskStatus EosWrapped(MeshData *md, eosMode mode);
TaskStatus EosWrappedBlk(MeshBlock *mb, eosMode mode);
}  // namespace kamayan::eos

#endif  // EOS_EOS_HPP_
