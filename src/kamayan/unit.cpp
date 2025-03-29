#include "unit.hpp"

#include <memory>

#include "physics/eos/eos.hpp"
#include "physics/physics.hpp"

namespace kamayan {
UnitCollection ProcessUnits() {
  UnitCollection unit_collection;
  unit_collection["physics"] = physics::ProcessUnit();
  unit_collection["eos"] = eos::ProcessUnit();

  // list out order of units that should be called during
  // RK stages & for operator splitting
  // unit_collection.rk_stage = {"hydro"};

  return unit_collection;
}
}  // namespace kamayan
