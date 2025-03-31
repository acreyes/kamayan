#include "unit.hpp"

#include <memory>

#include "grid/grid.hpp"
#include "physics/eos/eos.hpp"
#include "physics/physics.hpp"

namespace kamayan {
UnitCollection ProcessUnits() {
  UnitCollection unit_collection;
  unit_collection["eos"] = eos::ProcessUnit();
  unit_collection["grid"] = grid::ProcessUnit();
  unit_collection["physics"] = physics::ProcessUnit();

  // list out order of units that should be called during
  // RK stages & for operator splitting
  // unit_collection.rk_stage = {"hydro", "heat_exchange", "extended_mhd", "viscosity"};

  return unit_collection;
}
}  // namespace kamayan
