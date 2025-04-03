#include "unit.hpp"

#include <memory>

#include "grid/grid.hpp"
#include "physics/eos/eos.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/physics.hpp"

namespace kamayan {
UnitCollection ProcessUnits() {
  UnitCollection unit_collection;
  unit_collection["eos"] = eos::ProcessUnit();
  unit_collection["grid"] = grid::ProcessUnit();
  unit_collection["physics"] = physics::ProcessUnit();
  unit_collection["hydro"] = hydro::ProcessUnit();

  // list out order of units that should be called during
  // RK stages & for operator splitting
  unit_collection.rk_stage = {"hydro"};

  return unit_collection;
}
}  // namespace kamayan
