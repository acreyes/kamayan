#include "unit.hpp"

#include <list>
#include <memory>
#include <string>

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
  unit_collection.rk_fluxes = {"hydro"};

  // make sure that eos always is applied last when preparing our primitive vars!
  std::list<std::string> prepare_prim;
  for (const auto &unit : unit_collection) {
    if (unit.second->PreparePrimitive != nullptr) {
      prepare_prim.push_front(unit.first);
    }
  }
  prepare_prim.sort([](const std::string &first, const std::string &second) {
    return second == "eos";
  });
  for (const auto &key : prepare_prim) {
  }
  unit_collection.prepare_prim = prepare_prim;

  return unit_collection;
}
}  // namespace kamayan
