#include "unit.hpp"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/runtime_parameters.hpp"
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

  // --8<-- [start:rk_flux]
  // list out order of units that should be called during
  // RK stages & for operator splitting
  unit_collection.rk_fluxes = {"hydro"};
  // --8<-- [end:rk_flux]

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

std::stringstream RuntimeParameterDocs(const KamayanUnit *unit) {
  std::stringstream ss;
  if (unit->Setup != nullptr) {
    Config cfg;
    ParameterInput pin;
    runtime_parameters::RuntimeParameters rps(&pin);
    unit->Setup(&cfg, &rps);

    std::map<std::string, std::list<std::string>> block_keys;
    for (const auto &parm : rps.parms) {
      auto block = std::visit([](auto &p) { return p.block; }, parm.second);
      auto key = std::visit([](auto &p) { return p.key; }, parm.second);
      block_keys[block].push_back(block + key);
    }
    std::list<std::string> blocks;
    for (auto &bk : block_keys) {
      blocks.push_back(bk.first);
    }
    blocks.sort();

    ss << "| Paramter | Type | Default | Allowed | Description |\n";
    ss << "| -------  | ---- | ------  | ------- | ----------- |\n";
    for (const auto &block : blocks) {
      ss << "**<" << block << "\\>**\n";
      for (const auto &key : block_keys[block]) {
        ss << std::visit([](auto &parm) { return parm.DocString(); }, rps.parms.at(key));
      }
    }
  }

  return ss;
}
}  // namespace kamayan
