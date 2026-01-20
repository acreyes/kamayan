#include "unit.hpp"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "physics/eos/eos.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/physics.hpp"

namespace kamayan {

const UnitData &KamayanUnit::Data(const std::string &key) const {
  return unit_data_.at(key);
}

UnitData &KamayanUnit::AddData(const std::string &block) {
  if (unit_data_.count(block) == 0) {
    unit_data_.emplace(block,
                       UnitData(block, runtime_parameters_, config_, shared_from_this()));
  }
  return unit_data_.at(block);
}

bool KamayanUnit::HasData(const std::string &block) const {
  return unit_data_.count(block) > 0;
}

void KamayanUnit::InitResources(
    std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
    std::shared_ptr<Config> cfg) {
  runtime_parameters_ = rps;
  config_ = cfg;
}

void KamayanUnit::InitializePackage(std::shared_ptr<StateDescriptor> pkg) {
  for (auto &[name, ud] : unit_data_) {
    ud.SetPackage(pkg);
  }
}

void KamayanUnit::SetUnits(std::shared_ptr<const UnitCollection> units) {
  units_ = units;
}

const KamayanUnit &KamayanUnit::GetUnit(const std::string &name) const {
  PARTHENON_REQUIRE_THROWS(units_ != nullptr,
                           "UnitCollection not set. Call SetUnits() before GetUnit().");
  return *units_->Get(name);
}

std::shared_ptr<const KamayanUnit>
KamayanUnit::GetUnitPtr(const std::string &name) const {
  PARTHENON_REQUIRE_THROWS(
      units_ != nullptr, "UnitCollection not set. Call SetUnits() before GetUnitPtr().");
  return units_->Get(name);
}

std::shared_ptr<KamayanUnit> KamayanUnit::GetFromMesh(MeshData *md,
                                                      const std::string &name) {
  auto pkg = md->GetMeshPointer()->packages.Get(name);

  PARTHENON_REQUIRE_THROWS(typeid(*pkg) == typeid(KamayanUnit),
                           "Package '" + name + "' is not a KamayanUnit");

  return std::static_pointer_cast<KamayanUnit>(pkg);
}

void UnitCollection::AddTasks(std::list<std::string> unit_list,
                              std::function<void(KamayanUnit *)> function) const {
  for (const auto &unit : units) {
    auto found = std::find(unit_list.begin(), unit_list.end(), unit.first);
    if (found == unit_list.end()) function(unit.second.get());
  }

  for (const auto &key : unit_list) {
    function(Get(key).get());
  }
}

void UnitCollection::Add(std::shared_ptr<KamayanUnit> kamayan_unit) {
  units[kamayan_unit->Name()] = kamayan_unit;
}

UnitCollection ProcessUnits() {
  UnitCollection unit_collection;
  unit_collection["driver"] = driver::ProcessUnit();
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

std::stringstream RuntimeParameterDocs(KamayanUnit *unit, ParameterInput *pin) {
  std::stringstream ss;
  if (unit->SetupParams != nullptr) {
    auto cfg = std::make_shared<Config>();
    auto rps = std::make_shared<runtime_parameters::RuntimeParameters>(pin);
    unit->InitResources(rps, cfg);
    for (auto &[name, ud] : unit->AllData()) {
      ud.Setup(rps, cfg);
    }

    std::map<std::string, std::list<std::string>> block_keys;
    for (const auto &parm : rps->parms) {
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
        ss << std::visit([](auto &parm) { return parm.DocString(); }, rps->parms.at(key));
      }
    }
  }

  return ss;
}
}  // namespace kamayan
