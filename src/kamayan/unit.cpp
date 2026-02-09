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
#include "kamayan/callback_dag.hpp"
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
    // Pass nullptr instead of shared_from_this() to defer params_ assignment
    // This prevents AddParam() from running during SetupParams, avoiding duplicates
    unit_data_.emplace(block, UnitData(block, runtime_parameters_, config_, nullptr));
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
    ud.Initialize(pkg);  // Sets params_ AND calls AddParam for all parameters
  }
}

void KamayanUnit::SetUnits(std::shared_ptr<const UnitCollection> units) {
  units_ = units;
}

const KamayanUnit &KamayanUnit::GetUnit(const std::string &name) const {
  auto units = units_.lock();
  PARTHENON_REQUIRE_THROWS(units != nullptr,
                           "UnitCollection has been destroyed or not set.");
  return *units->Get(name);
}

std::shared_ptr<const KamayanUnit>
KamayanUnit::GetUnitPtr(const std::string &name) const {
  auto units = units_.lock();
  PARTHENON_REQUIRE_THROWS(units != nullptr,
                           "UnitCollection has been destroyed or not set.");
  return units->Get(name);
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
  unit_collection["Driver"] = driver::ProcessUnit();
  unit_collection["Eos"] = eos::ProcessUnit();
  unit_collection["Grid"] = grid::ProcessUnit();
  unit_collection["Physics"] = physics::ProcessUnit();
  unit_collection["Hydro"] = hydro::ProcessUnit();

  // Legacy ordering lists - will be removed once all callbacks use DAG
  // Dependencies are now expressed in unit ProcessUnit() functions via .Register()
  unit_collection.rk_fluxes = {"Hydro"};

  return unit_collection;
}

std::stringstream RuntimeParameterDocs(KamayanUnit *unit, ParameterInput *pin) {
  std::stringstream ss;
  if (unit->SetupParams.IsRegistered()) {
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

// Template method implementations for UnitCollection DAG functionality

}  // namespace kamayan
