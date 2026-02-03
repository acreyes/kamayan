#ifndef KAMAYAN_UNIT_HPP_
#define KAMAYAN_UNIT_HPP_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit_data.hpp"

namespace kamayan {

struct UnitCollection;
struct KamayanUnit;

}  // namespace kamayan

namespace kamayan {

struct KamayanUnit : public StateDescriptor,
                     public std::enable_shared_from_this<KamayanUnit> {
  explicit KamayanUnit(std::string name) : StateDescriptor(name), name_(name) {}

  ~KamayanUnit() = default;

  // Setup is called to add options into the kamayan configuration and to register
  // runtime parameters owned by the unit
  std::function<void(KamayanUnit *unit)> SetupParams = nullptr;

  // Initialize is responsible for setting up the parthenon StateDescriptor, registering
  // params , adding fields owned by the unit & registering any callbacks known to
  // parthenon
  std::function<void(KamayanUnit *unit)> InitializeData = nullptr;

  // Used as a callback during problem generation on the mesh
  std::function<void(MeshBlock *)> ProblemGeneratorMeshBlock = nullptr;

  // makes sure the conserved variables are ready before applying dudt
  std::function<TaskStatus(MeshData *md)> PrepareConserved = nullptr;

  // make sure primitive variables are ready after updating conserved
  std::function<TaskStatus(MeshData *md)> PreparePrimitive = nullptr;

  // Accumulates the fluxes in md, and the driver will handle the flux
  // correction and dudt
  std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md)> AddFluxTasks = nullptr;

  // These tasks get added to the tasklist that accumulate dudt for this unit based
  // on the current state in md, returning the TaskID of the final task for a single
  // stage in the multi-stage driver
  std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt)>
      AddTasksOneStep = nullptr;

  // These tasks are used to advance md by dt as one of the operators in the
  // operator splitting
  std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, const Real &dt)>
      AddTasksSplit = nullptr;

  const std::string Name() const { return name_; }

  // get a reference to the UnitData configured for a particular block
  const UnitData &Data(const std::string &key) const;
  UnitData &AddData(const std::string &block);
  bool HasData(const std::string &block) const;
  auto &AllData() { return unit_data_; }

  std::shared_ptr<Config> Configuration() { return config_; }
  std::shared_ptr<runtime_parameters::RuntimeParameters> RuntimeParameters() {
    return runtime_parameters_;
  }

  void InitResources(std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
                     std::shared_ptr<Config> cfg);
  void InitializePackage(std::shared_ptr<StateDescriptor> pkg);

  void SetUnits(std::shared_ptr<const UnitCollection> units);
  const KamayanUnit &GetUnit(const std::string &name) const;
  std::shared_ptr<const KamayanUnit> GetUnitPtr(const std::string &name) const;

  static std::shared_ptr<KamayanUnit> GetFromMesh(MeshData *md, const std::string &name);

 private:
  std::string name_;
  std::map<std::string, UnitData> unit_data_;
  std::shared_ptr<Config> config_;
  std::shared_ptr<runtime_parameters::RuntimeParameters> runtime_parameters_;
  std::weak_ptr<const UnitCollection> units_;
};
// --8<-- [end:unit]

struct UnitCollection {
  std::list<std::string> rk_fluxes, rk_stage, prepare_prim, operator_split;

  UnitCollection(const UnitCollection &uc)
      : units(uc.units), rk_fluxes(uc.rk_fluxes), rk_stage(uc.rk_stage),
        prepare_prim(uc.prepare_prim), operator_split(uc.operator_split) {}
  UnitCollection() {}

  std::shared_ptr<KamayanUnit> Get(const std::string &key) const { return units.at(key); }
  std::shared_ptr<KamayanUnit> &operator[](const std::string &key) { return units[key]; }

  auto GetMap() const { return &units; }

  // iterator goes over all registered units
  auto begin() const { return units.begin(); }
  auto end() const { return units.end(); }

  void AddTasks(std::list<std::string> unit_list,
                std::function<void(KamayanUnit *)> function) const;

  void Add(std::shared_ptr<KamayanUnit> kamayan_unit);

 private:
  std::map<std::string, std::shared_ptr<KamayanUnit>> units;
};

// gather up all the units in kamayan
UnitCollection ProcessUnits();

// write out all the doc strings for runtime parameters
// registered by a given unit
std::stringstream RuntimeParameterDocs(KamayanUnit *unit, ParameterInput *pin);

}  // namespace kamayan

#endif  // KAMAYAN_UNIT_HPP_
