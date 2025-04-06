#ifndef KAMAYAN_UNIT_HPP_
#define KAMAYAN_UNIT_HPP_
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "utils/map_list.hpp"

namespace kamayan {
struct KamayanUnit {
  // Setup is called to add options into the kamayan configuration and to register
  // runtime parameters owned by the unit
  std::function<void(Config *, runtime_parameters::RuntimeParameters *)> Setup = nullptr;

  // Initialize is responsible for setting up the parthenon StateDescriptor, registering
  // params , adding fields owned by the unit & registering any callbacks known to
  // parthenon
  std::function<std::shared_ptr<StateDescriptor>(
      const Config *, const runtime_parameters::RuntimeParameters *)>
      Initialize = nullptr;

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
};

struct UnitCollection {
  using MapList_t = MapList<std::string, std::shared_ptr<KamayanUnit>>;
  MapList_t rk_fluxes, rk_stage, prepare_prim, operator_split;

  UnitCollection(UnitCollection &uc)
      : units(uc.units), rk_fluxes(uc.rk_fluxes.Keys(), units),
        rk_stage(uc.rk_stage.Keys(), units), prepare_prim(uc.prepare_prim.Keys(), units),
        operator_split(uc.operator_split.Keys(), units) {}
  UnitCollection()
      : rk_fluxes(units), rk_stage(units), prepare_prim(units), operator_split(units) {}

  std::shared_ptr<KamayanUnit> &operator[](const std::string &key) { return units[key]; }

  auto GetMap() const { return &units; }

  // iterator goes over all registered units
  auto begin() const { return units.begin(); }
  auto end() const { return units.end(); }

 private:
  std::map<std::string, std::shared_ptr<KamayanUnit>> units;
};

// gather up all the units in kamayan
UnitCollection ProcessUnits();

}  // namespace kamayan

#endif  // KAMAYAN_UNIT_HPP_
