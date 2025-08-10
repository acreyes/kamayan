#ifndef PHYSICS_HYDRO_HYDRO_HPP_
#define PHYSICS_HYDRO_HYDRO_HPP_

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"

namespace kamayan::hydro {
std::shared_ptr<KamayanUnit> ProcessUnit();

void AddParams(KamayanUnit *unit);
void InitializeData(StateDescriptor *hydro_pkg, Config *cfg);

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md);
TaskID AddTasksOneStep(TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt);
Real EstimateTimeStepMesh(MeshData *md);

TaskStatus FillDerived(MeshData *md);

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_HYDRO_HPP_
