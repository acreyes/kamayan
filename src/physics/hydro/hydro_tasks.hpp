#ifndef PHYSICS_HYDRO_HYDRO_TASKS_HPP_
#define PHYSICS_HYDRO_HYDRO_TASKS_HPP_

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"

namespace kamayan::hydro {
TaskID CalculateFluxes(TaskID prev, TaskList &tl, MeshData *md);
}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_HYDRO_TASKS_HPP_
