#include "grid/pybind/grid_bindings.hpp"

#include "grid/grid_types.hpp"
#include "interface/mesh_data.hpp"
#include "tasks/tasks.hpp"

#include "kamayan/pybind/kamayan_nanobind.h"

namespace kamayan {

void parthenon_module(nanobind::module_ &m) {
  nanobind::class_<parthenon::TaskID>(m, "TaskID");
  nanobind::class_<parthenon::TaskList>(m, "TaskList");
  nanobind::class_<parthenon::MeshData<Real>>(m, "MeshData_d");
}

}  // namespace kamayan
