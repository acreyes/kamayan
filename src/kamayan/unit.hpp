#ifndef KAMAYAN_UNIT_HPP_
#define KAMAYAN_UNIT_HPP_
#include <functional>
#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "types.hpp"

namespace kamayan {
struct KamayanUnit {
  // Setup is called to add options into the kamayan configuration and to register
  // runtime parameters owned by the unit
  std::function<void(Config *cfg, runtime_parameters::RuntimeParameters)> Setup = nullptr;

  // Initialize is responsible for setting up the parthenon StateDescriptor, registering
  // params , adding fields owned by the unit & registering any callbacks known to
  // parthenon
  std::function<std::shared_ptr<StateDescriptor>(
      const runtime_parameters::RuntimeParameters &)>
      Initialize = nullptr;

  // These tasks get added to the tasklist that accumulate dudt for this unit based
  // on the current state in md, returning the TaskID of the final task for a single
  // stage in the multi-stage driver
  std::function<TaskID(TaskList &tl, MeshData *md, MeshData *dudt)> AddTasksOneStep =
      nullptr;

  // These tasks are used to advance md by dt as one of the operators in the
  // operator splitting
  std::function<TaskID(TaskList &tl, MeshData *md, const Real dt)> AddTasksSplit =
      nullptr;
};
}  // namespace kamayan

#endif  // KAMAYAN_UNIT_HPP_
