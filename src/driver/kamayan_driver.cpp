#include "kamayan_driver.hpp"

#include <memory>

#include "kamayan/config.hpp"

namespace kamayan {
KamayanDriver::KamayanDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm,
                             std::shared_ptr<Config> config)
    : parthenon::MultiStageDriver(pin, app_in, pm), config_(config) {
  // call thie runtime parameters
}

TaskCollection KamayanDriver::MakeTaskCollection(const BlockList_t &blocks,
                                                 const int &stage) {
  TaskCollection tc;
  TaskID none(0);

  // task region over partitions of meshdata
  // * get buffers for base, stage-1, stage,
  // * loop over units calling the AddTasksOneStage(base, dudt)
  // * update base with dudt
  // * check if final stage
  //    * loop over units calling the AddTasksSplit(base, dt)

  return tc;
}

}  // namespace kamayan
