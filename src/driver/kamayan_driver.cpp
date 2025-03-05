#include "kamayan_driver.hpp"

#include <memory>

#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "utils/error_checking.hpp"

namespace kamayan {
using RP = runtime_parameters::RuntimeParameters;
KamayanDriver::KamayanDriver(std::shared_ptr<ParameterInput> pin,
                             ApplicationInput *app_in, Mesh *pm)
    : parthenon::MultiStageDriver(pin.get(), app_in, pm),
      config_(std::make_shared<Config>()), parms_(std::make_shared<RP>(pin)) {}

void KamayanDriver::Setup() {
  PARTHENON_REQUIRE_THROWS(ProcessUnits != nullptr,
                           "[KamayanDriver] ProcessUnits not set!")

  units_ = ProcessUnits();
  for (const auto &ku : units_) {
    if (ku->Setup != nullptr) ku->Setup(config_.get(), parms_.get());
  }
}

TaskCollection KamayanDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
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

void KamayanDriver::AddUnit(std::shared_ptr<KamayanUnit> ku) { units_.push_back(ku); }

}  // namespace kamayan
