#include "kamayan_driver.hpp"

#include <list>
#include <memory>

#include <utils/instrument.hpp>

#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan_driver_types.hpp"

namespace kamayan {
using RP = runtime_parameters::RuntimeParameters;
KamayanDriver::KamayanDriver(const std::list<std::shared_ptr<KamayanUnit>> units,
                             std::shared_ptr<ParameterInput> pin,
                             ApplicationInput *app_in, Mesh *pm)
    : parthenon::MultiStageDriver(pin.get(), app_in, pm), units_(units),
      config_(std::make_shared<Config>()), parms_(std::make_shared<RP>(pin)) {}

void KamayanDriver::Setup() {
  for (const auto &kamayan_unit : units_) {
    if (kamayan_unit->Setup != nullptr) kamayan_unit->Setup(config_.get(), parms_.get());
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
  const auto &stage_name = integrator->stage_name;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(partitions.size());

  for (int i = 0; i < partitions.size(); i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &md1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);
    BuildTaskList(tl, dt, beta, stage, md0.get(), md1.get(), mdudt.get());
  }

  return tc;
}
void KamayanDriver::BuildTaskList(TaskList &task_list, const Real &dt, const Real &beta,
                                  const int &stage, MeshData *md0, MeshData *md1,
                                  MeshData *mdudt) const {
  // TODO(acreyes): is there a better way to do this?
  // auto start_send = tl.AddTask(none, parthenon::StartReceiveBoundaryBuffers, mc1);
  // auto start_flxcor = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
  TaskID next(0);
  for (const auto &kamayan_unit : units_) {
    if (kamayan_unit->AddTasksOneStep != nullptr)
      next = kamayan_unit->AddTasksOneStep(next, task_list, md0, mdudt);
  }
  // add mdudt -> md1
  if (stage == integrator->nstages) {
    for (const auto &kamayan_unit : units_) {
      next = kamayan_unit->AddTasksSplit(next, task_list, md0, dt);
    }
  }
}

}  // namespace kamayan
