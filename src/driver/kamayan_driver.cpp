#include "kamayan_driver.hpp"

#include <limits>
#include <memory>
#include <string>

#include <utils/instrument.hpp>

#include "bvals/comms/bvals_in_one.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan_driver_types.hpp"

namespace kamayan {
using RP = runtime_parameters::RuntimeParameters;
KamayanDriver::KamayanDriver(UnitCollection units, std::shared_ptr<RPs> rps,
                             ApplicationInput *app_in, Mesh *pm)
    : parthenon::MultiStageDriver(rps->GetPin(), app_in, pm), units_(units),
      config_(std::make_shared<Config>()), parms_(rps) {
  parms_->Add<std::string>("parthenon/time", "integrator", "rk2",
                           "Which multi-stage Runge-Kutta method to use",
                           {"rk1", "rk2", "rk3"});
  parms_->Add<Real>("parthenon/time", "dt_ceil", std::numeric_limits<Real>::max(),
                    "The maximum allowed timestep.");
  parms_->Add<Real>(
      "parthenon/time", "dt_factor", 2.0,
      "The maximum allowed relative increase of the timestep over the previous value.");
  parms_->Add<Real>("parthenon/time", "dt_floor", std::numeric_limits<Real>::min(),
                    "The minimum allowed timestep.");
  parms_->Add<Real>("parthenon/time", "dt_force", std::numeric_limits<Real>::lowest(),
                    "Force the timestep to this value, ignoring all other conditions.");
  parms_->Add<Real>("parthenon/time", "dt_init", std::numeric_limits<Real>::max(),
                    "The maximum allowed timestep during the first cycle.");
  parms_->Add<bool>(
      "parthenon/time", "dt_init_force", true,
      "If set to true, force the first cycle’s timestep to the value given by dt_init.");
  parms_->Add<Real>("parthenon/time", "dt_min", std::numeric_limits<Real>::lowest(),
                    "If the timestep falls below dt_min for dt_min_cycle_limit cycles, "
                    "Parthenon fatals.");
  parms_->Add<int>("parthenon/time", "dt_min_cycle_limit", 10,
                   "The maximum number of cycles the timestep can be below dt_min.");
  parms_->Add<Real>("parthenon/time", "dt_max", std::numeric_limits<Real>::max(),
                    "If the timestep falls above dt_max for dt_max_cycle_limit cycles, "
                    "Parthenon fatals.");
  parms_->Add<int>("parthenon/time", "dt_max_cycle_limit", 1,
                   "The maximum number of cycles the timestep an be above dt_max.");
  parms_->Add<Real>("parthenon/time", "dt_user", std::numeric_limits<Real>::max(),
                    "Set a global timestep limit.");
  parms_->Add<Real>(
      "parthenon/time", "ncrecv_bdry_buf_timeout_sec", -1.0,
      "Timeout in seconds for the ReceiveBoundaryBuffers tasks. Disabed (negative) by "
      "default. Typically no need in production runs. Useful for debugging MPI calls.");
  parms_->Add<int>(
      "parthenon/time", "ncycle_out", 1,
      "Number of cycles between short diagnostic output to standard out containing, "
      "e.g., current time, dt, zone-update/wsec. Default: 1 (i.e, every cycle).");
  parms_->Add<int>(
      "parthenon/time", "ncycle_out_mesh", 0,
      "Number of cycles between printing the mesh structure to standard out. "
      "Use a negative number to also print every time the mesh was modified. "
      "Default: 0 (i.e, off).");
  parms_->Add<int>("parthenon/time", "nlim", -1,
                   "Stop criterion on total number of steps taken. Ignored if < 0.");
  parms_->Add<int>(
      "parthenon/time", "perf_cycle_offset", 0,
      "Skip the first N cycles when calculating the final performance (e.g., "
      "zone-cycles/wall_second). Allows to hide the initialization overhead "
      "in Parthenon.");
  parms_->Add<Real>("parthenon/time", "tlim", std::numeric_limits<Real>::max(),
                    "Stop criterion on simulation time.");
}

void KamayanDriver::Setup() {
  for (const auto &kamayan_unit : units_) {
    if (kamayan_unit.second->Setup != nullptr)
      kamayan_unit.second->Setup(config_.get(), parms_.get());
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

    // guard cell fill?
    auto start_send = tl.AddTask(none, parthenon::StartReceiveBoundaryBuffers, md1);

    auto stage_tasks = BuildTaskListRKStage(tl, dt, beta, stage, md0, md1, mdudt);

    // auto boundaries = parthenon::AddBoundaryExchangeTasks(
    //     stage_tasks, tl, md0, md0->GetMeshPointer()->multilevel);

    // probably want another hook to do a cons2prim? Otherwise it can be on the individual
    // units to do any that they need
  }

  return tc;
}
TaskID KamayanDriver::BuildTaskList(TaskList &task_list, const Real &dt, const Real &beta,
                                    const int &stage, std::shared_ptr<MeshData> md0,
                                    std::shared_ptr<MeshData> md1,
                                    std::shared_ptr<MeshData> mdudt) const {
  auto rk_stage = BuildTaskListRKStage(task_list, dt, beta, stage, md0, md1, mdudt);
  auto next = rk_stage;
  if (stage == integrator->nstages) {
    for (const auto &kamayan_unit : units_.operator_split) {
      // these should be responsible for doing their own boundary fills
      next = kamayan_unit->AddTasksSplit(next, task_list, md0.get(), dt);
    }
  }
  return next;
}
TaskID KamayanDriver::BuildTaskListRKStage(TaskList &task_list, const Real &dt,
                                           const Real &beta, const int &stage,
                                           std::shared_ptr<MeshData> md0,
                                           std::shared_ptr<MeshData> md1,
                                           std::shared_ptr<MeshData> mdudt) const {
  TaskID next(0), none(0);
  if (units_.rk_fluxes.size() > 0) {
    auto start_flux_correction =
        task_list.AddTask(none, parthenon::StartReceiveFluxCorrections, md0);

    for (const auto &kamayan_unit : units_.rk_fluxes) {
      if (kamayan_unit->AddFluxTasks != nullptr)
        next = kamayan_unit->AddFluxTasks(next, task_list, md0.get());
    }
    auto set_fluxes = parthenon::AddFluxCorrectionTasks(
        next, task_list, md0, md0->GetMeshPointer()->multilevel);
    // now set dudt using flux-divergence / discrete stokes theorem
  }

  for (const auto &kamayan_unit : units_.rk_stage) {
    if (kamayan_unit->AddTasksOneStep != nullptr)
      next = kamayan_unit->AddTasksOneStep(next, task_list, md0.get(), mdudt.get());
  }
  // TODO(acreyes): add mdudt -> md1
  if (units_.rk_fluxes.size() + units_.rk_stage.size() > 0) {
    // then do boundary exchange
  }
  return next;
}

}  // namespace kamayan
