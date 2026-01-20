#ifndef DRIVER_KAMAYAN_DRIVER_HPP_
#define DRIVER_KAMAYAN_DRIVER_HPP_

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {
namespace driver {
std::shared_ptr<KamayanUnit> ProcessUnit(bool with_setup = false);
void PreStepUserWorkInLoop(Mesh *mesh, ParameterInput *pin, SimTime const &sim_time);
}  // namespace driver
class KamayanDriver : public parthenon::MultiStageDriver {
  using RPs = runtime_parameters::RuntimeParameters;

 public:
  KamayanDriver(std::shared_ptr<UnitCollection> units, std::shared_ptr<RPs> rps,
                ApplicationInput *app_in, Mesh *pm);

  void Setup();
  std::shared_ptr<Config> GetConfig() { return config_; }

  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
  TaskID BuildTaskListRKStage(TaskList &task_list, const Real &dt, const Real &beta,
                              const int &stage, std::shared_ptr<MeshData> mbase,
                              std::shared_ptr<MeshData> md0,
                              std::shared_ptr<MeshData> md1,
                              std::shared_ptr<MeshData> mdudt) const;
  TaskID BuildTaskList(TaskList &task_list, const Real &dt, const Real &beta,
                       const int &stage, std::shared_ptr<MeshData> mbase,
                       std::shared_ptr<MeshData> md0, std::shared_ptr<MeshData> md1,
                       std::shared_ptr<MeshData> mdudt) const;

  static const parthenon::SimTime GetSimTime();

 private:
  std::shared_ptr<Config> config_;
  std::shared_ptr<UnitCollection> units_;
  std::shared_ptr<RPs> parms_;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);  // NOLINT

}  // namespace kamayan

#endif  // DRIVER_KAMAYAN_DRIVER_HPP_
