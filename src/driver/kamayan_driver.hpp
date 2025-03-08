#ifndef DRIVER_KAMAYAN_DRIVER_HPP_
#define DRIVER_KAMAYAN_DRIVER_HPP_

#include <functional>
#include <list>
#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {
class KamayanDriver : public parthenon::MultiStageDriver {
 public:
  KamayanDriver(const std::list<std::shared_ptr<KamayanUnit>> units,
                std::shared_ptr<ParameterInput> pin, ApplicationInput *app_in, Mesh *pm);

  void Setup();
  std::shared_ptr<Config> GetConfig() { return config_; }

  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
  void BuildTaskList(TaskList &task_list, const Real &dt, const Real &beta,
                     const int &stage, MeshData *md0, MeshData *md1,
                     MeshData *mdudt) const;

 private:
  std::shared_ptr<Config> config_;
  const std::list<std::shared_ptr<KamayanUnit>> units_;
  std::shared_ptr<runtime_parameters::RuntimeParameters> parms_;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);  // NOLINT

}  // namespace kamayan

#endif  // DRIVER_KAMAYAN_DRIVER_HPP_
