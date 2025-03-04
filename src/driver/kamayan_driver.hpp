#ifndef DRIVER_KAMAYAN_DRIVER_HPP_
#define DRIVER_KAMAYAN_DRIVER_HPP_

#include <functional>
#include <list>
#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/unit.hpp"
#include "types.hpp"

namespace kamayan {
class KamayanDriver : public parthenon::MultiStageDriver {
 public:
  KamayanDriver(std::shared_ptr<ParameterInput> pin, ApplicationInput *app_in, Mesh *pm,
                std::shared_ptr<Config> config);
  parthenon::TaskCollection MakeTaskCollection(const BlockList_t &blocks,
                                               const int &stage);
  std::function<std::list<KamayanUnit>()> ProcessUnits = nullptr;

 private:
  void AddUnit(std::shared_ptr<KamayanUnit> ku);
  void Setup();
  std::shared_ptr<Config> config_;
  std::list<std::shared_ptr<KamayanUnit>> units_;
  std::shared_ptr<runtime_parameters::RuntimeParameters> parms_;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);  // NOLINT

}  // namespace kamayan

#endif  // DRIVER_KAMAYAN_DRIVER_HPP_
