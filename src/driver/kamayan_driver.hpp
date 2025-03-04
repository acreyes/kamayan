#ifndef DRIVER_KAMAYAN_DRIVER_HPP_
#define DRIVER_KAMAYAN_DRIVER_HPP_

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/config.hpp"
#include "types.hpp"

namespace kamayan {

class KamayanDriver : public parthenon::MultiStageDriver {
 public:
  KamayanDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm,
                std::shared_ptr<Config> config);
  parthenon::TaskCollection MakeTaskCollection(const BlockList_t &blocks,
                                               const int &stage);

 private:
  std::shared_ptr<Config> config_;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);  // NOLINT

}  // namespace kamayan

#endif  // DRIVER_KAMAYAN_DRIVER_HPP_
