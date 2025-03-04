#ifndef KAMAYAN_DRIVER_HPP_
#define KAMAYAN_DRIVER_HPP_

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "types.hpp"

namespace kamayan {
// import some parthenon types into our namespace
using ParameterInput = parthenon::ParameterInput;
using ApplicationInput = parthenon::ApplicationInput;
using TaskCollection = parthenon::TaskCollection;
using TaskList = parthenon::TaskList;
using TaskID = parthenon::TaskID;

class KamayanDriver : public parthenon::MultiStageDriver {
 public:
  KamayanDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  parthenon::TaskCollection MakeTaskCollection(const BlockList_t &blocks,
                                               const int &stage);
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);  // NOLINT

}  // namespace kamayan

#endif  // KAMAYAN_DRIVER_HPP_
