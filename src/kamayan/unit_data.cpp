#include "kamayan/unit_data.hpp"

#include <memory>
#include <string>

namespace kamayan {
void UnitData::Setup(std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
                     std::shared_ptr<Config> cfg) {
  runtime_parameters = rps;
  config = cfg;
  for (const auto &up : parameters) {
    up.second.AddRP(rps.get());
  }
}
void UnitData::Initialize(std::shared_ptr<StateDescriptor> pkg) {
  params = pkg;
  for (const auto &up : parameters) {
    up.second.AddParam();
  }
}

const UnitData::DataType UnitData::Get(const std::string &key) const {
  return parameters.at(key).Get();
}

}  // namespace kamayan
