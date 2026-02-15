#include "physics/material_properties/multi_species.hpp"

#include <memory>

#include "kamayan/unit.hpp"

namespace kamayan::multispecies {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto mspec = std::make_shared<KamayanUnit>("multispecies");
  // mspec->SetupParams = SetupParams;
  // mspec->InitializeData = InitializeData;
  // mspec->PrepareConserved = PrepareConserved;
  // mspec->PreparePrimitive = PreparePrimitive;
  return mspec;
}
};  // namespace kamayan::multispecies
