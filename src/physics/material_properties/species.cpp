#include "physics/material_properties/species.hpp"

#include <memory>

#include "kamayan/unit.hpp"

namespace kamayan::species {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto mspec = std::make_shared<KamayanUnit>("species");
  // mspec->SetupParams = SetupParams;
  // mspec->InitializeData = InitializeData;
  // mspec->PrepareConserved = PrepareConserved;
  // mspec->PreparePrimitive = PreparePrimitive;
  return mspec;
}
};  // namespace kamayan::species
