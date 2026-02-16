#include "physics/material_properties/material.hpp"

#include <memory>
#include <string>

#include "kamayan/unit.hpp"
#include "physics/material_properties/eos/eos.hpp"
#include "physics/material_properties/eos/equation_of_state.hpp"
#include "utils/strings.hpp"

namespace kamayan::material {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto mspec = std::make_shared<KamayanUnit>("material");
  mspec->SetupParams = SetupParams;
  mspec->InitializeData = InitializeData;
  // mspec->PrepareConserved = PrepareConserved;
  // mspec->PreparePrimitive = PreparePrimitive;
  return mspec;
}

void SetupParams(KamayanUnit *unit) {
  auto &material = unit->AddData("material");
  material.AddParm<std::string>("species", "single", "Comma separated list of species.");

  auto species = strings::split(material.Get<std::string>("species"), ',');
  unit->AddParam("species", species);
  unit->AddParam("nspecies", species.size());
  for (const auto &spec : species) {
    auto &spec_data = unit->AddData("material/" + spec);
    spec_data.AddParm<Real>("Z", 1.0, "Atomic number of " + spec);
    spec_data.AddParm<Real>("Abar", 1.0, "Atomic mass of " + spec);
    eos::SetupSpeciesParams(spec_data, spec);
  }
}

void InitializeData(KamayanUnit *unit) {
  auto species = unit->Param<std::vector<std::string>>("species");
  auto nspecies = species.size();

  eos::EOS_t eos;
  if (nspecies > 1) {
    // build the multi species eos
  } else {
    eos = eos::MakeEosSingleSpecies(species[0], unit);
  }
  unit->AddParam("eos", eos);
}
};  // namespace kamayan::material
