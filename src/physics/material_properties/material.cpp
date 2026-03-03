#include "physics/material_properties/material.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/fields.hpp"
#include "kamayan/unit.hpp"
#include "physics/material_properties/eos/eos.hpp"
#include "physics/material_properties/eos/equation_of_state.hpp"
#include "physics/material_properties/material_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/parallel.hpp"
#include "utils/strings.hpp"

namespace kamayan::material {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto mspec = std::make_shared<KamayanUnit>("material");
  mspec->SetupParams = SetupParams;
  mspec->InitializeData = InitializeData;
  mspec->PostMeshInitialization.Register(PostMeshInitialization, {}, {"eos"});
  // mspec->PrepareConserved = PrepareConserved;
  // mspec->PreparePrimitive = PreparePrimitive;
  return mspec;
}

void SetupParams(KamayanUnit *unit) {
  auto &material = unit->AddData("material");
  material.AddParm<std::string>("species", "single", "Comma separated list of species.");

  material.AddParm<Real>("allocation_threshold", 1.0e-6,
                         "Mass fraction allocation threshold per species.");
  material.AddParm<Real>("deallocation_threshold", 5.0e-7,
                         "Mass fraction deallocation threshold per species.");
  material.AddParm<Real>("default_mass_fraction", 0.0,
                         "default value for mass fractions on allocation");

  auto species = strings::split(material.Get<std::string>("species"), ',');
  auto unique_species = std::set(species.begin(), species.end());
  PARTHENON_REQUIRE_THROWS(species.size() == unique_species.size(),
                           "material/species must be unique")
  unit->AddParam("species", species);
  unit->AddParam("nspecies", species.size());

  // placeholder to make sure we have the single species defined
  // until a multispecies eos is added
  if (std::find(species.begin(), species.end(), "single") == species.end()) {
    species.push_back("single");
  }
  for (const auto &spec : species) {
    auto &spec_data = unit->AddData("material/" + spec);
    spec_data.AddParm<Real>("Z", 1.0, "Atomic number for the " + spec + " species");
    spec_data.AddParm<Real>("Abar", 1.0, "Atomic mass for the " + spec + " species");
    eos::SetupSpeciesParams(spec_data, spec);
  }
}

void InitializeSparseFields(const std::vector<std::string> &species, KamayanUnit *unit) {
  auto &material = unit->Data("material");
  auto nspecies = species.size();

  // sparse ids for the species
  std::vector<int> spec_ids;
  for (int i = 0; i < nspecies; i++) {
    spec_ids.push_back(i);
  }

  auto alloc_threshold = material.Get<Real>("allocation_threshold");
  auto dealloc_threshold = material.Get<Real>("deallocation_threshold");
  auto default_value = material.Get<Real>("default_mass_fraction");
  Metadata meta_data(
      {CENTER_FLAGS(Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse)},
      MFRAC::Shape());
  meta_data.SetSparseThresholds(alloc_threshold, dealloc_threshold, default_value);

  // add sparsepools for each field, with allocations controlled
  // by the mass fractions
  unit->AddSparsePool<MFRAC>(meta_data, spec_ids);
}

void InitializeData(KamayanUnit *unit) {
  auto &material = unit->Data("material");
  auto species = strings::split(material.Get<std::string>("species"), ',');
  std::size_t nspecies = species.size();

  eos::EOS_t eos;
  if (nspecies > 1) {
    // build the multi species eos
    // placeholder until we have a multispecies eos
    eos = eos::MakeEosSingleSpecies("single", unit);
  } else {
    eos = eos::MakeEosSingleSpecies(species[0], unit);
  }
  unit->AddParam("eos", eos);

  if (nspecies > 1) InitializeSparseFields(species, unit);
}

TaskStatus PostMeshInitialization(MeshData *md) {
  // make sure our mass fractions sum to 1
  auto material = md->GetMeshPointer()->packages.Get("material");
  const auto nspecies = material->Param<std::size_t>("nspecies");
  if (nspecies == 1) return TaskStatus::complete;

  auto pack = grid::GetPack<MFRAC>(md);

  const int nblocks = pack.GetNBlocks();
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);

  par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real sum = 0;
        for (int s = 0; s <= pack.GetUpperBound(b, MFRAC()); s++) {
          sum += pack(b, MFRAC(s), k, j, i);
        }
        for (int s = 0; s <= pack.GetUpperBound(b, MFRAC()); s++) {
          pack(b, MFRAC(s), k, j, i) *= 1. / sum;
        }
      });

  return TaskStatus::complete;
}
};  // namespace kamayan::material
