#ifndef PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_HPP_

#include <memory>
#include <string>
#include <vector>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "physics/material_properties/eos/eos_types.hpp"
#include "physics/material_properties/eos/equation_of_state.hpp"

namespace kamayan::eos {

std::shared_ptr<KamayanUnit> ProcessUnit();

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps);
void SetupParams(KamayanUnit *unit);

std::shared_ptr<StateDescriptor>
Initialize(const Config *cfg, const runtime_parameters::RuntimeParameters *rps);
void InitializeData(KamayanUnit *unit);

TaskStatus EosWrapped(MeshData *md, EosMode mode);
TaskStatus EosWrapped(MeshBlock *mb, EosMode mode);
TaskStatus PreparePrimitive(MeshData *md);
TaskStatus PrepareConserved(MeshData *md);

// Add all parameters needed for a single species' eos
void SetupSpeciesParams(UnitData &ud, std::string spec);
// build an equation of state for a list of species
EOS_t MakeEos(std::vector<std::string> species);

}  // namespace kamayan::eos

#endif  // PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_HPP_
