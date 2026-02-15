#ifndef PHYSICS_MATERIAL_PROPERTIES_SPECIES_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_SPECIES_HPP_
#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/unit.hpp"
namespace kamayan::species {
std::shared_ptr<KamayanUnit> ProcessUnit();
void SetupParams(KamayanUnit *unit);
void InitializeData(KamayanUnit *unit);

TaskStatus PrepareConserved(MeshData *md);
TaskStatus PreparePrimitive(MeshData *md);
}  // namespace kamayan::species

#endif  // PHYSICS_MATERIAL_PROPERTIES_SPECIES_HPP_
