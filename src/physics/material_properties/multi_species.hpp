#ifndef PHYSICS_MULTI_SPECIES_HPP_
#define PHYSICS_MULTI_SPECIES_HPP_
#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/unit.hpp"
namespace kamayan::multispecies {
std::shared_ptr<KamayanUnit> ProcessUnit();
void SetupParams(KamayanUnit *unit);
void InitializeData(KamayanUnit *unit);

TaskStatus PrepareConserved(MeshData *md);
TaskStatus PreparePrimitive(MeshData *md);
}  // namespace kamayan::multispecies

#endif  // PHYSICS_MULTI_SPECIES_HPP_
