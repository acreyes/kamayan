#ifndef PHYSICS_PHYSICS_HPP_
#define PHYSICS_PHYSICS_HPP_
#include <memory>

#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
namespace kamayan::physics {
// The point of this unit is to manage some common configurations that
// are needed by multiple units organized here.
// For example the choice of 3T effects what kind of EoS we want to use
// in addition to hydro
std::shared_ptr<KamayanUnit> ProcessUnit();
void SetupParams(UnitDataCollection &udc);

}  // namespace kamayan::physics

#endif  // PHYSICS_PHYSICS_HPP_
