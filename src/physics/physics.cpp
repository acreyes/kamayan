#include "physics/physics.hpp"

#include <memory>
#include <string>

#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"
#include "physics/physics_types.hpp"

namespace kamayan::physics {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto physics = std::make_shared<KamayanUnit>("physics");
  physics->SetupParams = SetupParams;
  return physics;
}

void SetupParams(KamayanUnit &unit) {
  auto &physics = unit.AddData("physics");
  physics.AddParm<Fluid>("fluid", "1T", "physics model to use for our fluid",
                         {{"1t", Fluid::oneT}, {"3t", Fluid::threeT}});

  physics.AddParm<Mhd>("MHD", "off", "Mhd model", {{"off", Mhd::off}, {"ct", Mhd::ct}});
}
}  // namespace kamayan::physics
