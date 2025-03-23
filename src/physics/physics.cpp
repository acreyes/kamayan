#include "physics/physics.hpp"

#include <memory>
#include <string>

#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "physics/physics_types.hpp"

namespace kamayan::physics {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  std::shared_ptr<KamayanUnit> physics;
  physics->Setup = Setup;
  return physics;
}

namespace rp = runtime_parameters;
void Setup(Config *cfg, rp::RuntimeParameters *rps) {
  auto fluid_str = rps->GetOrAdd<std::string>(
      "physics", "fluid", "1T", "physics model to use for our fluid", {"1t", "3t"});

  Fluid fluid_type;
  if (fluid_str == "1t") {
    fluid_type = Fluid::oneT;
  } else if (fluid_str == "3t") {
    fluid_type = Fluid::threeT;
  }
  cfg->Add(fluid_type);
}
}  // namespace kamayan::physics
