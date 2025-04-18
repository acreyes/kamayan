#include "physics/physics.hpp"

#include <memory>
#include <string>
#include <utility>

#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "physics/physics_types.hpp"

namespace kamayan::physics {
std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto physics = std::make_shared<KamayanUnit>("physics");
  physics->Setup = Setup;
  return physics;
}

namespace rp = runtime_parameters;
void Setup(Config *cfg, rp::RuntimeParameters *rps) {
  auto fluid_str = rps->GetOrAdd<std::string>(
      "physics", "fluid", "1T", "physics model to use for our fluid", {"1t", "3t"});

  auto fluid_type = MapStrToEnum<Fluid>(fluid_str, std::make_pair(Fluid::oneT, "1t"),
                                        std::make_pair(Fluid::threeT, "3t"));
  cfg->Add(fluid_type);

  auto mhd_str =
      rps->GetOrAdd<std::string>("physics", "MHD", "off", "Mhd model", {"off", "ct"});

  auto Mhd_type = MapStrToEnum<Mhd>(mhd_str, std::make_pair(Mhd::off, "off"),
                                    std::make_pair(Mhd::ct, "ct"));
  cfg->Add(Mhd_type);
}
}  // namespace kamayan::physics
