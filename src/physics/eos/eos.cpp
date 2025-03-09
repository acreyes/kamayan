#include "physics/eos.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include <memory>

namespace kamayan::eos {
namespace rp = runtime_parameters;
std::shared_ptr<KamayanUnit> ProcessUnit() {
  std::shared_ptr<KamayanUnit> eos_unit;
  eos_unit->Setup = Setup;
  eos_unit->Initialize = Initialize;
  return eos_unit;
}

void Setup(Config *cfg, rp::RuntimeParameters *rps) {
  // general eos configurations
  auto eostype_str = rps->GetOrAdd<std::string>(
      "eos", "type", "single", "Type of Eos to use, Single or multitype.",
      {"single", "multitype"});
  eosType type;
  if (eostype_str == "single") {
    type = eosType::Single;
  } else if (eostype_str == "multitype") {
    type = eosType::MultiType;
  }
  cfg->Add(type);
  // used in single fluid EoS
  rps->Add<Real>("eos/single", "Abar", 1.0, "Mean molecular weight in g/mol");

  // gamma law gas eos
  rps->Add<Real>("eos/gamma", "gamma", 1.4, "adiabatic index used in ideal gas EoS");
}
}  // namespace kamayan::eos
