#include <memory>
#include <string>

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "physics/eos/eos.hpp"
#include "physics/eos/equation_of_state.hpp"
#include "utils/instrument.hpp"

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
  auto eos_model_str = rps->GetOrAdd<std::string>(
      "eos", "model", "single", "Type of Eos to use, single, tabulated or multitype.",
      {"single", "tabulated", "multitype"});
  EosModel type;
  if (eos_model_str == "single") {
    type = EosModel::gamma;
  } else if (eos_model_str == "tabulated") {
    type = EosModel::tabulated;
  } else if (eos_model_str == "multitype") {
    type = EosModel::multitype;
  }
  cfg->Add(type);
  // used in single fluid EoS
  rps->Add<Real>("eos/single", "Abar", 1.0, "Mean molecular weight in g/mol");

  // gamma law gas eos
  rps->Add<Real>("eos/gamma", "gamma", 1.4, "adiabatic index used in ideal gas EoS");

  // build the Eos Now
}

template <Fluid fluid>
requires(fluid == Fluid::oneT)
void AddEosImpl(const EosModel model, StateDescriptor *pkg,
                const runtime_parameters::RuntimeParameters *rps) {
  eos::EOS_t eos;
  if (model == EosModel::gamma) {
    auto gamma = rps->Get<Real>("eos/gamma", "gamma");
    auto abar = rps->Get<Real>("eos/single", "abar");
    eos = EquationOfState<EosModel::gamma>(gamma, abar);
  } else {
    std::string msg =
        "EosModel " + rps->Get<std::string>("eos", "model") + "not implemented\n";
    PARTHENON_THROW(msg.c_str())
  }
  pkg->AddParam("EoS", eos);
}

struct AddEos {
  using options = OptTypeList<OptList<Fluid, Fluid::oneT>>;
  using value = void;
  template <Fluid fluid>
  value dispatch(const EosModel model, StateDescriptor *pkg,
                 const runtime_parameters::RuntimeParameters *rps) {
    AddEosImpl<fluid>(model, pkg, rps);
  }
};

std::shared_ptr<StateDescriptor>
Initialize(const Config *cfg, const runtime_parameters::RuntimeParameters *rps) {
  std::shared_ptr<StateDescriptor> eos_pkg;
  auto model = cfg->Get<EosModel>();
  auto fluid = cfg->Get<Fluid>();

  Dispatcher<AddEos>(PARTHENON_AUTO_LABEL, fluid).execute(model, eos_pkg.get(), rps);

  return eos_pkg;
}
}  // namespace kamayan::eos
