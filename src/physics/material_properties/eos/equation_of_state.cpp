#include "equation_of_state.hpp"

#include <string>

namespace kamayan::eos {

EosVariant MakeEosSingleSpecies(std::string spec, KamayanUnit *material) {
  auto get_block = [&](const std::string &key) { return "material/" + spec + "/" + key; };
  auto eos_type = material->Param<std::string>(get_block("eos_type"));
  // currently only support gamma law so just build that
  auto Abar = material->Param<Real>(get_block("Abar"));
  auto Z = material->Param<Real>(get_block("Z"));
  auto gamma = material->Param<Real>(get_block("gamma"));

  if (eos_type == "gamma") {
    return EquationOfState<EosModel::gamma>(gamma, Abar);
  }
  return EquationOfState<EosModel::gamma>(gamma, Abar);
}
}  // namespace kamayan::eos
