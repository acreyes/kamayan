#include "physics/multi_species.hpp"

#include <memory>

#include "kamayan/unit.hpp"

namespace kamayan::multispecies {

void SetupParams(KamayanUnit *unit) {
  // TODO: Implement multispecies parameter setup
}

void InitializeData(KamayanUnit *unit) {
  // TODO: Implement multispecies data initialization
}

TaskStatus PrepareConserved(MeshData *md) {
  // TODO: Implement multispecies conserved variable preparation
  return TaskStatus::complete;
}

TaskStatus PreparePrimitive(MeshData *md) {
  // TODO: Implement multispecies primitive variable preparation
  return TaskStatus::complete;
}

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md) {
  // TODO: Implement multispecies flux tasks
  return prev;
}

std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto mspec = std::make_shared<KamayanUnit>("multispecies");
  mspec->SetupParams.Register(SetupParams);
  mspec->InitializeData.Register(InitializeData);
  mspec->PrepareConserved.Register(PrepareConserved);
  // Multispecies should run AFTER hydro but BEFORE eos when preparing primitives
  mspec->PreparePrimitive.Register(PreparePrimitive,
                                   /*after=*/{"hydro"},
                                   /*before=*/{"eos"});
  return mspec;
}
};  // namespace kamayan::multispecies
