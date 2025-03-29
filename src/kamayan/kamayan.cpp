#include "kamayan/kamayan.hpp"

#include <cstdlib>
#include <list>
#include <memory>
#include <utility>

#include "parthenon_manager.hpp"

namespace kamayan {
using parthenon::ParthenonManager;
using parthenon::ParthenonStatus;

std::shared_ptr<ParthenonManager> InitEnv(int argc, char *argv[]) {
  auto pman = std::make_shared<parthenon::ParthenonManager>();

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman->ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman->ParthenonFinalize();
    std::exit(0);
  }
  if (manager_status == ParthenonStatus::error) {
    pman->ParthenonFinalize();
    PARTHENON_THROW("Error during initialization");
  }

  return pman;
}

KamayanDriver InitPackages(std::shared_ptr<ParthenonManager> pman,
                           std::list<std::shared_ptr<KamayanUnit>> units) {
  std::shared_ptr<kamayan::ParameterInput> pin = std::move(pman->pinput);
  auto runtime_parameters =
      std::make_shared<kamayan::runtime_parameters::RuntimeParameters>(pin);

  // put together the configuration & runtime parameters
  std::shared_ptr<kamayan::Config> config;
  for (auto &kamayan_unit : units) {
    if (kamayan_unit->Setup != nullptr)
      kamayan_unit->Setup(config.get(), runtime_parameters.get());
  }

  pman->app_input->ProcessPackages = [&](std::unique_ptr<kamayan::ParameterInput> &pin) {
    parthenon::Packages_t packages;
    // start with the config package, then go into all of our units
    auto config_pkg = std::make_shared<kamayan::StateDescriptor>("Config");
    config_pkg->AddParam("config", config);
    packages.Add(config_pkg);
    for (auto &kamayan_unit : units) {
      if (kamayan_unit->Initialize != nullptr)
        packages.Add(kamayan_unit->Initialize(config.get(), runtime_parameters.get()));
    }
    return packages;
  };
  // may want to add unit callbacks for these as well...
  // app->PreFillDerivedBlock = advection_package::PreFill;
  // app->PostFillDerivedBlock = advection_package::PostFill;
  pman->ParthenonInitPackagesAndMesh();
  return KamayanDriver(units, runtime_parameters, pman->app_input.get(),
                       pman->pmesh.get());
}

void Finalize(std::shared_ptr<ParthenonManager> pman) { pman->ParthenonFinalize(); }
}  // namespace kamayan
