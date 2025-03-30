#include "kamayan/kamayan.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include "kamayan/unit.hpp"
#include "parthenon_manager.hpp"
#include "utils/error_checking.hpp"

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

KamayanDriver InitPackages(std::shared_ptr<ParthenonManager> pman, UnitCollection units) {
  auto pin = pman->pinput.get();
  auto runtime_parameters =
      std::make_shared<kamayan::runtime_parameters::RuntimeParameters>(pin);

  // put together the configuration & runtime parameters
  auto config = std::make_shared<Config>();
  for (auto &kamayan_unit : units) {
    if (kamayan_unit.second->Setup != nullptr)
      kamayan_unit.second->Setup(config.get(), runtime_parameters.get());
  }

  pman->app_input->ProcessPackages = [&](std::unique_ptr<kamayan::ParameterInput> &pin) {
    parthenon::Packages_t packages;
    // start with the config package, then go into all of our units
    auto config_pkg = std::make_shared<kamayan::StateDescriptor>("Config");
    config_pkg->AddParam("config", config);
    packages.Add(config_pkg);
    for (auto &kamayan_unit : units) {
      if (kamayan_unit.second->Initialize != nullptr)
        packages.Add(
            kamayan_unit.second->Initialize(config.get(), runtime_parameters.get()));
    }
    return packages;
  };

  pman->app_input->ProblemGenerator = [&](MeshBlock *mb, ParameterInput *pin) {
    for (auto &kamayan_unit : units) {
      if (kamayan_unit.second->ProblemGeneratorMeshBlock != nullptr) {
        kamayan_unit.second->ProblemGeneratorMeshBlock(mb);
      }
    }
  };

  // pman->app_input->MeshProblemGenerator = [&](Mesh *mesh, ParameterInput *pin,
  //                                             MeshData *md) {
  //   for (auto &kamayan_unit : units) {
  //     if (kamayan_unit.second->ProblemGeneratorMesh != nullptr) {
  //       PARTHENON_REQUIRE_THROWS(kamayan_unit.second->ProblemGeneratorMeshBlock ==
  //                                    nullptr,
  //                                "Kamayan Unit can have only one of
  //                                ProblemGeneratorMesh "
  //                                "& ProblemGeneratorMeshBlock")
  //       kamayan_unit.second->ProblemGeneratorMesh(mesh, md);
  //     }
  //   }
  // };
  // may want to add unit callbacks for these as well...
  // app->PreFillDerivedBlock = advection_package::PreFill;
  // app->PostFillDerivedBlock = advection_package::PostFill;
  pman->ParthenonInitPackagesAndMesh();
  return KamayanDriver(units, runtime_parameters, pman->app_input.get(),
                       pman->pmesh.get());
}

void Finalize(std::shared_ptr<ParthenonManager> pman) { pman->ParthenonFinalize(); }
}  // namespace kamayan
