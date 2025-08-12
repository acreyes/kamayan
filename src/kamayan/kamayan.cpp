#include "kamayan/kamayan.hpp"

#include <cstdlib>
#include <memory>
#include <utility>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/unit.hpp"
#include "parthenon_manager.hpp"
#include "physics/eos/eos.hpp"
#include "physics/eos/eos_types.hpp"
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
    // TODO(acreyes): I don't think we should be accessing unit_data_collection
    // directly?
    if (kamayan_unit.second->SetupParams != nullptr) {
      kamayan_unit.second->unit_data_collection.Init(runtime_parameters, config);
      kamayan_unit.second->SetupParams(kamayan_unit.second->unit_data_collection);
    }
  }

  pman->app_input->ProcessPackages = [&](std::unique_ptr<kamayan::ParameterInput> &pin) {
    parthenon::Packages_t packages;
    // start with the config package, then go into all of our units
    auto config_pkg = std::make_shared<kamayan::StateDescriptor>("Config");
    config_pkg->AddParam("config", config);
    packages.Add(config_pkg);
    for (auto &kamayan_unit : units) {
      if (kamayan_unit.second->InitializeData != nullptr) {
        auto unit = kamayan_unit.second;
        auto pkg = std::make_shared<StateDescriptor>(kamayan_unit.second->Name());
        unit->unit_data_collection.SetPackage(pkg);
        // TODO(acreyes): we shouldn't be the one iterating over all the UnitDatas
        for (auto &ud : kamayan_unit.second->unit_data_collection.Data()) {
          ud.second.Initialize(pkg);
        }
        kamayan_unit.second->InitializeData(kamayan_unit.second->unit_data_collection);
        packages.Add(pkg);
      }
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

  pman->app_input->MeshPostInitialization = [&](Mesh *mesh, ParameterInput *pin,
                                                MeshData *md) {
    for (auto &kamayan_unit : units) {
      if (kamayan_unit.second->PrepareConserved != nullptr) {
        kamayan_unit.second->PrepareConserved(md);
      }
    }
  };

  // maybe this should be a part of all the units...
  pman->app_input->PreStepMeshUserWorkInLoop = driver::PreStepUserWorkInLoop;

  // parthenon also has the option for problem generation using a MeshData object
  // instead of a meshblock. We can only have one or the other wrt the MeshBlock variant
  // we use above. This could be more performant, but also not as straighforward to
  // implement and it is just initialization after all...
  //
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
