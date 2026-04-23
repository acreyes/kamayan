#include "kamayan/kamayan.hpp"

#include <cstdlib>
#include <memory>
#include <utility>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "kamayan/unit.hpp"
#include "parthenon_manager.hpp"
#include "physics/material_properties/eos/eos.hpp"
#include "physics/material_properties/eos/eos_types.hpp"
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

KamayanDriver InitPackages(std::shared_ptr<ParthenonManager> pman,
                           std::shared_ptr<UnitCollection> units) {
  auto pin = pman->pinput.get();
  auto runtime_parameters =
      std::make_shared<kamayan::runtime_parameters::RuntimeParameters>(pin);

  grid::RegisterBoundaryConditions(pman->app_input.get());

  auto config = std::make_shared<Config>();
  for (auto &kamayan_unit : *units) {
    kamayan_unit.second->SetUnits(units);
    // At this point we can allow params in our units
    kamayan_unit.second->UnlockParams();

    // Always initialize resources first - this sets config_ which is needed
    // by InitializeData even if SetupParams is not registered
    kamayan_unit.second->InitResources(runtime_parameters, config);

    // TODO(acreyes): these callbacks should not depend on the order of execution
    // so we should add a CallBackRegistration constructor that raises a runtime
    // error if someone tries to suggest an order
    if (kamayan_unit.second->SetupParams.IsRegistered()) {
      kamayan_unit.second->SetupParams(kamayan_unit.second.get());
      // Sync UnitData parameters from RuntimeParameters (input file takes precedence)
      for (auto &[name, ud] : kamayan_unit.second->AllData()) {
        ud.Setup(runtime_parameters, config);
        ud.SetupComplete();
      }
    }
  }

  // Add app_input callbacks, but take care that these are released in
  // kamayan::Finalize below in case they're holding references to
  // any KamayanUnits
  pman->app_input->ProcessPackages =
      [units, config](std::unique_ptr<kamayan::ParameterInput> &pin) {
        parthenon::Packages_t packages;
        auto config_pkg = std::make_shared<kamayan::StateDescriptor>("Config");
        config_pkg->AddParam("config", config);
        packages.Add(config_pkg);
        // TODO(acreyes): these callbacks should not depend on the order of execution
        // so we should add a CallBackRegistration constructor that raises a runtime
        // error if someone tries to suggest an order
        for (auto &kamayan_unit : *units) {
          auto unit = kamayan_unit.second;
          if (unit->InitializeData.IsRegistered()) {
            unit->InitializePackage(unit);
            unit->InitializeData(unit.get());
          }
          packages.Add(unit);
        }
        return packages;
      };

  pman->app_input->ProblemGenerator = [units](MeshBlock *mb, ParameterInput *pin) {
    units->AddTasksDAG(
        [](KamayanUnit *u) -> auto & { return u->ProblemGeneratorMeshBlock; },
        [&](KamayanUnit *u) { u->ProblemGeneratorMeshBlock(mb); },
        "ProblemGeneratorMeshBlock");
  };

  pman->app_input->MeshPostInitialization = [units](Mesh *mesh, ParameterInput *pin,
                                                    MeshData *md) {
    units->AddTasksDAG([](KamayanUnit *u) -> auto & { return u->PostMeshInitialization; },
                       [&](KamayanUnit *u) { u->PostMeshInitialization(md); },
                       "PostMeshInitialization");
  };

  pman->app_input->InitMeshBlockUserData = [units](MeshBlock *mb, ParameterInput *pin) {
    units->AddTasksDAG([](KamayanUnit *u) -> auto & { return u->InitMeshBlockData; },
                       [&](KamayanUnit *u) { u->InitMeshBlockData(mb); },
                       "InitMeshBlockUserData");
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

void Finalize(std::shared_ptr<ParthenonManager> pman) {
  // Clear lambda callbacks that capture units to break reference cycles
  if (pman->app_input) {
    pman->app_input->ProcessPackages = nullptr;
    pman->app_input->ProblemGenerator = nullptr;
    pman->app_input->MeshPostInitialization = nullptr;
    pman->app_input->InitMeshBlockUserData = nullptr;
    pman->app_input->PreStepMeshUserWorkInLoop = nullptr;
  }
  pman->ProcessPackages = nullptr;
  pman->ParthenonFinalize();
}
}  // namespace kamayan
