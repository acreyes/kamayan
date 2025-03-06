#include <memory>
#include <utility>

#include <parthenon_manager.hpp>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos
  std::shared_ptr<kamayan::ParameterInput> pin = std::move(pman.pinput);
  auto runtime_parameters = kamayan::runtime_parameters::RuntimeParameters(pin);
  auto units = kamayan::ProcessUnits();

  // Redefine defaults
  pman.app_input->ProcessPackages = [&](std::unique_ptr<kamayan::ParameterInput> &pin) {
    parthenon::Packages_t packages;
    for (auto &kamayan_unit : units) {
      if (kamayan_unit->Initialize != nullptr)
        packages.Add(kamayan_unit->Initialize(&runtime_parameters));
    }
    return packages;
  };
  // may want to add unit callbacks for these as well...
  // app->PreFillDerivedBlock = advection_package::PreFill;
  // app->PostFillDerivedBlock = advection_package::PostFill;

  pman.ParthenonInitPackagesAndMesh();
  {
    kamayan::KamayanDriver driver(units, pin, pman.app_input.get(), pman.pmesh.get());
    auto driver_status = driver.Execute();
  }

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
