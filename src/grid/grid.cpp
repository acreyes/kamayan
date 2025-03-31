#include "grid.hpp"

#include <memory>
#include <string>

#include "kamayan/runtime_parameters.hpp"

namespace kamayan::grid {

std::shared_ptr<KamayanUnit> ProcessUnit() {
  auto grid_unit = std::make_shared<KamayanUnit>();
  grid_unit->Setup = Setup;
  return grid_unit;
}

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps) {
  // most of what we're doing here is wrapping the parthenon mesh related
  // input parameters as runtime parameters for the docs!
  // <parthenon/mesh>
  rps->Add<std::string>("parthenon/mesh", "refinement", "adaptive",
                        "Mesh refinement startegy.", {"adaptive", "static", "none"});
  rps->Add<int>("parthenon/mesh", "numlevel", 1, "Number of refinement levels.");

  rps->Add<int>("parthenon/mesh", "nx1", 32,
                "Number of cells across the domain at level 0.");
  rps->Add<int>("parthenon/mesh", "nx2", 32,
                "Number of cells across the domain at level 0. Set to 1 for 1D.");
  rps->Add<int>("parthenon/mesh", "nx3", 32,
                "Number of cells across the domain at level 0. Set to 1 for 2D.");

  rps->Add<Real>("parthenon/mesh", "x1min", 0.0, "Minimum x1 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x2min", 0.0, "Minimum x2 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x3min", 0.0, "Minimum x3 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x1max", 1.0, "Maximum x1 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x2max", 1.0, "Maximum x2 value of domain.");
  rps->Add<Real>("parthenon/mesh", "x3max", 1.0, "Maximum x3 value of domain.");

  rps->Add<std::string>("parthenon/mesh", "ix1_bc", "outflow",
                        "Inner boundary condition along x1.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ix2_bc", "outflow",
                        "Inner boundary condition along x2.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ix3_bc", "outflow",
                        "Inner boundary condition along x3.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox1_bc", "outflow",
                        "Outer boundary condition along x1.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox2_bc", "outflow",
                        "Outer boundary condition along x2.",
                        {"periodic", "outflow", "reflect", "user"});
  rps->Add<std::string>("parthenon/mesh", "ox3_bc", "outflow",
                        "Outer boundary condition along x3.",
                        {"periodic", "outflow", "reflect", "user"});

  // <parthenon/meshblock>
  rps->Add<int>("parthenon/meshblock", "nx1", 16, "Size of meshblocks in x1.");
  rps->Add<int>("parthenon/meshblock", "nx2", 16, "Size of meshblocks in x2.");
  rps->Add<int>("parthenon/meshblock", "nx3", 16, "Size of meshblocks in x3.");
}

}  // namespace kamayan::grid
