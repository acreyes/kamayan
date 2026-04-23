#ifndef GRID_TESTS_TEST_GEOMETRY_HPP_
#define GRID_TESTS_TEST_GEOMETRY_HPP_

#include <memory>

#include <mesh/meshblock.hpp>

namespace kamayan {

std::shared_ptr<parthenon::MeshBlock> MakeTestMeshBlockCartesian3D();
std::shared_ptr<parthenon::MeshBlock> MakeTestMeshBlockCylindrical2D();
parthenon::UniformCartesian MakeCoordinatesCartesian3D();
parthenon::UniformCartesian MakeCoordinatesCylindrical2D();

}  // namespace kamayan
#endif  // GRID_TESTS_TEST_GEOMETRY_HPP_
