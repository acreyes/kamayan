#ifndef GRID_TESTS_TEST_GRID_HPP_
#define GRID_TESTS_TEST_GRID_HPP_

#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "mesh/meshblock.hpp"

namespace kamayan {
// Builds a block list and initializes a MeshData from that block list
// We export these in case additional tests need to use a pack
// that can be built from the MeshData
parthenon::BlockList_t MakeTestBlockList(const std::shared_ptr<StateDescriptor> pkg,
                                         const int NBLOCKS, const int NSIDE,
                                         const int NDIM);
MeshData MakeTestMeshData(parthenon::BlockList_t block_list);
}  // namespace kamayan
#endif  // GRID_TESTS_TEST_GRID_HPP_
