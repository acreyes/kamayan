#ifndef TYPES_HPP_
#define TYPES_HPP_

#include "application_input.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include <parthenon/parthenon.hpp>

namespace kamayan {
using Real = parthenon::Real;
// TODO(acreyes): move these to the Grid header
using BlockList_t = parthenon::BlockList_t;
using Mesh = parthenon::Mesh;
using MeshData = parthenon::MeshData<Real>;
using MeshBlockData = parthenon::MeshBlockData<Real>;
using MeshBlock = parthenon::MeshBlock;
}  // namespace kamayan

#endif  // TYPES_HPP_
