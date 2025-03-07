#ifndef TYPES_HPP_
#define TYPES_HPP_

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
