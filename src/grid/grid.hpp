#ifndef GRID_GRID_HPP_
#define GRID_GRID_HPP_

#include <parthenon/parthenon.hpp>

#include "grid/grid_types.hpp"
#include "utils/type_list.hpp"

namespace kamayan::grid {
template <typename... Ts>
auto GetPack(MeshData *md) {
  auto desc = parthenon::MakePackDescriptor<Ts...>(md);
  return desc.GetPack(md);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshData *md) {
  return GetPack<Ts...>(md);
}

}  // namespace kamayan::grid

#endif  // GRID_GRID_HPP_
