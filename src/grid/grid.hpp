#ifndef GRID_GRID_HPP_
#define GRID_GRID_HPP_
#include <memory>
#include <type_traits>

#include <parthenon/parthenon.hpp>

#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "utils/type_list.hpp"

namespace kamayan::grid {

std::shared_ptr<KamayanUnit> ProcessUnit();

void Setup(Config *cfg, runtime_parameters::RuntimeParameters *rps);

template <typename... Ts, typename Container>
requires(std::is_same_v<Container, MeshData> || std::is_same_v<Container, MeshBlockData>)
auto GetPack(Container *md) {
  auto desc = parthenon::MakePackDescriptor<Ts...>(md);
  return desc.GetPack(md);
}

template <typename... Ts>
auto GetPack(MeshBlock *mb) {
  return GetPack<Ts...>(mb->meshblock_data.Get().get());
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshBlock *mb) {
  return GetPack<Ts...>(mb);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshData *md) {
  return GetPack<Ts...>(md);
}

}  // namespace kamayan::grid

#endif  // GRID_GRID_HPP_
