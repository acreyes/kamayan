#ifndef GRID_GRID_HPP_
#define GRID_GRID_HPP_
#include <memory>
#include <set>
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
auto GetPack(Container *md, std::set<PDOpt> pack_opts = {}) {
  static auto desc = parthenon::MakePackDescriptor<Ts...>(md, {}, pack_opts);
  return desc.GetPack(md);
}

template <typename... Ts>
auto GetPack(MeshBlock *mb, std::set<PDOpt> pack_opts = {}) {
  return GetPack<Ts...>(mb->meshblock_data.Get().get(), pack_opts);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshBlock *mb, std::set<PDOpt> pack_opts = {}) {
  return GetPack<Ts...>(mb, pack_opts);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshData *md, std::set<PDOpt> pack_opts = {}) {
  return GetPack<Ts...>(md, pack_opts);
}

}  // namespace kamayan::grid

#endif  // GRID_GRID_HPP_
