#ifndef GRID_GRID_HPP_
#define GRID_GRID_HPP_
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan_utils/type_list.hpp"

namespace kamayan::grid {

std::shared_ptr<KamayanUnit> ProcessUnit();

void SetupParams(KamayanUnit *unit);
void InitializeData(KamayanUnit *unit);
void InitMeshBlockData(MeshBlock *mb);

void RegisterBoundaryConditions(parthenon::ApplicationInput *app);

template <typename Container>
requires(std::is_same_v<Container, MeshData> || std::is_same_v<Container, MeshBlockData>)
auto GetPackDescriptor(Container *md, std::vector<parthenon::MetadataFlag> m = {},
                       std::set<PDOpt> pack_opts = {}) {
  auto resolved_pkg = md->GetMeshPointer()->resolved_packages.get();
  auto vars = resolved_pkg->GetVariableNames(parthenon::Metadata::FlagCollection(m));
  return parthenon::MakePackDescriptor(resolved_pkg, vars, {}, pack_opts);
}

namespace impl {

template <typename>
struct PackGetter {};

template <typename... Ts>
requires(UniqueTypes<Ts...>)
struct PackGetter<TypeList<Ts...>> {
  static auto Get(StateDescriptor *pkg, MeshData *md) {
    static auto desc = parthenon::MakePackDescriptor<Ts...>(pkg);
    return desc.GetPack(md);
  }

  template <typename Container>
  requires(std::is_same_v<Container, MeshData> ||
           std::is_same_v<Container, MeshBlockData>)
  static auto Get(Container *md, std::set<PDOpt> pack_opts = {}) {
    static auto desc = parthenon::MakePackDescriptor<Ts...>(md, {}, pack_opts);
    return desc.GetPack(md);
  }
};

template <typename... Ts, typename... Args>
auto GetPack(TypeList<Ts...>, Args &&...args) {
  return PackGetter<TypeSet<Ts...>>::Get(std::forward<Args>(args)...);
}

}  // namespace impl

// The state descriptor overloads are only really
// needed in testing to create a sparse pack without the parthenon
// resolved package existing
template <typename... Ts>
auto GetPack(StateDescriptor *pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), pkg, md);
}

template <typename... Ts>
auto GetPack(std::shared_ptr<StateDescriptor> pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), pkg.get(), md);
}

template <typename... Ts>
auto GetPack(KamayanUnit *pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), static_cast<StateDescriptor *>(pkg), md);
}

template <typename... Ts>
auto GetPack(std::shared_ptr<KamayanUnit> pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), static_cast<StateDescriptor *>(pkg.get()), md);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, StateDescriptor *pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), pkg, md);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, std::shared_ptr<StateDescriptor> pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), pkg.get(), md);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, KamayanUnit *pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), static_cast<StateDescriptor *>(pkg), md);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, std::shared_ptr<KamayanUnit> pkg, MeshData *md) {
  return impl::GetPack(TypeList<Ts...>(), static_cast<StateDescriptor *>(pkg.get()), md);
}

template <typename... Ts, typename Container>
auto GetPack(Container *md, std::set<PDOpt> pack_opts = {}) {
  return impl::GetPack(TypeList<Ts...>(), md, pack_opts);
}

template <typename... Ts>
auto GetPack(MeshBlock *mb, std::set<PDOpt> pack_opts = {}) {
  return impl::GetPack(TypeList<Ts...>(), mb->meshblock_data.Get().get(), pack_opts);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshBlock *mb, std::set<PDOpt> pack_opts = {}) {
  return impl::GetPack(TypeList<Ts...>(), mb->meshblock_data.Get().get(), pack_opts);
}

template <typename... Ts>
auto GetPack(TypeList<Ts...>, MeshData *md, std::set<PDOpt> pack_opts = {}) {
  return impl::GetPack(TypeList<Ts...>(), md, pack_opts);
}

template <typename... Ts, typename... Vs, typename... Args>
auto GetPack(TypeList<Ts...>, TypeList<Vs...>, Args &&...args) {
  return GetPack(TypeList<Ts..., Vs...>(), std::forward<Args>(args)...);
}

TaskStatus FluxesToDuDt(MeshData *md, MeshData *dudt);
TaskID ApplyDuDt(TaskID prev, TaskList &tl, MeshData *mbase, MeshData *md0, MeshData *md1,
                 MeshData *dudt_data, const Real &beta, const Real &dt);

}  // namespace kamayan::grid

#endif  // GRID_GRID_HPP_
