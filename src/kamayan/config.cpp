#include "kamayan/config.hpp"

#include <memory>

#include "grid/grid_types.hpp"

namespace kamayan {
std::shared_ptr<Config> GetConfig(MeshData *md) {
  return md->GetMeshPointer()->packages.Get("Config")->Param<std::shared_ptr<Config>>(
      "config");
}
}  // namespace kamayan
