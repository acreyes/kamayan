#ifndef KAMAYAN_CONFIG_HPP_
#define KAMAYAN_CONFIG_HPP_
#include <memory>

#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"

namespace kamayan {
class Config {
  // wrap parthenon::Params and limit the public API
  // to make it suitable for holding global runtime options
  // usable in the dispatcher
  // There should only ever be one value for a given option
  // and so we restrict the keys stored in the Params with
  // OptInfo<T>::key()
 public:
  Config() {}

  template <PolyOpt T>
  void Add(T value) {
    _params.Add(OptInfo<T>::key(), value, Mutability::Restart);
  }

  template <PolyOpt T>
  void Update(T value) {
    _params.Update(OptInfo<T>::key(), value);
  }

  template <PolyOpt T>
  const T &Get() const {
    return _params.template Get<T>(OptInfo<T>::key());
  }

 private:
  using Mutability = parthenon::Params::Mutability;
  parthenon::Params _params;
};

std::shared_ptr<Config> GetConfig(MeshData *md);
std::shared_ptr<Config> GetConfig(MeshBlock *mb);

}  // namespace kamayan

#endif  // KAMAYAN_CONFIG_HPP_
