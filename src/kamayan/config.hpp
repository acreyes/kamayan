#ifndef KAMAYAN_CONFIG_HPP_
#define KAMAYAN_CONFIG_HPP_

#include <string>

#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"

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

  template <poly_opt T>
  void Add(T value) {
    _params.Add(OptInfo<T>::key(), value, Mutability::Restart);
  }

  template <poly_opt T>
  void Update(T value) {
    _params.Update(OptInfo<T>::key(), value);
  }

  template <poly_opt T>
  const T &Get() const {
    return _params.template Get<T>(OptInfo<T>::key());
  }

 private:
  using Mutability = parthenon::Params::Mutability;
  parthenon::Params _params;
};
}  // namespace kamayan

#endif  // KAMAYAN_CONFIG_HPP_
