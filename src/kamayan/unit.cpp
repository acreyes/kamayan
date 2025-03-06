#include "unit.hpp"

#include <list>
#include <memory>

namespace kamayan {
std::list<std::shared_ptr<KamayanUnit>> ProcessUnits() {
  std::list<std::shared_ptr<KamayanUnit>> unit_list;
  // push additional units into unit_list here
  // take special note of the order that these get added, as this will be the order that
  // they are called by driver
  // *** first non-evolution units e.g., material properties, eos, etc. ***
  // *** Next those that contribute during the RK-stages ***
  // *** Finally those that are operator split ***
  return unit_list;
}
}  // namespace kamayan
