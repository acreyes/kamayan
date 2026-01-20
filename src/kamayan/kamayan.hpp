#ifndef KAMAYAN_KAMAYAN_HPP_
#define KAMAYAN_KAMAYAN_HPP_

#include <memory>

#include <parthenon_manager.hpp>

#include "driver/kamayan_driver.hpp"
#include "kamayan/unit.hpp"

namespace kamayan {

std::shared_ptr<parthenon::ParthenonManager> InitEnv(int argc, char *argv[]);
KamayanDriver InitPackages(std::shared_ptr<parthenon::ParthenonManager> pman,
                           std::shared_ptr<UnitCollection> units);
void Finalize();

}  // namespace kamayan

#endif  // KAMAYAN_KAMAYAN_HPP_
