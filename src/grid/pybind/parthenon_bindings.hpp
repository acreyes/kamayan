#ifndef GRID_PYBIND_PARTHENON_BINDINGS_HPP_
#define GRID_PYBIND_PARTHENON_BINDINGS_HPP_

#include "kamayan/pybind/kamayan_nanobind.h"
namespace kamayan {
void parthenon_module(nanobind::module_ &m);
}
#endif  // GRID_PYBIND_PARTHENON_BINDINGS_HPP_
