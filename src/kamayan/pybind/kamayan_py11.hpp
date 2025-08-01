#ifndef KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
#define KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
#include <pybind11/pybind11.h>

namespace kamayan {
void state_descrptor(pybind11::module_ &m);
void parthenon_manager(pybind11::module_ &m);
}  // namespace kamayan
#endif  // KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
