#ifndef KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
#define KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
#define kamayan_PYTHON
#include <nanobind/nanobind.h>

namespace kamayan {
void state_descrptor(nanobind::module_ &m);
void parthenon_manager(nanobind::module_ &m);
void unit_data_collection(nanobind::module_ &m);
}  // namespace kamayan
#endif  // KAMAYAN_PYBIND_KAMAYAN_PY11_HPP_
