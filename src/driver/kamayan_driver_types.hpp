#ifndef DRIVER_KAMAYAN_DRIVER_TYPES_HPP_
#define DRIVER_KAMAYAN_DRIVER_TYPES_HPP_
#include <parthenon/driver.hpp>
#include <parthenon/parthenon.hpp>

#include <dispatcher/options.hpp>

namespace kamayan {
// import some parthenon types into our namespace
using ParameterInput = parthenon::ParameterInput;
using ApplicationInput = parthenon::ApplicationInput;
using Packages_t = parthenon::Packages_t;
using StateDescriptor = parthenon::StateDescriptor;

using TaskCollection = parthenon::TaskCollection;
using TaskRegion = parthenon::TaskRegion;
using TaskList = parthenon::TaskList;
using TaskID = parthenon::TaskID;
using TaskListStatus = parthenon::TaskListStatus;
using TaskStatus = parthenon::TaskStatus;

using SimTime = parthenon::SimTime;

}  // namespace kamayan

#endif  // DRIVER_KAMAYAN_DRIVER_TYPES_HPP_
