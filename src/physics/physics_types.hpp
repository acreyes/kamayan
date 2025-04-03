#ifndef PHYSICS_PHYSICS_TYPES_HPP_
#define PHYSICS_PHYSICS_TYPES_HPP_

#include "dispatcher/options.hpp"

namespace kamayan {
POLYMORPHIC_PARM(Fluid, oneT, threeT);
POLYMORPHIC_PARM(Mhd, off, ct);

using FluidOptions = OptList<Fluid, Fluid::oneT>;  //, Fluid::threeT>;
using MhdOptions = OptList<Mhd, Mhd::off, Mhd::ct>;
}  // namespace kamayan

#endif  // PHYSICS_PHYSICS_TYPES_HPP_
