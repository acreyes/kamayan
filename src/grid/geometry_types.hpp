#ifndef GRID_GEOMETRY_TYPES_HPP_
#define GRID_GEOMETRY_TYPES_HPP_
#include "dispatcher/options.hpp"
namespace kamayan {
POLYMORPHIC_PARM(Geometry, cartesian, cylindrical);

namespace grid {
using GeometryOptions = OptList<Geometry, Geometry::cartesian, Geometry::cylindrical>;
}
}  // namespace kamayan
#endif  // GRID_GEOMETRY_TYPES_HPP_
