#ifndef PHYSICS_MATERIAL_PROPERTIES_MATERIAL_TYPES_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_MATERIAL_TYPES_HPP_

#include "dispatcher/options.hpp"

namespace kamayan {
namespace material {
// material properties use a variant to expose the runtime
// polymorphism of various material properties that need
// to be called on a single zone.
// Each implementation in the variant also needs to export
// the variables that it is expecting
//
// so then we need a type trait that can be built from the option factory method
// this built type trait should export the variable types that it needs
// as well as the type in the variant
//
// EOS is not a great example because the multitype variant depends
// on the number of species, rather than the

}  // namespace material
}  // namespace kamayan
#endif  // PHYSICS_MATERIAL_PROPERTIES_MATERIAL_TYPES_HPP_
