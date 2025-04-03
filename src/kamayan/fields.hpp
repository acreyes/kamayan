#ifndef KAMAYAN_FIELDS_HPP_
#define KAMAYAN_FIELDS_HPP_

#include <string>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "interface/state_descriptor.hpp"
#include "utils/strings.hpp"

namespace kamayan {
// import field related things from parthenon
using MetadataFlag = parthenon::MetadataFlag;
using Metadata = parthenon::Metadata;

//! @brief Define a variable
//! @details creates a variable struct that can be registered to the grid with the
//! grid::AddField method used to index into a pack
//! \code{.cpp} pack(lb, grid::TopologicalElement::CC, varname(), k, j, i)\endcode
//! @param[in]  varname Type name that will be used in the struct
//! @param[in]  varnameStr variable name used in outputs
//! @param[in]  optional comma separated list of user-flags that will be associated with
//! the variable. These can be passes as strings in the `userFlags` argument to AddField.
//! @return
//! @exception
using variable_base_t = parthenon::variable_names::base_t<false>;
#define VARIABLE(varname, ...)                                                           \
  struct varname : public variable_base_t {                                              \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : variable_base_t(std::forward<Ts>(args)...) {}                                  \
    static std::string name() { return strings::lower(#varname); }                       \
    static Metadata flags() { return {__VA_ARGS__}; }                                    \
  }

template <typename T>
void AddField(parthenon::StateDescriptor *pkg, Metadata m) {
  // can also add refinement ops here depending on the metadata
  pkg->AddField<T>(m);
}

template <typename... Ts>
void AddFields(parthenon::StateDescriptor *pkg, Metadata m) {
  (void)(AddField<Ts>(pkg, m), ...);
}

template <typename... Ts>
void AddFields(TypeList<Ts...>, parthenon::StateDescriptor *pkg, Metadata m) {
  AddFields<Ts...>(pkg, m);
}
//! @brief default flags for cell-centered variables
//! @details can be used as the flags argument in AddField
//! @param[in] additional comma separated flag_t types that will be appended to the
//! defaults.
//! @return
//! @exception
#define CENTER_FLAGS(...)                                                                \
  {Metadata::Cell, Metadata::Restart, Metadata::FillGhost, __VA_ARGS__}
//! @brief default flags for face-centered variables
//! @details can be used as the flags argument in AddField
//! @param[in] additional comma separated flag_t types that will be appended to the
//! defaults.
//! @return
//! @exception
#define FACE_FLAGS(...) {Metadata::Face, Metadata::FillGhost, __VA_ARGS__}

// all recognized kamayan fields

// conserved variables
VARIABLE(DENS);
VARIABLE(MOMENTUM);
VARIABLE(ENER);
VARIABLE(MAG);

// primitives & Eos should be FillGhost?
VARIABLE(MAGC);
VARIABLE(EINT);
VARIABLE(PRES);
VARIABLE(GAMC);
VARIABLE(GAME);
VARIABLE(TEMP);

VARIABLE(VELOCITY);

// 3T
VARIABLE(TELE);
VARIABLE(EELE);
VARIABLE(PELE);
VARIABLE(TION);
VARIABLE(EION);
VARIABLE(PION);
}  // namespace kamayan

#endif  // KAMAYAN_FIELDS_HPP_
