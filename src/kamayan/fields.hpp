#ifndef KAMAYAN_FIELDS_HPP_
#define KAMAYAN_FIELDS_HPP_

#include <string>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

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
// TODO(acreyes): make userFlags be metadata instead
using variable_base_t = parthenon::variable_names::base_t<false>;
#define VARIABLE(varname, ...)                                                           \
  struct varname : public variable_base_t {                                              \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : variable_base_t(std::forward<Ts>(args)...) {}                                  \
    static std::string name() { return strings::lower(#varname); }                       \
    static Metadata flags() { return {__VA_ARGS__}; }                                    \
  }
//! @brief default flags for cell-centered variables
//! @details can be used as the flags argument in AddField
//! @param[in] additional comma separated flag_t types that will be appended to the
//! defaults.
//! @return
//! @exception
#define CENTER_FLAGS(...)                                                                \
  {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, __VA_ARGS__}
//! @brief default flags for face-centered variables
//! @details can be used as the flags argument in AddField
//! @param[in] additional comma separated flag_t types that will be appended to the
//! defaults.
//! @return
//! @exception
#define FACE_FLAGS(...)                                                                  \
  {Metadata::Face, Metadata::Independent, Metadata::FillGhost, __VA_ARGS__}

// all recognized kamayan fields

// conserved variables
VARIABLE(DENS, CENTER_FLAGS());

// primitives & Eos should be FillGhost?
VARIABLE(EINT, {Metadata::Cell});
VARIABLE(PRES, {Metadata::Cell});
VARIABLE(GAMC, {Metadata::Cell});
VARIABLE(GAME, {Metadata::Cell});
VARIABLE(TEMP, {Metadata::Cell});

VARIABLE(VELOCITY, {Metadata::Cell}, std::vector<int>{3});

// 3T
VARIABLE(TELE, {Metadata::Cell});
VARIABLE(EELE, {Metadata::Cell});
VARIABLE(PELE, {Metadata::Cell});
VARIABLE(TION, {Metadata::Cell});
VARIABLE(EION, {Metadata::Cell});
VARIABLE(PION, {Metadata::Cell});
}  // namespace kamayan

#endif  // KAMAYAN_FIELDS_HPP_
