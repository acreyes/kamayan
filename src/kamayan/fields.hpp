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

constexpr std::size_t GetNComp(std::vector<std::size_t> shape) {
  std::size_t n = 1;
  for (const auto &s : shape) {
    n *= s;
  }
  return n;
}

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
#define VARIABLE_IMPL(varname, shape)                                                    \
  struct varname : public variable_base_t {                                              \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : variable_base_t(std::forward<Ts>(args)...) {}                                  \
    static std::string name() { return strings::lower(#varname); }                       \
    static std::vector<int> Shape() { return shape; }                                    \
    static constexpr std::size_t n_comps = GetNComp(shape);                              \
  }

// choose how to call VARIABLE_IMPL based on the number of args passed to VARIABLE
#define VARIABLE_1(varname) VARIABLE_IMPL(varname, {1})
#define VARIABLE_2(varname, shape) VARIABLE_IMPL(varname, shape)
#define GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define VARIABLE_CHOOSER(...) GET_3RD_ARG(__VA_ARGS__, VARIABLE_2, VARIABLE_1)
#define VARIABLE(...) VARIABLE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

template <typename T>
void AddField(parthenon::StateDescriptor *pkg, std::vector<MetadataFlag> m,
              std::vector<int> shape = T::Shape()) {
  // can also add refinement ops here depending on the metadata
  pkg->AddField<T>(Metadata(m, shape));
}

template <typename... Ts>
void AddFields(parthenon::StateDescriptor *pkg, std::vector<MetadataFlag> m) {
  (void)(AddField<Ts>(pkg, m), ...);
}

template <typename... Ts>
void AddFields(TypeList<Ts...>, parthenon::StateDescriptor *pkg,
               std::vector<MetadataFlag> m) {
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
VARIABLE(MOMENTUM, {3});
VARIABLE(ENER);
VARIABLE(MAG);

// primitives & Eos should be FillGhost?
VARIABLE(MAGC, {3});
VARIABLE(EINT);
VARIABLE(PRES);
VARIABLE(GAMC);
VARIABLE(GAME);
VARIABLE(TEMP);

VARIABLE(VELOCITY, {3});

// 3T
VARIABLE(TELE);
VARIABLE(EELE);
VARIABLE(PELE);
VARIABLE(TION);
VARIABLE(EION);
VARIABLE(PION);
}  // namespace kamayan

#endif  // KAMAYAN_FIELDS_HPP_
