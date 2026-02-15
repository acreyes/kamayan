#ifndef KAMAYAN_FIELDS_HPP_
#define KAMAYAN_FIELDS_HPP_

#include <concepts>
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

// dense variables are always allocated, and so have
// to statically declare their size/shape at compile time
template <typename T>
concept DenseVar = requires {
  { T::n_comps } -> std::same_as<const std::size_t &>;  // number of components
  { T::Shape() } -> std::same_as<std::vector<int>>;     // shape of the array
  requires std::same_as<decltype(std::declval<T &>().idx), const int>;
};

template <strings::CompileTimeString var_name, int... NCOMP>
struct VariableBase : public parthenon::variable_names::base_t<false, NCOMP...> {
  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION VariableBase(Ts &&...args)
      : parthenon::variable_names::base_t<false, NCOMP...>(std::forward<Ts>(args)...) {}

  static std::string name() { return std::string(var_name.value); }
  static std::vector<int> Shape() {
    if constexpr (sizeof...(NCOMP) > 0) {
      return {NCOMP...};
    }
    return {1};
  }
  static constexpr std::size_t n_comps = (1 * ... * NCOMP);
};

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

template <template <typename...> typename T, DenseVar... Ts>
constexpr int count_components(T<Ts...>) {
  return (0 + ... + Ts::n_comps);
}
// all recognized kamayan fields

// --8<-- [start:cons]
// conserved variables
using DENS = VariableBase<"dens">;
// will register with shape=std::vector<int>{3}
using MOMENTUM = VariableBase<"momentum", 3>;
using ENER = VariableBase<"ener">;
using MAG = VariableBase<"mag">;
// --8<-- [end:cons]

// primitives & Eos should be FillGhost?
using MAGC = VariableBase<"magc", 3>;
using EINT = VariableBase<"eint">;
using PRES = VariableBase<"pres">;
using GAMC = VariableBase<"gamc">;
using GAME = VariableBase<"game">;
using TEMP = VariableBase<"temp">;

using VELOCITY = VariableBase<"velocity", 3>;

// 3T
using TELE = VariableBase<"tele">;
using EELE = VariableBase<"eele">;
using PELE = VariableBase<"pele">;
using TION = VariableBase<"tion">;
using EION = VariableBase<"eion">;
using PION = VariableBase<"pion">;
using TRAD = VariableBase<"trad">;
using ERAD = VariableBase<"erad">;
using PRAD = VariableBase<"prad">;

// derived
using DIVB = VariableBase<"divb">;
}  // namespace kamayan

#endif  // KAMAYAN_FIELDS_HPP_
