#ifndef KAMAYAN_FIELDS_HPP_
#define KAMAYAN_FIELDS_HPP_

#include <concepts>
#include <string>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "grid/geometry.hpp"
#include "grid/refinement_operations.hpp"
#include "interface/state_descriptor.hpp"
#include "kamayan/unit.hpp"
#include "kamayan_utils/strings.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"

namespace kamayan {
// import field related things from parthenon
using MetadataFlag = parthenon::MetadataFlag;
using Metadata = parthenon::Metadata;

enum class VariableRank { scalar, vector };

template <strings::CompileTimeString var_name, VariableRank vr = VariableRank::scalar,
          int... NCOMP>
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
  static constexpr std::size_t n_dims = sizeof...(NCOMP);
  static constexpr auto vname = var_name;
  static constexpr VariableRank variable_rank = vr;
};

template <typename T>
concept DenseVar = requires {
  { T::n_comps } -> std::same_as<const std::size_t &>;  // number of components
  { T::Shape() } -> std::same_as<std::vector<int>>;     // shape of the array
  { T::variable_rank } -> std::same_as<const VariableRank &>;
  requires std::same_as<decltype(std::declval<T &>().idx), const int>;
  requires NonTypeTemplateSpecialization<T, VariableBase>;
};

template <strings::CompileTimeString var_name, int... NCOMP>
struct SparseBase : public VariableBase<var_name, VariableRank::scalar, NCOMP...> {
  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION SparseBase(Ts &&...args)
      : VariableBase<var_name, VariableRank::scalar, NCOMP...>(
            std::forward<Ts>(args)...) {}

  // tensor indexer for sparse variable at sparse index n, with tensor
  // index (ds...)
  template <typename... Ts>
  requires(std::integral<Ts> && ...)
  KOKKOS_INLINE_FUNCTION static auto TI(const std::size_t n, Ts &&...ds) {
    static_assert(sizeof...(Ts) == sizeof...(NCOMP),
                  "Number of indexers matches tensor dimensions");
    constexpr auto n_inds = sizeof...(NCOMP);
    const int ts[]{NCOMP...}, vs[]{ds...};
    std::size_t stride = ts[0];
    std::size_t ncomp = n;
    for (int i = 0; i < n_inds; i++) {
      stride *= ts[i];
      ncomp += stride * vs[i];
    }
    return SparseBase<var_name, NCOMP...>(ncomp);
  }
};

template <typename T, std::size_t N>
struct TI_check {
  template <std::size_t... Is>
  static constexpr bool check(std::index_sequence<Is...>) {
    return std::same_as<decltype(T::TI(std::size_t{0}, Is...)), T>;
  }
  static constexpr bool value = check(std::make_index_sequence<N>{});
};

template <typename T>
concept SparseVar = requires {
  { T::n_comps } -> std::same_as<const std::size_t &>;  // number of components
  { T::Shape() } -> std::same_as<std::vector<int>>;     // shape of the array
  requires std::same_as<decltype(std::declval<T &>().idx), const int>;
  requires NonTypeTemplateSpecialization<T, SparseBase>;
  requires TI_check<T, T::n_dims>::value;
};

template <typename T>
requires(DenseVar<T>)
void AddField(KamayanUnit *pkg, std::vector<MetadataFlag> m,
              std::vector<int> shape = T::Shape()) {
  // can also add refinement ops here depending on the metadata
  if (T::variable_rank == VariableRank::vector) m.push_back(Metadata::Vector);
  auto md = Metadata(m, shape);

  const auto geometry = pkg->Configuration()->Get<Geometry>();
  auto register_ops = [&]<Geometry geom>() {
    md.RegisterRefinementOps<grid::ProlongateSharedMinMod<geom>,
                             grid::RestrictAverage<geom>>();
  };
  const auto handled = grid::GeometryOptions::dispatch(register_ops, geometry);
  PARTHENON_REQUIRE_THROWS(handled, "Geometry not handled for refinement operations.");
  pkg->AddField<T>(md);
}

template <typename... Ts>
requires(DenseVar<Ts> && ...)
void AddFields(KamayanUnit *pkg, std::vector<MetadataFlag> m) {
  (void)(AddField<Ts>(pkg, m), ...);
}

template <typename... Ts>
requires(DenseVar<Ts> && ...)
void AddFields(TypeList<Ts...>, KamayanUnit *pkg, std::vector<MetadataFlag> m) {
  AddFields<Ts...>(pkg, m);
}

template <typename T, typename V>
requires(SparseVar<T> && SparseVar<V>)
void AddSparseField(V, KamayanUnit *pkg, const std::vector<MetadataFlag> m,
                    const std::vector<int> &ids) {
  std::vector<int> shape = T::Shape();
  auto m_plus = m;
  m_plus.push_back(Metadata::Sparse);
  pkg->AddSparsePool<T>(Metadata(m_plus, shape), V::name(), ids);
}

template <typename... Ts, typename V>
requires(SparseVar<V> && (SparseVar<Ts> && ...))
void AddSparseFields(TypeList<Ts...>, V, KamayanUnit *pkg,
                     const std::vector<MetadataFlag> m, const std::vector<int> &ids) {
  (void)(AddSparseField<Ts>(V(), pkg, m, ids), ...);
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
using MOMENTUM = VariableBase<"momentum", VariableRank::vector, 3>;
using ENER = VariableBase<"ener">;
using MAG = VariableBase<"mag">;
// --8<-- [end:cons]

// primitives & Eos should be FillGhost?
using MAGC = VariableBase<"magc", VariableRank::vector, 3>;
using EINT = VariableBase<"eint">;
using PRES = VariableBase<"pres">;
using BMOD = VariableBase<"bulk modulus">;
using TEMP = VariableBase<"temp">;

using VELOCITY = VariableBase<"velocity", VariableRank::vector, 3>;

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
