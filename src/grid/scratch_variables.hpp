#ifndef GRID_SCRATCH_VARIABLES_HPP_
#define GRID_SCRATCH_VARIABLES_HPP_

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <parthenon/parthenon.hpp>

#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"

namespace kamayan {
template <int... NCOMPS>
using scratch_base_t = parthenon::variable_names::base_t<true, NCOMPS...>;

// scratch variables are registered per-unit as sets of variables that
// can each have their own shape, with the goal that the allocated memory
// on each meshblock can be shared between packages, with no guarantees
// about the persistence between unit layers.
//
// Some loose requirements
// * scratch variables can have arbitrary shape
//    * shape
//    * TopologicalType
// * even though they point to the same memory each unit might have their own
//   variable types they will want to use to reference it
// * There can't be any sane way to use these unless a unit is only allowed
//   a single unique call to register its scratch variables
KOKKOS_INLINE_FUNCTION constexpr auto TopologicalTypeToMetaData(TopologicalType tt) {
  using TT = TopologicalType;
  if (tt == TT::Face) {
    return Metadata::Face;
  } else if (tt == TT::Edge) {
    return Metadata::Edge;
  } else if (tt == TT::Node) {
    return Metadata::Node;
  }
  return Metadata::Cell;
}

inline std::string TopologicalTypeToString(TopologicalType tt) {
  using TT = TopologicalType;
  if (tt == TT::Face) {
    return "face";
  } else if (tt == TT::Edge) {
    return "edge";
  } else if (tt == TT::Node) {
    return "node";
  }
  return "cell";
}

inline std::string range_regex(unsigned a, unsigned b) {
  std::ostringstream pattern;
  pattern << "((" << std::to_string(a) << ")";
  for (int i = a + 1; i <= b; i++) {
    pattern << "|(" << std::to_string(i) << ")";
  }
  pattern << ")";
  return pattern.str();
}

template <size_t N>
struct CompileTimeString {
  char value[N];

  explicit(false) constexpr CompileTimeString(const char (&str)[N]) {
    for (size_t i = 0; i < N; ++i)
      value[i] = str[i];
  }
};

template <CompileTimeString name, TopologicalType TT, int... NCOMPS>
struct ScratchVariable {
  using base_t = scratch_base_t<NCOMPS...>;
  static constexpr std::string_view str_name{name.value, sizeof(name.value)};
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
};

template <TopologicalType TT, typename SV>
concept ScratchType =
    requires {
      { SV::type } -> std::same_as<const TopologicalType &>;
      { SV::ncomps } -> std::same_as<const int &>;
      { SV::size } -> std::same_as<const int &>;
      { SV::shape } -> std::same_as<const std::array<int, SV::ncomps> &>;
    } && TT == SV::type && SV::ncomps == SV::ncomps && SV::size == SV::size &&
    SV::shape == SV::shape;

template <typename SV, int lower>
requires(NonTypeTemplateSpecialization<SV, ScratchVariable>)
struct ScratchVariable_impl : public SV::base_t {
  using type = SV;
  static constexpr int lb = lower;
  static constexpr int ub = lower + SV::size - 1;

  static std::string name() {
    return "scratch_" + TopologicalTypeToString(SV::type) + "_" + range_regex(lb, ub);
  }
};

namespace impl {
template <typename...>
struct SVList_impl {};

template <typename SV>
requires(NonTypeTemplateSpecialization<SV, ScratchVariable>)
struct SVList_impl<SV> {
  using type = ScratchVariable_impl<SV, 0>;
  using value = TypeList<type>;
};

template <typename SV, typename... SVs>
requires(ScratchType<SV::type, SVs> && ... &&
         NonTypeTemplateSpecialization<SV, ScratchVariable>)
struct SVList_impl<SV, SVs...> {
  using list = SVList_impl<SVs...>;
  using type = ScratchVariable_impl<SV, list::type::ub + 1>;
  using value = ConcatTypeLists_t<TypeList<type>, typename list::value>;
};
}  // namespace impl

// should take ScratchVariables...
template <TopologicalType TT, typename... SVs>
requires(ScratchType<TT, SVs> && ...)
struct ScratchVariableList {
  static constexpr int n_vars = (SVs::size + ...);
  using TL = TypeList<SVs...>;
  using list = impl::SVList_impl<SVs...>;

  template <typename SV>
  using type = list::value::template type<TL::template Idx<SV>()>;

  auto GetVarNames() {
    std::array<std::string, n_vars> vars;
    auto base = "scratch_" + TopologicalTypeToString(TT) + "_";
    for (int i = 0; i < n_vars; i++) {
      vars[i] = base + std::to_string(i);
    }
    return vars;
  }
};

}  // namespace kamayan

#endif  // GRID_SCRATCH_VARIABLES_HPP_
