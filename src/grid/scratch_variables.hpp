#ifndef GRID_SCRATCH_VARIABLES_HPP_
#define GRID_SCRATCH_VARIABLES_HPP_

#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

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
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  static constexpr std::string_view str_name{name.value, sizeof(name.value)};
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
  static std::string Name() { return std::string(name.value); }
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
  static constexpr auto shape = SV::shape;

  // Single int constructor - add lb offset for proper indexing
  KOKKOS_INLINE_FUNCTION explicit ScratchVariable_impl(int idx) : SV::base_t(lb + idx) {}

  // Default constructor - set idx to lb
  KOKKOS_INLINE_FUNCTION ScratchVariable_impl() : SV::base_t(lb) {}

  // Multi-index constructor - forward as-is (multi-dimensional indices)
  template <class... Ts>
  requires(sizeof...(Ts) > 1)
  KOKKOS_INLINE_FUNCTION ScratchVariable_impl(Ts &&...args)
      : SV::base_t(std::forward<Ts>(args)...) {}

  static std::string name() {
#ifdef KAMAYAN_DEBUG_SCRATCH
    return "scratch_" + SV::Name();
#else
    return "scratch_" + TopologicalTypeToString(SV::type) + "_" + range_regex(lb, ub);
#endif
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

template <typename V, typename... SVs>
requires(NonTypeTemplateSpecialization<V, ScratchVariable> &&
         (ScratchType<V::type, SVs> && ...))
struct ScratchVariableList {
  static constexpr TopologicalType TT = V::type;
  static constexpr int n_vars = V::size + (SVs::size + ... + 0);
  using TL = TypeList<V, SVs...>;
  using list = impl::SVList_impl<V, SVs...>;

  template <typename SV>
  using type = list::value::template type<TL::template Idx<SV>()>;

  static const auto GetVarNames() {
    std::array<std::string, n_vars> vars;
    auto base = "scratch_" + TopologicalTypeToString(TT) + "_";
    for (int i = 0; i < n_vars; i++) {
      vars[i] = base + std::to_string(i);
    }
    return vars;
  }
};

template <typename SL>
requires(TemplateSpecialization<SL, ScratchVariableList>)
void AddScratch(StateDescriptor *pkg) {
#ifdef KAMAYAN_DEBUG_SCRATCH
  // in debug mode each scratch variable has its own unique name
  type_for(typename SL::list::value(), [&]<typename T>(const T &) {
    auto m = Metadata(
        {TopologicalTypeToMetaData(SL::TT), Metadata::Derived, Metadata::Overridable},
        std::vector<int>(std::begin(T::shape), std::end(T::shape)));
    pkg->AddField<T>(m);
  });
#else
  auto m = Metadata(
      {TopologicalTypeToMetaData(SL::TT), Metadata::Derived, Metadata::Overridable});
  for (const auto var : SL::GetVarNames()) {
    pkg->AddField(var, m);
  }
#endif
}

}  // namespace kamayan

#endif  // GRID_SCRATCH_VARIABLES_HPP_
