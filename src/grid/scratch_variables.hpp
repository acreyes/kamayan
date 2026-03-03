#ifndef GRID_SCRATCH_VARIABLES_HPP_
#define GRID_SCRATCH_VARIABLES_HPP_

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <parthenon/parthenon.hpp>

#include "Kokkos_Core_fwd.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/subpack.hpp"
#include "kamayan/fields.hpp"
#include "pack/sparse_pack.hpp"
#include "utils/strings.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan {

namespace impl {
constexpr auto DebugScratch() {
#ifdef KAMAYAN_DEBUG_SCRATCH
  return true;
#else
  return false;
#endif
}
}  // namespace impl

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

template <TopologicalType TT>
constexpr auto TopologicalTypeToCTS() {
  if constexpr (TT == TopologicalType::Face) {
    return strings::make_cts("face");
  } else if constexpr (TT == TopologicalType::Edge) {
    return strings::make_cts("edge");
  } else if constexpr (TT == TopologicalType::Node) {
    return strings::make_cts("node");
  } else {
    return strings::make_cts("cell");
  }
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

template <strings::CompileTimeString var_name, TopologicalType TT, int... NCOMPS>
struct RuntimeScratchVariable : parthenon::variable_names::base_t<true, NCOMPS...> {
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION RuntimeScratchVariable(Args &&...args)
      : base_t(std::forward<Args>(args)...) {}
  static constexpr std::string_view str_name{var_name.value, sizeof(var_name.value)};
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
  static constexpr auto vname = var_name;
  static std::string ame() { return std::string(var_name.value); }
};

template <strings::CompileTimeString name, TopologicalType TT, int... NCOMPS>
struct ScratchVariable {
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  static constexpr std::string_view str_name{name.value, sizeof(name.value)};
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
  static constexpr auto vname = name;
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

  template <class... Ts>
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

template <typename V, typename... Vars>
class RuntimeScratchVariableList {
 public:
  using TL = TypeList<V, Vars...>;
  static constexpr TopologicalType TT = V::type;
  static constexpr int n_vars = sizeof...(Vars);

  template <std::size_t N>
  using CS = strings::CompileTimeString<N>;
  static constexpr auto base_name =
      strings::concat_cts(CS("scratch_"), TopologicalTypeToCTS<TT>(), CS("_"));

  // when we're not debugging we just always pack into all the scratch variables
  // since we can't know at compile time how large any of the variables are.
  // When we're debugging then these types are all unique, since we have a
  // label for each scratch variable.
  template <typename T>
  requires(TL::template Contains<T>())
  using VarType = std::conditional_t<
      impl::DebugScratch(), VariableBase<concat_cts(base_name, T::vname)>,
      RuntimeScratchVariable<strings::concat_cts(base_name, CS(".*")), TT>>;

  using list = TypeList<VarType<V>, VarType<Vars>...>;
  using Pack_t = SparsePack<VarType<V>, VarType<Vars>...>;

  RuntimeScratchVariableList()
      : shapes_(n_vars, std::vector<int>(1, 1)), offsets_("scratch_offsets", n_vars + 1),
        host_offsets_("scratch_offsets_host", n_vars + 1) {
    UpdateOffsets();
  }

  template <typename Var>
  void RegisterShape(const std::vector<int> &shape) {
    static_assert(TL::template Contains<Var>(),
                  "Var not found in RuntimeScratchVariableList");
    constexpr std::size_t idx = TL::template Idx<Var>();
    shapes_[idx] = shape;
    UpdateOffsets();
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION auto Idx(const T &instance) const {
    constexpr std::size_t tag_idx = TL::template Idx<decltype(instance)>();
    const auto offset = offsets_(tag_idx);
    PARTHENON_DEBUG_REQUIRE(offset >= 0, "scratch variable is allocated");
    return VarType<T>(offsets_(tag_idx) + instance.idx);
  }

  template <typename T>
  using Identity_ = T;

  // Idea here is to index into the pack using our scratch list, but we want
  // to index using variable Ts..., which might not necessarily be in our
  // scratch list, but we have a transformation from Ts -> Vars
  template <typename Pack_t, typename... Ts,
            template <typename T> typename Transform = Identity_>
  KOKKOS_INLINE_FUNCTION auto MakePackIndexer(TypeList<Ts...>, const Pack_t &pack,
                                              const int &b, const int &k, const int &j,
                                              const int &i) {
    auto indexer = MakePackIndexer(TypeList<Transform<Ts>...>(), pack, b, k, j, i);
    return
        [=, this]<typename T>(const T &t) { return indexer(Idx(Transform<T>(t.idx))); };
  }

  template <typename Var>
  int Size() const {
    static_assert(TL::template Contains<Var>(),
                  "Var not found in RuntimeScratchVariableList");
    constexpr std::size_t idx = TL::template Idx<Var>();
    return offsets_(idx + 1) - offsets_(idx);
  }

  int TotalSize() const { return offsets_(n_vars); }

  KOKKOS_INLINE_FUNCTION const auto &GetOffsets() const { return offsets_; }

  template <typename T>
  KOKKOS_INLINE_FUNCTION const auto &GetOffsets() const {
    return offsets_[TL::template Idx<T>()];
  }

  const auto &GetOffsetsHost() const { return host_offsets_; }

  template <typename T>
  const auto &GetOffsetsHost() const {
    return host_offsets_[TL::template Idx<T>()];
  }

  const std::vector<std::vector<int>> &GetShapes() const { return shapes_; }

  std::vector<std::string> GetVarNames() const {
    std::vector<std::string> vars;
    vars.reserve(n_vars);
    int idx = 0;
    (
        [&]<typename Var>() {
          vars.push_back("scratch_" + TopologicalTypeToString(TT) + "_" +
                         std::to_string(idx++));
        }.template operator()<Vars>(),
        ...);
    return vars;
  }

  static constexpr TopologicalType GetType() {
    if constexpr (sizeof...(Vars) > 0) {
      return TypeList<Vars...>::template type<0>::type;
    }
    return TopologicalType::Cell;
  }

 private:
  void UpdateOffsets() const {
    host_offsets_(0) = 0;
    for (int i = 0; i < n_vars; i++) {
      int size = 1;
      for (int dim : shapes_[i]) {
        size *= dim;
      }
      // when debugging scratch our labels are unique so zero offset
      host_offsets_(i + 1) = impl::DebugScratch() ? 0 : host_offsets_(i) + size;
      // if a variable is not allocated return -1;
      host_offsets_(i + 1) = size > 0 ? host_offsets_(i + 1) : -1;
    }
    Kokkos::deep_copy(offsets_, host_offsets_);
  }

  std::vector<std::vector<int>> shapes_;
  mutable Kokkos::View<int *, Kokkos::DefaultExecutionSpace> offsets_;
  mutable Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> host_offsets_;
};

template <typename... Ts>
struct ScratchPack {
  using ScratchType = RuntimeScratchVariableList<Ts...>;
  using TL = TypeList<Ts...>;

  ScratchPack(MeshData *md, const ScratchType &scratch_list)
      : scratch_list_(scratch_list) {
    pack_ = grid::GetPack(typename ScratchType::list(), md);
  }

  template <typename... Args>
  auto &operator()(Args &&...args) const {
    return pack_(std::forward<Args>(args)...);
  }

  auto SubPack(const int &b, const int &k, const int &j, const int &i) const {
    return [&, this]<typename T>(const T &v) -> Real & {
      if constexpr (TL::template Contains<T>()) {
        return pack_(b, v.idx + scratch_list_.template GetOffsets<T>(), k, j, i);
      } else {
        return pack_(b, v, k, j, i);
      }
    };
  }

  // return an indexer into our scratch pack at variable V, with tensor
  // index given by a variable indexed by the typelist order
  template <typename V, DenseVar... Vs>
  auto Indexer(const V &, TypeList<Vs...>, const int &b, const int &k, const int &j,
               const int &i) const {
    return [&, this]<typename T>(const T &t) -> Real & {
      static_assert(TypeList<Vs...>::template Contains<T>(),
                    "type index must be in list");

      using Var = ScratchType::template VarType<V>;
      using indexer = TypeVarIndexer<Vs...>;
      return pack_(b, Var(indexer::Idx(t)), k, j, i);
    };
  }

 private:
  ScratchType scratch_list_;
  ScratchType::Pack_t pack_;
};

template <typename... Ts>
void AddScratch(const RuntimeScratchVariableList<Ts...> &scratch_list,
                StateDescriptor *pkg) {
  using SL = decltype(scratch_list);
#ifdef KAMAYAN_DEBUG_SCRATCH
  int idx = 0;
  auto shapes = scratch_list.GetShapes();
  type_for(TypeList<Vars...>(), [&]<typename T>(const T &) {
    if (scratch_list.Size<T> == 0) return;
    auto m = Metadata({TopologicalTypeToMetaData(SL::GetType()), Metadata::Derived,
                      shapes[i]);
      using T_scratch = VariableBase<concat_cts(SL::base_name, T::vname)>;
      pkg->AddField<T_scratch>(m);
      i++;
    });
#else
  auto m = Metadata({TopologicalTypeToMetaData(SL::GetType()), Metadata::Derived,
                     Metadata::Overridable});
  for (const auto &var : scratch_list.GetVarNames()) {
    pkg->AddField(var, m);
  }

#endif
}

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
