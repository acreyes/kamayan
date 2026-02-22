#ifndef PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_TYPES_HPP_
#define PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_TYPES_HPP_
#include <concepts>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "physics/physics_types.hpp"
#include "utils/type_abstractions.hpp"
#include "utils/type_list.hpp"

namespace kamayan {
// recognized eos options
POLYMORPHIC_PARM(EosMode, ener, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather,
                 pres, none);
POLYMORPHIC_PARM(EosClosure, single, dalton, pte);
POLYMORPHIC_PARM(EosModel, gamma, tabulated, multitype);
POLYMORPHIC_PARM(EosComponent, oneT, ele, ion)

namespace eos {
using ClosureOptions = OptList<EosClosure, EosClosure::single>;

template <typename T>
concept AccessorLike = requires(T obj) {
  { &obj[int()] } -> std::convertible_to<Real *>;
};

struct NullIndexer {
  Real *operator[](int i) { return nullptr; }
};

// Use these as scratch space in singularityEoS' lambda argument. These
// are intended to be used with Kokkos' scratch pad memory
struct ViewIndexer {
  using View_t = ScratchPad1D;
  KOKKOS_INLINE_FUNCTION ViewIndexer(View_t data) : data_(data) {}
  KOKKOS_INLINE_FUNCTION Real &operator[](int i) { return data_(i); }

 private:
  View_t data_;
};

template <EosComponent>
struct EosVars {};

template <>
struct EosVars<EosComponent::oneT> {
  using temp = TEMP;
  using eint = EINT;
  using pres = PRES;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
  using modes = OptList<EosMode, EosMode::ener, EosMode::temp, EosMode::pres>;
};

template <>
struct EosVars<EosComponent::ion> {
  using temp = TION;
  using eint = EION;
  using pres = PION;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
};

template <>
struct EosVars<EosComponent::ele> {
  using temp = TELE;
  using eint = EELE;
  using pres = PELE;
  using types = TypeList<DENS, temp, eint, pres, GAMC, GAME>;
};

namespace impl {
template <Fluid fluid>
auto constexpr EosModes() {
  if constexpr (fluid == Fluid::oneT) {
    return OptList<EosMode, EosMode::ener, EosMode::temp, EosMode::pres>();
  } else {
    return OptList<EosMode, EosMode::ener, EosMode::temp_equi, EosMode::temp_gather,
                   EosMode::ei, EosMode::ei_scatter, EosMode::ei_gather>();
  }
}
template <Fluid fluid>
auto constexpr EosVarsImpl() {
  if constexpr (fluid == Fluid::oneT) {
    return EosVars<EosComponent::oneT>::types();
  } else {
    return ConcatTypeLists_t<EosVars<EosComponent::ion>::types,
                             EosVars<EosComponent::ele>::types>();
  }
}
}  // namespace impl
template <Fluid fluid>
using EosModeOptions = decltype(impl::EosModes<fluid>());
template <Fluid fluid>
using EosVariables = decltype(impl::EosVarsImpl<fluid>());

template <Fluid, EosClosure>
struct EosTraits {};

template <EosClosure closure>
struct EosTraits<Fluid::oneT, closure> {
  using vars = EosVars<EosComponent::oneT>::types;
};

// can be used if a kernel knows what mode it needs to call
struct EosFactory : OptionFactory {
  using options = OptTypeList<FluidOptions, ClosureOptions>;

  template <Fluid fluid, EosClosure closure>
  using composite = EosTraits<fluid, closure>;
  using type = EosFactory;
};

}  // namespace eos
}  // namespace kamayan

#endif  // PHYSICS_MATERIAL_PROPERTIES_EOS_EOS_TYPES_HPP_
