#ifndef PHYSICS_EOS_EOS_TYPES_HPP_
#define PHYSICS_EOS_EOS_TYPES_HPP_
#include <concepts>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan {
// recognized eos options
POLYMORPHIC_PARM(EosMode, ener, temp, temp_equi, temp_gather, ei, ei_scatter, ei_gather,
                 pres, none);
POLYMORPHIC_PARM(EosType, Single, MultiType);
POLYMORPHIC_PARM(EosModel, gamma, tabulated, multitype);
POLYMORPHIC_PARM(EosComponent, oneT, ele, ion)

namespace eos {
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

}  // namespace eos
}  // namespace kamayan

#endif  // PHYSICS_EOS_EOS_TYPES_HPP_
