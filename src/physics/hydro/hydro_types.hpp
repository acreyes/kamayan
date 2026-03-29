#ifndef PHYSICS_HYDRO_HYDRO_TYPES_HPP_
#define PHYSICS_HYDRO_HYDRO_TYPES_HPP_

#include <concepts>

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "grid/scratch_variables.hpp"
#include "kamayan/fields.hpp"
#include "kamayan_utils/strings.hpp"
#include "kamayan_utils/type_list.hpp"
#include "physics/material_properties/material_types.hpp"
#include "physics/physics_types.hpp"

namespace kamayan {
// Reconstruction & Riemann solve
POLYMORPHIC_PARM(Reconstruction, fog, plm, ppm, wenoz);
POLYMORPHIC_PARM(SlopeLimiter, minmod, van_leer, mc);
POLYMORPHIC_PARM(RiemannSolver, hll, hllc, hlld);
POLYMORPHIC_PARM(ReconstructVars, primitive);
POLYMORPHIC_PARM(ReconstructionStrategy, scratchpad, scratchvar);
// MHD
POLYMORPHIC_PARM(EMFAveraging, arithmetic);
}  // namespace kamayan
namespace kamayan::hydro {

using ReconstructionOptions =
    OptList<Reconstruction, Reconstruction::fog, Reconstruction::plm, Reconstruction::ppm,
            Reconstruction::wenoz>;
using SlopeLimiterOptions =
    OptList<SlopeLimiter, SlopeLimiter::minmod, SlopeLimiter::van_leer, SlopeLimiter::mc>;
using RiemannOptions = OptList<RiemannSolver, RiemannSolver::hll, RiemannSolver::hllc>;
using ReconstructVarsOptions = OptList<ReconstructVars, ReconstructVars::primitive>;
using EMFOptions = OptList<EMFAveraging, EMFAveraging::arithmetic>;

struct RiemannScratch {
  static constexpr auto TT = TopologicalType::Cell;
  using Minus = RuntimeScratchVariable<"minus", TT>;
  using Plus = RuntimeScratchVariable<"plus", TT>;

  using type = RuntimeScratchVariableList<Minus, Plus>;
};

struct HydroBase {
  // variables that have fluxes and are independent on all mesh containers
  using WithFlux = TypeList<>;
  using Conserved = TypeList<>;
  // variables that won't have fluxes and are replicated on mesh containers
  // being references to the same
  using NonFlux = TypeList<>;
  // primitives of the system
  using Primitive = TypeList<>;
  static constexpr std::size_t ncons = 0;  // # of scalar flux/cons variables
};

// unimplemented, so has no variables
template <typename T>
struct HydroVars : HydroBase {};

template <>
struct HydroVars<Opt_t<Fluid::oneT>> : HydroBase {
  using WithFlux = TypeList<DENS, MOMENTUM, ENER>;
  using Conserved = WithFlux;
  using NonFlux = TypeList<VELOCITY, PRES, BMOD, EINT>;
  using Primitive = ConcatTypeLists_t<TypeList<DENS>, NonFlux>;
  static constexpr std::size_t ncons = 5;  // dens + mom[123] + ener
};

template <>
struct HydroVars<Opt_t<Mhd::ct>> : HydroBase {
  using WithFlux = TypeList<MAGC>;
  using Conserved = TypeList<MAG, MAGC>;
  using Primitive = TypeList<MAGC>;
  using NonFlux = TypeList<DIVB>;
  static constexpr std::size_t ncons = 3;  // mag[123]
};

template <auto option>
using hydro_vars = HydroVars<Opt_t<option>>;

// trait for recon_vars?? prim, cons, char
template <typename, typename, auto>
struct ReconVars {};

template <DenseVar... Cs, DenseVar... Vs>
struct ReconVars<TypeList<Cs...>, TypeList<Vs...>, ReconstructVars::primitive> {
  using type = TypeList<Vs...>;
};

// --8<-- [start:traits]
template <Fluid fluid, Mhd mhd, ReconstructVars recon_vars>
struct HydroTraits {
  using fluid_vars = hydro_vars<fluid>;
  using mhd_vars = hydro_vars<mhd>;

  // Mass Scalars are advected like d_t phi + div . (rho * u * phi) = 0
  // these may not always be allocated / registered as fields so we
  // need to check pack.GetUpperBound(b, T) >= 0
  using MassScalars = TypeList<material::MFRAC>;

  using WithFlux =
      ConcatTypeLists_t<typename fluid_vars::WithFlux, typename mhd_vars::WithFlux>;
  using NonFlux =
      ConcatTypeLists_t<typename fluid_vars::NonFlux, typename mhd_vars::NonFlux>;
  using Conserved =
      ConcatTypeLists_t<typename fluid_vars::Conserved, typename mhd_vars::Conserved>;
  using Primitive =
      ConcatTypeLists_t<typename fluid_vars::Primitive, typename mhd_vars::Primitive>;
  using Reconstruct = typename ReconVars<Conserved, Primitive, recon_vars>::type;

  using ConsPrim = ConcatTypeLists_t<Conserved, Primitive>;
  using All = ConcatTypeLists_t<ConsPrim, NonFlux>;
  static constexpr auto FLUID = fluid;
  static constexpr auto MHD = mhd;
  static constexpr std::size_t ncons = Conserved::n_types;
};
// --8<-- [end:traits]

// declare the signature for all HydroTraits instantiations.
// adding these concepts allows for much better LSP autocompletion
// on template parameters as well as not needing to use typename before everything
template <typename T>
concept HydroTrait =
    requires {
      // Type aliases that must be TypeList specializations
      typename T::WithFlux;
      typename T::NonFlux;
      typename T::Conserved;
      typename T::Primitive;
      typename T::Reconstruct;
      typename T::ConsPrim;
      typename T::All;
      typename T::fluid_vars;
      typename T::mhd_vars;

      // Static constexpr members with specific types
      { T::FLUID } -> std::same_as<const Fluid &>;
      { T::MHD } -> std::same_as<const Mhd &>;
      { T::ncons } -> std::same_as<const std::size_t &>;
    } && TemplateSpecialization<typename T::WithFlux, TypeList> &&
    TemplateSpecialization<typename T::NonFlux, TypeList> &&
    TemplateSpecialization<typename T::Conserved, TypeList> &&
    TemplateSpecialization<typename T::Primitive, TypeList> &&
    TemplateSpecialization<typename T::Reconstruct, TypeList> &&
    TemplateSpecialization<typename T::ConsPrim, TypeList> &&
    TemplateSpecialization<typename T::All, TypeList>;

struct HydroFactory : OptionFactory {
  using options = OptTypeList<FluidOptions, MhdOptions, ReconstructVarsOptions>;

  template <Fluid fluid, Mhd mhd, ReconstructVars recon_vars>
  using composite = HydroTraits<fluid, mhd, recon_vars>;
  using type = HydroFactory;
};

template <Reconstruction recon, SlopeLimiter limiter>
struct ReconstructTraits {
  static constexpr auto reconstruction = recon;
  static constexpr auto slope_limiter = limiter;
};

template <typename T>
concept ReconstructTrait = requires {
  { T::reconstruction } -> std::same_as<const Reconstruction &>;
  { T::slope_limiter } -> std::same_as<const SlopeLimiter &>;
};

struct ReconstructionFactory : OptionFactory {
  using options = OptTypeList<ReconstructionOptions, SlopeLimiterOptions>;

  template <Reconstruction recon, SlopeLimiter limiter>
  using composite = ReconstructTraits<recon, limiter>;
  using type = ReconstructionFactory;
};

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_HYDRO_TYPES_HPP_
