#ifndef PHYSICS_HYDRO_HYDRO_TYPES_HPP_
#define PHYSICS_HYDRO_HYDRO_TYPES_HPP_

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "kamayan/fields.hpp"
#include "physics/physics_types.hpp"
#include "utils/type_list.hpp"

namespace kamayan {
POLYMORPHIC_PARM(Reconstruction, fog, plm, ppm);
POLYMORPHIC_PARM(SlopeLimiter, minmod, van_leer, mc);
POLYMORPHIC_PARM(RiemannSolver, hll);
POLYMORPHIC_PARM(ReconstructVars, primitive);
}  // namespace kamayan
namespace kamayan::hydro {

using ReconstructionOptions = OptList<Reconstruction, Reconstruction::fog,
                                      Reconstruction::plm, Reconstruction::ppm>;
using SlopeLimiterOptions =
    OptList<SlopeLimiter, SlopeLimiter::minmod, SlopeLimiter::van_leer, SlopeLimiter::mc>;
using RiemannOptions = OptList<RiemannSolver, RiemannSolver::hll>;
using ReconstructVarsOptions = OptList<ReconstructVars, ReconstructVars::primitive>;

struct HydroBase {
  // variables that have fluxes
  using Conserved = TypeList<>;
  using Primitive = TypeList<>;
  static constexpr std::size_t ncons = 0;  // # of scalar flux/cons variables
};

// unimplemented, so has no variables
template <typename T>
struct HydroVars : HydroBase {};

template <>
struct HydroVars<Opt_t<Fluid::oneT>> : HydroBase {
  using Conserved = TypeList<DENS, MOMENTUM, ENER>;
  using Primitive = TypeList<VELOCITY, PRES, GAMC, GAME, EINT>;
  static constexpr std::size_t ncons = 5;  // dens + mom[123] + ener
};

template <>
struct HydroVars<Opt_t<Mhd::ct>> : HydroBase {
  using Conserved = TypeList<MAG>;
  using Primitive = TypeList<MAGC>;
  static constexpr std::size_t ncons = 3;  // mag[123]
};

template <auto option>
using hydro_vars = HydroVars<Opt_t<option>>;

// trait for recon_vars?? prim, cons, char
template <typename, typename, auto>
struct ReconVars {};

template <typename... Cs, typename... Vs>
struct ReconVars<TypeList<Cs...>, TypeList<Vs...>, ReconstructVars::primitive> {
  using type = TypeList<DENS, Vs...>;
};

// --8<-- [start:traits]
template <Fluid fluid, Mhd mhd, ReconstructVars recon_vars>
struct HydroTraits {
  using fluid_vars = hydro_vars<fluid>;
  using mhd_vars = hydro_vars<mhd>;

  using Conserved =
      ConcatTypeLists_t<typename fluid_vars::Conserved, typename mhd_vars::Conserved>;
  using Primitive =
      ConcatTypeLists_t<typename fluid_vars::Primitive, typename mhd_vars::Primitive>;
  using Reconstruct = typename ReconVars<Conserved, Primitive, recon_vars>::type;

  using ConsPrim = ConcatTypeLists_t<Conserved, Primitive>;
  static constexpr auto FLUID = fluid;
  static constexpr auto MHD = mhd;
  static constexpr std::size_t ncons = Conserved::n_types;
};
// --8<-- [end:traits]

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

struct ReconstructionFactory : OptionFactory {
  using options = OptTypeList<ReconstructionOptions, SlopeLimiterOptions>;

  template <Reconstruction recon, SlopeLimiter limiter>
  using composite = ReconstructTraits<recon, limiter>;
  using type = ReconstructionFactory;
};

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_HYDRO_TYPES_HPP_
