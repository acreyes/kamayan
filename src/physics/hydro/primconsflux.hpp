#ifndef PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
#define PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
#include <Kokkos_Core.hpp>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/physics_types.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::hydro {

// these will prepare any U <-> V in our data at the beginning/end of the hydro cycle
TaskStatus PrepareConserved(MeshData *md);
TaskStatus PreparePrimitive(MeshData *md);

template <Mhd mhd, typename Prim>
KOKKOS_INLINE_FUNCTION Real TotalPres(const Prim &V) {
  Real pres = V(PRES());
  if constexpr (mhd != Mhd::off) {
    pres += 0.5 *
            (V(MAGC(0)) * V(MAGC(0)) + V(MAGC(1)) * V(MAGC(1)) + V(MAGC(2)) * V(MAGC(2)));
  }

  return pres;
}
template <Mhd mhd, typename Prim>
KOKKOS_INLINE_FUNCTION Real FastSpeed(const int &dir1, const Prim &V) {
  const Real idens = 1. / V(DENS());
  const Real a2 = V(GAMC()) * V(PRES()) * idens;

  Real cfast2;
  if constexpr (mhd == Mhd::off) {
    // sound speed
    cfast2 = a2;
  } else {
    // fast magneto-sonic speed
    const Real bb2 = V(MAGC(dir1)) * V(MAGC(dir1)) * idens;
    Real b2 = 0.0;
    for (int dir = 0; dir < 3; dir++) {
      b2 += V(MAGC(dir)) * V(MAGC(dir));
    }
    b2 *= idens;
    cfast2 =
        0.5 * ((a2 + b2) + Kokkos::sqrt((a2 - b2) * (a2 - b2) + 4. * a2 * (b2 - bb2)));
  }

  return Kokkos::sqrt(cfast2);
}

template <typename hydro_traits, typename Prim, typename Cons>
KOKKOS_INLINE_FUNCTION void Prim2Cons(const Prim &V, Cons &U) {
  // --8<-- [start:use-idx]
  U(DENS()) = V(DENS());
  Real emag = 0.;
  Real ekin = 0.;
  for (int dir = 0; dir < 3; dir++) {
    U(MOMENTUM(dir)) = V(DENS()) * V(VELOCITY(dir));
    ekin += V(VELOCITY(dir)) * V(VELOCITY(dir));
    if constexpr (hydro_traits::MHD != Mhd::off) {
      U(MAGC(dir)) = V(MAGC(dir));
      emag += V(MAGC(dir)) * V(MAGC(dir));
    }
  }
  // --8<-- [end:use-idx]
  const Real eint = V(PRES()) / (V(GAME()) - 1.0);
  ekin *= 0.5 * V(DENS());
  emag *= 0.5;
  U(ENER()) = eint + ekin + emag;
}

template <typename hydro_traits, typename Prim, typename Cons>
requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
KOKKOS_INLINE_FUNCTION void Cons2Prim(const Cons &U, Prim &V) {
  V(DENS()) = U(DENS());
  const Real idens = 1.0 / V(DENS());
  V(VELOCITY(0)) = idens * U(MOMENTUM(0));
  Real emag = 0.;
  Real ekin = 0.;
  for (int dir = 0; dir < 3; dir++) {
    V(VELOCITY(dir)) = idens * U(MOMENTUM(dir));
    ekin += V(VELOCITY(dir)) * V(VELOCITY(dir));
    if constexpr (hydro_traits::MHD != Mhd::off) {
      V(MAGC(dir)) = U(MAGC(dir));
      emag += V(MAGC(dir)) * V(MAGC(dir));
    }
  }
  ekin *= 0.5 * V(DENS());
  emag *= 0.5;
  const Real eint = U(ENER()) - ekin - emag;
  V(EINT()) = idens * eint;
  V(PRES()) = (V(GAME()) - 1.0) * eint;
}

template <std::size_t dir1, typename hydro_traits, typename Prim, typename Flux>
requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
KOKKOS_INLINE_FUNCTION void Prim2Flux(const Prim &V, Flux &F) {
  constexpr std::size_t dir2 = (dir1 + 1) % 3;
  constexpr std::size_t dir3 = (dir1 + 2) % 3;

  F(DENS()) = V(DENS()) * V(VELOCITY(dir1));
  F(MOMENTUM(dir1)) = F(DENS()) * V(VELOCITY(dir1));
  F(MOMENTUM(dir2)) = F(DENS()) * V(VELOCITY(dir2));
  F(MOMENTUM(dir3)) = F(DENS()) * V(VELOCITY(dir3));

  Real ptot = V(PRES());
  Real B2 = 0.;
  Real uB = 0.;
  Real ekin = 0.;
  for (int dir = 0; dir < 3; dir++) {
    F(MOMENTUM(dir)) = F(DENS()) * V(VELOCITY(dir));
    ekin += V(VELOCITY(dir)) * V(VELOCITY(dir));
    if constexpr (hydro_traits::MHD != Mhd::off) {
      B2 += V(MAGC(dir)) * V(MAGC(dir));
      uB += V(VELOCITY(dir)) * V(MAGC(dir));
      F(MOMENTUM(dir)) -= V(MAGC(dir1)) * V(MAGC(dir));
    }
  }
  ekin *= 0.5 * V(DENS());
  const Real emag = 0.5 * B2;
  ptot += 0.5 * B2;
  const Real etot = V(PRES()) / (V(GAME()) - 1.0) + ekin + emag;

  F(MOMENTUM(dir1)) += ptot;
  F(ENER()) = (etot + ptot) * V(VELOCITY(dir1));

  if constexpr (hydro_traits::MHD != Mhd::off) {
    F(ENER()) -= uB * V(MAGC(dir1));
    F(MAGC(dir1)) = 0.;
    F(MAGC(dir2)) = V(VELOCITY(dir1)) * V(MAGC(dir2)) - V(VELOCITY(dir2)) * V(MAGC(dir1));
    F(MAGC(dir3)) = V(VELOCITY(dir1)) * V(MAGC(dir3)) - V(VELOCITY(dir3)) * V(MAGC(dir1));
  }
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
