#ifndef PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
#define PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "kamayan/fields.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan::hydro {

template <typename Scratch, typename hydro_traits>
KOKKOS_INLINE_FUNCTION void Prim2Cons(const Scratch &V, TypeListArray<hydro_traits> &U) {
  U(DENS()) = V(DENS());
  U(MOMENTUM(0)) = V(DENS()) * V(VELOCITY(0));
  U(MOMENTUM(1)) = V(DENS()) * V(VELOCITY(1));
  U(MOMENTUM(2)) = V(DENS()) * V(VELOCITY(2));
  const Real eint = V(PRES()) / (V(GAME()) - 1.0);
  const Real ekin = 0.5 * V(DENS()) *
                    (V(VELOCITY(0)) * V(VELOCITY(0)) + V(VELOCITY(1)) * V(VELOCITY(1)) +
                     V(VELOCITY(2)) * V(VELOCITY(2)));
  U(ENER()) = eint + ekin;
}

template <std::size_t dir1, typename Scratch, typename hydro_traits>
KOKKOS_INLINE_FUNCTION void Prim2Flux(const Scratch &V, TypeListArray<hydro_traits> &F) {
  constexpr std::size_t dir2 = (dir1 + 1) % 3;
  constexpr std::size_t dir3 = (dir1 + 2) % 3;

  F(DENS()) = V(DENS()) * V(VELOCITY(dir1));
  F(MOMENTUM(dir1)) = F(DENS()) * V(VELOCITY(dir1));
  F(MOMENTUM(dir2)) = F(DENS()) * V(VELOCITY(dir2));
  F(MOMENTUM(dir3)) = F(DENS()) * V(VELOCITY(dir3));

  Real ptot = V(PRES());
  Real etot = V(PRES()) / (V(GAME()) - 1.0);
  etot += 0.5 * V(DENS()) *
          (V(VELOCITY(0)) * V(VELOCITY(0)) + V(VELOCITY(1)) * V(VELOCITY(1)) +
           V(VELOCITY(2)) * V(VELOCITY(2)));

  F(MOMENTUM(dir1)) += ptot;
  F(ENER()) = (etot + ptot) * V(VELOCITY(dir1));
}

}  // namespace kamayan::hydro
#endif  // PHYSICS_HYDRO_PRIMCONSFLUX_HPP_
