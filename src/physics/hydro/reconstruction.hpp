#ifndef PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#define PHYSICS_HYDRO_RECONSTRUCTION_HPP_
#include <Kokkos_Core.hpp>

#include "grid/indexer.hpp"
#include "physics/hydro/hydro_types.hpp"

namespace kamayan::hydro {

template <Reconstruction recon, typename Container>
void Reconstruct(Container stencil, Real &vM, Real &vP) {}

template <Reconstruction recon, typename Container>
requires(recon == Reconstruction::fog, Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  vM = stencil(0);
  vP = stencil(0);
}

template <Reconstruction recon, typename Container>
requires(recon == Reconstruction::plm, Stencil1D<Container>)
void Reconstruct(Container stencil, Real &vM, Real &vP) {
  // --8<-- [start:use-stncl]
  const Real dvL = stencil(0) - stencil(-1);
  const Real dvR = stencil(1) - stencil(0);
  // --8<-- [end:use-stncl]
}

}  // namespace kamayan::hydro

#endif  // PHYSICS_HYDRO_RECONSTRUCTION_HPP_
