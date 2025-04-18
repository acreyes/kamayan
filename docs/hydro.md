# Hydro Unit

Kamayan's hydro unit uses a high-order Godunov scheme to solve the
compressible Euler equations as the building block that can be extended
to the ideal MHD and three temperature systems of equations. The basic
principles are to

* Reconstruct high-order Riemann states biased to the left and right
of the cell-interfaces in all directions from the cell-centered volume
averaged fields.
* Use an approximate Riemann solver at the cell-face centers to get the
upwinded high-order fluxes from the reconstructed Riemann states.
* Update the conserved fluid variables with a discrete divergence theorem
to difference the face-centered fluxes.

## Hydro Traits

The above described method can apply to a number of possible combinations of
systems of equations from linear advection, to the compressible Euler equations,
ideal MHD, three-temperature hydrodynamics and advection of material mixtures.
Even though all these systems of equations can be solved with the same
methodology supporting all of them in the same code isn't so trivial.

Kamayan tries to address this by constructing a type trait `HydroTraits`
that is templated on the runtime options used by hydro to define the 
variables need for the reconstruct-evolve-average method.

```cpp title="physics/hydro/hydro_types.hpp:traits"
--8<-- "physics/hydro/hydro_types.hpp:traits"
```

With the `HydroTraits` kernels used in the hydro unit can dispatch
to specialized code that is specific to the system of equations
only when it is needed

```cpp title="physics/hydro/riemann_solver.hpp:tl-arr"
--8<-- "physics/hydro/riemann_solver.hpp:tl-arr"
```

## Parameters
{!assets/generated/hydro_parms.md!}
