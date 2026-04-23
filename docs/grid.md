# Grid Unit

![](assets/images/grid_initialization.svg)

Kamayan is built on parthenon's grid, which can support uniform, static
and adaptively refined meshes as a runtime option. The domain is initialized
as a uniform grid of `nx1`x`nx2`x`nx3` cells. These can then be
partitioned onto an initial grid of `MeshBlocks`, which can be further
refined based on various conditions. See [Parameters](#parameters) for more 
details on the runtime parameters that control this behavior.

## Fields

Kamayan strongly prefers to reference field data stored on the `MeshBlock`s through
the type based interface. Fields are associated with a `struct`, and instantiations
of that type can be used to reference a desired element of that field if it 
is declared with some tensor shape. Kamayan exports all possible fields in
`kamayan/fields.hpp`, but which fields have been registered to the simulation
will depend on the configuration of the various units. Here fields are declared
with the `VARIABLE` macro, which takes in the type name in all capital letters,
which also serves as the string label associated with the field, and optionally
a list of `int`s that will become the tensor shape of the field when registered
with the `kamayan::AddField[s]` interfaces.

```cpp title="kamayan/fields.hpp:cons"
--8<-- "kamayan/fields.hpp:cons"
```

!!! note

    `KamayanUnit`s will register fields through the `Initialize` callback interface
    that returns a `StateDescriptor`.

## Packs

Field data can be accessed through the [`SparsePack`s](https://parthenon-hpc-lab.github.io/parthenon/develop/src/sparse_packs.html#building-and-using-a-sparsepack)
provided by parthenon that will pack variables by type into a single data structure
that can be used to index into cell level data on `MeshBlock`s and `MeshData` 
partitions. Kamayan provides an interface to directly get these packs, either
directly through listing explicitly the desired variables or by passing a 
`TypeList` of the variable structs you wish to access.

```cpp title="problems/isentropic_vortex.cpp:pack<br>physics/hydro/hydro_add_flux_tasks.cpp:pack"
// explicitly specify the fields we want
--8<-- "problems/isentropic_vortex.cpp:pack"
// packing with a decl'ed `TypeList` from the HydroTraits
--8<-- "physics/hydro/hydro_add_flux_tasks.cpp:pack"
```

!!! note
    
    Additional pack options can be specified. In this example the `PDOpt::WithFluxes`
    option means that the associated fluxes if any for the packed variables will
    also be accessible with the `pack_flux.flux` interface.

Packs can then be indexed by block, variable and (k,j,i) index

```cpp title="problems/isentropic_vortex.cpp:index"
--8<-- "problems/isentropic_vortex.cpp:index"
```

## `Subpack`s

Often we don't need full access to all block, and cell indices of the pack. Rather
it is advantageous only worry about a relative offset along the dimensions of interest
for a particular function. Some examples are the conversion of primitive to 
conservative variables, which only cares about indexing into different fields at the
same cell, or the reconstruction of Riemann states, which need only to index
into the same field at neighboring zones along a one dimensional stencil. 

Kamayan provides the `SubPack` abstractions to wrap packs and
`Kokkos::View`s to do just that.

```cpp title="physics/hydro/primconsflux.cpp:make-idx"
--8<-- "physics/hydro/primconsflux.cpp:make-idx"
```
```cpp title="physics/hydro/primconsflux.hpp:use-idx"
--8<-- "physics/hydro/primconsflux.hpp:use-idx"
```

```cpp title="physics/hydro/hydro_add_flux_tasks.cpp:make-stncl"
--8<-- "physics/hydro/hydro_add_flux_tasks.cpp:make-stncl"
```

```cpp title="physics/hydro/reconstruction.hpp:use-stncl"
--8<-- "physics/hydro/reconstruction.hpp:use-stncl"
```

## Scratch Variables

Kamayan provides the ability to represent fields on the mesh that are temporary
in nature using collections called `ScratchVariableList`s. These provide types that
can be registered to a package that will alias common `Metadata::derived` fields 
across different regions where they are used. In this way temporary block storage is 
achieved without needing to allocate more memory than is minimally required to
satisfy a single `ScratchVariableList`.

`ScratchVarableList`s are constructed from specializations of the `ScratchVariable` 
type that is used to describe the variable's `TopologicalType` and vector/tensor shape,
and finally are used to index into the `ScratchVariableList` in order to get a type
that can be used exactly as any other field through the type-based packing.

```cpp title="grid/grid_refinement.hpp:scratch"
--8<-- "grid/grid_refinement.hpp:scratch"
```
```cpp title="grid/grid.cpp:addscratch"
--8<-- "grid/grid.cpp:addscratch"
```

In the above example a single scratch variable to represent the first-order
derivatives in each 3D direction is registered as a cell centered variable as a 
3 component vector. When it is registered it will define three fields, 
`scratch_cell_n`, that can be reused by other scratch variables, possibly of different
shape.
The alias `FirstDer` is pulled out of the `ScratchVariableList` can then be used with the pack overloads just like any other field.

```cpp title="grid/grid_refinement.cpp:FirstDer"
--8<-- "grid/grid_refinement.cpp:FirstDer"
```

!!! note

    If cmake is configured with `-Dkamayan_DEBUG_SCRATCH` then each scratch variable
    will be independently registered using the `name` string template parameter
    to the `ScratchVariable`. In the above example a shape `{3}` field 
    `scratch_firstder` will be registered.


## Coordinates

Kamayan builds on Parthenon's mesh and coordinate layout (currently
`parthenon::UniformCartesian`), but wraps it behind a small geometry layer so that
code can be written against a consistent coordinate/metric API while the geometry
is selected at runtime.

At the moment the grid unit supports:

- `Geometry::cartesian`: standard 1D/2D/3D Cartesian
- `Geometry::cylindrical`: 2D axisymmetric r-z with an implicit azimuthal direction
  (the third direction uses `dphi = 2*pi`)

### `CoordinateSystem`

The coordinate wrappers implement the `CoordinateSystem` concept
(`src/grid/geometry.hpp`), which is the practical "contract" used throughout the code.
It includes both compile-time and runtime axis dispatch:

- Compile-time: `template <Axis ax> Dx()`, `Xc(idx)`, `Xf(idx)`, `FaceArea(k,j,i)`, ...
- Runtime: `Dx(Axis)`, `Xc(Axis, idx)`, `Xf(Axis, idx)`, `FaceArea(Axis,k,j,i)`, ...

The main coordinate wrapper is `grid::Coordinates<geom>` (`src/grid/geometry.hpp`). It is
a thin layer over Parthenon's coordinates that provides geometry-specific fixes for
metric factors:

- `X`/`Xi`: coordinate at the Parthenon cell-center location
- `Xc`: coordinate at the cell centroid (differs from `Xi` for cylindrical r)
- `Xf`: coordinate at the face location

- In Cartesian geometry, methods delegate to Parthenon.
- In cylindrical geometry:
  - `Dx<Axis::KAXIS>()` returns `2*pi` (implicit azimuthal direction)
  - `Xc<Axis::IAXIS>(idx)` computes the radial *centroid* of the cell (not just the
    midpoint); `Xi<Axis::IAXIS>(idx)` remains the "cell center" value from Parthenon
  - `FaceArea`, `EdgeLength`, and `CellVolume` use r-z metric factors (e.g.
    `V = 0.5 * dphi * (r_{i+1/2}^2 - r_{i-1/2}^2) * dz`)

### CoordinatePacks

Parthenon's coordinate methods are largely inline calculations. For geometry-dependent
metric factors (areas, volumes, centroids, etc.) it is often cheaper and simpler to
precompute them once per `MeshBlock` and store them as normal fields.

The grid unit registers and fills these coordinate fields:

- Field types live in `src/grid/coordinates.hpp` under `kamayan::grid::coords::*` and
  the full list is `grid::CoordFields`.
- The fields are allocated with geometry-aware *degenerate* shapes via
  `grid::CoordinateShape`:
  - Cartesian: `Dx*`, `FaceArea*`, `EdgeLength*`, `Volume` are scalars; `X*`, `Xc*`, `Xf*`
    are 1D arrays per-axis.
  - Cylindrical: `Dx*` are scalars; r-dependent quantities (`Volume`, `FaceArea*`,
    `EdgeLength*`, and `X*`/`Xc*`/`Xf*` for r) are stored as 1D arrays in the radial direction.
- `grid::CalculateCoordinates` (`src/grid/coordinates.cpp`) fills all `CoordFields` for each
  block (using `grid::CoordinateIndexRanges` to respect degenerate/face extents), and is
  called from the grid unit's `InitMeshBlockData` callback.

`grid::CoordinatePack<geom, ...>` (`src/grid/coordinates.hpp`) wraps a `SparsePack` holding
these `geom.*` fields and exposes the same API as `grid::Coordinates<geom>` but indexed by
`(k,j,i)`. Internally it maps `(k,j,i)` onto the degenerate storage layout (scalar/1D), so
call sites do not need to care about how each metric is stored.

Example usage (runtime geometry):

```cpp title="problems/isentropic_vortex.cpp:isen_coords_pack"
--8<-- "problems/isentropic_vortex.cpp:isen_coords_pack"
```

### Generic Coordinates (Packs)

When the geometry is only known at runtime, Kamayan provides variant-based wrappers:

- `grid::GenericCoordinate` (`src/grid/geometry.hpp`) wraps `grid::Coordinates<geom>` in a
  `variant` and forwards calls via `visit`.
- `grid::GenericCoordinatePack` (`src/grid/coordinates.hpp`) does the same for
  `grid::CoordinatePack<geom, ...>`.

In practice `GenericCoordinatePack` is constructed from a coordinate-field pack (e.g.
`grid::GetPack(grid::CoordFields(), ...)`) and then used like a normal coordinate object.

These are convenient in code like problem generators, but they do add a small overhead
compared to templating on `Geometry`. For performance-critical kernels prefer templating
on `geom` and constructing the matching `grid::Coordinates<geom>`/`grid::CoordinatePack<geom>`.

## Parameters
{!assets/generated/grid_parms.md!}
