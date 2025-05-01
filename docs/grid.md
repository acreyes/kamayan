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

## Indexers & Stencils

Often we don't need full access to all block, and cell indices of the pack. Rather
it is advantageous only worry about a relative offset along the dimensions of interest
for a particular function. Some examples are the conversion of primitive to 
conservative variables, which only cares about indexing into different fields at the
same cell, or the reconstruction of Riemann states, which need only to index
into the same field at neighboring zones along a one dimensional stencil. 

Kamayan provides the `Indexer` and `Stencil` abstractions to wrap packs and
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

## Parameters
{!assets/generated/grid_parms.md!}
