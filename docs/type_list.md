# `TypeList`

The `TypeList` struct provides utilities for dealing with
template parameter packs, and in particular is heavily used
in the [dispatcher](dispatcher.md) and abstractions around
the [type-based packs](grid.md#packs) to perform
compile time logic. Some utilities include concatenating
`TypeList`s, picking out types by index and splitting into separate
lists.

## `TypeListArray`

It can also be useful to index into a regular array using the types in
a `TypeList`. This is commonly used in the [hydro](hydro.md) as a way
to abstract the size of the system used depending on runtime options
used around MHD/3T physics.

```cpp title="physics/hydro/riemann_solver.hpp:tl-arr"
--8<-- "physics/hydro/riemann_solver.hpp:tl-arr"
```

```cpp title="physics/hydro/primconsflux.hpp:use-idx"
--8<-- "physics/hydro/primconsflux.hpp:use-idx"
```

Types in a `TypeList` can also be iterated over using the `type_for` interface
together with a templated lambda function.

```cpp title="physics/hydro/riemann_solver.hpp:type_for"
--8<-- "physics/hydro/riemann_solver.hpp:type_for"
```
