# Kamayan

This document describes the core infrastructure of Kamayan, focusing on the C++ implementation. 
For Python API documentation, see the [API Reference](api.md).

## `Config`

Kamayan tracks a global configuration that is made up of kamayan runtime
options that are compatible with the [dispatcher](dispatcher.md).
The `Config` wraps parthenon's `Params` class, but only holds kamayan
`POLYMORPHIC_PARM` `enum class`s, and only allows a single param
of each type in the `Config`. Kamayan provides an API for checking out
the global `Config` from the corresponding `StateDescriptor` with the
`GetConfig` interface.

## Runtime Parameters

Parthenon provides a method for [parameter input](https://parthenon-hpc-lab.github.io/parthenon/develop/src/inputs.html),
that can be set with an input file as well as from the command line.
Kamayan wraps this system with the `RuntimeParameters` class. The
primary reason for doing this is to provide optionally defined constraints
for validating parameters on read, as well as to store doc strings
for runtime parameters at the location that they are defined. This
enables kamayan to generate parameter documentation for each unit
directly from the source code, and enforcing ownership of parameters
to their respective `KamayanUnit`.

!!! note

    Kamayan provides another wrapper, through `UnitData`, around input parameters, parthenon package `params`, and the kamayan [`Config`](#config) that integrates seamlessly with the 
    [`KamayanUnit`](#kamayanunit)s described later. This is the preferred
    method for setting/adding runtime parameters when using `pyKamayan`.

The minimum required for a runtime parameter to be defined is a block name,
a parm name, default value, and a doc string.

```cpp title="problems/isentropic_vortex.cpp:parms"
--8<-- "problems/isentropic_vortex.cpp:parms"
```

The parms then can be set in the input file under the corresponding `<block_name>` 
section
``` title="problems/isentropic_vortex.in:parms"
--8<-- "problems/isentropic_vortex.in:parms"
```

Lastly kamayan supports some simple rules that can be used for validating
values read in from the input file using an initializer list.

* `std::string` parameters can add a list of allowed values
```cpp title="kamayan/tests/test_runtime_parameters.cpp:string"
--8<-- "kamayan/tests/test_runtime_parameters.cpp:string"
```
* `int` & `Real` parameters can be either allowed values or allowed ranges specified
by inclusive bounds
```cpp title="kamayan/tests/test_runtime_parameters.cpp:int"
--8<-- "kamayan/tests/test_runtime_parameters.cpp:int"
```

## `KamayanUnit`

The `KamayanUnit` is the building block of a kamayan simulation, providing the 
interface into the initialization and driver evolution cycle through the
defined callbacks.

```cpp title="kamayan/unit.hpp:unit"
--8<-- "kamayan/unit.hpp:unit"
```

* `SetupParams` -- Used to register runtime parameters to the simulation, and set options
used in the global `Config`. These are added through the [`UnitData`](#unitdata) interface.
* `InitializeData` -- Main place to initialize the underlying `parthenon::StateDescriptor` that `KamayanUnit`s inherit from. To register parthenon callbacks, 
add fields to be allocated on the mesh blocks and add data structures to the `Params`.
* `ProblemGeneratorMeshBlock` -- Called during mesh initialization to set the
initial conditions on a single `MeshBlock`.
* `PrepareConserved` -- Called before starting the evolution cycle to ensure
conserved variables are properly initialized, if not done so by the problem
generator.
* `PreparePrimitive` -- called at the end of an RK-stage, after applying `DuDt`
to convert from the updated conservative variables to primitives. Equation
of state should always be called last.
* The following control how the mesh data is evolved. See [driver](driver.md) for
more detail on how these are used.
   * `AddFluxTasks`
   * `AddTasksOneStep`
   * `AddTasksSplit`

## `UnitCollection`

`KamayanUnit`s are composed together into a `UnitCollection`, which is used 
to construct a `KamayanDriver`. In some case it could be desirable to 
express the order that some of the `KamayanUnit` callbacks get executed,
in which case there are some named `std::list<std::string>`s that will set
the execution order for some or all of their corresponding callbacks. Those
that are listed will be called last and in order using the `UnitCollection::AddTasks`
method.

## `UnitData`

There is a natural relationship between input parameters and `StateDescriptor` [params](https://parthenon-hpc-lab.github.io/parthenon/develop/src/interface/state.html#statedescriptor), and `Config` options, that all have different entrypoints in terms of data structures (`RuntimeParameters`, `Config` and `StateDescriptor`).
The registering of variables and `PolyOpt<T>`s to these are commonly handled in the `Setup` & `Init` `KamayanUnit` callbacks, and so kamayan provides `UnitData` to handle the registering and mapping between the three.

A `UnitData` object manages all of the parameters inside of a given input block. Depending on the template type passed to the `AddParm` method,
these will either get mapped to the owning unit's `StateDescriptor::Params`, or to the global kamayan `Config` object. 


```cpp title="physics/hydro/hydro.cpp:add_parm"
--8<-- "physics/hydro/hydro.cpp:add_parm"
```

```cpp title="physics/hydro/hydro_time_step.cpp:get_param"
--8<-- "physics/hydro/hydro_time_step.cpp:get_param"
```

Additionally params can be registered with `UnitData::Mutability` flag, which
defaults to `Immutable`, meaning that once the simulation has initialized
those parms should not be changed. If they are registered as immutable,
if the registered parameter is updated with `UnitData::UpdateParm` then
the parameter is updated both in the underlying package `Params` as well
as the runtime parameters.

All of the input blocks described by individual `UnitData` objects are owned
by their respective `KamayanUnit`s and made accessible during each unit's `Initialize`. 
The separation between `Setup` and `Initialize` is then in registering
`UnitData` parameters, and resolving the `Config` options first across all units, making them accessible to the initialization of each unit.

Parameters are uniquely owned by their respective `KamayanUnit`s, and the preferred 
way to access parameters from other units is directly through the unit's parameter 
interface using `unit->GetUnit(name)` to retrieve the desired unit, and then accessing 
its `UnitData` block.

```cpp title="problems/sedov.cpp:access_grid_params"
--8<-- "problems/sedov.cpp:access_grid_params"
```

This pattern allows units to access configuration from other units during initialization,
maintaining clear ownership while enabling cross-unit parameter dependencies.

Bindings to `UnitData` are provided through `pyKamayan`, and the `KamayanManager`
allows for type validation in setting input parameters and python object based setting.

```python title="problems/sedov.py:py_add_parms"
--8<-- "problems/sedov.py:py_add_parms"
```

```python title="problems/sedov.py:py_get_parms"
--8<-- "problems/sedov.py:py_get_parms"
```

```python title="problems/sedov.py:py_set_param"
--8<-- "problems/sedov.py:py_set_param"
```

```python title="problems/sedov.py:py_get_param"
--8<-- "problems/sedov.py:py_get_param"
```

For complete examples of using UnitData and building simulations, see 
[Setting up a Simulation](simulation_setup.md).

