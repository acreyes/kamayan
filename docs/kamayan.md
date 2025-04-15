# Kamayan

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
--8<--"kamayan/tests/test_runtime_parameters.cpp:string"
```
* `int` & `Real` parameters can be either allowed values or allowed ranges specified
by inclusive bounds
```cpp title="kamayan/tests/test_runtime_parameters.cpp:int"
--8<-- "kamayan/tests/test_runtime_parameters.cpp:int"

## `KamayanUnit`

The `KamayanUnit` is the building block of a kamayan simulation, providing the 
interface into the initialization and driver evolution cycle through the
defined callbacks.

```cpp title="kamayan/unit.hpp:unit"
--8<-- "kamayan/unit.hpp:unit"
```

* `Setup` -- Used to register runtime parameters to the simulation, and set options
used in the global `Config`. It may be helpful to use the `GetOrAdd` interface
to also set the `Config` parameters.
```cpp title="physics/hydro/hydro.cpp:getoradd"
--8<-- "physics/hydro/hydro.cpp:getoradd"
```
* `Initialize` -- Place for a kamayan unit to create a parthenon 
[`StateDescriptor`](https://parthenon-hpc-lab.github.io/parthenon/develop/src/interface/state.html#statedescriptor).
In particular to register fields and add in any `Params` or callback functions
used by parthenon. Like for filling derived fields, or contributing to 
mesh refinement and time step calculation.
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

## Building a Simulation

```cpp
--8<-- "problems/isentropic_vortex.cpp:isen_main"
```
