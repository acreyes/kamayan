# Driver Unit

The `KamayanDriver` implements the parthenon `MultiStageDriver`, and so works with any of the
provided multistage integrators. Kamayan allows any `KamayanUnit` to hook into the main evolution
loop through the provided callback interfaces. The primary evolution hooks can be classified as

* Flux-based multi-stage operators
* Non-flux multi-stage operators
* Split operators

For multi-stage integrators we require our primary independent variables to be replicated
across buffers corresponding to the various stages. Parthenon buffers will only duplicate
fields marked `Metadata::Independent` across these various buffers, while all others will
point to the same data.

```cpp title="physics/hydro/hydro.cpp:hydro_add_fields"
--8<-- "physics/hydro/hydro.cpp:hydro_add_fields"
```

## RK-Stages

The multi-stage integrator is based of a method-of-lines approach where the temporal
discretization at each point in space is determined from the state of the system
at the current time, resulting in an ODE system of equations for each point on the 
grid.

```math
\dfrac{du^n_{i,j,k}}{dt} = \mathcal{L}_{i,j,k}(u^n_{l,m,n},...).
```

The `KamayanDriver` will accumulate $`\mathcal{L}`$ over all the units that have been
registered to it's `UnitCollection`. The driver will call into each unit's 
`AddTasksOneStep` to accumulate the right hand side into a provided `MeshData` 
container `dudt`, and the driver handles apply the changes to the solution.

### Flux-based Tasks

For the case that terms in $`\mathcal{L}`$ can be expressed as a difference of 
fluxes then it can be advantageous to have the driver accumulate the fluxes 
over all the units that provide a callback for `AddFluxTasks`.

```math
\begin{align*}
\mathcal{L}_{i,j,k} = \frac{1}{\mathcal{V}_{i,j,k}} 
    & \left (
                  A_{i+1/2,j,k}f^{x_1}_{i+1/2,j,k} - A_{i-1/2,j,k}f^{x_1}_{i-1/2,j,k} \right . \\
      & \left . + A_{i,j+1/2,k}f^{x_2}_{i,j+1/2,k} - A_{i,j-1/2,k}f^{x_2}_{i,j-1/2,k} \right . \\
      & \left . + A_{i,j,k+1/2}f^{x_3}_{i,j,k+1/2} - A_{i,j,k-1/2}f^{x_3}_{i,j,k-1/2} 
   \right )
\end{align*}
```

The driver will then take care of calling into the parthenon routines for flux 
correction at block and fine-coarse boundaries.

## Tasks

![Tasks in a single RK driver Stage](assets/generated/driver_tasks.svg)
///caption
Tasks from a final RK stage, with operator split tasks.
///

The `KamayanDriver` uses parthenon's [taksing](https://parthenon-hpc-lab.github.io/parthenon/develop/src/tasks.html) infrastructure to build out the 
execution for the components making up an evolution cycle. 
It is helpful to add named labels when registering tasks into the
`TaskList`s, as that is how the driver can generate the above task graph.

```cpp title="physics/hydro/hydro_add_flux_tasks.cpp:add_task"
--8<-- "physics/hydro/hydro_add_flux_tasks.cpp:add_task"
```

The `UnitCollection` holds all the various `KamayanUnits` as a `std::map` and so 
can not guarantee the order of execution. Instead the order that tasks
are added can be specified with the various lists owned by the `UnitCollection`,

```cpp title="kamayan/unit.cpp:rk_flux"
--8<-- "kamayan/unit.cpp:rk_flux"
```


## Parameters

{!assets/generated/driver_parms.md!}
