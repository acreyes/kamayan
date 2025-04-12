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

```cpp
--8<-- "physics/hydro/hydro.cpp:hydro_add_fields"
```



## Flux-based



![Tasks in a single RK driver Stage](assets/generated/driver_tasks.svg)

## Parameters

{!assets/generated/driver_parms.md!}
