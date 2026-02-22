# Equation Of State Unit

Kamayan uses [singularity-eos](https://lanl.github.io/singularity-eos/main/index.html) 
to perform equation of state calls. The equation of state
converts between a set of thermodynamic variables, namely
density, internal energy, pressure and temperature for a 
given fluid component. 


!!! note

    Only the gamma law equation of state from singularity is currently supported

In order to support a wide range of possible equation of state capabilities (e.g.,
 1 & 3 temperature, and multiple species) a single equation of state is wrapped
 in the `EOS_t` type that is owned by the `material` unit. 
 The type wraps the visitor pattern needed to call the implementations in the variant.
The eos gets called with a [subpack](../grid.md#subpacks)
that can index into the required variables for the `EosComponent`.
The variables associated with a given component are 
defined in a `TypeList` `EosVars<eos_component>::types` type trait
that is specialized for each option.
There is an `EosVariables` typedef that can be used to determine the 
variables needed for a given `Fluid` method.

Finally the equation of state needs to be told what fields to treat as input,
and what to fill as the output into the indexer. 

### Eos modes

| mode | input | output |
| ---  | ----  | ------ |
| ener | `EINT` `DENS` | `PRES` `TEMP` |
| temp| `TEMP` `DENS` | `PRES` `EINT` |
| pres | `PRES` `DENS` | `EINT` `TEMP` |

## `EosWrapped`

Often there isn't a need to call the equation of state directly for a given cell,
rather one wishes to have the equation of state called over all cells
on the domain to make the state thermodynamically consistent. When this is the case
one can just depend on the `EosWrapped` call to call the equation of state
on an entire `MeshData` or `MeshBlock`.


## Parameters
{!assets/generated/eos_parms.md!}
