# Material Properties

Material Properties is used to manage various physics for computing derived
quantities, that are not directly evolved, but that can be calculated from the
rest of the state vector. 

!!! info
    
    Currently the equation of state is the only implemented material property.

## Species

Materials in the simulation can be composed of species, potentially each
with their own material properties. Species are declared as a comma separated
list of strings in the input block, and will correspondingly generate a species
input block for each species in the list.


!!! warning

    Currently only the single species is supported.

## Parameters
{!assets/generated/material_parms.md!}

