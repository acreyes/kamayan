# Setting up a Simulation

This guide covers how to create and configure Kamayan simulations using either C++ or Python.

## C++

```cpp
--8<-- "problems/isentropic_vortex.cpp:isen_main"
```

Kamayan itself is more of a library of `KamayanUnit`s and a driver that is
steered by user code. Simulations must provide their own `main` function,
which will initialize kamayan, build a driver and execute the evolution 
loop. The main work of creating a new simulation is in building the 
driver from a `UnitCollection`. Kamayan provides a `kamayan::ProcessUnits()`
function that will build the default set of units, which can be modified
however one wishes. Additionally, units may be added into the `UnitCollection`,
which can hook into any of the interfaces described in [Kamayan Infrastructure](kamayan.md#kamayanunit).
Most importantly a unit that provides a `ProblemGenerator` should be added
to set the initial conditions.

Finally the new problem can be added to the build with a provided cmake function.

```cmake title="problems/CMakeLists.txt:add"
--8<-- "problems/CMakeLists.txt:add"
```

## Python

### Introduction

The design of kamayan as a library of component units that can be used to build a standalone
C++ program to run a simulation allows for a very similar program to be constructed using
the provided Python bindings (pyKamayan).

!!! info "API Reference"
    For detailed documentation of all Python classes and functions, see the [API Reference](api.md).

### Building a Simulation with pyKamayan

```python title="problems/sedov.py"
--8<-- "problems/sedov.py:py_sedov"
```

In complete analogy to the C++ approach, a `UnitCollection` is made using
`kamayan_manager.process_units` to construct a `"sedov"` unit using the provided
Python callbacks and taking the default unit collection provided by kamayan.
This is used to build the `KamayanManager` object that is used to set
input parameters and run the simulation. The `KamayanManager` owns a set of
properties that can be set with the various objects provided by the `kamayan.code_units`
module to set various input parameters in common combinations. Finally,
any arbitrary input parameter can be set through the `KamayanManager.params` property.

See the [Simulation Manager](api.md#simulation-manager) and [Core Module](api.md#core-module) 
in the API reference for detailed documentation.

### KamayanManager

::: kamayan.kamayan_manager.KamayanManager
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### process_units

::: kamayan.kamayan_manager.process_units
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

For details on UnitData, callbacks, and component architecture, see 
[Kamayan Infrastructure](kamayan.md).

### Adding the CLI Interface

The `@kamayan_app` decorator automatically generates a command-line interface with common commands
for running, testing, and inspecting your simulation. See the [CLI Interface](api.md#cli-interface) 
module in the API reference for detailed documentation.

### kamayan_app Decorator

::: kamayan.cli.app.kamayan_app
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Add the decorator and entry point to your simulation:

```python
from kamayan import kamayan_app

@kamayan_app(description="Sedov blast wave simulation")
def sedov() -> KamayanManager:
    # ... (simulation setup as shown above)
    return km

if __name__ == "__main__":
    sedov.app()
```

### Running Your Simulation

**Single process:**
```bash
uv run python src/problems/sedov.py
```

**Parallel (MPI):**
```bash
mpirun -np 4 uv run python src/problems/sedov.py
```

!!! warning "MPI Command Order"
    Always put `mpirun` before `uv run`:
    
    ✓ `mpirun -np 4 uv run python ...`  
    ✗ `uv run mpirun -np 4 python ...` (broken pipe error)

**Dry run (generate input only):**
```bash
uv run python src/problems/sedov.py run --dry-run
```

### Available Commands

The `@kamayan_app` decorator provides these commands:

| Command | Description |
|---------|-------------|
| `run` | Execute simulation (default) |
| `run --dry-run` | Generate input file only |
| `generate-input` | Generate input file |
| `info` | Show configuration |
| `version` | Show Kamayan version |

### Parthenon Arguments

Additional arguments are forwarded to Parthenon's argument parser. Kamayan 
automatically generates its own input file, so the `-i` flag is not used.

| Argument | Description | Example |
|----------|-------------|---------|
| `-r <file>` | Restart from checkpoint | `-r sedov.out0.00500.rhdf` |
| `-a <file>` | Analyze/postprocess | `-a output.phdf` |
| `-d <dir>` | Run directory | `-d /scratch/run` |
| `-t hh:mm:ss` | Wall time limit | `-t 01:30:00` |
| `-n` | Parse and quit | `-n` |
| `-m <n>` | Output mesh structure | `-m 4` |
| `-c` | Show config | `-c` |
| `block/par=val` | Override parameters | `sedov/density=2.0` |

**Example:**
```bash
mpirun -np 8 uv run python src/problems/sedov.py -r sedov.out0.00500.rhdf
```

### Typical Workflow

```bash
# Check configuration
uv run python src/problems/sedov.py info

# Test run
uv run python src/problems/sedov.py run --dry-run

# Production run
mpirun -np 16 uv run python src/problems/sedov.py

# Restart
mpirun -np 16 uv run python src/problems/sedov.py -r sedov.out0.00500.rhdf
```
