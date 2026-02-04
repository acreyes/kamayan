# Building Kamayan

This guide covers how to obtain, configure, and build kamayan from source.

## Requirements

Kamayan has the following dependencies:

- **[UV](https://docs.astral.sh/uv/)** - Python package manager (required for testing, recommended for all builds)
- **CMake** 3.25 or higher
- **C++ Compiler** with C++20 support
- **HDF5** - For I/O operations
- **MPI** - Message Passing Interface implementation (OpenMPI, MPICH, Intel MPI, or compatible)

### Installing UV

UV is used to manage Python dependencies for kamayan's tooling and bindings. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods, see the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting the Source

Clone the kamayan repository and initialize its submodules (Parthenon, singularity-eos, and nanobind):

```bash
git clone https://github.com/acreyes/kamayan.git
cd kamayan
git submodule update --init --recursive
```

## CMake Build Options

Kamayan provides several CMake options to control the build:

| Option | Description | Default |
|--------|-------------|---------|
| `kamayan_ENABLE_TESTING` | Enable kamayan test suite (requires UV) | `ON` |
| `kamayan_BUILD_DOCS` | Build the kamayan documentation | `OFF` |
| `kamayan_BUILD_PYKAMAYAN` | Build Python bindings (pyKamayan) | `ON` |
| `KAMAYAN_ENSURE_MPI4PY` | Automatically rebuild mpi4py when MPI mismatch detected | `OFF` |
| `kamayan_NP_TESTING` | Number of MPI ranks to use in testing | `2` |
| `KAMAYAN_VERBOSE_MPI_CHECK` | Enable verbose MPI compatibility checking | `OFF` |

Additional Parthenon and singularity-eos options are also available. See their respective documentation for details.

## Building Kamayan

### Standard Build (with Python bindings)

Create a build directory and configure the project:

```bash
mkdir build && cd build
cmake ..
cmake --build . -j4
```

If UV is installed, you can prefix the cmake commands with `uv run` to ensure consistency:

```bash
mkdir build && cd build
uv run cmake ..
uv run cmake --build . -j4
```

This will build:

- The core kamayan library
- Python bindings (pyKamayan)
- Example problems in `src/problems/` (see [Setting up a Simulation](simulation_setup.md))

### C++ Only Build

To build without Python bindings:

```bash
mkdir build && cd build
cmake -Dkamayan_BUILD_PYKAMAYAN=OFF ..
cmake --build . -j4
```

### Build with Testing Disabled

If UV is not available and you don't need tests:

```bash
mkdir build && cd build
cmake -Dkamayan_ENABLE_TESTING=OFF ..
cmake --build . -j4
```

!!! note
    Disabling testing also disables the documentation build, as docs require the test infrastructure.

### Build Types

Kamayan defaults to `Release` build. To change the build type:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..      # Debug build with symbols
cmake -DCMAKE_BUILD_TYPE=Release ..    # Optimized release build (default)
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. # Release with debug info
```

## Running a Simulation

After building, you can run one of the example problems:

```bash
# From the build directory
mpirun -np 4 isentropic_vortex -i ../src/problems/isentropic_vortex.in
```

For information on creating your own simulations, see 
[Setting up a Simulation](simulation_setup.md).

## Troubleshooting

### MPI/mpi4py Compatibility

Kamayan verifies that the MPI implementation used by mpi4py matches the one discovered by CMake. If there's a mismatch, you'll see an error like:

```
CMake Error: MPI implementation mismatch detected!
  CMake found: OpenMPI
  mpi4py uses: MPICH
```

**Solutions:**

1. **Automatic rebuild** (recommended): Reconfigure with `KAMAYAN_ENSURE_MPI4PY=ON`:
   ```bash
   cmake -DKAMAYAN_ENSURE_MPI4PY=ON ..
   ```
   This will automatically rebuild mpi4py with the correct MPI implementation.

2. **Manual rebuild**: Use the provided target:
   ```bash
   make rebuild-mpi4py
   ```

3. **Install matching mpi4py**: Install mpi4py manually with your MPI compiler wrappers:
   ```bash
   MPICC=mpicc uv pip install --no-build-isolation --force-reinstall mpi4py
   ```

### UV Not Found Warnings

If you see warnings about UV not being found:

- **For testing**: UV is required. Install it from https://docs.astral.sh/uv/ or disable testing with `-Dkamayan_ENABLE_TESTING=OFF`
- **For Python bindings**: Install UV or disable bindings with `-Dkamayan_BUILD_PYKAMAYAN=OFF`
- **For docs**: Install UV or disable docs with `-Dkamayan_BUILD_DOCS=OFF`

### Submodules Not Initialized

If you see errors about missing Parthenon or other dependencies:

```bash
git submodule update --init --recursive
```

### Python Module Not Found

If you encounter Python import errors when running simulations with pyKamayan, ensure you're using UV to manage the environment:

```bash
uv run python your_simulation.py
```

Or activate the UV-managed environment explicitly.

## Building Documentation

To build the kamayan documentation (requires UV and testing enabled):

```bash
cmake -Dkamayan_BUILD_DOCS=ON ..
cmake --build . --target docs
uv run mkdocs serve
```

This will start a python server for the docs, and should give a local url that will dynamically serve the docs for development.

## Advanced Build Options

### Specifying MPI Implementation

If CMake doesn't find the correct MPI implementation:

```bash
cmake -DMPI_C_COMPILER=/path/to/mpicc \
      -DMPI_CXX_COMPILER=/path/to/mpicxx \
      ..
```

### Custom HDF5 Location

If HDF5 is installed in a non-standard location:

```bash
cmake -DHDF5_DIR=/path/to/hdf5 ..
```

### Parallel Build

Speed up compilation by using more parallel jobs:

```bash
cmake --build . -j$(nproc)  # Use all available cores
```

## Next Steps

- [Set up a simulation (C++ or Python)](simulation_setup.md)
- [Python API Reference](api.md) - Detailed Python API documentation
- [Learn about Kamayan's architecture](kamayan.md)
- [Explore physics modules](physics.md)
- [Understand the driver system](driver.md)
- [Check out example problems](https://github.com/acreyes/kamayan/tree/main/src/problems)
