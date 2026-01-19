# Kamayan

A parthenon code for playing with concepts for a modular multiphysics and/or high-order hydro code.

## Quick Start

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Get kamayan and its dependencies
git clone https://github.com/acreyes/kamayan.git
cd kamayan
git submodule update --init --recursive

# Build kamayan
mkdir build && cd build
uv run cmake ..
uv run cmake --build . -j4

# Run an example simulation
mpirun -np 4 isentropic_vortex -i ../src/problems/isentropic_vortex.in
```

## Documentation

For detailed build instructions, configuration options, and API documentation, see the [full documentation](https://acreyes.github.io/kamayan).

Key documentation sections:

- [Building Kamayan](https://acreyes.github.io/kamayan/building/) - Detailed build instructions and troubleshooting
- [Kamayan Architecture](https://acreyes.github.io/kamayan/kamayan/) - Core concepts and API
- [Building Simulations](https://acreyes.github.io/kamayan/kamayan/#building-a-simulation) - Creating your own simulations

## Python Tooling

Kamayan uses `uv` to manage python dependencies. All python scripts and tooling should be executed prefixed with `uv run`.

For example, linting can be done with:
```bash
uv run cpplint --recursive src/
```
