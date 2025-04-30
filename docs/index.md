# Welcome to Kamayan

<p align="center">
   <img src="assets/images/paw-amr-25.png">
</p>

A [kamayan](https://en.wikipedia.org/wiki/Kamayan) is a Filipino cultural practice where a feast
is shared communally, spread on top of a table, with diners eating with their
hands, unhindered by the formalities often associated with western dining etiquette. 
The [kamayan](https://github.com/acreyes/kamayan) here aims to follow in that tradition as a hydrodynamics code,
easing the combination of novel numerical methods with multi-physics applications. 

The code is built on the [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) Adaptive Mesh Refinement(AMR) framework for performance
portability. 

## Quick Start

Kamayan uses [uv](https://docs.astral.sh/uv/) to manage python dependencies used for tooling. Independently any
dependencies for parthenon must be built externally, e.g., hdf5 & mpi. A quick start
to getting kamayan and building the code can be done with
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/acreyes/kamayan.git
cd kamayan
git submodule update --init --recursive
mkdir build && cd build
uv run cmake ..
uv run cmake --build . -j4
mpirun -np 4 isentropic_vortex -i ../src/problems/isentropic_vortex.in
```

