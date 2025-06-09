# Kamayan

A parthenon code for playing with concepts for a modular multiphysics and/or high-order hydro code.

## quick start
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
`kamayan` uses `uv` to manage the python dependencies. All python scripts/tooling should be executed prefixed with `uv run`.
For example the lintting can be done with
```
uv run cpplint --recursive src/
```
```
git clone https://github.com/acreyes/kamayan.git
cd kamayan
git submodule update --init --recursive
mkdir build && cd build
uv run cmake ..
uv run cmake --build . -j4
mpirun -np 4 isentropic_vortex -i ../src/problems/isentropic_vortex.in
```
