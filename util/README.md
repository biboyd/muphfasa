# Utilities

Standalone helpers that build without the full solver.

## input_info

Reads a muphfasa input file with the solver's own parser and prints what a run would start with, without running anything: grid and memory, unit conversion and lattice parameters (dx, dt, u_lat, tau), boundary conditions, estimated step count and end time, and disk usage of the plot/checkpoint output. Out-of-range `tau` / `u_lat` produce warnings with a suggested fix instead of an abort. If `amr.restart` is set, it also checks the checkpoint's Header against the input (grid size, number of distributions) and reports the steps that remain.

Use it to sanity-check an input file or a restart before submitting a job.

The executable is built for one dimensionality, which must match the input file (2 or 3 values in `grid.ncells`).

### Build

CMake (builds `muphfasa_inputinfo`):

```sh
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_UTIL=ON
cmake --build build2d --target muphfasa_inputinfo -j$(nproc)
./build2d/util/muphfasa_inputinfo <inputs file>
```

GNUMake (needs `AMREX_HOME`, or `external/amrex`):

```sh
cd util/input_info
make DIM=2 -j$(nproc)
./input_info2d.gnu.ex <inputs file>
```

### Run

```sh
muphfasa_inputinfo example_inputs/inputs.2d
muphfasa_inputinfo example_inputs/inputs.2d lbm.tau=0.8 amr.restart=chk0002000
```

Extra `key=value` arguments override the file, as for the solver. Run it from the directory the solver will run in so a relative `amr.restart` path resolves the same way.

## powerspec

Computes the shell-averaged kinetic energy spectrum E(k) of a muphfasa plotfile using `amrex::FFT` (based on the AMReX tutorial `FFT/EnergySpectrum`). It reads `vx`, `vy`, `vz` (m/s) from the plotfile, so E is per unit mass in SI units, and writes `shell, k [rad/m], E_shell, E(k)` to a text file. It also prints the kinetic energy in physical and spectral space as a Parseval check.

The field is assumed periodic, so it is meant for periodic flows such as Taylor-Green or forced turbulence in a periodic box. Shells have width `2 pi / L_max`. Domains with unequal sides work, and the odd-sized grids FFTW supports work too. Shell 0 is the mean flow.

Built for 3D by default. Like `input_info`, the dimension must match the plotfile; a mismatch is reported with the `DIM` to rebuild with.

Needs FFTW (or the vendor FFT library on GPUs); set `FFTW_HOME` if it is not on the default search path.

### Build

GNUMake (MPI by default, `USE_MPI=FALSE` for serial):

```sh
cd util/powerspec
make -j$(nproc)            # 3D: powerspec3d.gnu.MPI.ex
make DIM=2 -j$(nproc)      # 2D plotfiles
```

CMake (enables AMReX's FFT module, so it needs FFTW at configure time):

```sh
cmake -S . -B build -Dmuphfasa_DIM=3 -Dmuphfasa_BUILD_POWERSPEC=ON
cmake --build build --target muphfasa_powerspec -j$(nproc)
```

### Run

```sh
./powerspec3d.gnu.MPI.ex plt0005000 outfile=spectrum.txt
mpirun -np 8 ./powerspec3d.gnu.MPI.ex plt0005000
```
