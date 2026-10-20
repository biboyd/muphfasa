.. _testing:

Testing
=======

MuPhFASa's test suite lives under ``tests/`` at the repository root and is
split into two tiers that use different build systems:

1. **Unit tests** (``tests/unit/``) — small, fast, single-process checks of
   individual LBM kernels (lattice descriptors, macroscopic moments,
   equilibrium distributions, the physical/lattice unit converter). These are
   registered with CTest and built through the project's normal CMake build
   when the ``muphfasa_BUILD_TESTS`` option is enabled.

2. **Regression / physics-validation tests** (``tests/regression/``) — each
   one runs (a build of) the solver against a classic analytic LBM benchmark
   and compares the simulated field against the known solution. Some of these
   (``poiseuille``, ``lid_driven_cavity``) are wired into CTest as plain
   "does the solver run to completion on this inputs file" checks; others
   (``couette``) build their own dedicated analysis executable and are run
   via CTest *or* standalone, with extra Python tooling for convergence
   studies and plotting.

.. note::

   The ``couette`` regression test (and its Python tooling) can also be built
   and run completely independently of the main CMake build, using its own
   classic-AMReX ``GNUmakefile``. This is a different build system from the
   top-level CMake build described in :doc:`getting_started`, and is useful
   for iterating on the test itself without reconfiguring the full project.

Build wiring
------------

The top-level ``CMakeLists.txt`` gates the entire test suite behind one
option:

.. code-block:: cmake

   option(muphfasa_BUILD_TESTS "Build the validation and unit test suite" OFF)
   if(muphfasa_BUILD_TESTS)
       enable_testing()
       add_subdirectory(tests)
   endif()

When ``muphfasa_BUILD_TESTS=ON``, ``tests/CMakeLists.txt`` does three things:

* For each name in ``test_lattice``, ``test_macroscopic``, ``test_equilibrium``,
  ``test_units`` it builds a standalone executable (each ``test_*.cpp`` defines
  its own ``main()``, the AMReX idiom) linked directly against the solver's
  source files (excluding ``main.cpp``) so the tests stay in sync with the
  main binary without a shared library, and registers it with
  ``add_test(NAME <name> COMMAND <name>)``.
* For ``poiseuille`` and ``lid_driven_cavity`` it registers
  ``regression_<name>`` CTest cases that simply invoke the main ``muphfasa``
  executable with the ``inputs`` file in that test's directory (no dedicated
  pass/fail analysis beyond the run completing).
* For ``couette`` it builds a dedicated ``test_couette`` executable from
  ``regression/couette/test_couette.cpp`` plus the same solver sources, and
  registers it as ``regression_couette``, run from the
  ``regression/couette/`` directory so it can find its own ``inputs`` file.

``tests/GNUmakefile`` is a separate, classic-AMReX-style makefile that builds
all four unit tests together into one ``muphfasa_unit_tests`` executable
(mirroring the top-level ``GNUmakefile``'s options); it is an alternative to
the CMake path above, not used by CTest.

Unit tests (``tests/unit/``)
-----------------------------

.. warning::

   As of this writing, all four unit test files below compile and link
   correctly (and will report "pass" to CTest, since they exit 0), but their
   actual check bodies are **stubbed out** with ``// TODO`` comments. None of
   them currently exercises solver behavior beyond printing a banner. The
   descriptions here record exactly what each file documents it *intends* to
   check (and, where helper functions already exist, what those helpers
   would check once wired in), so this section stays accurate as the stubs
   get filled in.

test_lattice.cpp
~~~~~~~~~~~~~~~~

Intends to check mathematical properties that must hold for any valid
velocity set, via two templated helper functions parameterized on the
``LatticeDescriptor`` tag ``L``:

* ``testWeightSum<L>()`` — asserts ``sum_q w_q == 1`` to within ``1e-12``.
* ``testOppositeInvolution<L>()`` — asserts ``opposite[opposite[q]] == q``
  for every direction ``q`` (i.e. ``opposite[]`` is a self-consistent
  involution).

The file's header comment also lists checking ``cs2 == 1/3``, though no
helper for that exists yet. In ``main()``, calls for both helpers against
``muphfasa::D2Q9`` and ``muphfasa::D3Q19`` are present but **commented out**
with a ``// TODO: uncomment as implementations are filled in`` marker, so
neither D2Q9 nor D3Q15/D3Q19/D3Q27 is actually probed at present.

test_macroscopic.cpp
~~~~~~~~~~~~~~~~~~~~~

States its intent to verify, for a known distribution array ``f``:

* ``computeDensity`` returns the correct sum over ``f``.
* ``computeVelocity`` returns the correct first moment of ``f``.
* Round-tripping an equilibrium state (building ``f^eq`` from a given
  ``rho``, ``u`` and then computing moments back from it) recovers the
  original ``rho`` and ``u``.

The body of ``main()`` is a single ``// TODO`` comment; no ``f`` array is
constructed and no assertion is made.

test_equilibrium.cpp
~~~~~~~~~~~~~~~~~~~~~

States its intent to verify, for the equilibrium distribution function:

* Mass conservation: ``sum_q f^eq_q == rho``.
* Momentum conservation: ``sum_q c_{q,alpha} f^eq_q == rho * u_alpha``.
* Non-negativity of ``f^eq`` for ``Ma << 1``.
* Recovery of the rest-state limit when ``u = 0``.

As with the other files, ``main()`` is a ``// TODO`` placeholder; no call to
``computeEquilibrium<D2Q9>`` is made.

test_units.cpp
~~~~~~~~~~~~~~~

States its intent to verify round-trip physical→lattice→physical unit
conversions and the consistency of derived dimensionless parameters. The
comment in ``main()`` specifically calls out:

* ``toLatticeLength(refLength())`` returning ``N_ref``.
* ``relaxationTime() > 0.5`` (the LBM stability requirement).
* ``machNumber() < 0.1`` (the low-Mach assumption).

It does **not** currently exercise, even as a stated intent, the
``tau``/``u_lat`` mutual-exclusivity abort path or any stability-bound
warning messages in ``src/LBM/Units.cpp`` — those are not mentioned anywhere
in this file. A small ``assertClose(a, b, tol, msg)`` helper is defined
(throws ``std::runtime_error`` on mismatch) but is never called, since the
body of ``main()`` is, again, only a ``// TODO`` comment.

Regression test: plane Couette flow (``tests/regression/couette/``)
---------------------------------------------------------------------

Physical problem
~~~~~~~~~~~~~~~~~

Plane Couette flow in a 2-D channel (D2Q9): periodic in ``x``, a stationary
(``NoSlip``) wall at ``y = lo``, and a wall moving tangentially at ``U0``
(``MovingWall``) at ``y = hi``. With ``H`` equal to ``grid.ncells[1]`` in
lattice units (the halfway-bounce-back convention places each wall half a
cell beyond the last fluid node, so ``H == Ny`` exactly), the flow obeys the
1-D unsteady diffusion equation

.. code-block:: text

   du/dt = nu * d^2u/dy^2,   u(0,t) = 0,  u(H,t) = U0 (t>0),  u(y,0) = 0

whose solution is the classic plane-Couette-startup ("doubly-bounded
Rayleigh") series

.. code-block:: text

   u(y,t)/U0 = y/H + sum_{n=1}^inf [2*(-1)^n/(n*pi)] * sin(n*pi*y/H)
                     * exp(-n^2*pi^2*nu*t/H^2)

which reduces to the steady linear profile ``u(y) = U0*y/H`` as ``t -> inf``.
This exercises both the solver's transient startup behavior and its
steady-state accuracy against a known closed-form solution — unlike
``poiseuille``/``lid_driven_cavity``, which (today) only check that the
solver runs.

What ``test_couette.cpp`` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``test_couette.cpp`` builds its own executable (``test_couette``, or
``test_couette2d.gnu.ex`` when built via the standalone GNUmakefile) with a
custom ``main()``. Each step, it calls ``sim->advance()``/``writeOutput()``
and, on plot-interval steps, evaluates the analytic series above
(``analyticCouette()``, 200 terms) on the same grid, writes it out as a
matching ``analytic_pltXXXXXXX`` plotfile alongside the solver's own
``pltXXXXXXX``, and records the inf-norm error between the two velocity
fields (``max_err``). The final entry of ``max_err`` is printed as the
"Final Steady state error".

How to run it
~~~~~~~~~~~~~~

Standalone (own GNUmakefile, DIM=2, no MPI by default):

.. code-block:: bash

   cd tests/regression/couette
   make                       # builds test_couette2d.gnu.ex
   ./test_couette2d.gnu.ex inputs

This produces ``pltXXXXXXX``/``analytic_pltXXXXXXX`` plotfile pairs that can
also be diffed directly with AMReX's ``fcompare`` tool.

Via CTest (after a CMake configure with ``muphfasa_BUILD_TESTS=ON``), it runs
as the ``regression_couette`` test, executing the ``test_couette`` binary
with ``inputs`` from the ``tests/regression/couette/`` working directory.

Plotting (``plot_couette.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd tests/regression/couette
   make                    # build test_couette2d.gnu.ex, if not already built
   python3 plot_couette.py # builds faverage (DIM=2) on first run, then plots

Runs the solver with the base ``inputs`` file in its own ``profile_run/``
subdirectory (solver log in ``run.log``), laterally averages ``vx`` in each
``plt``/``analytic_plt`` pair using AMReX's ``faverage`` tool to get a
``vx(y)`` profile, and renders ``couette_decay.png`` with two panels:

1. Velocity profiles ``u/U0`` vs. ``y/H`` at each snapshot time — simulated
   (markers) against the analytic solution (lines), both read straight from
   the solver's own plotfiles.
2. The dominant (``n=1``) Fourier mode's log-amplitude vs. time, with a
   fitted decay rate compared against the analytic rate ``pi^2*nu/H^2``.

Useful flags: ``--inputs <path>``, ``--override sim.max_step=8000 ...``,
``--no-run`` (re-plot from an existing ``profile_run/`` without rerunning),
``--out <path>``, ``--max-profile-err <tol>`` / ``--max-rate-rel-err <tol>``
(nonzero exit if the max profile error or the fitted decay rate's relative
error against the analytic rate exceeds the given tolerance; used in CI).

Convergence study (``convergence.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd tests/regression/couette
   make                       # build test_couette2d.gnu.ex, if not already built
   python3 convergence.py     # builds fcompare (DIM=2) on first run, then runs the study

Runs the same problem at several resolutions (``Ny = 16, 32, 64`` by
default, overridable with ``--ny-list``), pinning ``lbm.tau`` to the same
value (default ``0.8``, via ``--tau``) at every resolution rather than
letting it drift with ``Ny`` — this keeps the lattice wall Reynolds number
resolution-independent. Every run is stopped at the same dimensionless time
``nu*t/H^2 = 0.05`` (``--target-time``), so the comparison across
resolutions is apples-to-apples (well before steady state). Each resolution
is run in its own ``convergence_runs/Ny<N>/`` subdirectory (generated
output, not checked in as source), and its ``plt``/``analytic_plt`` pair is
diffed with AMReX's ``fcompare`` tool.

The script:

* Prints a table of ``Ny, dx, target_step, abs_err_vx, order_vx`` (the
  observed order of accuracy fit between consecutive resolutions, assuming
  each ``Ny`` doubles the previous one) and writes it to
  ``couette_convergence.csv``.
* Produces a log-log plot of ``abs_err_vx`` vs. ``dx`` against a reference
  2nd-order line, saved to ``couette_convergence.png`` (skip with
  ``--no-plot``).
* Only tracks ``vx`` — ``vy`` is identically zero for this flow, so its
  ``fcompare`` "error" is pure round-off noise with no real convergence
  signal.
* Supports ``--min-order <N>`` to fail (nonzero exit) if the worst observed
  order falls below the given threshold; CI uses ``--min-order 2.5``.

CI
~~~

Per the test's ``README.md``, ``.github/workflows/couette-regression.yml``
builds ``test_couette2d.gnu.ex`` and runs both scripts on every push/PR
touching ``src/``, ``tests/regression/couette/``, or the AMReX submodule:
``convergence.py --min-order 2.5`` and
``plot_couette.py --max-profile-err 0.01 --max-rate-rel-err 0.01``. Plots and
the convergence CSV are uploaded as workflow artifacts regardless of
pass/fail.

.. note::

   ``tests/regression/couette/tmp_build_dir/`` and ``convergence_runs/`` are
   generated build and output directories, not source — they are not
   described further here.

Other regression cases
-----------------------

lid_driven_cavity and poiseuille
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/regression/lid_driven_cavity/inputs`` and
``tests/regression/poiseuille/inputs`` are input decks only — there is no
accompanying ``.cpp`` or ``.py`` driver, analysis, or plotting script in
either directory. ``tests/CMakeLists.txt`` does register each of them as a
CTest case (``regression_lid_driven_cavity``, ``regression_poiseuille``),
but those cases currently only run the main ``muphfasa`` solver binary
against the ``inputs`` file and check that it completes; there is no
automated comparison against the analytic solutions documented in the input
files' own comments (Ghia et al. (1982) tabulated streamfunction/vorticity
values at Re=100 for the cavity; the parabolic profile
``u(y) = (dp/dx)/(2*mu) * y*(H-y)`` for Poiseuille flow). In short: **input
decks present, automation not yet wired up** — treat these as staged
benchmarks, not as full regression tests on par with ``couette``.

tgv2d
~~~~~~

``tests/regression/tgv2d/`` contains only four committed artifact files —
``tgv2d_convergence.csv``, ``tgv2d_convergence.png``, ``tgv2d_decay.png``,
and ``tgv2d_snapshots.csv`` — with no ``inputs`` file and no driver script
(``.cpp`` or ``.py``) checked in anywhere in the directory. These appear to
be leftover output from a prior Taylor-Green-vortex convergence study; there
is currently no way to reproduce or re-run that study from what's checked
into this directory.

Running the test suite
------------------------

**Unit tests**, via CMake + CTest:

.. code-block:: bash

   cmake -S . -B build -Dmuphfasa_BUILD_TESTS=ON
   cmake --build build
   ctest --test-dir build

This also builds and registers the ``regression_poiseuille``,
``regression_lid_driven_cavity``, and ``regression_couette`` CTest cases
described above, so a plain ``ctest --test-dir build`` run covers both
tiers in one go.

**Couette regression test**, standalone (its own GNUmakefile, independent of
the CMake build above):

.. code-block:: bash

   cd tests/regression/couette
   make
   ./test_couette2d.gnu.ex inputs

   # or, for the convergence study / plots:
   python3 convergence.py
   python3 plot_couette.py
