.. _getting_started:

Getting Started
===============

What is MuPhFASa
-----------------

MuPhFASa (the MultiPHase Flow Adaptive Simulator) is a Lattice-Boltzmann
solver for multiphase flows built on `AMReX
<https://amrex-codes.github.io/amrex/>`_, the block-structured
adaptive-mesh-refinement (AMR) framework. Currently, development is focused
on single-phase flow; multiphase capability is planned but not yet
implemented.

Dependencies
------------

MuPhFASa depends on a single external library: **AMReX**. AMReX is vendored
as a git submodule at ``external/amrex`` rather than being expected as a
system install, so AMReX's own build machinery (CMake or GNUMake) is pulled
in directly from the submodule checkout.

Because AMReX is a submodule, a fresh clone of MuPhFASa does **not**
automatically contain its source — you must initialize it before building::

    git submodule update --init --recursive

Both build systems described below check for this: the CMake build aborts
with a ``external/amrex is empty`` message if ``external/amrex/CMakeLists.txt``
is missing, and the GNUmakefile build will fail to find
``$(AMREX_HOME)/Tools/GNUMake/Make.defs`` if the submodule hasn't been
checked out.

Building with CMake
--------------------

The top-level ``CMakeLists.txt`` defines a set of ``muphfasa_*`` options that
mirror the backend choices AMReX itself exposes, and forwards them to AMReX's
own CMake cache variables before calling ``add_subdirectory(external/amrex)``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - CMake option
     - Default
     - Meaning
   * - ``muphfasa_DIM``
     - ``3``
     - Dimensionality of the solver (``1``, ``2``, or ``3``). Forwarded to
       AMReX as ``AMReX_SPACEDIM``.
   * - ``muphfasa_ENABLE_MPI``
     - ``ON``
     - Enable MPI distributed parallelism. Forwarded as ``AMReX_MPI``.
   * - ``muphfasa_ENABLE_OMP``
     - ``OFF``
     - Enable OpenMP threading. Forwarded as ``AMReX_OMP``.
   * - ``muphfasa_ENABLE_CUDA``
     - ``OFF``
     - Enable the NVIDIA CUDA GPU backend. Sets AMReX's ``AMReX_GPU_BACKEND``
       to ``CUDA``.
   * - ``muphfasa_ENABLE_HIP``
     - ``OFF``
     - Enable the AMD HIP GPU backend. Sets AMReX's ``AMReX_GPU_BACKEND`` to
       ``HIP``.
   * - ``muphfasa_ENABLE_EB``
     - ``ON``
     - Enable AMReX embedded-boundary support. Forwarded as ``AMReX_EB``.

.. warning::

   ``muphfasa_ENABLE_CUDA`` and ``muphfasa_ENABLE_HIP`` are mutually
   exclusive. Enabling both triggers a ``FATAL_ERROR`` at configure time
   (``CUDA and HIP backends are mutually exclusive.``). If neither is
   enabled, AMReX is configured with ``AMReX_GPU_BACKEND=NONE`` (CPU-only).

A typical out-of-source build looks like::

    mkdir build && cd build
    cmake .. -Dmuphfasa_DIM=3
    cmake --build . -j

This produces a ``muphfasa`` executable linked against
``AMReX::amrex_<DIM>d`` (e.g. ``AMReX::amrex_3d``). Any additional AMReX
CMake option not already covered by a ``muphfasa_*`` wrapper (for instance,
AMReX precision or logging options) can still be set directly on the
``cmake ..`` command line as a normal AMReX cache variable (e.g.
``-DAMReX_PRECISION=DOUBLE``), since AMReX's ``CMakeLists.txt`` is included
via ``add_subdirectory`` and reads the same CMake cache.

Building the test suite
~~~~~~~~~~~~~~~~~~~~~~~~

Tests are off by default. Pass ``-Dmuphfasa_BUILD_TESTS=ON`` to also
configure ``tests/``:

.. code-block:: ini

    cmake .. -Dmuphfasa_DIM=3 -Dmuphfasa_BUILD_TESTS=ON

When enabled, the top-level ``CMakeLists.txt`` calls ``enable_testing()``
and ``add_subdirectory(tests)``, which builds the unit tests
(``test_lattice``, ``test_macroscopic``, ``test_equilibrium``,
``test_units``) against the solver sources (excluding ``main.cpp``), plus
regression tests (``poiseuille``, ``lid_driven_cavity``, ``couette``) that
run the ``muphfasa`` binary (or a dedicated analysis executable, for
``couette``) against reference ``inputs`` files under ``tests/regression/``.
Tests can then be run from the build directory with ``ctest``.

Building with GNUmakefile
--------------------------

As an alternative to CMake, MuPhFASa can be built directly with AMReX's
GNUMake build system via the top-level ``GNUmakefile``. This mirrors the
CMake options above, using AMReX's native GNUMake variable names:

.. code-block:: ini

    make DIM=2                       # 2-D solver, MPI, no GPU
    make DIM=3                       # 3-D solver, MPI, no GPU
    make DIM=2 DEBUG=TRUE             # debug build
    make DIM=2 USE_MPI=FALSE          # single-process build
    make DIM=2 USE_OMP=TRUE           # OpenMP-threaded build
    make DIM=2 USE_CUDA=TRUE          # NVIDIA GPU build
    make DIM=2 USE_HIP=TRUE           # AMD GPU build

Relevant variables, with their defaults from the ``GNUmakefile``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Variable
     - Default
     - Meaning
   * - ``AMREX_HOME``
     - ``$(CURDIR)/external/amrex``
     - Path to the AMReX checkout (the vendored submodule, by default).
   * - ``COMP``
     - ``gnu``
     - Compiler collection: ``gnu`` \| ``intel`` \| ``cray`` \| ``pgi`` \| ``llvm``.
   * - ``DIM``
     - ``3``
     - Dimensionality (``2`` or ``3``).
   * - ``DEBUG``
     - ``FALSE``
     - Debug build.
   * - ``USE_MPI``
     - ``TRUE``
     - Enable MPI.
   * - ``USE_OMP``
     - ``FALSE``
     - Enable OpenMP.
   * - ``USE_CUDA``
     - ``FALSE``
     - Enable CUDA GPU backend.
   * - ``USE_HIP``
     - ``FALSE``
     - Enable HIP GPU backend.
   * - ``USE_EB``
     - ``TRUE``
     - Enable embedded-boundary support.

The GNUmakefile sets ``BL_NO_FORT = TRUE`` (there are no Fortran sources in
this project), compiles the same translation units as the CMake build
(``src/main.cpp`` plus the ``LBM``, ``Geometry``, ``IO``, and ``Simulation``
sources), and links against the AMReX library modules ``Base``, ``Boundary``,
``AmrCore``, and ``EB`` (via ``Make.package`` files included from
``$(AMREX_HOME)/Src/<module>``). It ultimately includes
``$(AMREX_HOME)/Tools/GNUMake/Make.defs`` and ``Make.rules``, so it must be
run from a checkout where ``external/amrex`` has already been initialized.
The resulting executable base name is ``muphfasa`` (AMReX's GNUMake system
appends a suffix encoding ``DIM``/``COMP``/``DEBUG``/etc.).

A GNUmakefile also exists under ``tests/`` (``EBASE = muphfasa_unit_tests``)
for building the unit tests the same way; it can be invoked with
``make -C tests``.

Running a simulation
---------------------

Once built, run the executable with a single positional argument: the path
to an input file::

    ./muphfasa <inputs-file>

Input files are plain-text ``ParmParse``-style key/value files (parsed by
AMReX). Two worked examples ship under ``example_inputs/``:

- ``example_inputs/inputs.2d`` — a 2-D case using lattice ``D2Q9``, with a
  128x64 grid, periodic in x and walls (``NoSlip``) in y.
- ``example_inputs/inputs.3d`` — a 3-D case using lattice ``D3Q19``, with a
  64x32x32 grid, periodic in x and z and walls (``NoSlip``) in y.

Run them, for example, as::

    ./muphfasa example_inputs/inputs.2d     # requires a 2-D build (DIM=2)
    ./muphfasa example_inputs/inputs.3d     # requires a 3-D build (DIM=3)

See :doc:`input_files` for the full parameter reference.

.. warning::

   The lattice requested via ``lbm.lattice_type`` must match the
   dimensionality the executable was *built* for
   (``muphfasa_DIM``/``AMREX_SPACEDIM``). This is enforced at simulation
   start-up in ``src/Simulation/SimulationFactory.cpp``:

   - In a **2-D build**, only ``D2Q9`` is accepted. Any other value causes
     ``amrex::Abort`` with the message:
     ``Unknown or unsupported lbm.lattice_type '<value>' for a 2D build
     (only D2Q9 is available).``
   - In a **3-D build**, ``D3Q15``, ``D3Q19``, or ``D3Q27`` are accepted.
     Any other value aborts with:
     ``Unknown or unsupported lbm.lattice_type '<value>' for a 3D build
     (expected D3Q15 | D3Q19 | D3Q27).``

   There is no runtime dimension switch: a 2-D-lattice input file cannot be
   run with a 3-D-built executable and vice versa; you must rebuild with the
   matching ``muphfasa_DIM``/``DIM``.
