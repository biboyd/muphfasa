.. _input_files:

Input Files
===========

MuPhFASa input files are plain-text key/value files parsed by AMReX's
``ParmParse``, loaded from the path given as the executable's first
positional argument (``./muphfasa <inputs-file>``). Keys are grouped into
namespaces (``grid.*``, ``sim.*``, ``lbm.*``, ``units.*``, ``geometry.*``,
``bc.*``, ``amr.*``); this page documents every key read by
``src/IO/InputParser.cpp``, which is the sole gateway between the input
file and the solver.

Keys not recognized by any ``InputParser`` accessor are silently ignored by
``ParmParse`` — there is no "unknown key" error. Several such vestigial
keys exist in the shipped example inputs; they are flagged explicitly below
where they occur.

Worked examples
----------------

2-D example (``example_inputs/inputs.2d``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 128x64 ``D2Q9`` case on the domain :math:`[0,2]\times[0,1]`: periodic in
x, solid (``NoSlip``) walls at y = lo/hi — i.e. a periodic channel between
two no-slip plates.

.. code-block:: ini

    # inputs.2d — Sample input file for a 2-D muphfasa simulation.
    # Edit values to suit your problem; see each section for available keys.

    # ── Grid ──────────────────────────────────────────────────────────────────
    grid.ncells        = 128 64     # cells in x and y
    grid.max_grid_size = 32          # max tile size for load balancing
    grid.n_ghost       = 1           # ghost layers (1 is sufficient for BGK)

    # ── Simulation ─────────────────────────────────────────────────────────────
    sim.max_step       = 10000
    sim.stop_time      = -1.0        # negative means run to max_step

    # ── LBM ────────────────────────────────────────────────────────────────────
    lbm.lattice_type   = D2Q9
    lbm.tau            = 0.8         # BGK relaxation time  (tau > 0.5 for stability)

    # ── Physical / unit parameters ─────────────────────────────────────────────
    units.Re           = 100.0       # Reynolds number
    units.Ma           = 0.05        # Mach number (keep < 0.1 for accuracy)
    units.ref_length   = 1.0         # [m]
    units.ref_velocity = 1.0         # [m/s]
    units.ref_density  = 1.0         # [kg/m^3]

    # ── Geometry ───────────────────────────────────────────────────────────────
    geometry.prob_lo    = 0.0  0.0
    geometry.prob_hi    = 2.0  1.0
    geometry.type       = box         # "box" | "sphere" | "stl"
    geometry.use_eb     = 0

    # ── Boundary conditions ────────────────────────────────────────────────────
    bc.lo_type = Periodic  NoSlip
    bc.hi_type = Periodic  NoSlip

    # ── Output ─────────────────────────────────────────────────────────────────
    amr.plot_int        = 500
    amr.chk_int         = 2000
    amr.plot_file       = plt
    amr.check_file      = chk

3-D example (``example_inputs/inputs.3d``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 64x32x32 ``D3Q19`` case on :math:`[0,2]\times[0,1]\times[0,1]`: periodic
in x and z, ``NoSlip`` walls at y = lo/hi — a periodic duct between two
no-slip plates (plane-Couette/Poiseuille-like setup).

.. code-block:: ini

    # inputs.3d — Sample input file for a 3-D muphfasa simulation.

    # ── Grid ──────────────────────────────────────────────────────────────────
    grid.ncells        = 64 32 32
    grid.max_grid_size = 32
    grid.n_ghost       = 1

    # ── Simulation ─────────────────────────────────────────────────────────────
    sim.max_step       = 5000
    sim.stop_time      = -1.0

    # ── LBM ────────────────────────────────────────────────────────────────────
    lbm.lattice_type   = D3Q19      # D3Q15 | D3Q19 | D3Q27
    lbm.tau            = 0.8

    # ── Physical / unit parameters ─────────────────────────────────────────────
    units.Re            = 100.0
    units.Ma            = 0.05
    units.ref_length    = 1.0
    units.ref_velocity  = 1.0
    units.ref_density   = 1.0

    # ── Geometry ───────────────────────────────────────────────────────────────
    geometry.prob_lo    = 0.0  0.0  0.0
    geometry.prob_hi    = 2.0  1.0  1.0
    geometry.type       = box
    geometry.use_eb     = 0

    # ── Boundary conditions ────────────────────────────────────────────────────
    bc.lo_type = Periodic  NoSlip  Periodic
    bc.hi_type = Periodic  NoSlip  Periodic

    # ── Output ─────────────────────────────────────────────────────────────────
    amr.plot_int        = 500
    amr.chk_int         = 2000
    amr.plot_file       = plt
    amr.check_file      = chk

``grid.*``
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Key
     - Type
     - Default
     - Meaning
   * - ``grid.ncells``
     - int array, length ``AMREX_SPACEDIM``
     - *(required)*
     - Number of cells in each spatial direction. Read with ``ParmParse::getarr``,
       so the key must be present or the run aborts.
   * - ``grid.max_grid_size``
     - int
     - ``64``
     - Maximum box ("tile") size used when chopping the domain for load
       balancing (``GeometryHandler::buildBoxArray``).
   * - ``grid.n_ghost``
     - int
     - ``1``
     - Number of ghost-cell layers on the distribution-function MultiFab.
       One layer is sufficient for a single-relaxation-time (BGK) stencil.

``sim.*``
---------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Key
     - Type
     - Default
     - Meaning
   * - ``sim.max_step``
     - int
     - *(required)*
     - Maximum number of timesteps to run. Read with ``ParmParse::get``, so
       the key must be present.
   * - ``sim.stop_time``
     - real
     - ``-1.0``
     - Physical end time. A negative value (the default) means "ignore this
       and run to ``max_step``".

``lbm.*``
---------

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Key
     - Type
     - Default
     - Meaning
   * - ``lbm.lattice_type``
     - string
     - ``D3Q19``
     - Lattice stencil. ``D2Q9`` is the only value accepted by a 2-D build;
       ``D3Q15``, ``D3Q19``, ``D3Q27`` are accepted by a 3-D build. See the
       warning below.
   * - ``lbm.collision_model``
     - string
     - ``BGK``
     - Collision operator. ``BGK`` is implemented; ``TRT`` and ``MRT`` are
       recognized tags but not yet implemented, and selecting them aborts
       with ``lbm.collision_model '<x>' is a recognised tag but not yet
       implemented.``. Any other value aborts with ``Unknown
       lbm.collision_model '<x>' (expected BGK | TRT | MRT).``
   * - ``lbm.tau``
     - optional real
     - *(unset)*
     - BGK relaxation time, specified directly.
   * - ``lbm.u_lat``
     - optional real
     - *(unset)*
     - Dimensionless lattice reference velocity, specified directly.

.. warning::

   ``lbm.tau`` and ``lbm.u_lat`` are **not independent parameters** — they
   are two different ways of pinning the same lattice-spacing choice inside
   ``UnitConverter`` (``src/LBM/Units.cpp``), given the other physical
   reference scales (``units.ref_length``, ``units.ref_velocity``,
   ``units.Re``, grid resolution). Supply **at most one**:

   - If only ``lbm.tau`` is given, ``u_lat`` is derived from it.
   - If only ``lbm.u_lat`` is given, it is used directly.
   - If **both** are given, ``UnitConverter``'s constructor aborts
     unconditionally with: ``UnitConverter: lbm.u_lat and lbm.tau were both
     specified.specify only one and let the other be derived.`` (this is an
     unconditional ``AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, ...)`` — there
     is no code path where supplying both succeeds, even if the two values
     happen to be numerically consistent).
   - If **neither** is given, ``UnitConverter`` falls back to a default
     ``u_lat = 0.1``.

   Once resolved, ``UnitConverter`` derives ``tau`` and enforces/warns on
   stability bounds (see ``units.*`` below).

``units.*``
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Key
     - Type
     - Default
     - Meaning
   * - ``units.Re``
     - real
     - ``0.0``
     - Reynolds number, :math:`Re = U_{ref} L_{ref} / \nu_{ref}`. Used by
       ``UnitConverter`` to derive the reference kinematic viscosity
       ``nu_ref = U_ref * L_ref / Re`` — note ``Re = 0`` (the default if the
       key is omitted) causes a division by zero, so this key should always
       be set in practice even though ``InputParser`` only ``query``'s it.
   * - ``units.Ma``
     - real
     - ``0.0``
     - Read into ``InputParser::machNumber()``, but see the warning below:
       this value is **not** passed into ``UnitConverter`` and has no effect
       on the simulation.
   * - ``units.ref_length``
     - real
     - ``1.0``
     - Reference length :math:`L_{ref}` [m] (e.g. channel half-width).
   * - ``units.ref_velocity``
     - real
     - ``1.0``
     - Reference velocity :math:`U_{ref}` [m/s].
   * - ``units.ref_density``
     - real
     - ``1.0``
     - Reference density :math:`\rho_{ref}` [kg/m³]. Also used as the wall/
       inflow density for ``MovingWall`` and ``Inflow`` boundary faces (see
       ``bc.*`` below).

.. warning::

   **``units.Ma`` is an unused input key in the current code.**
   ``InputParser::machNumber()`` does read ``units.Ma`` via ``ParmParse``,
   but grep of the solver sources shows it is never called anywhere in the
   simulation construction path (``src/Simulation/SimulationSetup.cpp`` only
   calls ``refLength()``, ``refVelocity()``, ``reynoldsNumber()``,
   ``refDensity()``, ``ncells()``, ``relaxationTime()``, and
   ``latticeVelocity()`` on the ``InputParser``). The Mach number that
   actually governs the simulation is a derived, read-only *output* of
   ``UnitConverter::machNumber()`` (:math:`Ma = u_{lat}/c_s`, printed to
   stdout at start-up as part of the "Starting Simulation with Lattice
   Units" banner) — it cannot be set as an input. Setting ``units.Ma`` in an
   input file has no effect on the run; it is accepted silently and
   ignored.

The reference kinematic viscosity used by the solver is *derived*, not
read from an input key, inside ``UnitConverter``'s constructor
(``src/LBM/Units.cpp``):

.. code-block:: cpp

    m_nu_ref = m_U_ref * m_L_ref / m_Re;   // nu_ref = U_ref * L_ref / Re

so the effective viscosity is whatever ``Re``, ``ref_velocity``, and
``ref_length`` imply; there is no separate ``units.ref_viscosity`` key.

Stability bounds enforced by ``UnitConverter`` (``src/LBM/Units.cpp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After resolving ``tau``/``u_lat`` (see ``lbm.*`` above), the constructor:

- Prints a warning if the resolved ``tau`` falls outside ``0.6 < tau < 1.5``
  (its "ideal" range for accuracy), but does not abort.
- Prints a warning if the resolved ``u_lat > 0.15`` (close to violating the
  low-Mach assumption), but does not abort.
- **Aborts** (``AMREX_ALWAYS_ASSERT_WITH_MESSAGE``) if ``tau`` is not within
  the hard stability range ``0.5 < tau < 2.0``, with message: ``tau is not
  within the stability range 0.5 < tau < 2.0.Alter lbm.tau or lbm.u_lat to
  fix.``
- **Aborts** if ``u_lat >= 0.3`` (the low-Mach-number limit), with message:
  ``u_lat does not satisfy low Mach number limitAlter lbm.u_lat to fix.``

``geometry.*``
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Key
     - Type
     - Default
     - Meaning
   * - ``geometry.prob_lo``
     - real array, length ``AMREX_SPACEDIM``
     - *(required)*
     - Physical-domain lower corner [m]. Read with ``getarr``; must be
       present.
   * - ``geometry.prob_hi``
     - real array, length ``AMREX_SPACEDIM``
     - *(required)*
     - Physical-domain upper corner [m]. Read with ``getarr``; must be
       present.
   * - ``geometry.type``
     - string
     - ``box``
     - Geometry kind (``"box"`` \| ``"sphere"`` \| ``"stl"``). Only consumed
       by ``GeometryHandler::initEB()`` when ``geometry.use_eb`` is set, and
       that function is currently a ``TODO`` stub (``initEB`` just sets
       ``m_has_eb = true`` and does not yet build an actual embedded
       boundary from this type).
   * - ``geometry.use_eb``
     - int (as bool)
     - ``0``
     - If nonzero, calls ``GeometryHandler::initEB()``. See the TODO note
       above — EB geometry construction is not yet implemented.

Periodicity is not set via a ``geometry.*`` key; it is derived
per-direction in ``GeometryHandler::GeometryHandler()``
(``src/Geometry/GeometryHandler.cpp``) from whether ``bc.lo_type`` reads
``"Periodic"`` for that direction:

.. code-block:: cpp

    auto const bc_names = params.bcNames();
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
        is_periodic[dim] = (bc_names[2 * dim] == "Periodic") ? 1 : 0;
    }

So periodicity is controlled entirely by ``bc.lo_type`` / ``bc.hi_type``
(see ``bc.*`` below). Note also that ``SimulationSetup::buildBCManager()``
asserts that any face named ``"Periodic"`` in ``bc.*`` corresponds to a
direction that ``amrex::Geometry`` actually considers periodic, and aborts
otherwise with ``buildBCManager: bc input requests "Periodic" on a face
whose Geometry is not periodic in that direction`` — so ``bc.lo_type``/
``hi_type`` must agree per-direction (only the lo-face name is consulted to
set periodicity, so the hi-face name for a periodic direction is not
independently checked against geometry, only against this assertion).

``bc.*``
--------

Boundary conditions are specified per spatial direction via two
``ParmParse`` arrays:

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 30

   * - Key
     - Type
     - Default
     - Meaning
   * - ``bc.lo_type``
     - string array, length ``AMREX_SPACEDIM``
     - *(required)*
     - BC name for the low face of each direction, in order ``[x, y, (z)]``.
   * - ``bc.hi_type``
     - string array, length ``AMREX_SPACEDIM``
     - *(required)*
     - BC name for the high face of each direction, in order ``[x, y, (z)]``.

Both are read with ``getarr`` and asserted to have exactly ``AMREX_SPACEDIM``
entries (``InputParser::bcNames``, which also reorders them internally into
``[xlo, xhi, ylo, yhi, zlo, zhi]`` to match
``muphfasa::faceIndex(dir, bound)``).

Accepted BC names are exactly the four enumerators of ``lbmBCtype``
(``src/LBM/BoundaryConditions.H``), as parsed by ``parselbmBCtype()``
(``src/LBM/BoundaryConditions.cpp``), plus the special name ``"Periodic"``
which ``buildBCManager()`` intercepts before ever calling
``parselbmBCtype()``:

- ``Periodic`` — handled by AMReX's own ``FillBoundary``/periodicity
  machinery, not by a custom kernel. Requires the corresponding
  ``amrex::Geometry`` direction to actually be periodic (see the
  periodicity-derivation note under ``geometry.*`` above) or the run aborts.
- ``NoSlip`` — stationary solid wall (bounce-back-style kernel). Takes no
  velocity/density parameters.
- ``MovingWall`` — solid wall translating tangentially at a prescribed
  velocity.
- ``Inflow`` — prescribed-velocity inflow.
- ``Outflow`` — convective/extrapolation-style outflow. Takes no
  velocity/density parameters.

Any other name aborts in ``parselbmBCtype`` with: ``parselbmBCtype:
unrecognized BC name "<name>"``.

Per-face velocity keys
~~~~~~~~~~~~~~~~~~~~~~~

``MovingWall`` and ``Inflow`` faces additionally take a velocity, given in
**physical units** [m/s] (it is converted to lattice units internally via
``UnitConverter::toLatticeVelocity`` before being handed to the boundary
kernel). The key names follow the pattern ``bc.<face>_<kind>``, where
``<face>`` is one of ``lox``, ``hix``, ``loy``, ``hiy``, ``loz``, ``hiz``
(low/high face of the x/y/z direction — ``loz``/``hiz`` only apply in 3-D
builds):

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - BC kind
     - Velocity keys
     - Component restriction
   * - ``MovingWall``
     - ``bc.lox_Uwall``, ``bc.hix_Uwall``, ``bc.loy_Uwall``, ``bc.hiy_Uwall``,
       ``bc.loz_Uwall``, ``bc.hiz_Uwall``
     - The component **normal** to the face must be exactly ``0``. A
       nonzero normal component aborts with: ``InputParser: <key> has a
       nonzero component along the wall normal (component <d>), which
       implies an expanding/contracting wall. Only wall velocities
       tangential to the face are supported for Moving walls.``
   * - ``Inflow``
     - ``bc.lox_Uin``, ``bc.hix_Uin``, ``bc.loy_Uin``, ``bc.hiy_Uin``,
       ``bc.loz_Uin``, ``bc.hiz_Uin``
     - All components **tangential** to the face must be exactly ``0``. A
       nonzero tangential component aborts with: ``InputParser: <key> has a
       nonzero component tangent to the wall (component <d>), which implies
       a moving Inflow/outflow. Only velocities normal to the face are
       supported for Inflows.``

Each key, if given, must list exactly ``AMREX_SPACEDIM`` components (e.g.
``bc.loy_Uwall = 0.0 0.0`` in 2-D); if the key is absent for a face, that
face's velocity defaults to zero. Example (a lid-driven cavity top wall
moving in +x at 1 m/s, 2-D)::

    bc.hiy_Uwall = 1.0  0.0

``Inflow`` and ``MovingWall`` faces also use ``units.ref_density`` as their
prescribed wall/inflow density (``m_units->toLatticeDensity(params.refDensity())``
in ``SimulationSetup::buildBCManager()`` — there is no separate
``bc.*_rho`` input key for this).

``amr.*``
---------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Key
     - Type
     - Default
     - Meaning
   * - ``amr.plot_int``
     - int
     - ``100``
     - Write a plotfile every N steps (via AMReX's plotfile format).
   * - ``amr.chk_int``
     - int
     - ``1000``
     - Write a checkpoint file every N steps.
   * - ``amr.plot_file``
     - string
     - ``plt``
     - Plotfile directory-name prefix (AMReX appends the step number).
   * - ``amr.check_file``
     - string
     - ``chk``
     - Checkpoint directory-name prefix (AMReX appends the step number).
