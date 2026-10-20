.. _api-simulation:

Simulation Orchestration
==========================

This chapter documents the top-level control flow of the solver: ``main.cpp``
(the entry point), ``ISimulation`` (the non-templated interface ``main.cpp``
drives), ``SimulationFactory`` (the runtime-to-compile-time dispatch), and
``SimulationImpl<LatticeTag, CollisionModel>`` (``SimulationSetup.H/.cpp``,
the class that actually owns all persistent AMReX state and implements the
time-step loop).

main.cpp — the whole program
-------------------------------

The entire executable is about 15 lines. It initializes AMReX, builds an
``InputParser``, asks ``makeSimulation`` for a concrete simulation object,
initializes it, and then runs a simple step loop calling ``advance`` and
``writeOutput`` up to ``max_step``:

.. code-block:: cpp

    int main(int argc, char* argv[])
    {
        amrex::Initialize(argc, argv);
        {
            muphfasa::InputParser params;
            std::unique_ptr<muphfasa::ISimulation> sim = muphfasa::makeSimulation(params);

            sim->initialize();

            const int max_step = params.maxStep();
            for (int step = 0; step <= max_step; ++step) {
                sim->advance(step);
                sim->writeOutput(step);
            }
        }
        amrex::Finalize();
        return 0;
    }

This is the best "big picture" entry point for a reader trying to understand
how the pieces fit together: ``main.cpp`` never mentions a ``LatticeTag`` or a
``CollisionModel`` by name — it only ever talks to the abstract
``ISimulation`` interface, with the concrete choice of lattice and collision
model resolved once, inside ``makeSimulation``.

ISimulation — the non-templated interface
--------------------------------------------

``src/Simulation/ISimulation.H`` declares the abstract interface that lets
``main.cpp`` (and any other caller) drive the solver without knowing which
``LatticeTag`` was selected at runtime. It declares the lifecycle methods:

- ``initialize()`` — allocate state and set initial conditions.
- ``advance(int step)`` — perform one LBM time step.
- ``writeOutput(int step) const`` — write plotfile/checkpoint output if due.

and read-only accessors to the solver's current state:

- ``f()``, ``rho()``, ``vel()`` — the distribution function and macroscopic
  fields, as ``const amrex::MultiFab&``.
- ``time()`` — current simulation time (lattice units).
- ``omega()`` — the BGK relaxation parameter actually in use
  (``omega = 1/tau``), derived from ``UnitConverter::relaxationTime()``.
- ``units()`` — the ``UnitConverter`` used for this run.
- ``geom()`` — the ``amrex::Geometry`` of the run.

Every one of these is implemented by ``SimulationImpl<LatticeTag,
CollisionModel>``, templated on the lattice and collision model, but exposed
here with no template parameters at all — this is what allows a single,
ordinary ``std::unique_ptr<ISimulation>`` in ``main.cpp`` to hold whichever
concrete instantiation was selected at runtime.

SimulationFactory — runtime-to-compile-time dispatch
--------------------------------------------------------

``src/Simulation/SimulationFactory.cpp`` is the single place in the codebase
where a ``SimulationImpl<LatticeTag, CollisionModel>`` actually gets
instantiated. The dispatch happens in two stages.

**Stage 1 — lattice selection** (``makeSimulation``): the ``lbm.lattice_type``
input string picks the ``LatticeTag``, gated by the build's
``AMREX_SPACEDIM``. A 2D build only accepts ``"D2Q9"``; a 3D build only
accepts ``"D3Q15"``, ``"D3Q19"``, or ``"D3Q27"``. Any other string aborts
with an exact, dimension-specific message:

.. code-block:: cpp

    #if (AMREX_SPACEDIM == 2)
        if (lattice == "D2Q9") {
            return makeSimulationForLattice<D2Q9>(params);
        }
        amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                     "' for a 2D build (only D2Q9 is available).");
    #else
        if (lattice == "D3Q15") {
            return makeSimulationForLattice<D3Q15>(params);
        }
        if (lattice == "D3Q19") {
            return makeSimulationForLattice<D3Q19>(params);
        }
        if (lattice == "D3Q27") {
            return makeSimulationForLattice<D3Q27>(params);
        }
        amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                     "' for a 3D build (expected D3Q15 | D3Q19 | D3Q27).");
    #endif

**Stage 2 — collision model selection** (``makeSimulationForLattice<LatticeTag>``):
with ``LatticeTag`` now fixed as a template parameter, the ``lbm.collision_model``
input string picks the ``CollisionModel`` tag. Only ``"BGK"`` is actually
implemented. ``"TRT"`` and ``"MRT"`` are recognized as valid tag names but
deliberately abort, since their ``collide()``/``collideArr()`` kernels don't
exist yet (see :doc:`core_kernels`):

.. code-block:: cpp

    template <typename LatticeTag>
    static std::unique_ptr<ISimulation> makeSimulationForLattice(const InputParser& params)
    {
        const std::string collision = params.collisionModel();

        if (collision == "BGK") {
            return std::make_unique<SimulationImpl<LatticeTag, BGK>>(params);
        }
        if (collision == "TRT" || collision == "MRT") {
            amrex::Abort("lbm.collision_model '" + collision +
                         "' is a recognised tag but not yet implemented.");
        }
        amrex::Abort("Unknown lbm.collision_model '" + collision +
                     "' (expected BGK | TRT | MRT).");
        return nullptr; // unreachable: amrex::Abort does not return
    }

The call ``std::make_unique<SimulationImpl<LatticeTag, BGK>>(params)`` is the
only place in the entire codebase where ``SimulationImpl`` is instantiated
with a live set of template arguments picked from runtime strings — every
other template parameter combination only exists if it appears in the
explicit-instantiation block at the bottom of ``SimulationSetup.cpp`` (see
below).

SimulationImpl<LatticeTag, CollisionModel>
---------------------------------------------

``SimulationImpl`` (``src/Simulation/SimulationSetup.H`` /
``SimulationSetup.cpp``) is the top-level owner of all persistent AMReX
state for a run: the ``Geometry``/``BoxArray``/``DistributionMapping``, the
distribution-function and macroscopic-field ``MultiFab``\ s, the unit
converter, and the boundary-condition machinery.

Constructor
~~~~~~~~~~~~

The constructor does five things, in order:

1. **Builds the** ``UnitConverter`` from ``InputParser`` quantities —
   ``params.refLength()``, ``params.refVelocity()``,
   ``params.reynoldsNumber()``, ``params.refDensity()`` — plus the largest
   grid dimension (computed in-constructor as ``N_ref``/``max_N``),
   ``LatticeTag::cs2``, and the optional ``lbm.u_lat`` /
   ``lbm.tau`` overrides (``params.latticeVelocity()`` /
   ``params.relaxationTime()``). ``UnitConverter`` validates mutual
   consistency between these if both are given, and falls back to a default
   ``u_lat`` if neither is.
2. **Computes** ``m_omega = 1 / tau`` from
   ``m_units->relaxationTime()``.
3. **Builds the** ``GeometryHandler`` (:doc:`geometry_io`), optionally
   calling ``initEB()`` if ``params.useEB()`` is true (see the warning in
   :doc:`geometry_io` about ``initEB`` being a stub), then
   ``buildBoxArray(params.maxGridSize())``, and copies out the resulting
   ``Geometry``/``BoxArray``/``DistributionMapping``.
4. **Allocates the four MultiFabs**: ``m_f`` with ``LatticeTag::Q``
   components and ``LatticeTag::nGhost`` ghost cells (the distribution
   function); ``m_feq``, an equilibrium scratch MultiFab with the same
   component count but **0** ghost cells; ``m_rho``, 1 component, 0 ghost
   cells; and ``m_vel``, ``AMREX_SPACEDIM`` components, 0 ghost cells.
5. **Calls** ``buildBCManager()`` to set up the physical boundary-condition
   functor.

``initialize()``
~~~~~~~~~~~~~~~~~~

Sets quiescent initial conditions — ``rho = 1``, ``u = 0`` everywhere, then
``f = f^eq(rho, u)`` via ``computeEquilibriumMF`` (:doc:`core_kernels`) — and
fills ghost cells once so the first call to ``advance()`` sees a consistent
state:

.. code-block:: cpp

    void SimulationImpl<LatticeTag, CollisionModel>::initialize()
    {
        // Restart support needs a restart-file key on InputParser, which doesn't
        // exist yet; always start fresh for now.
        setInitialConditions();
        fillGhostCells();
    }

As detailed in :doc:`geometry_io`, restart from a checkpoint is **not**
wired up here yet — every run starts from this fresh, quiescent state
regardless of any checkpoints that may exist on disk from a previous run.

``advance(step)``
~~~~~~~~~~~~~~~~~~~

This is the authoritative per-timestep orchestration, quoted in full. A
parallel chapter, :doc:`../theory`, covers the physics of each stage; this
is purely the code-level ordering:

.. code-block:: cpp

    void SimulationImpl<LatticeTag, CollisionModel>::advance(int step)
    {
        amrex::ignore_unused(step);

        // collide
        computeEquilibriumMF<LatticeTag>(m_rho, m_vel, m_feq);
        collideMF<LatticeTag, CollisionModel>(m_f, m_feq, m_omega);

        // fill ghost and stream
        fillGhostCells();
        streamInPlace<LatticeTag>(m_f, m_geom);

        // compute macro
        computeMacroscopic<LatticeTag>(m_f, m_rho, m_vel);

        m_time += amrex::Real(1.0);
    }

The four stages, in order, are: (1) compute equilibrium ``m_feq`` from the
*current* ``m_rho``/``m_vel``; (2) collide ``m_f`` toward ``m_feq`` using
``m_omega``; (3) fill ghost cells (periodic + physical BCs) and then stream
``m_f`` in place; (4) recompute ``m_rho``/``m_vel`` from the streamed
``m_f``. ``step`` itself is unused in the body (only
``amrex::ignore_unused(step)``) — time tracking is purely via the
accumulated ``m_time += 1.0``.

``writeOutput(step)``
~~~~~~~~~~~~~~~~~~~~~~~

Two independent, unrelated modulo-interval checks, one for plotfiles and one
for checkpoints — a step can trigger either, both, or neither:

.. code-block:: cpp

    void SimulationImpl<LatticeTag, CollisionModel>::writeOutput(int step) const
    {
        if (step % m_params.plotInterval() == 0) {
            const std::string fname = plotfileName(m_params.plotfilePrefix(), step);
            writeMacroPlotfile(fname, m_rho, m_vel, m_geom, m_time, step);
        }

        if (step % m_params.checkpointInterval() == 0) {
            const std::string cname = checkpointName(m_params.checkpointPrefix(), step);
            writeCheckpoint(cname, m_f, m_geom, step, m_time);
        }
    }

``plotInterval()``/``plotfilePrefix()`` and ``checkpointInterval()``/
``checkpointPrefix()`` come from the ``amr.plot_int`` and ``amr.chk_int``
input keys respectively; see :doc:`geometry_io` for the plotfile/checkpoint
write formats themselves and the digit-padding difference between
``plotfileName`` and ``checkpointName``.

``buildBCManager()``
~~~~~~~~~~~~~~~~~~~~~~

Builds the ``BCManager<LatticeTag>`` and wraps it in the
``amrex::PhysBCFunct`` that ``fillGhostCells()`` invokes each step. See
:doc:`boundary_conditions` for the BC kernel details themselves; this
section covers only the setup-time wiring.

First, wall and inflow velocities and the reference density are converted
from physical to lattice units, via ``m_units->toLatticeVelocity`` /
``m_units->toLatticeDensity``, *before* being handed to the ``BCManager``:

.. code-block:: cpp

    const auto rhoWall = m_units->toLatticeDensity(m_params.refDensity());
    auto Uwall = m_params.Uwall();
    auto Uin = m_params.Uinflow();
    for (int wall_idx=0; wall_idx < 2 * AMREX_SPACEDIM; ++wall_idx) {
        for (int dim=0; dim < AMREX_SPACEDIM; ++dim) {
            Uwall[wall_idx][dim] = m_units->toLatticeVelocity(Uwall[wall_idx][dim]);
            Uin[wall_idx][dim]   = m_units->toLatticeVelocity(Uin[wall_idx][dim]);
        }
    }

Then, the function walks every face of the domain (every ``dim`` crossed
with ``bound`` in ``{-1, 1}``). A face configured as ``"Periodic"`` is
skipped — but only after asserting that the ``Geometry`` really is periodic
in that direction, since ``FillBoundary`` (not the ``BCManager``) is what
actually handles periodic faces:

.. code-block:: cpp

    if (name == "Periodic") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_geom.isPeriodic(dir),
            "buildBCManager: bc input requests \"Periodic\" on a "
            "face whose Geometry is not periodic in that direction");
        continue; // no lbmBCtype to set; FillBoundary handles it
    }

Non-periodic faces get a BC type (``bc_manager.setFaceBC``), and
``"MovingWall"``/``"Inflow"`` faces additionally get the (now
lattice-unit) wall/inflow density and velocity set on that face via
``setFaceDensity``/``setFaceVel``.

Finally, the configured ``BCManager`` is wrapped in an
``amrex::GpuBndryFuncFab<BCManager<LatticeTag>>`` and that, in turn, in the
``amrex::PhysBCFunct`` stored as ``m_phys_bc``:

.. code-block:: cpp

    amrex::Vector<amrex::BCRec> bcs = makeBCRecVector(names, LatticeTag::Q);

    amrex::GpuBndryFuncFab<BCManager<LatticeTag>> gpu_bndry_func(bc_manager);
    m_phys_bc = std::make_unique<PhysBCFunctType>(m_geom, bcs, gpu_bndry_func);

Explicit instantiation
~~~~~~~~~~~~~~~~~~~~~~~~

Because ``SimulationImpl`` is a class template, it must be explicitly
instantiated somewhere so its member functions are actually compiled and
linkable. The block at the bottom of ``SimulationSetup.cpp`` does this,
dimension-matched only (mirroring the factory's own gating), and only ever
paired with ``BGK`` — since, as noted above, ``TRT``/``MRT`` have no working
``collide()``:

.. code-block:: cpp

    #if (AMREX_SPACEDIM == 2)
    template class SimulationImpl<D2Q9, BGK>;
    #else
    template class SimulationImpl<D3Q15, BGK>;
    template class SimulationImpl<D3Q19, BGK>;
    template class SimulationImpl<D3Q27, BGK>;
    #endif

Adding a new collision model requires implementing it in ``Collision.cpp``,
adding its instantiation line(s) here, and wiring it into
``SimulationFactory.cpp``'s collision-model dispatch (see above).
