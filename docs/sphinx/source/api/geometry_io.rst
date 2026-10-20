.. _api-geometry-io:

Geometry and I/O
=================

This chapter documents three small, self-contained modules: ``GeometryHandler``
(src/Geometry/GeometryHandler.{H,cpp}), which builds the AMReX ``Geometry``,
``BoxArray``, and ``DistributionMapping`` for a run, and ``Checkpoint`` /
``Plotfile`` (src/IO/Checkpoint.{H,cpp}, src/IO/Plotfile.{H,cpp}), which write
(and, for checkpoints, read) solver state to disk.

GeometryHandler
-----------------

``GeometryHandler`` wraps the construction of ``amrex::Geometry``, the
domain's ``amrex::BoxArray``, and its ``amrex::DistributionMapping`` from an
``InputParser``, and exposes a stubbed-out hook for embedded-boundary (EB)
setup.

Constructing the Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~

The constructor (``GeometryHandler::GeometryHandler``) builds the domain
``amrex::Box`` from ``grid.ncells`` (``params.ncells()``), the physical
``amrex::RealBox`` from ``geometry.prob_lo`` / ``geometry.prob_hi``
(``params.geometryProbLo()`` / ``params.geometryProbHi()``), and then
determines periodicity **per direction**, independently, by checking whether
the boundary-condition type configured on the low face of that direction is
the string ``"Periodic"``. The exact loop, quoted from
``src/Geometry/GeometryHandler.cpp``:

.. code-block:: cpp

    // check for periodicity
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{};
    auto const bc_names = params.bcNames();
    for (int dim=0; dim < AMREX_SPACEDIM; ++dim){
        if (bc_names[2*dim] == "Periodic"){
            is_periodic[dim] = 1;
        } else {
            is_periodic[dim] = 0;
        }

    }

Note that only the *low* face's name (``bc_names[2*dim]``, i.e. index
``2*dim``, not ``2*dim+1``) is consulted to decide periodicity for direction
``dim``; the high face's type is checked elsewhere (see
``SimulationImpl::buildBCManager`` in :doc:`simulation`, which asserts that
the ``Geometry`` really is periodic in a direction before skipping any face
named ``"Periodic"``).

The resulting ``amrex::Geometry`` is built with:

.. code-block:: cpp

    m_geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

``buildBoxArray(max_grid_size)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``buildBoxArray`` creates the ``BoxArray`` from the domain's single box and
chops it into tiles no larger than ``max_grid_size`` cells on a side via
``amrex::BoxArray::maxSize()``, then builds a matching
``amrex::DistributionMapping``:

.. code-block:: cpp

    void GeometryHandler::buildBoxArray(int max_grid_size)
    {
        m_ba = amrex::BoxArray(m_geom.Domain());
        m_ba.maxSize(max_grid_size);
        m_dm = amrex::DistributionMapping(m_ba);
    }

``initEB(geom_type)`` — currently a stub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   ``initEB`` is **not implemented**. Its entire body
   (``src/Geometry/GeometryHandler.cpp``, lines 51-55) is:

   .. code-block:: cpp

       void GeometryHandler::initEB(const std::string& /*geom_type*/)
       {
           // TODO: call AMReX EB2::Build with the requested implicit-function geometry
           m_has_eb = true;
       }

   The ``geom_type`` parameter is not even named (it is commented out as
   ``/*geom_type*/``), so no implicit-function geometry is ever built, no
   call is made to AMReX's ``EB2::Build``, and no EB factory is attached to
   the grid. The function only flips ``m_has_eb`` to ``true`` and returns.

Cross-referencing :doc:`simulation`, ``SimulationImpl``'s constructor calls
``initEB()`` only when the input file requests it:

.. code-block:: cpp

    if (params.useEB()) {
        m_geom_handler->initEB(params.geometryType());
    }

``params.useEB()`` reflects the ``geometry.use_eb`` input key. Because
``initEB`` does no real work, **setting** ``geometry.use_eb = 1`` **today has
no physical effect** beyond recording ``hasEB() == true`` on the handler — no
embedded boundary is actually carved into the domain, and no EB-aware
factory or kernels are engaged.

``physDomain()``
~~~~~~~~~~~~~~~~~

Returns the ``amrex::RealBox`` physical extent of the domain
(``m_geom.ProbDomain()``) — a thin convenience accessor for callers that want
the physical bounds without going through ``geometry()``.

Checkpoint
-----------

``src/IO/Checkpoint.{H,cpp}`` implements checkpoint write/read for restarting
a run. Only the distribution function ``f`` is saved — density and velocity
are not, since both are derivable from ``f`` via the macroscopic-moment
calculation (see :doc:`core_kernels`) and so would be redundant to store.

Header format
~~~~~~~~~~~~~~

``writeCheckpoint`` creates the checkpoint directory
(``amrex::UtilCreateCleanDirectory``) and, on the I/O processor, writes a
plain-text ``Header`` file with a version string, the step, the time, the
domain box, the number of components, and the ghost-cell vector, in that
order:

.. code-block:: cpp

    std::ofstream hdr(dir_name + "/Header");
    hdr << kCheckpointVersion << "\n";
    hdr << step << "\n";
    hdr << std::setprecision(17) << time << "\n";
    hdr << geom.Domain() << "\n";
    hdr << f_mf.nComp() << "\n";
    hdr << f_mf.nGrowVect() << "\n";

The version string is ``"muphfasa_checkpoint_v1"``
(``constexpr const char* kCheckpointVersion``). After the ``Header`` is
written and a ``ParallelDescriptor::Barrier()`` synchronizes all ranks, the
actual field data is written with ``amrex::VisMF::Write(f_mf, dir_name +
"/f")`` — the standard AMReX distributed-MultiFab-to-disk format.

``readCheckpoint`` is the inverse: it checks ``checkpointExists()``, reads
back the version string (aborting if it doesn't match
``kCheckpointVersion``), reads ``step`` and ``time``, and loads the field
data with ``amrex::VisMF::Read(f_mf, dir_name + "/f")``. Per the header
comment in ``Checkpoint.H``, the ``BoxArray``/``DistributionMapping`` already
present in the caller's ``f_mf`` are used as-is — it is the caller's
responsibility to make sure they match the checkpoint's grid before calling
``readCheckpoint``.

``checkpointExists(dir_name)`` simply tests whether ``dir_name + "/Header"``
can be opened.

Naming: ``checkpointName()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``checkpointName(prefix, step)`` zero-pads the step to **5 digits**, e.g.
``checkpointName("chk", 100)`` → ``"chk00100"``:

.. code-block:: cpp

    std::string checkpointName(const std::string& prefix, int step)
    {
        std::ostringstream ss;
        ss << prefix << std::setw(5) << std::setfill('0') << step;
        return ss.str();
    }

.. warning::

   **Restart is not wired up.** ``readCheckpoint`` exists and works, but
   nothing in the solver currently calls it. ``SimulationImpl::initialize()``
   (``src/Simulation/SimulationSetup.cpp``) contains this comment and always
   starts from a fresh, quiescent initial condition:

   .. code-block:: cpp

       // Restart support needs a restart-file key on InputParser, which doesn't
       // exist yet; always start fresh for now.
       setInitialConditions();
       fillGhostCells();

   Meanwhile, checkpoints **are** written periodically: ``writeOutput(step)``
   in the same file calls ``writeCheckpoint`` whenever the step count hits
   the ``amr.chk_int`` interval:

   .. code-block:: cpp

       if (step % m_params.checkpointInterval() == 0) {
           const std::string cname = checkpointName(m_params.checkpointPrefix(), step);
           writeCheckpoint(cname, m_f, m_geom, step, m_time);
       }

   So the write side of checkpointing is fully functional and exercised on
   every run with ``amr.chk_int`` set, but there is currently no way to feed
   a checkpoint back in to resume a run — every invocation of the solver
   starts from scratch. Adding restart support requires a new
   ``InputParser`` key (e.g. a ``amr.restart`` file path) and a call to
   ``readCheckpoint`` from ``SimulationImpl::initialize()``.

Plotfile
---------

``src/IO/Plotfile.{H,cpp}`` writes macroscopic fields in AMReX's native
plotfile format (readable by VisIt and ParaView), via
``amrex::WriteSingleLevelPlotfile``. Per the header comment in
``Plotfile.H``, **distribution functions are not written to plotfiles** —
only the checkpoint format (above) stores ``f`` directly.

``writePlotfile`` (generic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``writePlotfile`` takes an arbitrary list of named ``MultiFab``\ s plus a
flat list of variable names (one name per component, across all of the
supplied MultiFabs), asserts that the total component count matches
``var_names.size()``:

.. code-block:: cpp

    int ncomp_total = 0;
    for (const auto* mf : mfs) { ncomp_total += mf->nComp(); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ncomp_total == static_cast<int>(var_names.size()),
        "writePlotfile: var_names size does not match total component count");

and then concatenates all of the input MultiFabs' components into a single
scratch ``plot_mf`` via repeated ``amrex::MultiFab::Copy`` calls (all inputs
must share the same ``BoxArray``/``DistributionMapping``), before handing
``plot_mf`` to ``amrex::WriteSingleLevelPlotfile``.

``writeMacroPlotfile`` (convenience wrapper)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``writeMacroPlotfile`` is the wrapper actually used by
``SimulationImpl::writeOutput`` (see :doc:`simulation`). It writes exactly
density plus velocity components — ``"density"``, ``"vx"``, ``"vy"`` in 2D,
or ``"density"``, ``"vx"``, ``"vy"``, ``"vz"`` in 3D — from the solver's
``m_rho`` and ``m_vel`` MultiFabs:

.. code-block:: cpp

    amrex::Vector<std::string> names = {"density"};
    #if   (AMREX_SPACEDIM == 2)
        names.push_back("vx"); names.push_back("vy");
    #elif (AMREX_SPACEDIM == 3)
        names.push_back("vx"); names.push_back("vy"); names.push_back("vz");
    #endif

Again, note that the distribution function ``f`` is explicitly **not**
among the written fields — only checkpoints carry ``f``.

Naming: ``plotfileName()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``plotfileName(prefix, step)`` zero-pads the step to **7 digits**, e.g.
``plotfileName("plt", 100)`` → ``"plt0000100"``:

.. code-block:: cpp

    std::string plotfileName(const std::string& prefix, int step)
    {
        std::ostringstream ss;
        ss << prefix << std::setw(7) << std::setfill('0') << step;
        return ss.str();
    }

.. note::

   This is a real, if minor, inconsistency worth knowing about:
   ``checkpointName`` zero-pads to **5** digits (``"chk00100"``) while
   ``plotfileName`` zero-pads to **7** digits (``"plt0000100"``). The two
   naming schemes are otherwise structurally identical (prefix + zero-padded
   step), so the differing digit width is easy to overlook when matching up
   checkpoint and plotfile directories for the same step by eye, or when
   writing tooling that globs for both.
