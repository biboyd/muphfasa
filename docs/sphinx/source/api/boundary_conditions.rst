.. _boundary_conditions:

Boundary Conditions
====================

MuPhFASa's boundary conditions are implemented as a ghost-cell-fill functor,
``BCManager<LatticeTag>``, plugged into AMReX's generic boundary-filling
machinery. This page documents the fill contract, the dispatch mechanism
that routes a face to one of the four boundary kernels, and each kernel's
exact ghost-fill formula.

The relevant sources are ``src/LBM/BoundaryConditions.H``,
``src/LBM/BoundaryConditions.cpp``, and the kernel headers under
``src/LBM/BCTypes/``: ``BounceBack.H``, ``MovingWallKernel.H``,
``NoSlipKernel.H``, ``InflowKernel.H``, and ``OutflowKernel.H``. Input-file
wiring lives in ``src/IO/InputParser.H``/``.cpp`` and
``src/Simulation/SimulationSetup.cpp``.

Design and contract
--------------------

The ghost-fill contract
~~~~~~~~~~~~~~~~~~~~~~~~

``BCManager<LatticeTag>::operator()`` is only ever invoked by AMReX, via
``amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCManager<LatticeTag>>>``, for
ghost cells that lie outside the valid box. The header comment at the top of
``BoundaryConditions.H`` states the contract precisely:

    Contract: this functor is only ever invoked by AMReX for ghost cells
    (``iv`` outside the valid box). Its job is to fill that ghost cell with
    whatever value makes the next ``stream()`` produce the right boundary
    physics at the adjacent fluid node — it never corrects a valid interior
    cell directly.

In other words, every kernel below writes only to the ghost cell ``iv``
passed in; it never touches the valid interior, and the boundary physics
only materializes once the subsequent ``stream()`` call pulls the filled
ghost value into the adjacent fluid cell.

The custom dispatch mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MuPhFASa keeps its own boundary-kind tag entirely separate from
``amrex::BCType``:

.. code-block:: cpp

    // Our own dispatch tag — never amrex::BCType. Adding a BC means adding an
    // enumerator here, nowhere else in amrex's namespace.
    enum class lbmBCtype { MovingWall, NoSlip, Inflow, Outflow };

The ``amrex::BCRec`` objects actually handed to ``PhysBCFunct`` carry only
one of two possible codes, never the real boundary kind:

.. code-block:: cpp

    // The only amrex::BCType code our custom BCs ever use. Chosen from the
    // `user_*` range AMReX reserves for exactly this purpose; never ext_dir/
    // foextrap/hoextrap/hoextrapcc/reflect_even/reflect_odd — those drive
    // AMReX's own generic fill/interpolation machinery, which we don't want
    // touching this field.
    inline constexpr int kCustomBCSentinel = amrex::BCType::user_1;
    static_assert(kCustomBCSentinel != amrex::BCType::int_dir &&
                  kCustomBCSentinel != amrex::BCType::ext_dir &&
                  kCustomBCSentinel != amrex::BCType::foextrap &&
                  kCustomBCSentinel != amrex::BCType::hoextrap &&
                  kCustomBCSentinel != amrex::BCType::hoextrapcc &&
                  kCustomBCSentinel != amrex::BCType::reflect_even &&
                  kCustomBCSentinel != amrex::BCType::reflect_odd,
                  "custom BC sentinel must not collide with a semantically-defined amrex::BCType");

``kCustomBCSentinel`` is ``amrex::BCType::user_1``, a code from the range
AMReX explicitly reserves for user-defined boundary kinds that carry no
built-in meaning. The ``static_assert`` exists precisely so this sentinel
can never accidentally collide with one of AMReX's semantically-meaningful
codes (``ext_dir``, ``foextrap``, ``hoextrap``, ``hoextrapcc``,
``reflect_even``, ``reflect_odd``) — those codes drive AMReX's own generic
fill/interpolation machinery (e.g. Dirichlet extrapolation, reflection
across the boundary), which must never run on the distribution-function
field, since all of that physics is instead implemented by hand in the LBM
kernels below.

The one ``amrex::BCType`` code that *is* reused for real is ``int_dir``
(interior/periodic). For a periodic face, AMReX's own
``MultiFab::FillBoundary(geom.periodicity())`` has already filled the ghost
cell correctly before ``BCManager::operator()`` ever runs; the functor just
detects ``amrex::BCType::int_dir`` and skips that face:

.. code-block:: cpp

    if (amrex_code == amrex::BCType::int_dir) {
        continue; // periodic — FillBoundary already handled it
    }

Face indexing convention
~~~~~~~~~~~~~~~~~~~~~~~~~

Every per-face table in the boundary-condition code — ``BCManager``'s
internal storage, ``InputParser::bcNames()``/``Uwall()``/``Uinflow()``, and
``makeBCRecVector()`` — uses the same canonical ordering, computed by a
single free function:

.. code-block:: cpp

    inline constexpr int faceIndex(int dir, int bound) noexcept {
        return 2 * dir + (bound < 0 ? 0 : 1);
    }

With ``dir`` in ``{0=x, 1=y, 2=z}`` and ``bound`` in ``{-1 (lo), +1 (hi)}``,
this yields the fixed layout ``[xlo, xhi, ylo, yhi, zlo, zhi]``.

BCManager configuration and ``buildBCManager()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BCManager<LatticeTag>`` exposes three setters used to populate its
per-face tables once at setup:

.. code-block:: cpp

    void setFaceBC(int dir, int bound, lbmBCtype kind);
    void setFaceDensity(int dir, int bound, amrex::Real rho);
    void setFaceVel(int dir, int bound, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uwall);

As the header notes, the density/velocity tables are reused for two
purposes: for a ``MovingWall`` face they hold the wall density/velocity;
for an ``Inflow`` face they hold ``rho_in``/``Uin`` instead (``NoSlip`` and
``Outflow`` need no stored density or velocity at all).

``SimulationImpl<LatticeTag, CollisionModel>::buildBCManager()`` (in
``src/Simulation/SimulationSetup.cpp``) is the only call site that wires the
input-file configuration (via ``InputParser``) into a live ``BCManager``.
It reads the per-face boundary-kind names and the wall/inflow velocities,
converts the wall/inflow velocity and the reference density into lattice
units through ``UnitConverter``, and populates the manager face by face:

.. code-block:: cpp

    const auto names = m_params.bcNames();

    // wall/inflow params in lattice units
    const auto rhoWall = m_units->toLatticeDensity(m_params.refDensity());
    auto Uwall = m_params.Uwall();
    auto Uin = m_params.Uinflow();
    for (int wall_idx=0; wall_idx < 2 * AMREX_SPACEDIM; ++wall_idx) {
        for (int dim=0; dim < AMREX_SPACEDIM; ++dim) {
            Uwall[wall_idx][dim] = m_units->toLatticeVelocity(Uwall[wall_idx][dim]);
            Uin[wall_idx][dim]   = m_units->toLatticeVelocity(Uin[wall_idx][dim]);
        }
    }
    BCManager<LatticeTag> bc_manager{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int bound : {-1, 1}) {
            const std::string& name = names[faceIndex(dir, bound)];
            if (name == "Periodic") {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_geom.isPeriodic(dir),
                    "buildBCManager: bc input requests \"Periodic\" on a "
                    "face whose Geometry is not periodic in that direction");
                continue; // no lbmBCtype to set; FillBoundary handles it
            }
            bc_manager.setFaceBC(dir, bound, parselbmBCtype(name));
            if (name == "MovingWall") {
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir+1]);
            } else if (name == "Inflow") {
                // reuses the reference density as rho_in; see InputParser::Uinflow()
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir+1]);
            }
        }
    }

Note that ``Inflow`` reuses the simulation's reference density
(``m_params.refDensity()``, converted via ``toLatticeDensity``) as
``rho_in`` rather than exposing a separate inflow-density input parameter.

On the input-parsing side, ``InputParser::bcNames()`` reads ``bc.lo_type``
and ``bc.hi_type`` (one name per spatial direction: ``NoSlip`` | ``Inflow``
| ``Outflow`` | ``Periodic`` | ``MovingWall``) and repacks them into the
``[xlo, xhi, ylo, yhi, zlo, zhi]`` order via ``faceIndex``.
``InputParser::Uwall()`` reads per-face keys ``bc.lox_Uwall``,
``bc.hix_Uwall``, ``bc.loy_Uwall``, ... and aborts if a face's velocity has
a nonzero component along its own face normal (a ``MovingWall`` may only
slide tangentially — no expanding/contracting walls).
``InputParser::Uinflow()`` reads the analogous ``bc.lox_Uin``, ... keys with
no such restriction, since inflow velocity is expected to point into the
domain along the face normal.

Name parsing and the ``BCRec`` vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two free functions, both in ``BoundaryConditions.cpp``, close the loop
between the input-file strings and the two separate tables AMReX/muphfasa
each use:

``parselbmBCtype(const std::string& name)`` maps a boundary-kind string
(``"MovingWall"``, ``"NoSlip"``, ``"Inflow"``, ``"Outflow"``) to the
corresponding ``lbmBCtype`` enumerator, and ``amrex::Abort``s on anything
else — the caller is expected to have already filtered out ``"Periodic"``
faces before calling it (``buildBCManager()`` does this via its ``continue``
above).

``makeBCRecVector(names, ncomp)`` builds the ``amrex::Vector<amrex::BCRec>``
that is actually handed to ``PhysBCFunct``. For each direction it sets the
lo/hi code to ``amrex::BCType::int_dir`` when the corresponding name is
``"Periodic"``, and to ``kCustomBCSentinel`` for every other name
(``MovingWall``, ``NoSlip``, ``Inflow``, ``Outflow`` are all
indistinguishable at this level — the real kind is only recovered inside
``BCManager::operator()`` from its own ``m_face_bc`` table, keyed again by
``faceIndex``):

.. code-block:: cpp

    amrex::Vector<amrex::BCRec> makeBCRecVector(
        const std::array<std::string, 2 * AMREX_SPACEDIM>& names, int ncomp)
    {
        amrex::BCRec bcr;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const std::string& lo_name = names[faceIndex(dir, -1)];
            const std::string& hi_name = names[faceIndex(dir, 1)];

            bcr.setLo(dir, (lo_name == "Periodic") ? amrex::BCType::int_dir
                                                    : kCustomBCSentinel);
            bcr.setHi(dir, (hi_name == "Periodic") ? amrex::BCType::int_dir
                                                    : kCustomBCSentinel);
        }
        return amrex::Vector<amrex::BCRec>(ncomp, bcr);
    }

The resulting vector has one identical ``BCRec`` entry per lattice component
(``ncomp = LatticeTag::Q``), since — as ``BCManager::operator()`` notes in
its own comment — a wall applies to the whole set of populations together,
not per component.

Boundary kernels
------------------

All four boundary kernels share the same geometric setup. The face sits
halfway between the last fluid node ``x`` and the first ghost node
``g = x + bound*e_dir`` (``bound = -1`` for a lo-``dir`` face, ``+1`` for a
hi-``dir`` face); ``iv`` is always such a ghost node ``g``. Every kernel
touches only populations ``n`` whose velocity points *into* the face along
``dir`` — i.e. ``c_n[dir] == -bound`` — since those are exactly the
populations that would otherwise stream out of the domain and are therefore
unknown after streaming.

BounceBack
~~~~~~~~~~~

``src/LBM/BCTypes/BounceBack.H`` implements halfway bounce-back with a
prescribed face density ``rho`` and velocity ``U``, and is the shared
implementation underneath ``NoSlipKernel``, ``MovingWallKernel``, and
``InflowKernel`` (see below for how each wraps it).

For every population ``n`` with ``c_n[dir] == -bound``:

.. math::

   f(g, n) = f(x, \overline{n}) + \text{contrib}(n, \overline{n}, \rho, U)

where :math:`\overline{n}` = ``opposite(n)`` and

.. math::

   \text{contrib}(m, \rho, U) = -2 \, \rho \, w(m) \, \frac{\sum_i c_i(m)\, U_i}{c_s^2}

implemented as:

.. code-block:: cpp

    static amrex::Real contrib(int n, int opp_n, amrex::Real rho,
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& U) {
        amrex::Real cdotu = AMREX_D_TERM(LatticeTag::cx(opp_n) * U[0],
                                        + LatticeTag::cy(opp_n) * U[1],
                                        + LatticeTag::cz(opp_n) * U[2]);
        return -2. * rho * LatticeTag::weights(n) * cdotu * LatticeTag::cs2inv;
    }

The dot product in ``contrib`` runs over every spatial component of
``U``, including the component along ``dir`` — so ``U`` may carry a
component normal to the face (needed by ``InflowKernel``) as well as
tangential components (needed by ``MovingWallKernel``); passing ``U = 0``
reduces the formula to plain halfway bounce-back (``NoSlipKernel``). This
derivation (and the ``contrib`` term) follows Krüger et al., *The Lattice
Boltzmann Method*, ch. 5.

NoSlipKernel
~~~~~~~~~~~~~

``src/LBM/BCTypes/NoSlipKernel.H`` is plain halfway bounce-back — a
stationary, no-slip wall. Its own header derivation states the formula
without a momentum-correction term:

.. math::

   f(g, n) = f(x, \overline{n})

``NoSlipKernel`` is implemented as ``MovingWallKernel`` called with a dummy
density of 1 and zero velocity:

.. code-block:: cpp

    amrex::Real dummy_rho=1;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dummy_Uwall={AMREX_D_DECL(0., 0., 0.)};

    MovingWallKernel<LatticeTag>::apply(iv, f, dir, bound, dummy_rho, dummy_Uwall);

Since ``contrib`` is proportional to :math:`U`, passing ``U = 0`` makes the
correction term vanish identically, recovering the plain bounce-back formula
above — ``NoSlipKernel`` is a genuine special case of ``MovingWallKernel``,
not a separately-derived formula.

MovingWallKernel
~~~~~~~~~~~~~~~~~

``src/LBM/BCTypes/MovingWallKernel.H`` is halfway bounce-back with a
prescribed wall velocity, restricted (on the input-parsing side — see
``InputParser::Uwall()`` above) to motion tangential to the wall's own face
normal, as in Couette flow or a lid-driven cavity:

.. math::

   f(g, n) = f(x, \overline{n}) - 2 \, \rho_{\text{wall}} \, w(\overline{n}) \,
             \frac{\sum_i c_i(\overline{n})\, U_{\text{wall},i}}{c_s^2}

``MovingWallKernel::apply`` is a thin wrapper that forwards directly to
``BounceBack<LatticeTag>::apply`` with no additional logic of its own:

.. code-block:: cpp

    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound,
                       amrex::Real rho_wall,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uwall
                       ) noexcept
    {
        BounceBack<LatticeTag>::apply(iv, f, dir, bound, rho_wall, Uwall);
    }

Unlike ``BounceBack.H`` and ``NoSlipKernel.H`` — both of which restate the
halfway-bounce-back derivation in their own header comments — there is no
independent arithmetic implemented here: ``MovingWallKernel.H`` is purely a
named, documentation-bearing wrapper around ``BounceBack.H``. (This is
worth flagging explicitly because the two headers' comments each describe
the same derivation in their own words, which reads, on first pass, as if
two independent implementations existed; in the actual code there is only
one, in ``BounceBack.H``.)

InflowKernel
~~~~~~~~~~~~~

``src/LBM/BCTypes/InflowKernel.H`` reuses the exact same halfway
bounce-back + momentum-correction formula as ``MovingWallKernel``/
``BounceBack``, with a prescribed inflow density ``rho_in`` and velocity
``Uin``:

.. math::

   f(g, n) = f(x, \overline{n}) - 2 \, \rho_{\text{in}} \, w(\overline{n}) \,
             \frac{\sum_i c_i(\overline{n})\, U_{\text{in},i}}{c_s^2}

The difference from ``MovingWallKernel`` is purely at the input-validation
level: ``Uin`` is expected to carry a nonzero component along the face
normal (fluid entering the domain), in addition to any tangential
components, and ``BounceBack``'s momentum correction sums over *all*
``AMREX_SPACEDIM`` components, so that normal component is not dropped.
``InflowKernel::apply`` forwards directly to ``BounceBack<LatticeTag>::apply``:

.. code-block:: cpp

    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound,
                       amrex::Real rho_in,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uin
                       ) noexcept
    {
        BounceBack<LatticeTag>::apply(iv, f, dir, bound, rho_in, Uin);
    }

The header comment is explicit about the limits of this approximation:

    This is a *simplified* bounce-back-based Dirichlet-velocity BC, not full
    Zou-He: it reproduces the prescribed Uin to leading order by reusing the
    same halfway bounce-back + momentum-correction formula as NoSlip/
    MovingWall

OutflowKernel
~~~~~~~~~~~~~~

``src/LBM/BCTypes/OutflowKernel.H`` implements a zero-gradient /
extrapolation outflow condition. Unlike the other three kernels, it performs
**no** reflection through ``opposite()`` and no momentum correction — it
simply copies the same population index from the adjacent fluid cell:

.. math::

   f(g, n) = f(x, n)

.. code-block:: cpp

    for (int n = 0; n < LatticeTag::Q; ++n) {
        if (dir == 0) {
            if (LatticeTag::cx(n) == -bound) {
                f(iv, n) = f(iv[0]-bound, iv[1], iv[2], n);
            }
        } // ... analogous for dir == 1, 2

so that ``stream()``'s pull step reproduces ``f(x, n)`` unchanged at ``x``,
i.e. :math:`\partial f/\partial n = 0` across the face. As with
``InflowKernel``, this is explicitly a simplified condition: a plain
zero-gradient/extrapolation outflow, not a full Zou-He pressure boundary
condition.

Summary of kernel relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Kernel
     - Formula
     - Implementation
   * - ``BounceBack``
     - :math:`f(g,n) = f(x,\overline{n}) + \text{contrib}(\overline{n}, \rho, U)`
     - Shared implementation (``BounceBack.H``)
   * - ``NoSlipKernel``
     - Same, with :math:`U=0,\ \rho=1`
     - Calls ``MovingWallKernel::apply`` with dummy ``rho``/``U``
   * - ``MovingWallKernel``
     - Same as ``BounceBack``, tangential :math:`U` only
     - Forwards directly to ``BounceBack<LatticeTag>::apply``
   * - ``InflowKernel``
     - Same as ``BounceBack``, :math:`U` may have a normal component
     - Forwards directly to ``BounceBack<LatticeTag>::apply``
   * - ``OutflowKernel``
     - :math:`f(g,n) = f(x,n)` (no reflection, no correction)
     - Independent implementation

.. note::

   All four kernels are called from ``BCManager<LatticeTag>::operator()``'s
   switch over ``lbmBCtype``, once per spatial direction that is a ghost
   direction for the cell being filled, using the per-face density/velocity
   stored by ``setFaceDensity``/``setFaceVel``.

Adding a new boundary condition
---------------------------------

Based on how the four existing kernels are wired together, adding a new
boundary condition mechanically requires:

1. Add a new enumerator to ``lbmBCtype`` in ``BoundaryConditions.H``.
2. Implement an ``AMREX_GPU_HOST_DEVICE`` ``apply()`` kernel struct (under
   ``src/LBM/BCTypes/``, following the existing ``static void apply(iv, f,
   dir, bound, ...)`` signature convention), either as a new standalone
   implementation (as ``OutflowKernel`` is) or as a thin wrapper around an
   existing kernel such as ``BounceBack`` (as ``MovingWallKernel``,
   ``InflowKernel``, and ``NoSlipKernel`` are).
3. Add a ``case`` for the new enumerator to the ``switch
   (m_face_bc[faceIndex(dir, bound)])`` in
   ``BCManager<LatticeTag>::operator()``, calling the new kernel's
   ``apply()`` with whatever per-face parameters it needs.
4. Add a string-matching branch to ``parselbmBCtype()`` in
   ``BoundaryConditions.cpp`` so the new BC's input-file name resolves to
   the new ``lbmBCtype`` enumerator.
5. If the new BC needs extra per-face parameters beyond the existing
   density/velocity tables, extend ``InputParser`` with accessors for the
   new ``bc.*`` keys and wire them through
   ``SimulationImpl::buildBCManager()`` in
   ``src/Simulation/SimulationSetup.cpp``, following the same
   per-face/``faceIndex`` pattern used for ``MovingWall``'s and
   ``Inflow``'s density/velocity.
