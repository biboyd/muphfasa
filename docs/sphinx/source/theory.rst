.. _theory:

LBM Theory & Algorithm
========================

This chapter describes the lattice-Boltzmann method (LBM) as it is actually
implemented in MuPhFASa: the discrete velocity set, the macroscopic moments,
the equilibrium distribution, the BGK collision operator, the streaming
step, and the physical-to-lattice unit conversion that ties simulation
parameters to SI inputs. Every formula below is taken directly from the
source in ``src/LBM/`` and cross-checked against the code, not from a
generic textbook derivation.

For a complete treatment of the lattice-Boltzmann method beyond what is
summarized here, see Krüger et al., *The Lattice Boltzmann Method:
Principles and Practice*, Springer, 2017 — cited directly in several
header comments in this codebase (``src/LBM/Units.cpp``,
``src/LBM/BCTypes/BounceBack.H``, ``src/LBM/BCTypes/NoSlipKernel.H``).

The lattice-Boltzmann method at a high level
----------------------------------------------

The lattice-Boltzmann method replaces the continuous Boltzmann transport
equation with a discretized kinetic equation defined on a regular lattice.
Instead of tracking fluid velocity and pressure fields directly (as in a
Navier-Stokes solver), LBM tracks a set of **particle distribution
functions** :math:`f_q(\mathbf{x}, t)`, one for each discrete velocity
direction :math:`q` in a fixed velocity set :math:`\{\mathbf{c}_q\}_{q=0}^{Q-1}`.

MuPhFASa implements the D2Q9 lattice in 2D (9 discrete velocities) and
provides (partially stubbed, see :doc:`api/lattice_descriptor`) D3Q15,
D3Q19, and D3Q27 lattices for 3D. Each discrete velocity :math:`\mathbf{c}_q`
has an associated quadrature weight :math:`w_q` used both to recover the
continuous Maxwell-Boltzmann moments and to construct the discrete
equilibrium distribution.

Every timestep advances the distributions through the two-phase
**collide-stream** cycle:

1. **Collision**: at each lattice site, independently relax each
   :math:`f_q` toward a local equilibrium :math:`f_q^{eq}` computed from
   the local density and velocity.
2. **Streaming**: propagate each post-collision distribution
   :math:`f_q^{*}` to the neighboring lattice site in the direction
   :math:`\mathbf{c}_q`.

Macroscopic density and velocity are recovered at each site as the zeroth
and first velocity moments of the distributions. This recovers the
Navier-Stokes equations in the low-Mach, nearly-incompressible limit via a
Chapman-Enskog expansion (see Krüger et al., Ch. 3).

Macroscopic moments
----------------------

Implemented in ``src/LBM/Macroscopic.H`` / ``Macroscopic.cpp``
(``computeDensityCell``, ``computeVelocityCell2D``/``3D``,
``computeDensity``, ``computeVelocity``, ``computeMacroscopic``).

The density is the zeroth moment of the distribution functions:

.. math::

   \rho(\mathbf{x}) = \sum_{q} f_q(\mathbf{x})

The velocity is the first moment, normalized by density:

.. math::

   u_\alpha(\mathbf{x}) = \frac{1}{\rho(\mathbf{x})} \sum_{q} c_{q,\alpha}\, f_q(\mathbf{x})
   \qquad \alpha = 0, \dots, D-1

where :math:`D` is the spatial dimension (``LatticeTag::D``, 2 or 3) and
:math:`c_{q,\alpha}` is the :math:`\alpha`-component of the discrete
velocity :math:`\mathbf{c}_q` (``LatticeTag::cx(q)``, ``cy(q)``,
``cz(q)``).

In code, ``computeDensityCell<LatticeTag>`` sums over
``LatticeTag::Q`` components of ``f_arr(i,j,k,n)`` for a single cell, and
``computeVelocityCell2D``/``3D`` accumulate :math:`c_{q,\alpha} f_q` for
each component before dividing by :math:`\rho`. The MultiFab-level
``computeMacroscopic`` loops over all tiles/boxes, calling
``computeDensity`` and then ``computeVelocity`` (which needs the just-
computed :math:`\rho`) for every cell in the box.

The equilibrium distribution
--------------------------------

Implemented in ``src/LBM/Equilibrium.H`` / ``Equilibrium.cpp``. The header
comment states the implemented formula exactly:

.. math::

   f_q^{eq}(\rho, \mathbf{u}) = w_q\, \rho \left[
       1 + \frac{\mathbf{c}_q \cdot \mathbf{u}}{c_s^2}
         + \frac{(\mathbf{c}_q \cdot \mathbf{u})^2}{2 c_s^4}
         - \frac{\mathbf{u}^2}{2 c_s^2}
   \right]

This is the standard second-order (Hermite) truncation of the continuous
Maxwell-Boltzmann distribution, evaluated on the discrete velocity set.
The code (``computeEquilibriumCell2D``/``3D`` in ``Equilibrium.H``)
computes this term-for-term:

.. code-block:: cpp

   const amrex::Real u_sq  = u[0]*u[0] + u[1]*u[1];                 // u^2
   const amrex::Real cdotu = LatticeTag::cx(n)*u[0] + LatticeTag::cy(n)*u[1];

   return LatticeTag::weights(n) * rho *
          (1 + LatticeTag::cs2inv * cdotu
             + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
             - 0.5 * LatticeTag::cs2inv * u_sq);

where ``cs2inv = 1/cs2`` (so ``cs2inv * cdotu`` is
:math:`(\mathbf{c}_q\cdot\mathbf{u})/c_s^2`, and
``cs2inv*cs2inv*cdotu*cdotu`` is :math:`(\mathbf{c}_q\cdot\mathbf{u})^2 /
c_s^4`), confirming this matches the header-comment formula exactly.

Each term has a physical role:

- :math:`w_q\,\rho`: the **hydrostatic/pressure** contribution — the
  equilibrium at rest (:math:`\mathbf{u}=0`) recovers the quadrature
  weighting of a uniform density field.
- :math:`w_q\,\rho\,(\mathbf{c}_q\cdot\mathbf{u})/c_s^2`: the linear
  **advection** term, giving the equilibrium a first-order directional
  bias toward the bulk flow direction.
- :math:`w_q\,\rho\,(\mathbf{c}_q\cdot\mathbf{u})^2 / (2 c_s^4)`: a
  **quadratic momentum correction** capturing the directional anisotropy
  of the advected flux (the "convective" nonlinearity of the Euler/NS
  momentum flux tensor).
- :math:`-w_q\,\rho\,\mathbf{u}^2/(2 c_s^2)`: a **kinetic-energy
  correction**, independent of direction :math:`q`, that keeps the zeroth
  moment of :math:`f_q^{eq}` equal to :math:`\rho` regardless of
  :math:`\mathbf{u}`.

This is a second-order expansion in :math:`\mathbf{u}`/the Mach number,
and is therefore only valid in the **low-Mach-number** regime
(:math:`\mathrm{Ma} = u/c_s \ll 1`); see the Units section below for the
concrete bounds MuPhFASa enforces on :math:`u_{lat}` to stay inside the
expansion's validity.

``computeEquilibriumCell`` dispatches to the 2D or 3D variant based on
``AMREX_SPACEDIM``, reading ``u`` out of the velocity MultiFab via an
``Array4``. ``computeEquilibriumMF`` loops over all MFIter tiles and calls
``computeEquilibrium`` (a ``ParallelFor`` over the box and the ``Q``
velocity directions) to fill the full ``feq`` MultiFab from ``rho`` and
``vel``.

The BGK collision operator
------------------------------

Implemented in ``src/LBM/Collision.H`` / ``Collision.cpp``. MuPhFASa
currently implements a single collision model: **BGK**
(Bhatnagar-Gross-Krook) single-relaxation-time collision. Each
distribution relaxes linearly toward its local equilibrium:

.. math::

   f_q^{*} = (1 - \omega)\, f_q + \omega\, f_q^{eq}, \qquad \omega = \frac{1}{\tau}

where :math:`\tau` is the BGK relaxation time (in lattice units). In code
(``BGK::collide``, ``Collision.cpp``):

.. code-block:: cpp

   amrex::Real BGK::collide(const amrex::Real f,
                             const amrex::Real feq,
                             const amrex::Real omega) noexcept
   {
       return (1 - omega) * f + omega * feq;
   }

``BGK::collideArr`` applies this per-cell, per-direction over a box via
``ParallelFor``; ``collideMF`` is the MultiFab-level host-side driver that
loops over MFIter tiles and dispatches to
``CollisionModel::collideArr<LatticeTag>``.

.. note::

   ``Collision.H`` declares two additional collision-model tag types,
   ``struct TRT {};`` and ``struct MRT {};``, intended for two-relaxation-
   time and multiple-relaxation-time collision operators. **Neither has a
   ``collide()``/``collideArr()`` implementation** — they are empty tag
   structs reserved for future extension. ``SimulationSetup.cpp`` only
   explicitly instantiates ``SimulationImpl<LatticeTag, BGK>``, and
   ``SimulationFactory.cpp`` aborts at runtime if ``lbm.collision_model``
   requests ``TRT`` or ``MRT``. This is a known current limitation, not a
   bug: only BGK is a usable collision model today.

An alternate, unused fused path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Collision.H``/``Collision.cpp`` also define a second, **fused**
collision path: ``collideFullArr`` / ``collideFullMF``. Instead of reading
precomputed ``rho``/``vel``/``feq`` MultiFabs, ``collideFullArr`` computes
:math:`\rho`, :math:`\mathbf{u}`, and :math:`f_q^{eq}` from :math:`f`
directly, per cell, as local scratch values (via
``computeDensityCell``, ``computeVelocityCell2D``/``3D``, and
``computeEquilibriumCell2D``/``3D``), then immediately relaxes :math:`f`
into the output MultiFab — all in one ``MFIter``/``ParallelFor`` pass,
without separate ``rho``, ``vel``, or ``feq`` MultiFabs.

This is a legitimate, explicitly-instantiated code path (for both BGK and,
once implemented, future collision models), but **it is not called by the
main simulation loop**. ``SimulationImpl::advance()`` in
``src/Simulation/SimulationSetup.cpp`` uses the non-fused
``computeEquilibriumMF`` + ``collideMF`` pair instead (see the
"Putting it together" section below). Readers of the source should treat
``collideFullArr``/``collideFullMF`` as an alternate, currently-unused
building block worth knowing about — perhaps useful for a future
memory-bandwidth optimization that avoids materializing ``feq`` — rather
than as part of the active algorithm.

The streaming (propagation) step
------------------------------------

Implemented in ``src/LBM/Streaming.H`` / ``Streaming.cpp``. Streaming is
implemented in **pull form**: each lattice site reads the post-collision
distribution that was located one step upstream (in the direction opposite
to propagation), rather than pushing its own value forward into a
neighbor:

.. math::

   f_q^{out}(\mathbf{x}) = f_q^{in}(\mathbf{x} - \mathbf{c}_q)

In code (``streamCell``, ``Streaming.cpp``):

.. code-block:: cpp

   const int i_old = i - LatticeTag::cx(n);
   const int j_old = j - LatticeTag::cy(n);
   return f_in_arr(i_old, j_old, k, n);   // (+ k_old in 3D)

Because the pull-form read for a boundary cell needs the value of
:math:`f_q` at a neighboring cell that may lie just across an MPI-rank
boundary or a physical domain boundary, the header comment for
``Streaming.H`` is explicit about the precondition:

    "Requires ghost-cell-filled MultiFabs; the caller is responsible for
    calling FillBoundary / FillPatch before streaming."

If ghost cells are stale or unfilled when ``stream``/``streamInPlace`` is
called, cells adjacent to box/domain boundaries would pull from garbage or
outdated data. This is why ``SimulationImpl::advance()`` always calls
``fillGhostCells()`` (which calls ``m_f.FillBoundary(...)`` for periodic/
inter-box exchange and then invokes the physical boundary-condition
manager) immediately *before* streaming, never after.

``stream<LatticeTag>`` is the two-MultiFab form (`f_in` to a separate
`f_out`); ``streamInPlace<LatticeTag>`` builds a scratch MultiFab
internally, streams into it, then copies back into `f`, letting the
in-place call be used safely even though pull-form streaming cannot write
into the same buffer it reads from in-place without aliasing hazards.

Units and lattice-physical conversion
------------------------------------------

Implemented in ``src/LBM/Units.H`` / ``Units.cpp`` (class
``UnitConverter``). MuPhFASa fixes the lattice unit system by convention:
:math:`\Delta x_{lat} = \Delta t_{lat} = \rho_{lat} = 1`; everything else
is derived from a small set of physical reference scales supplied at
construction.

Construction contract
~~~~~~~~~~~~~~~~~~~~~~~~

The current ``UnitConverter`` constructor signature is:

.. code-block:: cpp

   UnitConverter(amrex::Real L_ref,
                 amrex::Real U_ref,
                 amrex::Real Re,
                 amrex::Real rho_ref,
                 int         N_ref,
                 amrex::Real cs2,
                 std::optional<amrex::Real> u_lat = std::nullopt,
                 std::optional<amrex::Real> tau   = std::nullopt);

It takes the Reynolds number ``Re`` **directly** as a constructor
argument (it does not take ``nu_ref``). The reference kinematic viscosity
is derived internally from ``Re``:

.. math::

   \nu_{ref} = \frac{U_{ref}\, L_{ref}}{Re}

and the lattice spacing is derived from the reference length and the
number of lattice cells spanning it:

.. math::

   \Delta x = \frac{L_{ref}}{N_{ref}}

Deriving ``tau``/``u_lat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exactly one of ``tau`` or ``u_lat`` may be supplied (both are
``std::optional``); the time step :math:`\Delta t` is ultimately pinned by
whichever lattice reference velocity, :math:`u_{lat}`, results:

- **If ``tau`` is given**: the lattice kinematic viscosity follows from
  the requested relaxation time,

  .. math::

     \nu_{lat} = (\tau - 0.5)\, c_s^2

  and :math:`u_{lat}` is then back-derived from :math:`\nu_{lat}` using
  the as-yet-undetermined :math:`\Delta t` relation
  (:math:`\nu_{lat} = \nu_{ref}\, \Delta t / \Delta x^2` combined with
  :math:`\Delta t = u_{lat}\,\Delta x / U_{ref}`), giving

  .. math::

     u_{lat} = \nu_{lat}\, \frac{U_{ref}\, \Delta x}{\nu_{ref}}

  matching the code exactly:

  .. code-block:: cpp

     const amrex::Real nu_lat_from_tau = (*tau - amrex::Real(0.5)) * cs2;
     const amrex::Real u_lat_from_tau  = nu_lat_from_tau * m_U_ref * m_dx / m_nu_ref;

- **If ``u_lat`` is given** (and ``tau`` is not): :math:`u_{lat}` is used
  directly, with no derivation.

- **If neither is given**: :math:`u_{lat}` defaults to the conventional
  value :math:`u_{lat} = 0.1` (``kDefaultLatticeVelocity``), chosen to
  stay well below the lattice sound speed
  (:math:`c_s = 1/\sqrt{3} \approx 0.577`) while remaining comfortably
  inside the BGK stability region (see Krüger et al., Ch. 7, on unit
  conversion).

.. warning::

   Supplying **both** ``tau`` and ``u_lat`` is a **hard, unconditional
   error**, not a tolerance check. The current implementation does:

   .. code-block:: cpp

      if (tau) {
          const amrex::Real nu_lat_from_tau = (*tau - amrex::Real(0.5)) * cs2;
          const amrex::Real u_lat_from_tau  = nu_lat_from_tau * m_U_ref * m_dx / m_nu_ref;

          if (u_lat) {
              AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                  "UnitConverter: lbm.u_lat and lbm.tau were both specified."
                  "specify only one and let the other be derived.");
          }
          m_u_lat = u_lat_from_tau;
      } else if (u_lat) {
          m_u_lat = *u_lat;
      } else {
          m_u_lat = kDefaultLatticeVelocity;
      }

   Earlier versions of this code allowed both to be specified as long as
   they agreed within a tolerance; **that behavior no longer exists**.
   Specifying both ``lbm.tau`` and ``lbm.u_lat`` in an input file now
   always aborts the run via ``AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
   ...)``, regardless of whether the two values would have been
   numerically consistent.

Once :math:`u_{lat}` is fixed, the remaining lattice quantities follow in
sequence, exactly as in the constructor body:

.. math::

   \Delta t = \frac{u_{lat}\, \Delta x}{U_{ref}}
   \qquad
   \nu_{lat} = \frac{\nu_{ref}\, \Delta t}{\Delta x^2}
   \qquad
   \tau = 0.5 + \frac{\nu_{lat}}{c_s^2}

Stability and accuracy bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After :math:`\tau` and :math:`u_{lat}` are computed, ``UnitConverter``
applies the following checks, in order, exactly as implemented:

- :math:`\tau < 0.6` or :math:`\tau > 1.5`: **prints a warning** ("outside
  ideal range of 0.6 < tau < 1.5") but does **not** abort.
- :math:`u_{lat} > 0.15`: **prints a warning** ("close to violating low
  Mach number constraint") but does **not** abort.
- :math:`\tau \le 0.5` or :math:`\tau \ge 2.0`: **hard abort** via
  ``AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_tau > 0.5) && (m_tau < 2.0), ...)``.
- :math:`u_{lat} \ge 0.3`: **hard abort** via
  ``AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_u_lat < 0.3), ...)``.

.. note::

   The "ideal" range (0.6, 1.5) and the "low-Mach warning" threshold
   (0.15) are soft guidance printed to stdout; the hard stability range
   (0.5, 2.0) for :math:`\tau` and the hard cutoff (0.3) for
   :math:`u_{lat}` are the only conditions that actually terminate the
   run.

Lattice sound speed and Mach number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every lattice currently defined in ``LatticeDescriptor.H``,

.. math::

   c_s^2 = \frac{1}{3}, \qquad c_s = \frac{1}{\sqrt{3}} \approx 0.577

and the Mach number of the simulation is

.. math::

   \mathrm{Ma} = \frac{u_{lat}}{\sqrt{c_s^2}}

(``UnitConverter::machNumber()``, implemented exactly as
``m_u_lat / std::sqrt(m_cs2)``).

Putting it together: the per-step algorithm order
------------------------------------------------------

The authoritative definition of the per-timestep update order is
``SimulationImpl<LatticeTag, CollisionModel>::advance()`` in
``src/Simulation/SimulationSetup.cpp``. Each call to ``advance(step)``
performs, in this exact order:

.. code-block:: text

   1. computeEquilibriumMF<LatticeTag>(m_rho, m_vel, m_feq)
        -- build f^eq from the macroscopic state left over from the
           previous step's computeMacroscopic call (or the initial
           condition, on step 0)

   2. collideMF<LatticeTag, CollisionModel>(m_f, m_feq, m_omega)
        -- relax f toward f^eq in place:  f <- (1-omega) f + omega f^eq

   3. fillGhostCells()
        -- m_f.FillBoundary(periodicity)       (periodic / inter-box)
        -- (*m_phys_bc)(...)                   (physical BCs, e.g. walls,
                                                 inflow/outflow)

   4. streamInPlace<LatticeTag>(m_f, m_geom)
        -- pull-form propagation: f_out(x,q) = f_in(x - c_q, q)

   5. computeMacroscopic<LatticeTag>(m_f, m_rho, m_vel)
        -- recompute rho, u from the newly-streamed f, ready for next
           step's equilibrium evaluation

   m_time += 1.0

This is the **non-fused** path: it keeps ``rho``, ``vel``, and ``feq`` as
persistent MultiFabs (``m_rho``, ``m_vel``, ``m_feq``) and calls the
separate ``computeEquilibriumMF``/``collideMF`` functions rather than the
fused ``collideFullMF`` described above. Note that step 1 uses the
``rho``/``vel`` computed by step 5 of the *previous* iteration (or by
``setInitialConditions()``, which calls ``computeEquilibriumMF`` once
against the quiescent initial state :math:`\rho=1`, :math:`\mathbf{u}=0`
to seed ``m_f`` before the first ``advance()``).
