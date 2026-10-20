.. _api-core-kernels:

Core LBM Kernels
=================

This chapter is the API reference for the four modules that implement the
per-timestep Lattice-Boltzmann work: **Equilibrium**, **Macroscopic**,
**Collision**, and **Streaming**. Each module lives in ``src/LBM/`` as a
header/``.cpp`` pair (``Equilibrium.H``/``.cpp``, ``Macroscopic.H``/``.cpp``,
``Collision.H``/``.cpp``, ``Streaming.H``/``.cpp``).

For the underlying physics — what the equilibrium distribution, collision
operator, and streaming step actually mean and why the lattice constants
take the values they do — see :doc:`../theory`. This chapter focuses on the
mechanics: exact signatures, how the host/array/MultiFab layers call into
one another, template instantiation, and GPU portability.

The three-tier layering pattern
--------------------------------

All four modules are organized into the same three tiers, from innermost
to outermost. Understanding this pattern once makes each per-module section
below a short exercise in "which functions occupy which tier."

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Tier
     - Qualifiers / context
     - Role
   * - Single-cell kernel
     - ``AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE``
     - Computes the result for one cell (and, where relevant, one velocity
       direction ``n``), given plain arguments or an ``Array4``. Callable
       directly on host or device, with no ``amrex::ParallelFor`` context —
       e.g. from unit tests on the host, or from another single-cell kernel
       in a different module.
   * - Per-box array kernel
     - Free function taking an ``amrex::Box`` and ``Array4`` views
     - Dispatches an ``amrex::ParallelFor`` over every cell (and sometimes
       every ``Q`` component) in a single box, calling the single-cell
       kernel inside the lambda. This is the level at which GPU/CPU
       portability is actually achieved: the lambda is tagged
       ``AMREX_GPU_DEVICE`` and AMReX decides at compile time whether it
       runs as a CUDA/HIP/SYCL kernel or a host loop.
   * - MultiFab-level host function
     - Loops ``amrex::MFIter`` over ``amrex::TilingIfNotGPU()``
     - The outermost, host-only driver. Iterates tiles/boxes of one or more
       ``MultiFab``\ s, wraps the loop body in ``#ifdef _OPENMP`` /
       ``#pragma omp parallel`` for CPU threading, extracts an ``Array4``
       view per tile via ``mf[mfi].array()`` / ``.const_array()``, and calls
       the array-level kernel once per tile.

A consequence of this layering is that the "math" is written exactly once,
in the single-cell kernel, and reused by both the array-level ``ParallelFor``
dispatch and any fused kernel in another module that needs the same
computation inline (for example, ``Collision.cpp``'s ``collideFullArr``
calls ``computeEquilibriumCell2D``/``3D`` directly rather than going through
``computeEquilibrium``'s ``ParallelFor``).

Several single-cell kernels come in two flavors:

- An **Array4-based variant** that reads its inputs (e.g. velocity) directly
  out of a ``Array4<amrex::Real const>`` view at ``(i, j, k, *)``. This is
  the variant called from an array-level kernel that already has the
  MultiFab's per-tile ``Array4`` in hand.
- A **GpuArray-returning/taking variant** (suffixed ``2D``/``3D``) that
  passes the small per-cell vector (velocity, or a direction's displacement)
  as an ``amrex::GpuArray<amrex::Real, N>`` value instead of indexing an
  ``Array4``. These exist for *fused* kernels that compute an intermediate
  quantity (like velocity) into a register/local variable and immediately
  reuse it, without ever materializing it into a MultiFab — see
  ``collideFullArr`` in :ref:`core-kernels-collision` for the canonical
  example chaining ``computeDensityCell`` → ``computeVelocityCell2D/3D`` →
  ``computeEquilibriumCell2D/3D`` → ``BGK::collide`` entirely in registers.

Template instantiation for lattice tags
-----------------------------------------

Every function across all four modules is templated on ``LatticeTag``
(``Collision.H``'s collision-related functions are additionally templated
on ``CollisionModel``). See :doc:`lattice_descriptor` for what a
``LatticeTag`` provides (``Q``, ``cx``/``cy``/``cz``, ``weights``,
``cs2inv``, etc.).

Because templates must be instantiated somewhere to be linkable from other
translation units, each ``.cpp`` file ends with a block of explicit
instantiations, one per concrete lattice (``D2Q9``, ``D3Q15``, ``D3Q19``,
``D3Q27``). These blocks are gated by ``#if (AMREX_SPACEDIM == 2) ... #else
... #endif`` so that a 2D build only instantiates ``D2Q9`` and a 3D build
only instantiates ``D3Q15``/``D3Q19``/``D3Q27`` — the dimension-mismatched
lattices are never compiled in, which also matters because several
single-cell kernels (e.g. ``computeEquilibriumCell``, ``computeVelocityCell``)
branch on ``AMREX_SPACEDIM`` internally and would not make sense instantiated
against a lattice of the other dimensionality. For example, from
``Equilibrium.cpp``:

.. code-block:: cpp

   #if (AMREX_SPACEDIM == 2)
   template amrex::Real computeEquilibriumCell<D2Q9>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
   #else
   template amrex::Real computeEquilibriumCell<D3Q15>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
   template amrex::Real computeEquilibriumCell<D3Q19>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
   template amrex::Real computeEquilibriumCell<D3Q27>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
   #endif

Not every function in a module is dimension-dependent (e.g.
``Macroscopic.cpp``'s ``computeDensity``/``computeDensityCell`` loop over
``LatticeTag::Q`` generically and are instantiated unconditionally for all
four lattices), so some instantiation blocks in the ``.cpp`` files are
unconditional while others are ``#if``-gated — follow the per-function
pattern shown in each source file rather than assuming the whole file is
gated uniformly.

.. note::
   Functions marked ``AMREX_FORCE_INLINE`` and defined directly in a header
   (rather than in the ``.cpp``) are *not* part of this instantiation
   scheme — they are ordinary inline template functions, instantiated
   implicitly wherever they are called (including from other modules'
   ``.cpp`` files). Only functions defined in a ``.cpp`` file need the
   explicit instantiation block to be visible to other translation units.

Equilibrium
-----------

``src/LBM/Equilibrium.H`` / ``src/LBM/Equilibrium.cpp``

Computes the discrete Maxwell-Boltzmann equilibrium distribution
:math:`f^{eq}_q(\rho, \mathbf{u})` from a local density and velocity, using
the standard second-order low-Mach expansion:

.. math::

   f^{eq}_q = w_q \, \rho \left[ 1 + \frac{\mathbf{c}_q \cdot \mathbf{u}}{c_s^2}
              + \frac{(\mathbf{c}_q \cdot \mathbf{u})^2}{2 c_s^4}
              - \frac{\mathbf{u}^2}{2 c_s^2} \right]

(See :doc:`../theory` for the derivation and validity regime.)

Layering in this module:

- **Single-cell**: ``computeEquilibriumCell`` (Array4-based, reads ``vel``
  at ``(i,j,k,*)``), and the GpuArray-based ``computeEquilibriumCell2D`` /
  ``computeEquilibriumCell3D``.
- **Per-box array kernel**: ``computeEquilibrium``.
- **MultiFab-level host function**: ``computeEquilibriumMF``.

Equilibrium Signatures
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real computeEquilibriumCell(const amrex::Real rho,
                                      amrex::Array4<amrex::Real const> const& vel,
                                      int i, int j, int k, int n) noexcept;

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real computeEquilibriumCell2D(const amrex::Real rho,
                                        amrex::GpuArray<amrex::Real, 2> const& u,
                                        int n) noexcept

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real computeEquilibriumCell3D(const amrex::Real rho,
                                        amrex::GpuArray<amrex::Real, 3> const& u,
                                        int n) noexcept

.. code-block:: cpp

   template <typename LatticeTag>
   void computeEquilibrium(const amrex::Box& box,
                                 amrex::Array4<amrex::Real const> const& rho,
                                 amrex::Array4<amrex::Real const> const& vel,
                                 amrex::Array4<amrex::Real> const& feq_arr);

.. code-block:: cpp

   template <typename LatticeTag>
   void computeEquilibriumMF(const amrex::MultiFab& rho_mf,
                              const amrex::MultiFab& vel_mf,
                              amrex::MultiFab&       feq_mf);

``computeEquilibrium`` dispatches a ``ParallelFor`` over the box *and* over
``LatticeTag::Q`` in one call (a 4D index space ``(i, j, k, n)``), calling
``computeEquilibriumCell`` per ``(cell, direction)`` pair.
``computeEquilibriumMF`` loops ``MFIter`` over ``feq_mf`` (with
``TilingIfNotGPU()`` and an ``#ifdef _OPENMP`` ``#pragma omp parallel``),
pulls ``const_array()`` views of ``rho_mf``/``vel_mf`` and an ``array()``
view of ``feq_mf`` per tile, and calls ``computeEquilibrium`` once per tile.

Header-vs-cpp split: why ``computeEquilibriumCell2D``/``3D`` live in the header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``computeEquilibriumCell`` (the ``Array4``-based entry point) is declared in
the header but *defined* in ``Equilibrium.cpp``, since it is only ever
called from array-level kernels within the Equilibrium translation unit (or
via the explicit instantiations). ``computeEquilibriumCell2D``/``3D``, by
contrast, are defined directly in ``Equilibrium.H``. This is a deliberate
layering choice: they are marked ``AMREX_FORCE_INLINE``, which only has
effect if the function body is visible to the compiler at every call site
it should be inlined into. ``Collision.cpp``'s ``collideFullArr`` calls
``computeEquilibriumCell2D``/``3D`` directly (bypassing the ``Array4``-based
wrapper entirely, since it has ``u`` as a local ``GpuArray``, not a
MultiFab) — if these were defined in ``Equilibrium.cpp`` instead, that call
from a different translation unit would require either an explicit
instantiation (defeating ``AMREX_FORCE_INLINE``, which precludes a normal
out-of-line, separately-compiled instantiation) or duplicate inline
definitions. Keeping the GpuArray variants in the header makes them
ordinary header-only inline templates, instantiated at each call site
exactly like any ``AMREX_FORCE_INLINE`` function must be.

Explicit instantiations (``Equilibrium.cpp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``computeEquilibriumCell``, ``computeEquilibrium``, and
``computeEquilibriumMF`` are each instantiated for ``D2Q9`` in 2D builds and
``D3Q15``/``D3Q19``/``D3Q27`` in 3D builds. Example:

.. code-block:: cpp

   #if (AMREX_SPACEDIM == 2)
   template void computeEquilibriumMF<D2Q9>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
   #else
   template void computeEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
   template void computeEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
   template void computeEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
   #endif

.. note::
   ``computeEquilibriumCell2D``/``3D`` have **no** explicit instantiation
   block in ``Equilibrium.cpp`` — being header-defined inline templates,
   they don't need one; they are instantiated wherever called (e.g. in
   ``Collision.cpp``).

Macroscopic
-----------

``src/LBM/Macroscopic.H`` / ``src/LBM/Macroscopic.cpp``

Derives the macroscopic density :math:`\rho` and velocity :math:`\mathbf{u}`
from the distribution function ``f`` by taking its zeroth and first moments
over the discrete velocity set:

.. math::

   \rho = \sum_q f_q \,, \qquad
   u_\alpha = \frac{1}{\rho} \sum_q c_{q,\alpha} \, f_q

Layering in this module:

- **Single-cell**: ``computeDensityCell``, ``computeVelocityCell2D``,
  ``computeVelocityCell3D`` (all header-defined); ``computeVelocityCell``
  (Array4-writing, defined in ``Macroscopic.cpp``).
- **Per-box array kernel**: ``computeDensity``, ``computeVelocity``.
- **MultiFab-level host function**: ``computeMacroscopic``.

Macroscopic Signatures
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real computeDensityCell(amrex::Array4<const amrex::Real> const& f_arr,
                                  int i, int j, int k) noexcept

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   void computeVelocityCell(amrex::Array4<const amrex::Real> const& f_arr,
                            const amrex::Real rho,
                            amrex::Array4<amrex::Real> const& vel_arr,
                            int i, int j, int k) noexcept;

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::GpuArray<amrex::Real, 2> computeVelocityCell2D(amrex::Array4<const amrex::Real> const& f_arr,
                                                          const amrex::Real rho,
                                                          int i, int j, int k) noexcept

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::GpuArray<amrex::Real, 3> computeVelocityCell3D(amrex::Array4<const amrex::Real> const& f_arr,
                                                          const amrex::Real rho,
                                                          int i, int j, int k) noexcept

.. code-block:: cpp

   template <typename LatticeTag>
   void computeDensity(const amrex::Box& box,
                       amrex::Array4<const amrex::Real> const& f_arr,
                       amrex::Array4<amrex::Real> const& rho_arr);

.. code-block:: cpp

   template <typename LatticeTag>
   void computeVelocity(const amrex::Box& box,
                        amrex::Array4<const amrex::Real> const& f_arr,
                        amrex::Array4<const amrex::Real> const& rho_arr,
                        amrex::Array4<amrex::Real> const& vel_arr);

.. code-block:: cpp

   template <typename LatticeTag>
   void computeMacroscopic(const amrex::MultiFab& f_mf,
                           amrex::MultiFab&       rho_mf,
                           amrex::MultiFab&       vel_mf);

``computeMacroscopic`` loops ``MFIter`` once over ``f_mf`` per tile and, for
each tile, calls *both* ``computeDensity`` and ``computeVelocity`` in
sequence (velocity needs the just-computed density), writing into the
corresponding tiles of ``rho_mf`` and ``vel_mf``.

Header-vs-cpp split
~~~~~~~~~~~~~~~~~~~~

Exactly the same reasoning as in Equilibrium applies here, with one more
function in the mix:

- ``computeDensityCell``, ``computeVelocityCell2D``, ``computeVelocityCell3D``
  are defined in ``Macroscopic.H`` because they are ``AMREX_FORCE_INLINE``
  and are called from *other* translation units at their call sites — most
  notably ``Collision.cpp``'s ``collideFullArr``, which calls
  ``computeDensityCell`` and then ``computeVelocityCell2D``/``3D`` directly
  to build up ``rho``/``u`` as local scratch before computing ``f^{eq}``
  in-register.
- ``computeVelocityCell`` (the ``Array4``-writing variant, which calls
  ``computeVelocityCell2D``/``3D`` internally and writes the result into
  ``vel_arr(i,j,k,*)``) is defined in ``Macroscopic.cpp`` instead, because
  it is *only* ever called from ``computeVelocity`` within that same file —
  there is no cross-translation-unit call site requiring it to be visible
  inline elsewhere, so keeping it out-of-line in the ``.cpp`` is both
  sufficient and keeps the header smaller.

Explicit instantiations (``Macroscopic.cpp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``computeVelocityCell`` is dimension-gated (it branches on
``AMREX_SPACEDIM`` to call ``computeVelocityCell2D`` or ``3D``):

.. code-block:: cpp

   #if (AMREX_SPACEDIM == 2)
   template void computeVelocityCell<D2Q9>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
   #else
   template void computeVelocityCell<D3Q15>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
   template void computeVelocityCell<D3Q19>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
   template void computeVelocityCell<D3Q27>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
   #endif

``computeDensity`` has no such branch (it's a dimension-agnostic sum over
``LatticeTag::Q``), so it is instantiated unconditionally for all four
lattices in the same file; ``computeVelocity`` and ``computeMacroscopic``
*are* dimension-gated (they call into the dimension-gated
``computeVelocityCell``), following the same ``#if (AMREX_SPACEDIM == 2)``
pattern shown above.

.. _core-kernels-collision:

Collision
---------

``src/LBM/Collision.H`` / ``src/LBM/Collision.cpp``

Applies the collision operator, relaxing ``f`` toward the local equilibrium
``f^{eq}`` at a rate set by ``omega`` (the inverse relaxation time). The
collision model is selected as a template parameter (a tag type), so new
models can be added without changing call sites that are already generic
over ``CollisionModel``.

Layering in this module (for the ``BGK`` model):

- **Single-cell**: ``BGK::collide``.
- **Per-box array kernel**: ``BGK::collideArr``; also ``collideFullArr``
  (the fused variant, described below).
- **MultiFab-level host function**: ``collideMF``; also ``collideFullMF``.

Collision model tags
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   struct BGK {
       template <typename LatticeTag>
       static void collideArr(const amrex::Box& box,
                              amrex::Array4<amrex::Real> const& f_arr,
                              amrex::Array4<const amrex::Real> const& feq_arr,
                              const amrex::Real omega);

       AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
       static amrex::Real collide(const amrex::Real f,
                                  const amrex::Real feq,
                                  const amrex::Real omega) noexcept;
   };   // single-relaxation-time
   struct TRT {};   // two-relaxation-time     (not yet implemented)
   struct MRT {};   // multiple-relaxation-time (not yet implemented)

``BGK`` is the only model with implemented behavior:

.. math::

   f_q \leftarrow (1 - \omega) f_q + \omega f^{eq}_q

.. code-block:: cpp

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real BGK::collide(const amrex::Real f,
                            const amrex::Real feq,
                            const amrex::Real omega) noexcept

``TRT`` and ``MRT`` are empty tag structs with no members and no
``collide``/``collideArr`` methods. They exist purely as placeholders for
the template-parameter slot.

.. warning::
   ``TRT`` and ``MRT`` are **not implemented**. Because ``collideMF``,
   ``collideArr``, and ``collideFullArr``/``collideFullMF`` call
   ``CollisionModel::collide``/``CollisionModel::template collideArr<...>``
   through the template parameter, instantiating any of those function
   templates with ``CollisionModel = TRT`` or ``MRT`` is a **compile-time
   error** (no such member exists on an empty struct) — not a runtime
   failure. In practice this is never reached: ``SimulationFactory.cpp``
   validates the requested collision model at startup and aborts before
   any ``TRT``/``MRT``-templated code path could be instantiated or called.

MultiFab-level dispatch
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename LatticeTag, typename CollisionModel>
   void collideMF(amrex::MultiFab&       f_mf,
                  const amrex::MultiFab& feq_mf,
                  amrex::Real            omega);

``collideMF`` loops ``MFIter`` over ``f_mf`` and, per tile, calls
``CollisionModel::template collideArr<LatticeTag>(tileBox, f_arr, feq_arr,
omega)`` — i.e. it dispatches to whichever model's array-level kernel is
selected by the ``CollisionModel`` template parameter. This is the
separate, two-MultiFab path (``f`` relaxed toward a pre-computed ``feq``)
used by the main simulation loop (see below).

Fused path: ``collideFullArr`` / ``collideFullMF``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename LatticeTag, typename CollisionModel>
   void collideFullArr(const amrex::Box& box,
                       amrex::Array4<amrex::Real> const& fstar_arr,
                       amrex::Array4<const amrex::Real> const& f_arr,
                       amrex::Real omega);

   template <typename LatticeTag, typename CollisionModel>
   void collideFullMF(amrex::MultiFab&       fstar_mf,
                      const amrex::MultiFab& f_mf,
                      amrex::Real            omega);

This is a fully fused rho → u → :math:`f^{eq}` → :math:`f^\star` pipeline
in a single ``ParallelFor`` per cell: inside the lambda, ``collideFullArr``
calls ``computeDensityCell``, then ``computeVelocityCell2D``/``3D``, then
(per direction ``n``) ``computeEquilibriumCell2D``/``3D``, then
``CollisionModel::collide`` — all chained through local registers/scratch
(``rho``, ``u``) rather than intermediate ``rho_mf``/``vel_mf``/``feq_mf``
MultiFabs, since nothing downstream of this kernel needs those intermediate
quantities materialized. ``collideFullMF`` is the corresponding
``MFIter``-driven MultiFab wrapper, writing the fused result into
``fstar_mf``.

.. note::
   As a factual note on current call sites (the rationale is covered in
   :doc:`../theory`): ``SimulationSetup.cpp``'s main ``advance()`` loop
   currently uses the separate ``computeEquilibriumMF`` + ``collideMF``
   path, not ``collideFullMF``. The fused path is available and exercised
   independently (e.g. in tests) but is not what drives the production
   timestep today.

Explicit instantiations (``Collision.cpp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BGK::collideArr`` and ``collideMF`` are dimension-agnostic (``BGK::collide``
takes no spatial arguments) and are instantiated unconditionally for all
four lattices:

.. code-block:: cpp

   template void BGK::collideArr<D2Q9>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
   template void BGK::collideArr<D3Q15>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
   template void BGK::collideArr<D3Q19>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
   template void BGK::collideArr<D3Q27>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);

``collideFullArr``/``collideFullMF`` call the dimension-branching
``computeVelocityCell2D``/``3D`` and ``computeEquilibriumCell2D``/``3D``
internally, so — like the dimension-dependent functions in the other
modules — they are instantiated under the ``#if (AMREX_SPACEDIM == 2) ...
#else ... #endif`` guard (``D2Q9`` only in 2D; ``D3Q15``/``D3Q19``/``D3Q27``
in 3D). All instantiations here are for ``CollisionModel = BGK``; there are
no ``TRT``/``MRT`` instantiations, consistent with the warning above.

Streaming
---------

``src/LBM/Streaming.H`` / ``src/LBM/Streaming.cpp``

Performs the propagation (streaming) step, shifting each distribution
component by one lattice site along its own discrete velocity:

.. math::

   f_{out}(\mathbf{x}, q) = f_{in}(\mathbf{x} - \mathbf{c}_q, q)

Layering in this module:

- **Single-cell**: ``streamCell``.
- **Per-box array kernel**: ``default_stream``.
- **MultiFab-level host functions**: ``stream`` and ``streamInPlace``.

Streaming Signatures
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename LatticeTag>
   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   amrex::Real streamCell(amrex::Array4<const amrex::Real> const& f_in_arr,
                          int i, int j, int k, int n) noexcept;

.. code-block:: cpp

   template <typename LatticeTag>
   void default_stream(const amrex::Box& box,
                       amrex::Array4<const amrex::Real> const& f_in_arr,
                       amrex::Array4<amrex::Real> const& f_out_arr);

.. code-block:: cpp

   template <typename LatticeTag>
   void stream(const amrex::MultiFab& f_in,
               amrex::MultiFab&       f_out,
               const amrex::Geometry& geom);

.. code-block:: cpp

   template <typename LatticeTag>
   void streamInPlace(amrex::MultiFab&       f,
                      const amrex::Geometry& geom);

``streamCell`` reads the *upstream* node for direction ``n`` —
``f_in_arr(i - c_x(n), j - c_y(n), k - c_z(n), n)`` — i.e. it gathers rather
than scatters: each output cell pulls its value from the neighbor one
lattice vector *behind* it, rather than each input cell pushing to the
neighbor ahead. ``default_stream`` dispatches a ``ParallelFor`` over the
box and over ``LatticeTag::Q`` (a 4D index space), calling ``streamCell``
per ``(cell, direction)``.

``stream`` vs. ``streamInPlace``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   Both entry points require ``f_in`` (or, for ``streamInPlace``, ``f``) to
   have **at least one layer of valid ghost cells already filled**, since
   ``streamCell`` reads from ``(i - c_x(n), ...)``, which for boundary
   cells and directions pointing across an MPI/tile boundary lands in a
   ghost cell. Per the header comment, the *caller* is responsible for
   calling ``FillBoundary`` / ``FillPatch`` on the input MultiFab before
   invoking either function — neither ``stream`` nor ``streamInPlace``
   does this itself.

``stream`` is the two-MultiFab form: it reads from ``f_in`` and writes into
a distinct ``f_out``, looping ``MFIter`` over ``f_in`` and calling
``default_stream`` per tile. Because gather reads from neighboring cells
while writes land in the current cell, ``f_out`` must not alias ``f_in``
(the header documents this: "``f_out`` may alias ``f_in`` only if an
intermediate buffer is used internally").

``streamInPlace`` provides the in-place-looking variant for callers that
only want to maintain a single logical ``f`` MultiFab across the timestep.
Internally it is not actually in-place: it allocates a full scratch
``MultiFab f_scratch`` with the same ``BoxArray``/``DistributionMapping``/
``nComp()``/``nGrow()`` as ``f``, calls ``stream<LatticeTag>(f, f_scratch,
geom)``, and then copies the result back with ``amrex::MultiFab::Copy(f,
f_scratch, 0, 0, f.nComp(), 0)``.

.. note::
   ``streamInPlace`` **doubles the memory footprint of** ``f`` **for the
   duration of the step** (the live ``f`` plus the freshly allocated
   ``f_scratch``, both sized ``nComp() = Q`` components). Callers that are
   memory-constrained, or that already maintain two buffers (e.g. a
   ping-pong ``f``/``f_next`` pair) should prefer calling ``stream``
   directly rather than ``streamInPlace``.

Explicit instantiations (``Streaming.cpp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``streamCell`` branches on ``AMREX_SPACEDIM`` (it only forms a ``k_old``
offset in the 3D branch), so every function in this file — ``streamCell``,
``default_stream``, ``stream``, and ``streamInPlace`` — is instantiated
under the dimension guard:

.. code-block:: cpp

   #if (AMREX_SPACEDIM == 2)
   template void stream<D2Q9>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
   #else
   template void stream<D3Q15>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
   template void stream<D3Q19>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
   template void stream<D3Q27>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
   #endif
