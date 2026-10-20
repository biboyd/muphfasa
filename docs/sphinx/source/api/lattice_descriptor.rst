.. _api-lattice-descriptor:

LatticeDescriptor
====================

``src/LBM/LatticeDescriptor.H`` defines compile-time descriptors for the
discrete velocity sets ("lattices") used throughout MuPhFASa: the discrete
velocities :math:`\mathbf{c}_q`, quadrature weights :math:`w_q`,
opposite-direction indices (for bounce-back boundary conditions), and the
lattice sound speed. ``src/LBM/LatticeDescriptor.cpp`` is a **placeholder
translation unit** — it contains only an ``#include`` of the header, since
every accessor and every piece of lattice data is defined ``inline``
``constexpr`` directly in the header; nothing in this module needs to be
compiled into a separate object file.

Primary template and specializations
----------------------------------------

The primary template is declared but never defined:

.. code-block:: cpp

   template <int NDIM, int NQ>
   struct LatticeDescriptor;

Four full specializations are provided, matching the ``NDIM`` (spatial
dimension) and ``NQ`` (number of discrete velocities) of each supported
lattice, with convenience aliases defined at the bottom of the header:

.. code-block:: cpp

   using D2Q9  = LatticeDescriptor<2,  9>;
   using D3Q15 = LatticeDescriptor<3, 15>;
   using D3Q19 = LatticeDescriptor<3, 19>;
   using D3Q27 = LatticeDescriptor<3, 27>;

Only **D2Q9** is fully implemented today. The three 3D lattices
(D3Q15, D3Q19, D3Q27) are structurally present (correct ``D``, ``Q``,
``cs2``, ``cs2inv``, ``nGhost`` constants, and the right accessor
signatures so templated code compiles) but are **data stubs** — see the
warning below.

D2Q9 (fully implemented)
----------------------------

``LatticeDescriptor<2, 9>``: :math:`D = 2`, :math:`Q = 9`,
:math:`c_s^2 = 1/3`, :math:`c_s^2{}^{-1} = 3`, ``nGhost = 1``.

The header documents the velocity table in a comment directly above the
specialization:

.. code-block:: text

     q:  0   1   2   3   4   5   6   7   8
   (cx): 0   1   0  -1   0   1  -1  -1   1
   (cy): 0   0   1   0  -1   1   1  -1  -1

Reproduced as a table, together with the weights and opposite-direction
map (also taken verbatim from the ``weights`` and ``opposite`` accessor
arrays):

.. list-table:: D2Q9 velocity set
   :header-rows: 1

   * - :math:`q`
     - :math:`c_{q,x}`
     - :math:`c_{q,y}`
     - :math:`w_q`
     - ``opposite(q)``
   * - 0
     - 0
     - 0
     - 4/9
     - 0
   * - 1
     - 1
     - 0
     - 1/9
     - 3
   * - 2
     - 0
     - 1
     - 1/9
     - 4
   * - 3
     - -1
     - 0
     - 1/9
     - 1
   * - 4
     - 0
     - -1
     - 1/9
     - 2
   * - 5
     - 1
     - 1
     - 1/36
     - 7
   * - 6
     - -1
     - 1
     - 1/36
     - 8
   * - 7
     - -1
     - -1
     - 1/36
     - 5
   * - 8
     - 1
     - -1
     - 1/36
     - 6

Direction 0 is the rest ("zero") velocity. Directions 1-4 are the
axis-aligned ("D2Q9 nearest-neighbor") velocities with weight
:math:`1/9`; directions 5-8 are the diagonal velocities with weight
:math:`1/36`. The weights satisfy :math:`\sum_q w_q = 1` as required for a
valid quadrature:
:math:`4/9 + 4 \times 1/9 + 4 \times 1/36 = 4/9 + 4/9 + 1/9 = 1`.

The opposite-direction map satisfies :math:`\mathbf{c}_{\mathrm{opposite}(q)}
= -\mathbf{c}_q` for every :math:`q` (e.g. ``opposite(1) = 3``:
:math:`\mathbf{c}_1 = (1,0)`, :math:`\mathbf{c}_3 = (-1,0)`); this map is
used by bounce-back boundary condition kernels (see
``src/LBM/BCTypes/BounceBack.H``, ``NoSlipKernel.H``).

``nGhost = 1`` reflects that D2Q9 streaming only ever moves a distribution
by one lattice site per step, so a single layer of ghost cells is
sufficient for :func:`streamCell` to read every upstream neighbor it
needs (see :doc:`../theory`).

D3Q15 / D3Q19 / D3Q27 (unimplemented stubs)
------------------------------------------------

.. warning::

   The three 3D lattice specializations in ``src/LBM/LatticeDescriptor.H``
   are **not implemented**. Each one's ``cx``, ``cy``, and ``cz``
   accessors unconditionally ``return 0``, ``weights`` unconditionally
   ``return amrex::Real(0.0)``, and ``opposite(q)`` unconditionally
   ``return q`` (i.e. every direction is treated as its own opposite).
   The header marks each with a literal TODO comment:

   - D3Q15, ``src/LBM/LatticeDescriptor.H:74``: ``// TODO: fill in D3Q15 velocity data``
   - D3Q19, ``src/LBM/LatticeDescriptor.H:101``: ``// TODO: fill in D3Q19 velocity data``
   - D3Q27, ``src/LBM/LatticeDescriptor.H:128``: ``// TODO: fill in D3Q27 velocity data``

   **This is not a build-time or run-time abort.** Selecting one of these
   lattices in a 3D build compiles and runs without error — it will
   silently execute with a degenerate, physically meaningless velocity set
   (every discrete velocity zero, every quadrature weight zero, zero
   streaming displacement for every direction). The result is not a crash
   but a **physically wrong simulation** that may appear to run normally
   while producing meaningless output (e.g. a zero or NaN-propagating
   density field, since a weight of 0 everywhere breaks the quadrature
   normalization :math:`\sum_q w_q = 1` the equilibrium distribution and
   moment calculations rely on). **D3Q15, D3Q19, and D3Q27 must not be
   selected for production use until their velocity data is filled in.**

Each 3D specialization does correctly declare the shared lattice constants
(:math:`D=3`, the respective :math:`Q`, :math:`c_s^2 = 1/3`,
:math:`c_s^{-2} = 3`, ``nGhost = 1``) and the right accessor shapes
(including ``cz(q)``, which D2Q9 does not have), so that templated LBM
code (``Equilibrium.H``, ``Collision.H``, ``Macroscopic.H``,
``Streaming.H``) compiles against them in a 3D build. Only the actual
per-direction numeric data is missing.

Design: plain arrays instead of ``amrex::GpuArray``
--------------------------------------------------------

Every accessor (``cx``, ``cy``, ``cz``, ``weights``, ``opposite``) is
declared ``AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr`` and
indexes into a **local ``constexpr`` plain C array** declared inside the
function body (e.g. ``constexpr int d[9] = {...}; return d[q];``), rather
than an ``amrex::GpuArray``. This is a deliberate choice:
``amrex::GpuArray::operator[]`` is not itself ``constexpr`` by design,
because it includes debug-mode bounds-checking via ``index_assert()``
which is not constexpr-safe; using ``GpuArray::operator[]`` inside a
``constexpr`` function body would therefore never produce a valid constant
expression, whereas plain C-array indexing does, letting these accessors
remain genuinely ``constexpr`` (and thus usable in constant-evaluated
contexts) while still being callable from both host and GPU device code.
