# AMR implementation notes (branch `amr`)

Living document: design, decisions, status, and known limitations for
block-structured AMR in muphfasa.

## 1. Starting point

* Single-level, BGK, Guo forcing, halfway bounce-back implemented as
  *ghost-cell fills* (`BCManager` via `PhysBCFunct<GpuBndryFuncFab<...>>`).
* Default time step is the fused pull kernel:
  `fout(x) = collide( fin(x - c_n) )`, so `fin` always holds
  **post-collision** populations and needs one ghost layer of
  post-collision data before every step.
* `ISimulation` / `SimulationImpl<LatticeTag, BGK>` own one level of data.

## 2. Design goals

1. `amr.max_level = 0` (the default) must leave the existing code path
   **untouched** -> all current CI (unit + couette/lid2d/tgv2d/poiseuille/
   uniform_force) is unchanged. AMR lives in a separate class,
   `AmrSimulationImpl<LatticeTag, BGK>` (AmrCore-derived), selected by
   `makeSimulation()` only when `amr.max_level > 0`.
2. Second-order-consistent coupling across coarse/fine (C/F) interfaces for
   BGK + Guo forcing, robust for every tau (no singular rescaling factors).
3. Reuse the existing kernels/BCs as much as possible; per-level
   `PhysBCFunct` built from the *same* `BCManager` (wall velocities and
   densities are level-invariant in lattice units under acoustic scaling).

## 3. Numerical scheme

### 3.1 Scaling between levels (refinement ratio 2 only)

Acoustic scaling: `dx_{l+1} = dx_l/2`, `dt_{l+1} = dt_l/2`, so lattice
velocity and density are identical on all levels.

* Constant physical viscosity: `tau_{l+1} - 1/2 = 2 (tau_l - 1/2)`.
* Body force in lattice units: `F_{l+1} = F_l / 2` (`F_lat = a dt^2/dx`).
* Level `l` advances `2^l` substeps per level-0 step (recursive subcycling).
* All times stored in level-0 lattice units (`dt_l = 2^-l`), so
  `ISimulation::time()` keeps its meaning.

### 3.2 Non-equilibrium rescaling (Dupuis & Chopard, pre-collision form)

Chapman-Enskog for BGK + Guo (with the Guo half-force velocity `u`):

    f_pre - feq(rho,u) = fneq = -tau * D feq  +  (tau - 1/2) G(u).F

where `G_n(u).F = w_n/cs2 [c.F + (c.u)(c.F)/cs2 - u.F]` is the Guo source
without the `(1-omega/2)` factor (`guoForceTerm(n,u,F,0)`).
In lattice units the gradient part scales with `tau*dt`, while the force
part `(tau - 1/2) F_lat` is **level invariant** under acoustic scaling.
Hence converting a pre-collision population from source level s to target
level t is

    P      = (tau_s - 1/2) G(u).F_s                     (force part)
    fneq_t = r (fneq_s - P) + P,   r = tau_t dt_t / (tau_s dt_s)
    f_pre_t  = feq + fneq_t
    f_post_t = f_pre_t - omega_t (f_pre_t - feq) + (1 - omega_t/2) G(u).F_t

with `r = tau_f/(2 tau_c)` for coarse->fine and `2 tau_c/tau_f` for
fine->coarse. Momentum check: `sum c fneq_t = -F_t/2`, as required.

Why *pre*-collision: the classic post-collision factor
`(tau_f-1)/(2(tau_c-1))` and its inverse are singular at tau_c = 1 and
tau_c = 3/4, both common in practice. Working on pre-collision populations
gives factors that are always finite and positive.

To have pre-collision data available, the AMR path uses a variant of the
fused kernel, `pullCollideFusedPreMF`, which additionally stores the
streamed (pre-collision) populations `fpre` it computes in registers anyway.
`fin = collide(fpre)` at the same time level.

### 3.3 Coarse -> fine (fine ghost cells)

Fine ghost cells need fine **post-collision** populations at the fine time.
For a fill of level l at time t:

1. Build coarse temporaries `conv_old`, `conv_new` = coarse `fpre_old`,
   `fpre_new` converted by 3.2 into *fine post-collision* populations
   (cellwise at coarse resolution).
2. `amrex::FillPatchTwoLevels` with `(conv_old, conv_new)` at
   `(t_old, t_new)` (linear in time) and the spatial interpolater
   `amr.interp` (default `consistent`, see below). The coarse-patch
   physical BC on converted data is first-order extrapolation. It only
   matters where the interpolater reads outside the domain, which
   `consistent` never does: it switches to one-sided differences. The fine
   physical ghosts are filled afterwards by the real `BCManager` at the
   fine geometry.

**Spatial interpolation must be consistent with average-down.** Because
covered coarse cells are overwritten by the mean of their fine children
(3.4), the coarse values next to the C/F boundary are, for the smooth fine
field `P` and child offsets `xi_d` (coarse cell units, ratio `r`),

    c = P + sum_d <xi_d^2> P_dd / 2 + O(h^4),   <xi_d^2> = (r^2-1)/(12 r^2)

(an O(h^2) offset, `h^2 P''/32` for r = 2). The fine ghost needs `P(xi)`,
which to third order is

    f(xi) = c + sum_d [ s_d xi_d + k_d (xi_d^2 - <xi_d^2>)/2 ] + sum_{d<e} m_de xi_d xi_e

with central first (`s`), second (`k`) and mixed (`m`) differences of `c`.
For r = 2 the `k` terms vanish (`xi^2 = 1/16 = <xi^2>`), leaving unlimited
central-linear interpolation plus the cross terms. This is
`AvgConsistentInterp` (`src/LBM/AmrInterface.H`). It is conservative (the
children average back to `c`) and needs a 3^d coarse stencil, so
`n_proper = 2` is enough. Next to a non-periodic domain face the
differences become one-sided, second order when two interior neighbours
exist.

The AMReX interpolaters each make an O(h^2) error in the ghost
populations:

* `quadratic_interp` (pointwise, exact for parabolas) adds `+k/32`;
* `cell_bilinear_interp` adds `+k/8`;
* `lincc_interp` and `cell_cons_interp`: both are **limited** (linear
  limiting and per-component MC limiting respectively; neither is
  "unlimited"). They are correct away from extrema, but at an extremum of
  the populations (e.g. the centreline of a channel) the slope is clipped
  to zero.

An O(h^2) inconsistency between what the fine level receives (ghosts) and
what the coarse level receives (averaged cells) is a spurious momentum
source at the interface. In a steady forced channel it shifts the wall
stresses and gives a *global*, linear-in-y velocity error of relative size
O(h / nu): first order, growing as tau -> 1/2 (see 5.2). The interpolater
is selectable for comparison (`amr.interp = quadratic | lincc | cell_cons
| bilinear`). Without reflux (3.9) `cell_cons` also shows a mass drift of
~1e-7 on coarse grids. Fine levels are created (and newly refined regions
refilled at a regrid) with the same interpolater. A non-average-preserving
one (`quadratic`, `bilinear`) therefore changes the composite mass and
momentum by its averaging error whenever fine cells are created (TGV, N =
32: momentum 1.9e-4 of `u0 V` at t = 0). The totals are constant
afterwards (5.4).

The analysis and the selection were done with a 1-D (y only) Python model
of the exact coupling (D2Q9 BGK + Guo, halfway bounce-back, subcycling,
time interpolation, average-down). It reproduces the C++ errors. In it,
the error grows linearly with the coefficient of `k` in the ghost formula
and vanishes at the average-consistent value 0.

### 3.4 Fine -> coarse (covered coarse cells)

After the two fine substeps of a coarse step, the fine `fpre` (time
`t_new`) is conservatively averaged onto coarsened fine boxes, converted
by 3.2 (fine->coarse) into coarse `fpre` and coarse post-collision `fin`,
and copied over the covered coarse cells. Uncovered coarse cells next to
the fine region therefore stream in fine-consistent populations on the next
coarse step. (Overlap / "coarse computes everywhere" approach. On its own
this conserves mass and momentum only to truncation error; 3.9 restores
exact composite conservation.)

### 3.5 Time-stepping (per level-0 step)

    timeStep(l):
      if l > 0: fillGhosts(l, t_l)          // FillPatchTwoLevels (3.3)
      swap(fpre_old[l], fpre_new[l])
      fused AMR kernel: fin[l] -> fout[l], fpre_new[l]; swap(fin, fout)
      t_old[l] = t_l;  t_l += dt_l
      [reflux] accumulate crossings of this (sub)step               // 3.9
      if l < finest: timeStep(l+1) twice;  averageDown(l+1 -> l)   // 3.4
                     [reflux] apply correction to uncovered cells   // 3.9
      if l == 0: FillBoundary + physbc on level 0

Ghosts are filled at the *start* of each fine substep (so they use the
already averaged coarse data at `t_old`), and once more for all levels
before output.

### 3.6 Initialisation, tagging, regridding

* Level 0 is initialised with `initData` exactly as the single-level
  solver: `fin = fpre = feq(rho, u)` (plus Guo correction when forced).
  With `amr.init_fneq = 1` (default) the first-order Chapman-Enskog
  non-equilibrium part `fneq = -tau w rho/cs^2 Q:grad(u)` of the initial
  velocity field (central differences) is then added to `fpre`, and
  `(1 - 1/tau) fneq` to `fin` (`addInitialNonEquilibriumMF`). Finer levels
  are interpolated from converted coarse data (3.3), so an `initData`
  override only has to handle level 0. Starting from `feq` alone, each
  level builds its own initial non-equilibrium layer at its own tau and dt,
  so for the first steps the levels disagree at the interfaces. That
  mismatch radiates acoustic waves and gives a composite momentum jump in
  the first step; with the CE start the jump is O(h^2) without refluxing
  (TGV: 1.5e-8, 2.0e-9, 4.5e-10 of `u0 V` at N = 32/64/128) and round-off
  with it.
* Tagging (`ErrorEst`): `tagging.box_lo/box_hi` per level (physical-space
  static boxes, used by tests) and/or `tagging.vort_thresh` (lattice
  vorticity magnitude * 2^l).
* `amr.regrid_int` > 0 regrids every N level-0 steps.
  `MakeNewLevelFromCoarse` / `RemakeLevel` interpolate converted data
  (both `fin` and `fpre`).

### 3.7 Input keys (new)

| key | default | meaning |
| --- | --- | --- |
| `amr.max_level` | 0 | 0 = original single-level solver |
| `amr.blocking_factor` | 4 | AmrInfo blocking factor (all levels) |
| `amr.n_error_buf` | 2 | tag buffer cells |
| `amr.grid_eff` | 0.7 | grid efficiency |
| `amr.regrid_int` | 0 | regrid every N coarse steps (0 = static grids) |
| `amr.interp` | consistent | coarse->fine interpolater (3.3); `quadratic`, `lincc`, `cell_cons`, `bilinear` for comparison only (not second order: 5.2, 5.3) |
| `amr.init_fneq` | 1 | start level 0 from feq + Chapman-Enskog fneq (3.6) |
| `amr.reflux` | 1 | conservation correction (3.9): 0 none, 1 global, 2 local (per cell) |
| `tagging.box_lo_<l>` / `tagging.box_hi_<l>` | – | physical refine box for level l+1 |
| `tagging.vort_thresh` | – | vorticity tagging threshold |

`grid.max_grid_size` is reused for every level. AmrCore is built through
the `(Geometry, AmrInfo)` constructor so it reads **no** ParmParse keys
(no clash with existing `amr.plot_int`, `geometry.*`).

### 3.8 Output

Multi-level plotfiles (`WriteMultiLevelPlotfile`) in physical units with
the same variable names as single-level. Checkpoint: level-0 `f` only
(restart is not supported by either solver).

### 3.9 Conservation across coarse/fine boundaries (`amr.reflux`)

The conserved quantities are the composite sums of the post-collision
populations, `M = sum f dV` and `P = sum c f dV` over uncovered coarse plus
fine cells (`dV = 2^{-lD}`). Each level conserves both exactly by itself
(BGK collision; bounce-back returns a population to the cell it left). The
composite changes only through populations crossing the C/F boundary, and
the levels count those crossings differently. Per coarse step, for an
uncovered coarse cell `U`:

* coarse: `A - B`, where `A` sums `f*_i(U - c_i)` over covered `U - c_i`
  (gained from covered) and `B` sums `f*_i(U)` over covered `U + c_i` (lost
  to covered);
* fine (both substeps, weight `2^-D`): ghost populations pulled into fine
  valid cells minus fine populations streamed into ghost cells.

Without a correction the composite drifts by the sum of the two. Each
crossing is evaluated on the post-collision populations just before the
(sub)step's stream (`accumulateCoarseCrossingsMF`,
`accumulateFineCrossingsMF`). The two contributions are added
(`ParallelAdd`, two ghost cells for the fine register) after average-down,
and `applyRefluxMF` removes the result as the equilibrium-shaped increment
`w_i (drho + c_i.dj/cs^2)`. That increment carries exactly `drho`, `dj` and
is added to both `fin` and `fpre_new` of uncovered cells.

**Attribution of fine crossings.** For a local correction the fine
crossings must be assigned to the coarse link they correspond to, so that
for a uniform state coarse and fine cancel cell by cell. A fine link
leaving fine cell `x` along `c` maps to coarse source `X_d = t/2` with
`t = x_d - 1` (`c_d > 0`) or `t = x_d` (`c_d < 0`). An odd `t` lies halfway
between two coarse links, and the population is split 1/2-1/2 between them.
Dimensions with `c_d = 0` use `coarsen(x)`. Each part goes to the uncovered
endpoint of its coarse link. Parts on coarse links that touch the domain
exterior carry mass only (their momentum is a wall-force difference, not a
coupling defect). This was checked by hand for straight faces, convex
corners and corner-clipping links.

**Why the correction is global by default.** The per-cell registers are not
small truncation errors. Where the C/F boundary meets a wall, or turns a
corner of the fine region, the levels resolve a diagonal population
differently: the coarse corner cell bounces it back into itself (or sends
all of it across), while half of it crosses at the fine resolution. The
per-cell mismatch is ~`rho |u_corner|/6` per step and does not vanish with
dx. In a Couette flow with the interfaces at the moving wall it reaches
4e-3 per step, against a net (summed) defect of 2e-6. In the uncorrected
scheme each level is steady on its own and only the small left/right
residual drifts. Removing the per-cell values (`amr.reflux = 2`, local)
turns them into source/sink pairs next to the corners. That is exactly
conservative but makes the steady walled-flow error first order (5.4).
`amr.reflux = 1` (global, default) removes only the net defect of each
coarse/fine pair per coarse step, spread uniformly over that pair's
uncovered coarse cells. It is also exact to round-off, and its accuracy is
indistinguishable from no correction. In periodic problems all three modes
are second order; there the local mode also works. A locally conservative
alternative would need a different coupling (e.g. volumetric LBM
refinement), not a reflux on top of this one.

## 4. Testing

* All existing tests unchanged (max_level = 0); verified with ctest in a
  2D test build after every solver change.
* `tests/regression/couette_amr` (`test_couette_amr.cpp`): plane
  Couette(-Poiseuille) startup vs. the analytic series, evaluated on the
  **composite** solution (each level's cells not covered by a finer level,
  analytic profile at `y = (j+1/2) 2^-l`). Also tracks the composite mass
  `sum f dV` of the post-collision populations (3.9; `sum rho dV` is
  printed too, but `rho` is evaluated after one more stream, so it differs
  by one step's crossing mismatch) and the finest level reached. Fails via exit code when
  `couette_amr.max_err` / `max_mass_drift` / `min_finest` are violated.
  ctest configurations:
  * `inputs` (`regression_couette_amr`): 3 levels, both fine levels touch
    the moving wall, interfaces at y = 0.5 and 0.75 (`n_error_buf = 0`).
    3D: `inputs3d` (D3Q19).
  * `inputs_band`: one interior refined band (fine level without walls).
  * `inputs_regrid`: vorticity tagging + regridding every 10 steps (levels
    are created, remade and cleared during the run).
  * `inputs_poiseuille`: forced (Guo) channel, refinement at a wall, C/F
    interface on the centreline (`regression_couette_amr_poiseuille`).
  * `inputs_normal`: the same forced channel with the interfaces normal to
    the walls (every interface ends on a wall, level 2 has an interior
    corner). It fails with `amr.reflux = 0` (mass drift) and with
    `amr.reflux = 2` (error), so it pins down the global reflux (3.9, 5.4).
  The variants share a working directory and so hold a ctest
  `RESOURCE_LOCK`.
* `tests/unit/test_amr_convert.cpp`: conversion round trip, rho/u
  invariance, `collide(pre)` consistency (2D and 3D).
* `convergence_amr.py`: grid-convergence order of the composite error
  (`--inputs`, `--target-time`, `--exe`, extra `KEY=VALUE` overrides).
* `tests/regression/tgv2d_amr` (`test_tgv2d_amr.cpp`): doubly periodic
  Taylor-Green vortex decay vs. the analytic solution, evaluated on the
  composite solution (same acoustic-scaling convention, errors relative to
  the current amplitude `u0 exp(-2 nu k^2 t)`, both inf-norm and
  volume-weighted L2). The static refinement patches cut through the
  vortices with mixed derivatives `u_xy != 0` and patch corners crossed by
  diagonal populations, exercising the full 2-D interpolater (including
  cross terms and multi-box blocking). `convergence_tgv_amr.py` is the
  companion convergence study (`--n-list`, `--decay-time`, `amr.interp`
  overrides). Its thresholds require mass and momentum exact to 1e-10.
* `tests/regression/shearwave_amr` (`test_shearwave_amr.cpp`): fully
  periodic advected decaying shear wave `u_x = A exp(-nu k^2 t)
  sin(k (y - V t))`, `u_y = V = 2A`, so the wave crosses the interfaces and
  the patch corners. The patches are finite in every direction. Checks inf
  and L2 errors and the composite mass and momentum (to 1e-10). `inputs`
  is 2-D (D2Q9), `inputs3d` is 3-D (D3Q19, finite patches in z too).
  `convergence_shearwave_amr.py` is the convergence study.
* CI: `.github/workflows/amr-regression.yml` builds with GNU make and runs
  the four 2-D Couette inputs, the TGV AMR test, the Couette convergence
  study (Ny 32/64/128, order >= 1.8), the steady forced-channel convergence
  study (Ny 16/32/64, t = 1.5, order >= 1.8) and the TGV AMR convergence
  study (N 32/64/128, order >= 1.8), plus the 3-D Couette case.

## 5. Results

All with tau_0 = 0.8, inf-norm composite error / U_ref, `amr.interp =
consistent` unless stated.

### 5.1 Couette startup (nu t/H^2 = 0.05, tau pinned)

| Ny (level 0) | AMR, `inputs` (3 levels, wall) | AMR, `inputs_band` | AMR, bottom wall refined | single level |
| --- | --- | --- | --- | --- |
| 16  | 1.38e-3 | 2.84e-3 | 3.04e-3 | 3.1e-3 |
| 32  | 3.82e-4 | 7.28e-4 | 7.71e-4 | 7.9e-4 |
| 64  | 1.00e-4 | 1.83e-4 | 1.94e-4 | 2.0e-4 |
| 128 | 2.56e-5 | – | – | – |

Orders 1.85, 1.93, 1.97 / 1.96, 1.99 / 1.98, 1.99. With `lincc` the
`inputs` orders were 1.30, 1.86. With `quadratic` they were 1.27, 1.19,
and the band case gave 1.40, 1.26. Steady Couette (40000 steps) is exact
(~3e-12) for every interpolater, because the populations are linear in y.

Long runs (5000 steps): `inputs` 8.9e-6 (single level 1.8e-5),
`inputs_band` 1.3e-5, `inputs_regrid` 1.5e-5, 3-D `inputs3d` 8.9e-6. Mass
drift stays at about 4e-13.

### 5.2 Forced channel (Poiseuille, `inputs_poiseuille`, interface on the centreline)

Steady state (t = 1.5):

| Ny | consistent | quadratic | single level |
| --- | --- | --- | --- |
| 16 | 1.95e-3 | 7.23e-3 | 2.03e-3 |
| 32 | 4.98e-4 | 3.56e-3 | 5.08e-4 |
| 64 | 1.26e-4 | 1.77e-3 | 1.27e-4 |

That is order 2 for `consistent` (the error is the coarse wall's BGK
slip, the same as single level) and order 1 for `quadratic`. `lincc`
gives 1.19e-2 at Ny = 32 (slope clipped on the centreline). With
`quadratic` the relative error grows as tau -> 1/2 (Ny = 32: 8.2e-3 at
tau 0.6, 3.6e-3 at 0.8, 1.5e-3 at 1.0), consistent with the O(h/nu)
momentum-source mechanism of 3.3. With `lincc`, a band whose edge is away
from the centreline is fine; any edge at y = 0.5 fails.

Regression run (5000 steps) by interpolater:

| Case | consistent | lincc | cell_cons | quadratic | bilinear |
| --- | --- | --- | --- | --- | --- |
| `inputs` | 8.9e-6 | 8.7e-6 | 8.9e-6 | 9.5e-5 | 3.4e-4 |
| `inputs_band` | 1.27e-5 | 1.55e-5 | 1.29e-5 | 8.8e-5 | 3.0e-4 |
| `inputs_regrid` | 1.49e-5 | 1.55e-5 | 1.67e-5 | 3.35e-5 | 8.5e-5 |
| `inputs_poiseuille` | 4.98e-4 | 1.14e-2 | 1.14e-2 | 3.38e-3 | 1.31e-2 |

(single-level Poiseuille: 5.07e-4.)

### 5.3 Taylor-Green vortex, 2-D AMR (section 4, `tgv2d_amr`)

Level-0 N = 32/64/128/256, tau_0 = 0.8 pinned (`nu = 0.1`), static patches
`[1/8,5/8]x[1/4,3/4]` (level 1) and `[1/4,1/2]x[3/8,5/8]` (level 2), off-centre
so no interface lies on a symmetry line. Diffusive scaling: same
`nu k^2 t = 0.5` at every N, composite error relative to the decayed
amplitude.

| N | consistent inf | order | consistent L2 | order | quadratic inf | order | quadratic L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.01e-2 | – | 5.32e-3 | – | 1.93e-2 | – | 9.51e-3 |
| 64 | 2.55e-3 | 1.99 | 1.32e-3 | 2.01 | 7.47e-3 | 1.37 | 3.50e-3 |
| 128 | 6.35e-4 | 2.01 | 3.29e-4 | 2.01 | 3.26e-3 | 1.20 | 1.46e-3 |
| 256 | 1.59e-4 | 2.00 | 8.20e-5 | 2.00 | – | – | – |

(Defaults: `amr.init_fneq = 1`, `amr.reflux = 1`. Before the CE start the
consistent column was 1.37e-2/3.26e-3/8.43e-4/2.00e-4; the order is the
same, but the start-up acoustic error is smaller.) L2 is the
volume-weighted rms `sqrt(<e^2>)` of `e = max(|du|,|dv|)/amp` on the
composite solution. Single level at N = 64: inf 3.24e-3, L2 2.18e-3 (the
AMR L2 is lower because the patches are resolved more finely).

`consistent` is second order and matches the single-level inf error.
`quadratic` degrades to first order, the same failure mode as 5.2. See 5.4
for conservation and the stress cases.

**3-D comparison:** `tgv2d_amr` is 2-D only (the TGV `D2Q9` vortex has no
3-D analogue in this harness). The AMR machinery itself is
dimension-agnostic: `AmrSimulationImpl<D2Q9/D3Q15/D3Q19/D3Q27, BGK>` shares
`AmrInterface.cpp` and `AmrSimulation.cpp`. 3-D coverage:

* `couette_amr/inputs3d` (`D3Q19`, 3 levels at the moving wall): 8.49e-6,
  mass 4.6e-13, the same as 2-D.
* `shearwave_amr/inputs3d` (`D3Q19`, patches finite in x, y and z, so
  there are interfaces, edges and corners in every direction; advected
  wave). The wave does not depend on z, so the 3-D run must reproduce 2-D,
  and it does: N = 32 inf 4.957e-3 / L2 3.218e-3 (2-D: 4.953e-3 /
  3.214e-3), N = 64 1.237e-3 / 8.37e-4 (2-D: 1.224e-3 / 7.94e-4), so the
  3-D order is 2.00 / 1.94. Mass and momentum are exact (1e-14 to 1e-13)
  with reflux; without it they are 1.9e-8 / 6.6e-6 at N = 32. (At N = 16
  the z extent equals the blocking factor, so the patch geometry differs and
  the 16 -> 32 L2 order is not meaningful.)

No 3-D-specific AMR defect has been observed.

### 5.4 Conservation and stress tests

Conservation is measured on `sum f` and `sum c f` of the post-collision
populations (3.9). Relative to the initial mass and to `u0 V` (V = level-0
volume in cells) respectively. "exact" means below 1e-12 (the totals are
summed in double precision over up to 1e5 cells).

**Composite mass / momentum drift by reflux mode** (end of run):

| Case | reflux 0 (none) | reflux 1 (global, default) | reflux 2 (local) |
| --- | --- | --- | --- |
| TGV N = 32/64/128, momentum | 2.9e-4 / 7.3e-5 / 1.8e-5 | exact | exact |
| TGV + advection + regrid every 8 steps (N = 64) | mass 1.2e-8, mom 2.2e-5 | < 1e-9 (print precision) | – |
| shearwave 2-D (N = 64) | mass 2.6e-9, mom 8.6e-7 | 4.1e-14, 1.9e-13 | – |
| shearwave 3-D (N = 32) | mass 1.9e-8, mom 6.6e-6 | 1.1e-14, 4.5e-14 | – |
| TGV N = 256 (8300 steps), momentum | – | 1.4e-12 | – |
| `inputs_normal` (2000 steps) | mass 2.9e-5 | 1.8e-13 | 1.8e-13 |
| Poiseuille, interfaces normal to walls, fixed Re, N = 16/32/64/128 | mass 5.3e-4 / 9.8e-5 / 1.6e-5 / 2.4e-6 | exact | exact |
| `couette_amr` inputs / band / regrid / poiseuille | 4e-13 (interfaces parallel to walls, no corners) | 4e-13 | – |

The TGV momentum defect without reflux is O(h^2) relative to `u0 V`. The
mass drift where the interfaces meet walls converges at about 2.5 order at
fixed Re. It is secular in time, which matters for long runs. With
reflux both are round-off. `quadratic`/`bilinear` change the totals once
when fine cells are created (3.3) and hold them afterwards.

**Galilean check:** TGV with a uniform background velocity (0.5 `u0` in x)
keeps the momentum `P0 = 1.6` exact (defect 2e-15) and the error equals the
co-moving run (inf 1.013e-2 vs 1.0125e-2 at N = 32).

**Order of accuracy by reflux mode.** TGV (periodic, inf / L2):

| N | none | global | local |
| --- | --- | --- | --- |
| 32 | 9.60e-3 / 5.26e-3 | 1.01e-2 / 5.32e-3 | 9.51e-3 / 5.38e-3 |
| 64 | 2.39e-3 / 1.31e-3 | 2.55e-3 / 1.32e-3 | 2.33e-3 / 1.33e-3 |
| 128 | 5.97e-4 / 3.25e-4 | 6.35e-4 / 3.29e-4 | 5.85e-4 / 3.31e-4 |

All three are second order (2.0). Shear wave (advected through the
interfaces, global), 2-D N = 32/64/128: inf 4.95e-3 / 1.22e-3 / 3.05e-4,
L2 3.21e-3 / 7.94e-4 / 1.97e-4 (orders 2.02, 2.00 / 2.02, 2.01). With
advection plus vorticity tagging and regridding every 8 steps (`tgv.bg = 40 20`, `n_error_buf = 2`), global:
inf 4.21e-2 / 1.00e-2 / 2.48e-3, L2 2.61e-2 / 6.43e-3 / 1.59e-3 (orders
2.07, 2.02 / 2.02, 2.02). Without reflux N = 64 gives the same error
(1.000e-2), so moving interfaces do not cost accuracy either.

Walled, steady forced channel with the interfaces **normal** to the walls
(`inputs_normal` geometry; every interface ends on a wall, plus an interior
corner), tau pinned at 0.8 (diffusive scaling, Re = 4, 3 N^2 steps):

| N | none | global | local | single level |
| --- | --- | --- | --- | --- |
| 16 | 1.767e-3 | 1.748e-3 | 3.99e-3 | 1.99e-3 |
| 32 | 4.73e-4 | 4.71e-4 | 1.94e-3 | 5.02e-4 |
| 64 | 1.23e-4 | 1.22e-4 | 9.58e-4 | 1.26e-4 |
| order | 1.90, 1.95 | 1.89, 1.95 | 1.04, 1.01 | 1.99, 1.99 |

This is the case that decides the default. Local reflux is first order:
it injects the O(1) corner mismatch of 3.9 as a source/sink pair at
every wall/interface corner. Global reflux is as accurate as no reflux,
slightly better than single level (the walls near the patch are resolved
finer), and exact in mass. At fixed Re (acoustic scaling, so tau - 1/2
grows like N) the error stalls at ~4.6e-4 from N = 32 on for none/global.
Single level does not converge either (5.05e-4, 2.24e-4, 4.07e-4 at N =
32/64/128). That is the known BGK halfway-bounce-back wall slip, O((tau -
1/2)^2 h^2) = O(1) here, not an AMR effect. Couette with the interfaces at the moving wall (corner
`|u| = U`): none 5.43e-5, global 2.94e-5, local 3.1e-2.

## 6. Status log

* (step 1) Design written.
* (step 2) `AmrSimulationImpl` implemented (fused path only); single-level
  path verified unchanged (13/13 ctests).
* (step 3) Design review (3 lenses + adversarial verify): the only
  "major" finding (stale coarse data at t_new under the fine level used
  for the mid-step ghost fill) was refuted as standard Berger-Oliger
  subcycling error of O(dx^2); the periodic-ghost concern does not apply
  (FillPatchSingleLevel ParallelCopies valid coarse data with
  `geom.periodicity()`).
* (step 4) `couette_amr` regression added; vorticity tagging uses a
  sentinel to skip stencils that reach unfilled ghosts; `lincc_interp`
  chosen over `cell_cons_interp` (superseded in step 6).
* (step 5) 3-D (D3Q19) verified; `test_amr_convert` unit test;
  `inputs_poiseuille` exposed a first-order steady error at a centreline
  interface for every AMReX interpolater (5.2).
* (step 6) Root cause: the ghost interpolation must invert average-down
  (3.3). Added `AvgConsistentInterp` and made it the default
  (`amr.interp = consistent`). Poiseuille is now second order and as
  accurate as single level, and Couette convergence improved.
  2-D ctest 18/18 and 3-D ctest 13/13 pass.
* (step 7) 2-D TGV AMR regression (`tgv2d_amr`) to cover the genuinely
  2-D code paths (cross terms, multi-box patch corners). Convergence is
  second order with `consistent` and first order with `quadratic`, matching
  5.2. 3-D comparison via the existing `couette_amr/inputs3d` (`D3Q19`)
  confirms the same AMR kernels (3 cross-term pairs vs. 1 in 2-D) behave
  identically; no 3-D-only defect found. Adversarial review of
  `AvgConsistentInterp` (math / BC / impl lenses) returned no findings.
* (step 8) Chapman-Enskog initial fneq (`amr.init_fneq`, 3.6): removes the
  start-up level mismatch and its acoustic error (TGV inf error 25% lower,
  same order).
* (step 9) Conservation stress test. Without correction the composite
  mass and momentum drift (TGV momentum O(h^2); mass secularly wherever an
  interface meets a wall or turns a corner). Added crossing registers and
  reflux (3.9). The per-cell (local) correction is exact but first order in
  walled flows (corner topology mismatch). The global correction is exact
  and keeps every case second order, so it is the default. New tests:
  `couette_amr/inputs_normal`, `shearwave_amr` (2-D and 3-D, mass and
  momentum to 1e-10); TGV thresholds tightened to 1e-10. Stress results in
  5.4 (reflux modes, regridding with advection, Galilean shift, tau-pinned
  and fixed-Re walled convergence, 3-D). 2-D ctest 21/21 and 3-D ctest
  14/14 pass.
