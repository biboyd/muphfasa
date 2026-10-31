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
| bilinear`). `cell_cons` also shows a mass drift of ~1e-7 on coarse grids.

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
coarse step. (Overlap / "coarse computes everywhere" approach. Mass is
conserved to truncation error, not to round-off; tracked in tests.)

### 3.5 Time-stepping (per level-0 step)

    timeStep(l):
      if l > 0: fillGhosts(l, t_l)          // FillPatchTwoLevels (3.3)
      swap(fpre_old[l], fpre_new[l])
      fused AMR kernel: fin[l] -> fout[l], fpre_new[l]; swap(fin, fout)
      t_old[l] = t_l;  t_l += dt_l
      if l < finest: timeStep(l+1) twice;  averageDown(l+1 -> l)   // 3.4
      if l == 0: FillBoundary + physbc on level 0

Ghosts are filled at the *start* of each fine substep (so they use the
already averaged coarse data at `t_old`), and once more for all levels
before output.

### 3.6 Initialisation, tagging, regridding

* Level 0 is initialised with `initData` exactly as the single-level
  solver: `fin = fpre = feq(rho, u)` (plus Guo correction when forced).
  Finer levels are interpolated from converted coarse data (3.3), so an
  `initData` override only has to handle level 0.
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
| `amr.interp` | consistent | coarse->fine interpolater (3.3); `quadratic`, `lincc`, `cell_cons`, `bilinear` for comparison |
| `tagging.box_lo_<l>` / `tagging.box_hi_<l>` | – | physical refine box for level l+1 |
| `tagging.vort_thresh` | – | vorticity tagging threshold |

`grid.max_grid_size` is reused for every level. AmrCore is built through
the `(Geometry, AmrInfo)` constructor so it reads **no** ParmParse keys
(no clash with existing `amr.plot_int`, `geometry.*`).

### 3.8 Output

Multi-level plotfiles (`WriteMultiLevelPlotfile`) in physical units with
the same variable names as single-level. Checkpoint: level-0 `f` only
(restart is not supported by either solver).

## 4. Testing

* All existing tests unchanged (max_level = 0); verified with ctest in a
  2D test build after every solver change.
* `tests/regression/couette_amr` (`test_couette_amr.cpp`): plane
  Couette(-Poiseuille) startup vs. the analytic series, evaluated on the
  **composite** solution (each level's cells not covered by a finer level,
  analytic profile at `y = (j+1/2) 2^-l`). Also tracks composite mass
  `sum rho dV` and the finest level reached. Fails via exit code when
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
* `tests/unit/test_amr_convert.cpp`: conversion round trip, rho/u
  invariance, `collide(pre)` consistency (2D and 3D).
* `convergence_amr.py`: grid-convergence order of the composite error
  (`--inputs`, `--target-time`, `--exe`, extra `KEY=VALUE` overrides).
* CI: `.github/workflows/amr-regression.yml` builds with GNU make and runs
  the four 2-D inputs, the Couette convergence study (Ny 32/64/128, order
  >= 1.8) and the steady forced-channel convergence study (Ny 16/32/64,
  t = 1.5, order >= 1.8), plus the 3-D Couette case.

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
