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
   `(t_old, t_new)` (linear in time) and `lincc_interp` (unlimited,
   conservative, linear in space). The coarse-patch physical BC on
   converted data is first-order extrapolation (only feeds interpolation
   slopes); fine physical ghosts are filled afterward by the real
   `BCManager` at the fine geometry.

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

* Each level is initialised with `initData` on its own grids:
  `fin = fpre = feq(rho, u)` (plus Guo correction when forced).
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
| `tagging.box_lo_<l>` / `tagging.box_hi_<l>` | – | physical refine box for level l+1 |
| `tagging.vort_thresh` | – | vorticity tagging threshold |

`grid.max_grid_size` is reused for every level. AmrCore is built through
the `(Geometry, AmrInfo)` constructor so it reads **no** ParmParse keys
(no clash with existing `amr.plot_int`, `geometry.*`).

### 3.8 Output

Multi-level plotfiles (`WriteMultiLevelPlotfile`) in physical units with
the same variable names as single-level. Checkpoint: per-level `f` written
(restart not supported yet, as in single level).

## 4. Testing plan

* All existing tests unchanged (max_level = 0).
* New regression `couette_amr` (2D): Couette with a static refined band,
  compares all levels to the analytic profile; error threshold and
  convergence order across resolutions.
* Poiseuille / uniform-force with a refined band (forced path).
* Mass-drift check on a periodic AMR case.

## 5. Status log

* (step 1) Design written.
