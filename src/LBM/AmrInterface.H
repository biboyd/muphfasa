// AmrInterface.H — Kernels used only by the multi-level (AMR) solver.
//
//  * pullCollideFusedPreMF: the fused pull-stream-collide step of
//    Collision.H, which additionally stores the streamed (pre-collision)
//    populations. The AMR coarse/fine coupling rescales pre-collision
//    non-equilibrium parts (see amr_implementation.md, section 3.2), which
//    is non-singular for every tau, unlike the post-collision form.
//
//  * convertPopulationsMF: converts pre-collision populations from a source
//    level (tau_s, F_s) to the pre- and/or post-collision populations of a
//    target level (tau_t, F_t) under acoustic scaling:
//
//        P      = (tau_s - 1/2) G(u).F_s          (Guo force part of fneq)
//        fneq_t = r (fneq_s - P) + P,   r = tau_t dt_t / (tau_s dt_s)
//        pre_t  = feq + fneq_t
//        post_t = pre_t - omega_t (pre_t - feq) + (1 - omega_t/2) G(u).F_t
//
//    with u the Guo half-force velocity (sum c f + F_s/2)/rho. rho and u are
//    level invariant in lattice units under acoustic scaling.
//
//  * addInitialNonEquilibriumMF: adds the Chapman-Enskog non-equilibrium
//    part of an initial velocity field to equilibrium populations.
//
//  * accumulateCoarseCrossingsMF / accumulateFineCrossingsMF /
//    applyRefluxMF: exact mass and momentum refluxing across coarse/fine
//    boundaries (amr.reflux; amr_implementation.md, 3.9).
//
//  * AvgConsistentInterp: coarse -> fine interpolater for the populations
//    (amr.interp = consistent, the default; amr_implementation.md, 3.3).

#ifndef MUPHFASA_AMR_INTERFACE_H
#define MUPHFASA_AMR_INTERFACE_H

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// Parameters of one level, as needed by the conversion kernels.
struct LevelParams {
    amrex::Real tau = amrex::Real(1.0);
    amrex::Real dt  = amrex::Real(1.0);   // in level-0 lattice time units
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> F{};
};

// Fused pull-stream-collide (BGK, optional Guo forcing) that also writes
// the streamed pre-collision populations to fpre_mf. fout = collide(fpre).
// fin_mf must have its ghost cells filled.
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedPreMF(const amrex::MultiFab& fin_mf,
                           amrex::MultiFab& fout_mf,
                           amrex::MultiFab& fpre_mf,
                           amrex::Real omega,
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F,
                           bool has_forcing);

// Convert pre-collision populations src_pre (level src) into target level
// populations. post_mf (Q comps at post_comp) and/or pre_mf (Q comps at
// pre_comp) may be nullptr. Applied on every cell of each destination fab
// (valid + ghost) that src_pre also holds; src_pre and the destinations
// must share BoxArray and DistributionMapping.
template <typename LatticeTag>
void convertPopulationsMF(const amrex::MultiFab& src_pre,
                          amrex::MultiFab* post_mf, int post_comp,
                          amrex::MultiFab* pre_mf, int pre_comp,
                          LevelParams const& src, LevelParams const& tgt,
                          int ngrow);

// Adds the first-order Chapman-Enskog non-equilibrium part of the velocity
// field vel (lattice units, dx = dt = 1 on this level),
//
//     fneq_n = -tau w_n rho / cs^2  Q_n:grad(u),   Q_n = c_n c_n - cs^2 I,
//
// to pre-collision populations fpre and (1 - 1/tau) fneq_n to post-collision
// populations fpost, which must both hold feq(rho, u) on entry. grad(u) is
// taken by central differences (one-sided next to a non-periodic domain
// face); vel needs one ghost layer filled across box and periodic
// boundaries. Starting from feq alone leaves every level to build its own
// initial layer at its own tau and dt, so the coarse/fine interfaces are
// inconsistent for the first steps and radiate acoustic waves.
template <typename LatticeTag>
void addInitialNonEquilibriumMF(const amrex::MultiFab& rho, const amrex::MultiFab& vel,
                                amrex::MultiFab& fpre, amrex::MultiFab& fpost,
                                amrex::Real tau, const amrex::Geometry& geom);

// Coarse/fine refluxing. Over one coarse step the composite (uncovered
// coarse + fine) mass changes only through populations crossing the
// coarse/fine boundary, and the two levels see those crossings differently
// (in particular where the boundary meets a wall or a corner of the fine
// region). The registers (components: mass, then momentum) collect, per
// uncovered coarse cell U,
//
//   coarse: sum of f*_i(U - c_i) over covered U - c_i   (gained from covered)
//         - sum of f*_i(U)       over covered U + c_i   (lost to covered)
//   fine:   weight * [ ghost populations pulled into fine valid cells
//                    - fine populations streamed into ghost cells ],
//           each attributed to the coarse link it corresponds to (see
//           addFineCrossing in AmrInterface.cpp),
//
// all evaluated on the post-collision populations with ghosts filled, just
// before the (sub)step's stream. Masks: cmask is 1 on covered coarse cells,
// 0 on uncovered cells in the domain, -1 outside it; fmask (1 ghost) is 1 on
// fine valid cells, 0 on coarse/fine ghosts, -1 outside the domain. Coarse
// links leaving the domain are excluded: bounce-back returns those
// populations to the cell they left.
template <typename LatticeTag>
void accumulateCoarseCrossingsMF(const amrex::MultiFab& fin, const amrex::iMultiFab& cmask,
                                 amrex::MultiFab& reg);
// Refinement ratio 2. reg and cmask are defined on coarsen(fine grids) with
// two ghost cells; contributions land in reg's ghost cells.
template <typename LatticeTag>
void accumulateFineCrossingsMF(const amrex::MultiFab& fin, const amrex::iMultiFab& fmask,
                               const amrex::iMultiFab& cmask, amrex::MultiFab& reg,
                               amrex::Real weight);
// Removes the accumulated composite change (reg) from each uncovered coarse
// cell as an equilibrium-shaped increment w_i (drho + c_i.dj / cs2), which
// carries exactly mass drho and momentum dj, in both the post- and the
// pre-collision populations.
template <typename LatticeTag>
void applyRefluxMF(const amrex::MultiFab& reg, const amrex::iMultiFab& cmask,
                   amrex::MultiFab& fpost, amrex::MultiFab& fpre);

// Coarse -> fine interpolation consistent with average_down. Covered coarse
// cells hold the mean of their fine children, so near the coarse/fine
// boundary a coarse value is c = P + sum_d <xi_d^2> P_dd / 2 + O(h^4) for the
// smooth fine field P, with xi_d the child offsets in coarse cell units and
// <xi_d^2> = (r^2 - 1) / (12 r^2). Inverting this exactly to third order gives
//
//     f(xi) = c + sum_d [ s_d xi_d + k_d (xi_d^2 - <xi_d^2>) / 2 ]
//               + sum_{d<e} m_de xi_d xi_e
//
// with central first (s), second (k) and mixed (m) differences of c. For
// r = 2 the k terms vanish. The scheme is unlimited and conservative (the
// children average back to c). Pointwise interpolaters (quadratic,
// bilinear) and limited ones (lincc and cell_cons at an extremum) make an
// O(h^2) error in the ghost populations. That error acts as a momentum
// source at the interface and gives first-order, 1/nu-amplified velocity
// errors in steady forced flows. Next to a non-periodic domain face, the
// differences become one-sided.
class AvgConsistentInterp : public amrex::Interpolater
{
public:
    amrex::Box CoarseBox(const amrex::Box& fine, int ratio) override;
    amrex::Box CoarseBox(const amrex::Box& fine, const amrex::IntVect& ratio) override;
    void interp(const amrex::FArrayBox& crse, int crse_comp,
                amrex::FArrayBox& fine, int fine_comp, int ncomp,
                const amrex::Box& fine_region, const amrex::IntVect& ratio,
                const amrex::Geometry& crse_geom, const amrex::Geometry& fine_geom,
                amrex::Vector<amrex::BCRec> const& bcr,
                int actual_comp, int actual_state, amrex::RunOn runon) override;
};

AvgConsistentInterp& avgConsistentInterp();

} // namespace muphfasa

#endif // MUPHFASA_AMR_INTERFACE_H
