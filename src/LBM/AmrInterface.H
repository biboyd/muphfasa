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
//  * AvgConsistentInterp: coarse -> fine interpolater for the populations
//    (amr.interp = consistent, the default; amr_implementation.md, 3.3).

#ifndef MUPHFASA_AMR_INTERFACE_H
#define MUPHFASA_AMR_INTERFACE_H

#include <AMReX_Array.H>
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
