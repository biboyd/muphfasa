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

#ifndef MUPHFASA_AMR_INTERFACE_H
#define MUPHFASA_AMR_INTERFACE_H

#include <AMReX_Array.H>
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

} // namespace muphfasa

#endif // MUPHFASA_AMR_INTERFACE_H
