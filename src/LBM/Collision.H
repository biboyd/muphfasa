// Collision.H — Interface for LBM collision operators.
// Currently supports BGK (Bhatnagar-Gross-Krook) single-relaxation-time.
// Additional models (TRT, MRT, regularised) can be added as new policy types.

#ifndef MUPHFASA_COLLISION_H
#define MUPHFASA_COLLISION_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Collision model tag types ─────────────────────────────────────────────
struct BGK {};   // single-relaxation-time
struct TRT {};   // two-relaxation-time     (not yet implemented)
struct MRT {};   // multiple-relaxation-time (not yet implemented)

// ── BGK cell-local kernel ─────────────────────────────────────────────────

// f_q(t+1) = f_q - (1/tau) * (f_q - f^eq_q)
// tau: BGK relaxation time (tau = nu/cs2 + 0.5 in lattice units)
// f and feq are arrays of length LatticeTag::Q.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collideBGK(amrex::Real*       f,
                const amrex::Real* feq,
                amrex::Real        tau) noexcept;

// ── MultiFab-level collision (host-side) ─────────────────────────────────

// Apply the collision operator in-place to f_mf given feq_mf and tau.
// Dispatches on CollisionModel tag (currently only BGK is active).
template <typename LatticeTag, typename CollisionModel>
void collide(amrex::MultiFab&       f_mf,
             const amrex::MultiFab& feq_mf,
             amrex::Real            tau);

} // namespace muphfasa

#endif // MUPHFASA_COLLISION_H
