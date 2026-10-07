// Collision.cpp — Implementations of LBM collision operators.
// Currently only BGK is active; the TRT and MRT stubs are placeholders
// for future extension.

#include "Collision.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── BGK cell-local kernel ─────────────────────────────────────────────────
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collideBGK(amrex::Real*       f,
                const amrex::Real* feq,
                amrex::Real        omega_inv) noexcept
{
    return (1 - omega) * f + omega * feq;
}

// ── MultiFab-level dispatch ───────────────────────────────────────────────
template <typename LatticeTag>
void collide(amrex::MultiFab&       f_mf,
             const amrex::MultiFab& feq_mf,
             amrex::Real            omega)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (MFIter mfi(f_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        Array4<Real> const& f_arr= f_mf[mfi].array();
        Array4<Real const> const& feq_arr = feq_mf[mfi].const_array();
        ParallelFor(tileBox, LatticeTag::Q, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

            // collide
            f_arr(i, j, k, n) = collideBGK(f_arr(i, j, k, n), 
                                           feq_arr(i, j, k, n), omega);
        }
    }
}

// ── Explicit instantiations — BGK ────────────────────────────────────────
template void collideBGK<D2Q9> (amrex::Real*, const amrex::Real*, amrex::Real) noexcept;
template void collideBGK<D3Q15>(amrex::Real*, const amrex::Real*, amrex::Real) noexcept;
template void collideBGK<D3Q19>(amrex::Real*, const amrex::Real*, amrex::Real) noexcept;
template void collideBGK<D3Q27>(amrex::Real*, const amrex::Real*, amrex::Real) noexcept;

template void collide<D2Q9, BGK> (amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collide<D3Q15,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collide<D3Q19,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collide<D3Q27,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);

} // namespace muphfasa
