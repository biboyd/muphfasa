// Streaming.cpp — Implementation of the LBM streaming (propagation) step.
// Performs: f_out(x, q) = f_in(x - c_q, q) for every cell and direction.
// Relies on ghost cells being pre-filled by the caller (FillBoundary or
// FillPatch), so that cells adjacent to MPI boundaries are available.

#include "Streaming.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── Two-MultiFab streaming ────────────────────────────────────────────────
template <typename LatticeTag>
void stream(const amrex::MultiFab& f_in,
            amrex::MultiFab&       f_out,
            const amrex::Geometry& geom)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (MFIter mfi(f_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        Array4<Real> const& fo_arr = f_out[mfi].array();
        Array4<Real const> const& fi_arr = f_in[mfi].const_array();
        ParallelFor(tileBox, LatticeTag.Q, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

            // find node to stream from
            int i_old = i - LatticeTag.cx[n];
            int j_old = j - LatticeTag.cy[n];
#if (AMREX_SPACEDIM == 3)
            int k_old = k - LatticeTag.cz[n];
#endif

            // stream
            fo_arr(i, j, k, n) = fi_arr(i_old, j_old, k_old, n);
        }
    }
}

// ── In-place streaming (using scratch space) ──────────────────────────────
template <typename LatticeTag>
void streamInPlace(amrex::MultiFab&       f,
                   const amrex::Geometry& geom)
{
    amrex::MultiFab f_scratch(f.boxArray(), f.DistributionMap(),
                               f.nComp(), f.nGrow());
    stream<LatticeTag>(f, f_scratch, geom);
    amrex::MultiFab::Copy(f, f_scratch, 0, 0, f.nComp(), 0);
}

// ── Explicit instantiations ───────────────────────────────────────────────
template void stream<D2Q9> (const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q15>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q19>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q27>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);

template void streamInPlace<D2Q9> (amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q15>(amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q19>(amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q27>(amrex::MultiFab&, const amrex::Geometry&);

} // namespace muphfasa
