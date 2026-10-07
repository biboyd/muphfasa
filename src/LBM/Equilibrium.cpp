// Equilibrium.cpp — Implementations of the Maxwell-Boltzmann equilibrium.
// The second-order expansion reads:
//   f^eq_q = w_q * rho * [1 + (c_q·u)/cs2 + (c_q·u)^2/(2*cs2^2) - u^2/(2*cs2)]
// This is valid for Ma << 1; higher-order corrections can be added later.

#include "Equilibrium.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── Cell-local kernel ─────────────────────────────────────────────────────
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibrium( const int         n,
                                const amrex::Real rho,
                                const amrex::Real u_sq,
                                const amrex::Real u_x,
                                const amrex::Real u_y,
#if (AMREX_DIM == 3)
                                const amrex::Real uz
#endif
                            ) noexcept
{
    amrex::Real cdotu = LatticeTag.cx(n) * u_x + LatticeTag.cy(n) * u_y
#if (AMREX_DIM == 3)
                        + LatticeTag.cz(n) * u_z
#endif
                        ;
    return LatticeTag.weights(n) * rho * (1 + LatticeTag.cs2inv * cdotu 
                                                         + 0.5*LatticeTag.cs2inv*LatticeTag.cs2inv * cdotu*cdotu 
                                                         + 0.5 * LatticeTag.cs2inv * u_sq);
}

// ── MultiFab-level helper ─────────────────────────────────────────────────
template <typename LatticeTag>
void computeEquilibriumMF(const amrex::MultiFab& rho_mf,
                           const amrex::MultiFab& vel_mf,
                           amrex::MultiFab&       feq_mf)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (MFIter mfi(feq_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        Array4<Real> const& feq_arr = feq_mf[mfi].array();
        Array4<Real const> const& rho = rho_mf[mfi].const_array();
        Array4<Real const> const& vel = vel_mf[mfi].const_array();
        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k ) {

            // calc u squared
            amrex::Real u_sqr = 0.;
            for (int idim=0; idim < AMREX_DIM; ++idim){
                u_sqr += vel(i, j, k, l)*vel(i, j, k, idim);
            }

            
            // calc equilib for each comp
            for (int n=0; n < LatticeTag.Q; ++n){

                amrex::Real cdotu = LatticeTag.cx(n) * u_x 
                                  + LatticeTag.cy(n) * u_y
#if (AMREX_DIM == 3)
                                  + LatticeTag.cz(n) * u_z
#endif
                ;

                feq_arr(i, j, k, n) = LatticeTag.weights(n) * rho(i, j, k) * 
                                      (1 + LatticeTag.cs2inv * cdotu 
                                         + 0.5 * LatticeTag.cs2inv * LatticeTag.cs2inv * cdotu*cdotu 
                                         + 0.5 * LatticeTag.cs2inv * u_sq);
            }
                                  
        }
    }
}

// ── Explicit instantiations ───────────────────────────────────────────────
template amrex::Real computeEquilibrium<D2Q9> (int, amrex::Real, const amrex::Real*) noexcept;
template amrex::Real computeEquilibrium<D3Q15>(int, amrex::Real, const amrex::Real*) noexcept;
template amrex::Real computeEquilibrium<D3Q19>(int, amrex::Real, const amrex::Real*) noexcept;
template amrex::Real computeEquilibrium<D3Q27>(int, amrex::Real, const amrex::Real*) noexcept;

template void computeEquilibriumMF<D2Q9> (const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);

} // namespace muphfasa
