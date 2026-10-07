// Equilibrium.cpp — Implementations of the Maxwell-Boltzmann equilibrium.
// The second-order expansion reads:
//   f^eq_q = w_q * rho * [1 + (c_q·u)/cs2 + (c_q·u)^2/(2*cs2^2) - u^2/(2*cs2)]
// This is valid for Ma << 1; higher-order corrections can be added later.

#include "Equilibrium.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── compute kernel ─────────────────────────────────────────────────────
template <typename LatticeTag>
void computeEquilibrium(const amrex::Box& box,
                               amrex::Array4<Real const> const& rho,
                               amrex::Array4<Real const> const& vel,
                               amrex::Array4<Real> const& feq_arr) {

        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k ) {

            // calc u squared
            amrex::Real u_sq = 0.;
            for (int idim=0; idim < AMREX_SPACEDIM; ++idim){
                u_sq += vel(i, j, k, idim)*vel(i, j, k, idim);
            }

            
            // calc equilib for each comp
            for (int n=0; n < LatticeTag::Q; ++n){

                amrex::Real cdotu = LatticeTag::cx(n) * vel(i, j, k, 0)
                                  + LatticeTag::cy(n) * vel(i, j, k, 1)
#if (AMREX_SPACEDIM == 3)
                                  + LatticeTag::cz(n) * vel(i, j, k, 2)
#endif
                ;

                feq_arr(i, j, k, n) = LatticeTag::weights(n) * rho(i, j, k) * 
                                      (1 + LatticeTag::cs2inv * cdotu 
                                         + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu*cdotu 
                                         - 0.5 * LatticeTag::cs2inv * u_sq);
            }
                                  
        });
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

        amrex::Array4<Real> const& feq_arr = feq_mf[mfi].array();
        amrex::Array4<Real const> const& rho = rho_mf[mfi].const_array();
        amrex::Array4<Real const> const& vel = vel_mf[mfi].const_array();
        computeEquilibrium(tileBox, rho, vel, feq_arr);

    }
}

// ── Explicit instantiations ───────────────────────────────────────────────
template amrex::Real computeEquilibrium<D2Q9> (const amrex::Box&, amrex::Array4<const Real> const&, amrex::Array4<Real const> const&, amrex::Array4<Real> const&);
template amrex::Real computeEquilibrium<D3Q15>(const amrex::Box&, amrex::Array4<const Real> const&, amrex::Array4<Real const> const&, amrex::Array4<Real> const&);
template amrex::Real computeEquilibrium<D3Q19>(const amrex::Box&, amrex::Array4<const Real> const&, amrex::Array4<Real const> const&, amrex::Array4<Real> const&);
template amrex::Real computeEquilibrium<D3Q27>(const amrex::Box&, amrex::Array4<const Real> const&, amrex::Array4<Real const> const&, amrex::Array4<Real> const&);

template void computeEquilibriumMF<D2Q9> (const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);

} // namespace muphfasa
