// InitData.cpp — Advected decaying shear wave initial condition.
//
// Overrides src/Simulation/InitData.cpp: see tgv2d/InitData.cpp for the
// VPATH/CMake mechanism. Only level 0 is initialised here; finer levels are
// interpolated from it by AmrSimulation.
//
// Exact incompressible NS solution (fully periodic):
//   u_x = A exp(-nu k^2 t) sin(k (y - V t)),  u_y = V,  u_z = 0,  rho = 1
// with k = 2 pi / Ny,  Ny = number of level-0 cells in y.
// At t = 0: u_x = A sin(k y), u_y = V.
// Positions in level-0 lattice units: cell (i,j,k) at (i+1/2, j+1/2, k+1/2).
// A = units.toLatticeVelocity(params.refVelocity()), V = bg_frac * A where
// bg_frac is read from ParmParse "shearwave" (default 0).

#include "InitData.H"

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include "IO/InputParser.H"
#include "LBM/Units.H"

#include <cmath>

namespace muphfasa {

void initData(amrex::MultiFab& rho, amrex::MultiFab& vel, const UnitConverter& units)
{
    InputParser params;
    amrex::Real A = units.toLatticeVelocity(params.refVelocity());

    amrex::Real bg_frac = 0.0;
    {
        amrex::ParmParse pp("shearwave");
        pp.query("bg_frac", bg_frac);
    }
    amrex::Real V = bg_frac * A;

    const amrex::Box domain = rho.boxArray().minimalBox();
    // k uses the y-extent (one wavelength per Ny cells)
    const int Ny = domain.length(1);
    const amrex::Real k = amrex::Real(2.0) * amrex::Real(3.14159265358979323846) / amrex::Real(Ny);

    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.validbox();
        auto const& r = rho.array(mfi);
        auto const& u = vel.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k_) noexcept {
            const amrex::Real y = amrex::Real(j) + amrex::Real(0.5);
            const amrex::Real ux = A * std::sin(k * y);
            r(i, j, k_, 0) = amrex::Real(1.0);
            u(i, j, k_, 0) = ux;
            u(i, j, k_, 1) = V;
#if AMREX_SPACEDIM == 3
            u(i, j, k_, 2) = amrex::Real(0.0);
#endif
        });
    }
}

} // namespace muphfasa
