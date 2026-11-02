// test_shearwave_amr.cpp — Regression / stress test: advected decaying
// shear wave on an AMR hierarchy, 2D (D2Q9) and 3D (D3Q19/etc).
//
// Problem: fully periodic, exact incompressible NS solution
//   u_x = A exp(-nu k^2 t) sin(k (y - V t)),  u_y = V,  u_z = 0,  rho = 1
//   k = 2 pi / Ny,  nu = (tau0 - 1/2)/3,  A = units.toLatticeVelocity(refVelocity),
//   V = shearwave.bg_frac * A.
// Positions/times in level-0 lattice units; acoustic scaling keeps velocity
// level-invariant, level l has spacing 2^-l.
//
// Diagnostics on the composite hierarchy (uncovered cells only, via
// amrex::makeFineMask):
//  (1) composite inf-norm and volume-weighted L2 of |u - u_exact| normalised
//      by the current amplitude A exp(-nu k^2 t) (max over components);
//  (2) mass conservation from post-collision populations sim.f(lev):
//      M = sum_q f_q dV  (covered cells excluded, dV = 2^{-l D});
//  (3) momentum conservation: P_d = sum_q c_{q,d} f_q dV.
// Mass drift = |M-M0|/M0, momentum defect per component = |P_d - P_d0|/(A Vol),
// reported as max over components; Vol = level-0 domain volume in cells.
// The rho/vel MultiFabs are NOT used for conservation (they are taken after
// an extra streaming step that is not conservative at C/F faces).
//
// Thresholds (ParmParse "shearwave", negative = disabled):
//   max_err, max_l2_err, max_mass_drift, max_mom_defect, min_finest

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "LBM/LatticeDescriptor.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <vector>

using namespace muphfasa;

namespace {

constexpr amrex::Real kPi = amrex::Real(3.14159265358979323846);

// 1 on cells of level lev not covered by lev+1, 0 elsewhere.
amrex::MultiFab uncoveredMask(const ISimulation& sim, int lev)
{
    const amrex::MultiFab& v = sim.vel(lev);
    if (lev == sim.finestLevel()) {
        amrex::MultiFab m(v.boxArray(), v.DistributionMap(), 1, 0);
        m.setVal(amrex::Real(1.0));
        return m;
    }
    return amrex::makeFineMask(v.boxArray(), v.DistributionMap(),
                               sim.vel(lev + 1).boxArray(), amrex::IntVect(2),
                               amrex::Real(1.0), amrex::Real(0.0));
}

struct Diagnostics {
    std::vector<amrex::Real> lev_err;
    std::vector<amrex::IntVect> lev_loc;
    amrex::Real err = 0;   // composite inf-norm (normalised)
    amrex::Real l2 = 0;    // composite volume-weighted RMS (normalised)
    amrex::Real fmass = 0; // composite sum f dV
    amrex::Real fmom[AMREX_SPACEDIM] = {};
};

Diagnostics diagnose(const ISimulation& sim, amrex::Real k, amrex::Real A,
                     amrex::Real nu, amrex::Real V)
{
    Diagnostics d;
    const amrex::Real t = sim.time();
    const amrex::Real amp = A * std::exp(-nu * k * k * t);
    const amrex::Real inv_amp = (amp != amrex::Real(0.0)) ? amrex::Real(1.0) / amp : amrex::Real(0.0);

    amrex::Real sq = 0;
    amrex::Real vol = 0;

    for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
        const amrex::MultiFab& vel = sim.vel(lev);
        const amrex::MultiFab& f = sim.f(lev);
        const amrex::MultiFab mask = uncoveredMask(sim, lev);
        const amrex::Real h = amrex::Real(1.0) / amrex::Real(1 << lev);
        const amrex::Real dv = std::pow(h, AMREX_SPACEDIM);

        // Velocity error (normalised)
        amrex::MultiFab err(vel.boxArray(), vel.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(err, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox();
            auto const& e = err.array(mfi);
            auto const& u = vel.const_array(mfi);
            auto const& msk = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kk) noexcept {
                const amrex::Real y = (amrex::Real(j) + amrex::Real(0.5)) * h;
                const amrex::Real y_adv = y - V * t;
                const amrex::Real ux_e = A * std::exp(-nu * k * k * t) * std::sin(k * y_adv);
                const amrex::Real uy_e = V;
#if AMREX_SPACEDIM == 3
                const amrex::Real uz_e = amrex::Real(0.0);
#endif
                amrex::Real emax = 0;
                emax = amrex::max(emax, std::abs(u(i, j, kk, 0) - ux_e));
                emax = amrex::max(emax, std::abs(u(i, j, kk, 1) - uy_e));
#if AMREX_SPACEDIM == 3
                emax = amrex::max(emax, std::abs(u(i, j, kk, 2) - uz_e));
#endif
                if (A != amrex::Real(0.0)) {
                    e(i, j, kk) = msk(i, j, kk) * emax * inv_amp;
                } else {
                    e(i, j, kk) = msk(i, j, kk) * emax;
                }
            });
        }
        const amrex::Real lev_err = err.max(0);
        d.lev_err.push_back(lev_err);
        d.lev_loc.push_back(err.maxIndex(0));
        d.err = amrex::max(d.err, lev_err);
        sq  += amrex::MultiFab::Dot(err, 0, 1, 0) * dv;
        vol += mask.sum(0) * dv;

        // f-based mass and momentum
        const int Q = f.nComp();
        if (Q <= 0) { continue; }
        amrex::MultiFab mom_f(vel.boxArray(), vel.DistributionMap(), 1 + AMREX_SPACEDIM, 0);
        for (amrex::MFIter mfi(mom_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox();
            auto const& p = mom_f.array(mfi);
            auto const& fa = f.const_array(mfi);
            auto const& msk = mask.const_array(mfi);
#if AMREX_SPACEDIM == 2
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real s0 = 0, sx = 0, sy = 0;
                for (int q = 0; q < 9; ++q) {
                    const amrex::Real fq = fa(i, j, k, q);
                    s0 += fq;
                    sx += LatticeDescriptor<2,9>::cx(q) * fq;
                    sy += LatticeDescriptor<2,9>::cy(q) * fq;
                }
                p(i, j, k, 0) = msk(i, j, k) * s0;
                p(i, j, k, 1) = msk(i, j, k) * sx;
                p(i, j, k, 2) = msk(i, j, k) * sy;
            });
#else
            if (Q == 15) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kk) noexcept {
                    amrex::Real s0 = 0, sx = 0, sy = 0, sz = 0;
                    for (int q = 0; q < 15; ++q) {
                        const amrex::Real fq = fa(i, j, kk, q);
                        s0 += fq;
                        sx += LatticeDescriptor<3,15>::cx(q) * fq;
                        sy += LatticeDescriptor<3,15>::cy(q) * fq;
                        sz += LatticeDescriptor<3,15>::cz(q) * fq;
                    }
                    p(i, j, kk, 0) = msk(i, j, kk) * s0;
                    p(i, j, kk, 1) = msk(i, j, kk) * sx;
                    p(i, j, kk, 2) = msk(i, j, kk) * sy;
                    p(i, j, kk, 3) = msk(i, j, kk) * sz;
                });
            } else if (Q == 19) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kk) noexcept {
                    amrex::Real s0 = 0, sx = 0, sy = 0, sz = 0;
                    for (int q = 0; q < 19; ++q) {
                        const amrex::Real fq = fa(i, j, kk, q);
                        s0 += fq;
                        sx += LatticeDescriptor<3,19>::cx(q) * fq;
                        sy += LatticeDescriptor<3,19>::cy(q) * fq;
                        sz += LatticeDescriptor<3,19>::cz(q) * fq;
                    }
                    p(i, j, kk, 0) = msk(i, j, kk) * s0;
                    p(i, j, kk, 1) = msk(i, j, kk) * sx;
                    p(i, j, kk, 2) = msk(i, j, kk) * sy;
                    p(i, j, kk, 3) = msk(i, j, kk) * sz;
                });
            } else if (Q == 27) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kk) noexcept {
                    amrex::Real s0 = 0, sx = 0, sy = 0, sz = 0;
                    for (int q = 0; q < 27; ++q) {
                        const amrex::Real fq = fa(i, j, kk, q);
                        s0 += fq;
                        sx += LatticeDescriptor<3,27>::cx(q) * fq;
                        sy += LatticeDescriptor<3,27>::cy(q) * fq;
                        sz += LatticeDescriptor<3,27>::cz(q) * fq;
                    }
                    p(i, j, kk, 0) = msk(i, j, kk) * s0;
                    p(i, j, kk, 1) = msk(i, j, kk) * sx;
                    p(i, j, kk, 2) = msk(i, j, kk) * sy;
                    p(i, j, kk, 3) = msk(i, j, kk) * sz;
                });
            } else {
                amrex::Abort("test_shearwave_amr: unsupported Q=" + std::to_string(Q));
            }
#endif
        }
        d.fmass += mom_f.sum(0) * dv;
        for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2) {
            d.fmom[d2] += mom_f.sum(1 + d2) * dv;
        }
    }
    d.l2 = (vol > 0) ? std::sqrt(sq / vol) : amrex::Real(0.0);
    return d;
}

void report(const Diagnostics& d, amrex::Real t, amrex::Real mass0,
            const amrex::Real fmom0[], amrex::Real A, amrex::Real Vol)
{
    amrex::Real mass_drift = (mass0 != amrex::Real(0.0)) ? std::abs(d.fmass - mass0) / std::abs(mass0) : amrex::Real(0.0);
    amrex::Real mom_defect = 0;
    if (A != amrex::Real(0.0) && Vol != amrex::Real(0.0)) {
        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
            amrex::Real def = std::abs(d.fmom[dd] - fmom0[dd]) / (std::abs(A) * Vol);
            mom_defect = amrex::max(mom_defect, def);
        }
    }
    amrex::Print() << "At Time=" << t << ": Error=" << d.err << " L2=" << d.l2
                   << "  (levels:";
    for (std::size_t l = 0; l < d.lev_err.size(); ++l) {
        amrex::Print() << " " << d.lev_err[l] << " @" << d.lev_loc[l];
    }
    amrex::Print() << ")  MassDrift=" << mass_drift
                   << "  MomDefect=" << mom_defect
                   << "  Fmom=(";
    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
        if (dd) amrex::Print() << ", ";
        amrex::Print() << d.fmom[dd];
    }
    amrex::Print() << ") Fmass=" << d.fmass << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    {
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        const auto ncells = params.ncells();
        const amrex::Real Ny = amrex::Real(ncells[1]);
        const amrex::Real k = amrex::Real(2.0) * kPi / Ny;
        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu = (tau - amrex::Real(0.5)) / amrex::Real(3.0);
        const amrex::Real A = sim->units().toLatticeVelocity(params.refVelocity());
        amrex::Real bg_frac = 0.0;
        {
            amrex::ParmParse pp("shearwave");
            pp.query("bg_frac", bg_frac);
        }
        const amrex::Real V = bg_frac * A;
        amrex::Real Vol = amrex::Real(1.0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) Vol *= amrex::Real(ncells[d]);

        amrex::Print() << "Ny=" << Ny << " k=" << k << " A=" << A << " V=" << V
                       << " nu=" << nu << " Vol=" << Vol << "\n";

        amrex::Real max_err = -1, max_l2 = -1, max_mass_drift = -1, max_mom_defect = -1;
        int min_finest = 0;
        {
            amrex::ParmParse pp("shearwave");
            pp.query("max_err", max_err);
            pp.query("max_l2_err", max_l2);
            pp.query("max_mass_drift", max_mass_drift);
            pp.query("max_mom_defect", max_mom_defect);
            pp.query("min_finest", min_finest);
        }

        sim->updateMacroscopic();
        Diagnostics d0 = diagnose(*sim, k, A, nu, V);
        const amrex::Real mass0 = d0.fmass;
        amrex::Real fmom0[AMREX_SPACEDIM];
        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) fmom0[dd] = d0.fmom[dd];
        report(d0, sim->time(), mass0, fmom0, A, Vol);
        amrex::Print() << "Finest level: " << sim->finestLevel()
                       << "  initial mass: " << mass0
                       << "  initial fmom=(";
        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
            if (dd) amrex::Print() << ", ";
            amrex::Print() << fmom0[dd];
        }
        amrex::Print() << ")\n";

        const int max_step = params.maxStep();
        const int plot_int = params.plotInterval();
        int max_finest = sim->finestLevel();
        Diagnostics last = d0;
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);
            max_finest = amrex::max(max_finest, sim->finestLevel());
            if (plot_int > 0 && step % plot_int == 0) {
                last = diagnose(*sim, k, A, nu, V);
                report(last, sim->time(), mass0, fmom0, A, Vol);
            }
        }
        if (plot_int <= 0 || max_step % plot_int != 0) {
            sim->updateMacroscopic();
            last = diagnose(*sim, k, A, nu, V);
            report(last, sim->time(), mass0, fmom0, A, Vol);
        }

        const amrex::Real mass_drift = (mass0 != amrex::Real(0.0))
            ? std::abs(last.fmass - mass0) / std::abs(mass0) : amrex::Real(0.0);
        amrex::Real mom_defect = 0;
        if (A != amrex::Real(0.0) && Vol != amrex::Real(0.0)) {
            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                amrex::Real def = std::abs(last.fmom[dd] - fmom0[dd]) / (std::abs(A) * Vol);
                mom_defect = amrex::max(mom_defect, def);
            }
        }

        amrex::Print() << "Final velocity error: " << last.err << "\n"
                       << "Final L2 velocity error: " << last.l2 << "\n"
                       << "Final mass drift: " << mass_drift << "\n"
                       << "Final momentum defect: " << mom_defect << "\n"
                       << "Finest level reached: " << max_finest << "\n";

        if (max_err > 0 && !(last.err <= max_err)) {
            amrex::Print() << "FAIL: error " << last.err << " > " << max_err << "\n";
            status = 1;
        }
        if (max_l2 > 0 && !(last.l2 <= max_l2)) {
            amrex::Print() << "FAIL: L2 error " << last.l2 << " > " << max_l2 << "\n";
            status = 1;
        }
        if (max_mass_drift > 0 && !(mass_drift <= max_mass_drift)) {
            amrex::Print() << "FAIL: mass drift " << mass_drift << " > " << max_mass_drift << "\n";
            status = 1;
        }
        if (max_mom_defect > 0 && !(mom_defect <= max_mom_defect)) {
            amrex::Print() << "FAIL: momentum defect " << mom_defect << " > " << max_mom_defect << "\n";
            status = 1;
        }
        if (max_finest < min_finest) {
            amrex::Print() << "FAIL: finest level " << max_finest << " < " << min_finest << "\n";
            status = 1;
        }
    }
    amrex::Finalize();
    return status;
}
