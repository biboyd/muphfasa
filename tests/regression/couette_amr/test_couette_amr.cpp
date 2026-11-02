// test_couette_amr.cpp — Regression test: plane Couette startup on an AMR
// hierarchy (amr.max_level >= 1). Supports 2D and 3D builds.
//
// Same problem as tests/regression/couette (see test_couette.cpp for the
// analytic solution), optionally with a wall-parallel body force
// (forcing.accel, plane Couette-Poiseuille; with both walls at rest this is
// the Poiseuille startup). Here the error is measured on the composite
// solution: every level contributes the cells that are not covered by a
// finer level. Under acoustic scaling lattice velocities are level
// invariant and level l has spacing 2^-l in level-0 lattice units, so the
// analytic profile is evaluated at y = (j + 1/2) 2^-l with H, nu and t all
// in level-0 lattice units.
//
// Also checks the composite mass for drift. The conserved quantity is the
// sum of the post-collision populations over the composite grid; with
// amr.reflux = 1 (default) it is exact to round-off; without refluxing it
// drifts secularly where a coarse/fine boundary meets a wall or turns a
// corner (amr_implementation.md, 3.9). The
// macroscopic sum(rho dV) is reported too: it is evaluated after one more
// streaming step and therefore differs from it by one step's coarse/fine
// crossing mismatch.
//
// Optional pass/fail thresholds (exit code 1 when exceeded):
//   couette_amr.max_err        inf-norm |u - u_exact| / U_ref at the final time
//                              (U_ref = max(|U0|, F H^2 / (8 nu)))
//   couette_amr.max_mass_drift |M(t) - M(0)| / M(0) at the final time, M the
//                              composite sum of the post-collision populations
//   couette_amr.min_finest     minimum finest level that must exist

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "LBM/BoundaryConditions.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <vector>

using namespace muphfasa;

namespace {

constexpr amrex::Real kPi = amrex::Real(3.14159265358979323846);

// Wall-normal direction: y in 2D, z in 3D (same convention as test_couette).
constexpr int WALL_DIM = (AMREX_SPACEDIM == 2 ? 1 : 2);

// Startup of plane Couette-Poiseuille flow from rest: top wall moving at
// U0, body force F along x. u_t = nu u_yy + F is linear, so the solution is
// the Couette series (see test_couette.cpp) plus the Poiseuille series
//   F y (H - y) / (2 nu) - sum_{n odd} 4 F H^2 / (nu n^3 pi^3)
//                          sin(n pi y / H) exp(-n^2 pi^2 nu t / H^2).
AMREX_GPU_HOST_DEVICE
amrex::Real analyticChannel(amrex::Real y, amrex::Real H, amrex::Real U0,
                            amrex::Real F, amrex::Real nu, amrex::Real t)
{
    amrex::Real u = U0 * y / H + F * y * (H - y) / (amrex::Real(2.0) * nu);
    for (int n = 1; n <= 200; ++n) {
        const amrex::Real exp_arg = -amrex::Real(n * n) * kPi * kPi * nu * t / (H * H);
        if (exp_arg < -50) { break; }
        const amrex::Real sign = (n % 2 == 0) ? amrex::Real(1.0) : amrex::Real(-1.0);
        const amrex::Real npi  = n * kPi;
        amrex::Real a = amrex::Real(2.0) * U0 * sign / npi;
        if (n % 2 == 1) { a -= amrex::Real(4.0) * F * H * H / (nu * npi * npi * npi); }
        u += a * std::sin(npi * y / H) * std::exp(exp_arg);
    }
    return u;
}

// 1 on cells of level lev not covered by level lev+1, 0 elsewhere.
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
    std::vector<amrex::Real> lev_err;  // per level, uncovered cells only
    std::vector<amrex::Real> lev_ymax; // wall-normal location y/H of lev_err
    amrex::Real err  = 0;              // composite inf-norm
    amrex::Real mass  = 0;             // composite sum(rho dV), level-0 units
    amrex::Real fmass = 0;             // composite sum(f dV), post-collision
};

Diagnostics diagnose(const ISimulation& sim, amrex::Real H, amrex::Real U0,
                     amrex::Real F, amrex::Real nu, amrex::Real Uref)
{
    Diagnostics d;
    const amrex::Real t = sim.time();
    for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
        const amrex::MultiFab& vel = sim.vel(lev);
        const amrex::MultiFab& rho = sim.rho(lev);
        const amrex::MultiFab mask = uncoveredMask(sim, lev);
        const amrex::Real h = amrex::Real(1.0) / amrex::Real(1 << lev);

        amrex::MultiFab err(vel.boxArray(), vel.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(err, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox();
            auto const& e = err.array(mfi);
            auto const& u = vel.const_array(mfi);
            auto const& m = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const int iv[3] = {i, j, k};
                const amrex::Real y = (iv[WALL_DIM] + amrex::Real(0.5)) * h;
                amrex::Real emax = std::abs(u(i, j, k, 0) - analyticChannel(y, H, U0, F, nu, t));
                for (int c = 1; c < AMREX_SPACEDIM; ++c) {
                    emax = amrex::max(emax, std::abs(u(i, j, k, c)));
                }
                e(i, j, k) = m(i, j, k) * emax / Uref;
            });
        }
        const amrex::Real lev_err = err.max(0);
        d.lev_err.push_back(lev_err);
        d.lev_ymax.push_back((err.maxIndex(0)[WALL_DIM] + amrex::Real(0.5)) * h / H);
        d.err = amrex::max(d.err, lev_err);
        const amrex::Real dv = std::pow(h, AMREX_SPACEDIM);
        d.mass += amrex::MultiFab::Dot(mask, 0, rho, 0, 1, 0) * dv;
        const amrex::MultiFab& f = sim.f(lev);
        for (int q = 0; q < f.nComp(); ++q) {
            d.fmass += amrex::MultiFab::Dot(mask, 0, f, q, 1, 0) * dv;
        }
    }
    return d;
}

void report(const Diagnostics& d, int step, const Diagnostics& d0)
{
    amrex::Print() << "At Time=" << step << ": Error=" << d.err << "  (levels:";
    for (std::size_t l = 0; l < d.lev_err.size(); ++l) {
        amrex::Print() << " " << d.lev_err[l] << " @y/H=" << d.lev_ymax[l];
    }
    amrex::Print() << ")  MassDrift=" << std::abs(d.fmass - d0.fmass) / d0.fmass
                   << "  (rho: " << std::abs(d.mass - d0.mass) / d0.mass << ")\n";
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

        const amrex::Real H   = amrex::Real(params.ncells()[WALL_DIM]);
        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - amrex::Real(0.5)) / amrex::Real(3.0);
        const amrex::Real U0  = sim->units().toLatticeVelocity(
            params.Uwall()[faceIndex(WALL_DIM, 1)][0]);
        const amrex::Real F   = sim->units().toLatticeAcceleration(params.forcingAccel()[0]);
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.forcingAccel()[d] == amrex::Real(0.0),
                "test_couette_amr: only a wall-parallel (x) body force is supported");
        }
        // Errors are reported relative to the larger of the wall speed and
        // the steady Poiseuille centreline velocity F H^2 / (8 nu).
        const amrex::Real Uref = amrex::max(std::abs(U0), std::abs(F) * H * H / (amrex::Real(8.0) * nu));
        amrex::Print() << "U0=" << U0 << " F=" << F << " nu=" << nu << " Uref=" << Uref << "\n";

        amrex::Real max_err = -1, max_drift = -1;
        int min_finest = 0;
        {
            amrex::ParmParse pp("couette_amr");
            pp.query("max_err", max_err);
            pp.query("max_mass_drift", max_drift);
            pp.query("min_finest", min_finest);
        }

        sim->updateMacroscopic();
        const Diagnostics d0 = diagnose(*sim, H, U0, F, nu, Uref);
        amrex::Print() << "Finest level: " << sim->finestLevel()
                       << "  initial mass: " << d0.fmass << "\n";

        const int max_step = params.maxStep();
        const int plot_int = params.plotInterval();
        int max_finest = sim->finestLevel();
        Diagnostics last;
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);
            max_finest = amrex::max(max_finest, sim->finestLevel());
            if (plot_int > 0 && step % plot_int == 0) {
                last = diagnose(*sim, H, U0, F, nu, Uref);  // writeOutput updated macros
                report(last, step, d0);
            }
        }
        if (plot_int <= 0 || max_step % plot_int != 0) {
            sim->updateMacroscopic();
            last = diagnose(*sim, H, U0, F, nu, Uref);
            report(last, max_step, d0);
        }

        const amrex::Real drift = std::abs(last.fmass - d0.fmass) / d0.fmass;
        amrex::Print() << "Final Steady state error: " << last.err << "\n"
                       << "Final mass drift: " << drift << "\n"
                       << "Finest level reached: " << max_finest << "\n";

        if (max_err > 0 && !(last.err <= max_err)) {
            amrex::Print() << "FAIL: error " << last.err << " > " << max_err << "\n";
            status = 1;
        }
        if (max_drift > 0 && !(drift <= max_drift)) {
            amrex::Print() << "FAIL: mass drift " << drift << " > " << max_drift << "\n";
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
