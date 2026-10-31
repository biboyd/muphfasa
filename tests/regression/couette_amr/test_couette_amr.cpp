// test_couette_amr.cpp — Regression test: plane Couette startup on an AMR
// hierarchy (amr.max_level >= 1). Supports 2D and 3D builds.
//
// Same problem as tests/regression/couette (see test_couette.cpp for the
// analytic solution); here the error is measured on the composite
// solution: every level contributes the cells that are not covered by a
// finer level. Under acoustic scaling lattice velocities are level
// invariant and level l has spacing 2^-l in level-0 lattice units, so the
// analytic profile is evaluated at y = (j + 1/2) 2^-l with H, nu and t all
// in level-0 lattice units.
//
// Also checks the composite mass sum(rho dV) for drift: the coarse/fine
// coupling (overlap + averaging, no refluxing) conserves mass only to
// truncation error, while the single-level solver conserves it exactly.
//
// Optional pass/fail thresholds (exit code 1 when exceeded):
//   couette_amr.max_err        inf-norm |u - u_exact| / U0 at the final time
//   couette_amr.max_mass_drift |M(t) - M(0)| / M(0) at the final time
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

AMREX_GPU_HOST_DEVICE
amrex::Real analyticCouette(amrex::Real y, amrex::Real H, amrex::Real U0,
                            amrex::Real nu, amrex::Real t)
{
    amrex::Real u = U0 * y / H;
    for (int n = 1; n <= 200; ++n) {
        const amrex::Real exp_arg = -amrex::Real(n * n) * kPi * kPi * nu * t / (H * H);
        if (exp_arg < -50) { break; }
        const amrex::Real sign = (n % 2 == 0) ? amrex::Real(1.0) : amrex::Real(-1.0);
        u += (amrex::Real(2.0) * U0 * sign / (n * kPi)) * std::sin(n * kPi * y / H)
             * std::exp(exp_arg);
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
    amrex::Real mass = 0;              // composite sum(rho dV), level-0 units
};

Diagnostics diagnose(const ISimulation& sim, amrex::Real H, amrex::Real U0, amrex::Real nu)
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
                amrex::Real emax = std::abs(u(i, j, k, 0) - analyticCouette(y, H, U0, nu, t));
                for (int c = 1; c < AMREX_SPACEDIM; ++c) {
                    emax = amrex::max(emax, std::abs(u(i, j, k, c)));
                }
                e(i, j, k) = m(i, j, k) * emax / U0;
            });
        }
        const amrex::Real lev_err = err.max(0);
        d.lev_err.push_back(lev_err);
        d.lev_ymax.push_back((err.maxIndex(0)[WALL_DIM] + amrex::Real(0.5)) * h / H);
        d.err = amrex::max(d.err, lev_err);
        d.mass += amrex::MultiFab::Dot(mask, 0, rho, 0, 1, 0) * std::pow(h, AMREX_SPACEDIM);
    }
    return d;
}

void report(const Diagnostics& d, int step, amrex::Real mass0)
{
    amrex::Print() << "At Time=" << step << ": Error=" << d.err << "  (levels:";
    for (std::size_t l = 0; l < d.lev_err.size(); ++l) {
        amrex::Print() << " " << d.lev_err[l] << " @y/H=" << d.lev_ymax[l];
    }
    amrex::Print() << ")  MassDrift=" << std::abs(d.mass - mass0) / mass0 << "\n";
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

        amrex::Real max_err = -1, max_drift = -1;
        int min_finest = 0;
        {
            amrex::ParmParse pp("couette_amr");
            pp.query("max_err", max_err);
            pp.query("max_mass_drift", max_drift);
            pp.query("min_finest", min_finest);
        }

        sim->updateMacroscopic();
        const amrex::Real mass0 = diagnose(*sim, H, U0, nu).mass;
        amrex::Print() << "Finest level: " << sim->finestLevel()
                       << "  initial mass: " << mass0 << "\n";

        const int max_step = params.maxStep();
        const int plot_int = params.plotInterval();
        int max_finest = sim->finestLevel();
        Diagnostics last;
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);
            max_finest = amrex::max(max_finest, sim->finestLevel());
            if (plot_int > 0 && step % plot_int == 0) {
                last = diagnose(*sim, H, U0, nu);  // writeOutput updated macros
                report(last, step, mass0);
            }
        }
        if (plot_int <= 0 || max_step % plot_int != 0) {
            sim->updateMacroscopic();
            last = diagnose(*sim, H, U0, nu);
            report(last, max_step, mass0);
        }

        const amrex::Real drift = std::abs(last.mass - mass0) / mass0;
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
