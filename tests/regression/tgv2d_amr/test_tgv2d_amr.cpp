// test_tgv2d_amr.cpp — Regression test: 2-D Taylor-Green vortex decay on an
// AMR hierarchy (amr.max_level >= 1). 2D builds only.
//
// Same problem as tests/regression/tgv2d (InitData.cpp and TgvAnalytic.H are
// shared with that directory), with static refinement patches that cut
// through the vortices. Unlike the channel tests in couette_amr the flow
// varies in both directions, so the coarse/fine interfaces see velocity
// gradients normal and tangential to them as well as mixed derivatives
// (the cross terms of the average-consistent interpolation,
// amr_implementation.md, section 3.3), and the patch corners are crossed by
// the diagonal populations.
//
// The error is measured on the composite solution: every level contributes
// the cells that are not covered by a finer level. Under acoustic scaling
// lattice velocities are level invariant and level l has spacing 2^-l in
// level-0 lattice units, so the analytic solution is evaluated at
// x = (i + 1/2) 2^-l, y = (j + 1/2) 2^-l with k, nu and t all in level-0
// lattice units. Errors are relative to the current vortex amplitude
// u0 exp(-2 nu k^2 t).
//
// Also checks conservation of the composite totals of the post-collision
// populations, M = sum(f dV) and P = sum(c f dV) (no body force here, so
// sum(c f) is the momentum). With amr.reflux = 1 or 2 both are exact to
// round-off; with amr.reflux = 0 they are conserved to truncation error
// (amr_implementation.md, 3.9). The momentum defect is
// |P(t) - P(0)| / (u0 Vol), the largest component. The macroscopic
// sum(rho dV) is reported as MassDrift(rho); it is evaluated after one more
// streaming step and so includes one step's coarse/fine crossing mismatch.
//
// tgv.bg = bx by adds a uniform background velocity (bx, by) u0 so the
// vortices are carried across the coarse/fine interfaces (TgvAnalytic.H).
//
// Optional pass/fail thresholds (exit code 1 when exceeded):
//   tgv_amr.max_err        composite inf-norm max(|u-u_e|, |v-v_e|) / amplitude
//   tgv_amr.max_l2_err     composite volume-weighted rms of the same quantity
//   tgv_amr.max_mass_drift |M(t) - M(0)| / M(0) at the final time (post-collision M)
//   tgv_amr.max_mom_defect momentum defect (above) at the final time
//   tgv_amr.min_finest     minimum finest level that must exist

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "../tgv2d/TgvAnalytic.H"
#include "IO/InputParser.H"
#include "LBM/LatticeDescriptor.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <vector>

using namespace muphfasa;

namespace {

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
    std::vector<amrex::Real> lev_err;  // per level inf-norm, uncovered cells only
    std::vector<amrex::IntVect> lev_loc; // cell of lev_err (level-l index space)
    amrex::Real err  = 0;              // composite inf-norm
    amrex::Real l2   = 0;              // composite volume-weighted rms
    amrex::Real mass = 0;              // composite sum(rho dV), level-0 units
    amrex::Real mom[2] = {0, 0};       // composite sum(rho u dV)
    // Same composite quadrature applied to the analytic field: nonzero at
    // O(h^2) amp because the mixed-resolution midpoint sum is not exact.
    amrex::Real mom_exact[2] = {0, 0};
    // Composite sum(f dV) and sum(c f dV) of the post-collision populations:
    // the state the scheme advances (the macroscopic fields above are taken
    // after one more streaming step, which at a coarse/fine face pulls from
    // covered coarse cells and fine ghosts and so is not itself conservative).
    amrex::Real fmass = 0;
    amrex::Real fmom[2] = {0, 0};
};

Diagnostics diagnose(const ISimulation& sim, amrex::Real k, amrex::Real u0, amrex::Real nu)
{
    Diagnostics d;
    amrex::Real U, V;
    tgv::backgroundVelocity(u0, U, V);
    const amrex::Real t   = sim.time();
    const amrex::Real amp = u0 * std::exp(-amrex::Real(2.0) * nu * k * k * t);
    amrex::Real sq = 0, vol = 0;
    for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
        const amrex::MultiFab& vel = sim.vel(lev);
        const amrex::MultiFab& rho = sim.rho(lev);
        const amrex::MultiFab mask = uncoveredMask(sim, lev);
        const amrex::Real h = amrex::Real(1.0) / amrex::Real(1 << lev);

        // Per-cell max-component velocity error relative to the current amplitude.
        amrex::MultiFab err(vel.boxArray(), vel.DistributionMap(), 1, 0);
        amrex::MultiFab mom(vel.boxArray(), vel.DistributionMap(), 4, 0);
        for (amrex::MFIter mfi(err, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox();
            auto const& e  = err.array(mfi);
            auto const& p  = mom.array(mfi);
            auto const& u  = vel.const_array(mfi);
            auto const& r  = rho.const_array(mfi);
            auto const& m  = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kc) {
                const amrex::Real x = (i + amrex::Real(0.5)) * h;
                const amrex::Real y = (j + amrex::Real(0.5)) * h;
                amrex::Real ue, ve;
                tgv::exactUV(x - U * t, y - V * t, k, u0, nu, t, ue, ve);
                ue += U;
                ve += V;
                e(i, j, kc) = m(i, j, kc) *
                    amrex::max(std::abs(u(i, j, kc, 0) - ue),
                               std::abs(u(i, j, kc, 1) - ve)) / amp;
                p(i, j, kc, 0) = m(i, j, kc) * r(i, j, kc) * u(i, j, kc, 0);
                p(i, j, kc, 1) = m(i, j, kc) * r(i, j, kc) * u(i, j, kc, 1);
                p(i, j, kc, 2) = m(i, j, kc) * ue;
                p(i, j, kc, 3) = m(i, j, kc) * ve;
            });
        }
        const amrex::Real dv = h * h;
        const amrex::Real lev_err = err.max(0);
        d.lev_err.push_back(lev_err);
        d.lev_loc.push_back(err.maxIndex(0));
        d.err = amrex::max(d.err, lev_err);
        sq  += amrex::MultiFab::Dot(err, 0, 1, 0) * dv;  // sum e^2; masked cells are 0
        vol += mask.sum(0) * dv;
        d.mass += amrex::MultiFab::Dot(mask, 0, rho, 0, 1, 0) * dv;
        d.mom[0] += mom.sum(0) * dv;
        d.mom[1] += mom.sum(1) * dv;
        d.mom_exact[0] += mom.sum(2) * dv;
        d.mom_exact[1] += mom.sum(3) * dv;

        using L = LatticeDescriptor<2, 9>;
        const amrex::MultiFab& f = sim.f(lev);
        amrex::MultiFab mom_f(vel.boxArray(), vel.DistributionMap(), 3, 0);
        for (amrex::MFIter mfi(mom_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox();
            auto const& p  = mom_f.array(mfi);
            auto const& fa = f.const_array(mfi);
            auto const& m  = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kc) {
                amrex::Real s0 = 0, sx = 0, sy = 0;
                for (int q = 0; q < L::Q; ++q) {
                    s0 += fa(i, j, kc, q);
                    sx += L::cx(q) * fa(i, j, kc, q);
                    sy += L::cy(q) * fa(i, j, kc, q);
                }
                p(i, j, kc, 0) = m(i, j, kc) * s0;
                p(i, j, kc, 1) = m(i, j, kc) * sx;
                p(i, j, kc, 2) = m(i, j, kc) * sy;
            });
        }
        d.fmass   += mom_f.sum(0) * dv;
        d.fmom[0] += mom_f.sum(1) * dv;
        d.fmom[1] += mom_f.sum(2) * dv;
    }
    d.l2 = std::sqrt(sq / vol);
    return d;
}

void report(const Diagnostics& d, amrex::Real t, amrex::Real mass0)  // mass0: sum(rho dV)
{
    amrex::Print() << "At Time=" << t << ": Error=" << d.err << " L2=" << d.l2 << "  (levels:";
    for (std::size_t l = 0; l < d.lev_err.size(); ++l) {
        amrex::Print() << " " << d.lev_err[l] << " @" << d.lev_loc[l];
    }
    amrex::Print() << ")  MassDrift(rho)=" << std::abs(d.mass - mass0) / mass0
                   << "  Mom=(" << d.mom[0] << ", " << d.mom[1] << ")"
                   << "  MomExact=(" << d.mom_exact[0] << ", " << d.mom_exact[1] << ")\n"
                   << "    post-collision state: mass " << d.fmass
                   << "  mom (" << d.fmom[0] << ", " << d.fmom[1] << ")\n";
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
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ncells[0] == ncells[1],
            "test_tgv2d_amr: domain must be square (one vortex wavelength per side)");
        const amrex::Real k   = tgv::waveNumber(ncells[0]);
        const amrex::Real u0  = sim->units().toLatticeVelocity(params.refVelocity());
        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - amrex::Real(0.5)) * tgv::kCs2;
        amrex::Print() << "N=" << ncells[0] << " u0=" << u0 << " nu=" << nu << "\n";

        amrex::Real max_err = -1, max_l2 = -1, max_drift = -1, max_mom = -1;
        int min_finest = 0;
        {
            amrex::ParmParse pp("tgv_amr");
            pp.query("max_err", max_err);
            pp.query("max_l2_err", max_l2);
            pp.query("max_mass_drift", max_drift);
            pp.query("max_mom_defect", max_mom);
            pp.query("min_finest", min_finest);
        }

        sim->updateMacroscopic();
        const Diagnostics d0 = diagnose(*sim, k, u0, nu);
        const amrex::Real mass0 = d0.mass;
        report(d0, sim->time(), mass0);
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
                last = diagnose(*sim, k, u0, nu);  // writeOutput updated macros
                report(last, sim->time(), mass0);
            }
        }
        if (plot_int <= 0 || max_step % plot_int != 0) {
            sim->updateMacroscopic();
            last = diagnose(*sim, k, u0, nu);
            report(last, sim->time(), mass0);
        }

        const amrex::Real drift = std::abs(last.fmass - d0.fmass) / d0.fmass;
        const amrex::Real vol0  = amrex::Real(ncells[0]) * amrex::Real(ncells[1]);
        const amrex::Real mom_defect =
            amrex::max(std::abs(last.fmom[0] - d0.fmom[0]),
                       std::abs(last.fmom[1] - d0.fmom[1])) / (u0 * vol0);
        amrex::Print() << "Final velocity error: " << last.err << "\n"
                       << "Final L2 velocity error: " << last.l2 << "\n"
                       << "Final mass drift: " << drift << "\n"
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
        if (max_drift > 0 && !(drift <= max_drift)) {
            amrex::Print() << "FAIL: mass drift " << drift << " > " << max_drift << "\n";
            status = 1;
        }
        if (max_mom > 0 && !(mom_defect <= max_mom)) {
            amrex::Print() << "FAIL: momentum defect " << mom_defect << " > " << max_mom << "\n";
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
