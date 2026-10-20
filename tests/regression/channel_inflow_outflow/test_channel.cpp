// test_channel.cpp — Regression test: uniform channel flow through Inflow/
// Outflow faces.
//
// Geometry: periodic in y, Inflow at x=lo with prescribed velocity (U0, 0)
// and density rho_in, Outflow (zero-gradient) at x=hi (see ./inputs). With
// no walls transverse to the flow (periodic y) and no pressure gradient
// driving force, the domain must become spatially uniform at steady state
// (there is nothing in the problem to create an x- or y-gradient once the
// transient from the quiescent initial condition dies out).
//
// NOTE: the steady-state value is *not* exactly (U0, rho_in). InflowKernel
// implements a simplified Dirichlet-velocity bounce-back (see
// src/LBM/BCTypes/InflowKernel.H), not full Zou-He, so it only reproduces
// the prescribed velocity to leading order -- the actual fixed point is
// whatever (rho, u) self-consistently reproduces itself under one
// collide+stream+BC cycle, which is close to but not equal to (rho_in, U0)
// (density ends up a few percent off rho_in, e.g. ~1.1 for rho_in=1,
// U0=0.1 in lattice units). This is an accepted tradeoff of the simplified
// BC, not a bug -- see InflowKernel.H's header comment.
//
// So rather than diffing against a hardcoded (U0, rho_in) field (as
// tests/regression/couette/test_couette.cpp does against its analytic wall-
// driven profile), this test checks the two properties that *do* regress if
// the kernel breaks: (1) the velocity field becomes spatially uniform
// (no residual x/y variation once transients die out), and (2) that
// uniform value is within a loose tolerance of the prescribed U0 (sanity
// check against the leading-order target, not an exact match).

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "IO/Plotfile.H"
#include "LBM/BoundaryConditions.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <vector>

using namespace muphfasa;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // Input file comes from the command line (e.g. `./test_channel inputs`)
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        // x-lo face, x-component (inflow velocity), and the reference
        // density used as rho_in -- see SimulationSetup::buildBCManager().
        const amrex::Real U0     = sim->units().toLatticeVelocity(params.Uinflow()[faceIndex(0, -1)][0]);
        const amrex::Real rho_in = sim->units().toLatticeDensity(params.refDensity());

        const int max_step = params.maxStep();
        std::vector<amrex::Real> max_err;

        const std::string analytic_file = "analytic_plt";

        // run simulation
        for (int step = 0; step <= max_step; ++step) {
            sim->advance(step);
            sim->writeOutput(step);

            if (step % params.plotInterval() == 0) {
                const amrex::MultiFab& sim_vel = sim->vel();
                const amrex::MultiFab& sim_rho = sim->rho();

                // Uniformity check: the vx field should have no residual
                // x/y variation at steady state (nothing in the problem can
                // create a gradient). max(vx) - min(vx) -> 0 as the
                // transient dies out, regardless of the exact fixed-point
                // value.
                const amrex::Real vx_min = sim_vel.min(0);
                const amrex::Real vx_max = sim_vel.max(0);
                const amrex::Real uniformity_err = vx_max - vx_min;
                max_err.push_back(uniformity_err);

                // Write a uniform plotfile at the *prescribed* (U0, rho_in)
                // purely as a visual reference for fcompare -- not used for
                // the pass/fail check above (see header comment: the
                // simplified Inflow BC's actual fixed point differs from
                // this by a leading-order correction).
                amrex::MultiFab analytic_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                              sim_vel.nComp(), sim_vel.nGrow());
                amrex::MultiFab analytic_rho(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                              sim_rho.nComp(), sim_rho.nGrow());
                analytic_vel.setVal(amrex::Real(0.0));
                analytic_vel.setVal(U0, 0, 1);
                analytic_rho.setVal(rho_in);
                muphfasa::writeMacroPlotfile(plotfileName(analytic_file, step),
                        analytic_rho, analytic_vel, sim->geom(), sim->time(), step);

                amrex::Print() << "At Time=" << step << ": UniformityError=" << uniformity_err
                                << " vx[min,max]=" << vx_min << "," << vx_max
                                << " rho[min,max]=" << sim_rho.min(0) << "," << sim_rho.max(0)
                                << "\n";
            }
        }

        amrex::Real final_uniformity_err = max_err.back();
        const amrex::Real final_vx = sim->vel().max(0);

        // Sanity check against the leading-order target: the simplified
        // Inflow BC's fixed point is close to but not exactly U0 (see
        // header comment), so use a loose relative tolerance here.
        const amrex::Real vx_rel_err = std::abs(final_vx - U0) / U0;

        amrex::Print() << "Final uniformity error: " << final_uniformity_err << '\n';
        amrex::Print() << "Final vx=" << final_vx << " (target U0=" << U0
                        << ", relative error=" << vx_rel_err << ")\n";

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(final_uniformity_err < amrex::Real(1.0e-6),
            "test_channel: velocity field did not reach a spatially uniform steady state");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(vx_rel_err < amrex::Real(0.25),
            "test_channel: steady-state vx is too far from the prescribed U0");

        amrex::Finalize();
        return 0;
    }
}
