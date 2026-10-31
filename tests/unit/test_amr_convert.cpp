// test_amr_convert.cpp — Unit tests for the AMR coarse/fine population
// conversion (convertPopulationsMF, src/LBM/AmrInterface.H).
// Host-only, one small box, every lattice of the active DIM, several tau
// (including the values where the classic post-collision rescaling is
// singular, tau_c = 1 and 3/4). For perturbed-equilibrium pre-collision
// populations and a random body force, verifies under acoustic scaling
// (tau_f - 1/2 = 2 (tau_c - 1/2), dt_f = dt_c/2, F_f = F_c/2):
//   - coarse -> fine -> coarse is the identity
//   - density and the Guo velocity (sum c f + F/2)/rho are level invariant
//   - post-collision output = BGK + Guo collision of the pre-collision output
//   - unforced: fneq_f = tau_f / (2 tau_c) fneq_c
//   - identical source/target parameters reproduce the input

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/AmrInterface.H"
#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"

#include <cmath>
#include <random>

using namespace muphfasa;

namespace {

using Vec = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

int g_failures = 0;

void checkClose(double a, double b, double tol, const char* lattice, const char* what,
                double tau)
{
    if (!(std::abs(a - b) <= tol)) {
        ++g_failures;
        amrex::Print() << "  FAIL [" << lattice << ", tau_c=" << tau << "] " << what
                       << " (err=" << std::abs(a - b) << ")\n";
    }
}

template <typename LatticeTag>
amrex::Real feqCell(amrex::Real rho, Vec const& u, int n)
{
#if (AMREX_SPACEDIM == 2)
    return computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
    return computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
}

template <typename LatticeTag>
Vec guoVel(amrex::Array4<const amrex::Real> const& f, int i, int j, int k, amrex::Real rho,
           Vec const& F)
{
#if (AMREX_SPACEDIM == 2)
    return computeVelocityCell2D<LatticeTag>(f, rho, i, j, k, F);
#else
    return computeVelocityCell3D<LatticeTag>(f, rho, i, j, k, F);
#endif
}

template <typename LatticeTag>
amrex::Real rhoCell(amrex::Array4<const amrex::Real> const& f, int i, int j, int k)
{
    amrex::Real r = 0;
    for (int n = 0; n < LatticeTag::Q; ++n) { r += f(i, j, k, n); }
    return r;
}

template <typename LatticeTag>
void testLattice(const char* name, std::mt19937& rng)
{
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const amrex::Box bx(amrex::IntVect(0), amrex::IntVect(3));
    const amrex::BoxArray ba(bx);
    const amrex::DistributionMapping dm(ba);
    constexpr int Q = LatticeTag::Q;

    amrex::MultiFab src(ba, dm, Q, 0), fine_pre(ba, dm, Q, 0), fine_post(ba, dm, Q, 0),
                    back(ba, dm, Q, 0), same(ba, dm, Q, 0);

    for (const double tau_c : {0.51, 0.75, 0.8, 1.0, 1.7}) {
        for (const bool forced : {false, true}) {
            LevelParams lc, lf;
            lc.tau = tau_c;
            lc.dt  = 1.0;
            lf.tau = 2.0 * (tau_c - 0.5) + 0.5;
            lf.dt  = 0.5;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                lc.F[d] = forced ? 1.e-3 * uni(rng) : 0.0;
                lf.F[d] = lc.F[d] / 2;
            }

            // perturbed equilibrium, different in every cell
            auto const& s = src.array(0);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                const amrex::Real rho = 1.0 + 0.05 * uni(rng);
                Vec u{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { u[d] = 0.08 * uni(rng); }
                for (int n = 0; n < Q; ++n) {
                    s(i, j, k, n) = feqCell<LatticeTag>(rho, u, n) * (1.0 + 0.02 * uni(rng));
                }
            });

            convertPopulationsMF<LatticeTag>(src, &fine_post, 0, &fine_pre, 0, lc, lf, 0);
            convertPopulationsMF<LatticeTag>(fine_pre, nullptr, 0, &back, 0, lf, lc, 0);
            convertPopulationsMF<LatticeTag>(src, nullptr, 0, &same, 0, lc, lc, 0);

            auto const& sc = src.const_array(0);
            auto const& fp = fine_pre.const_array(0);
            auto const& fq = fine_post.const_array(0);
            auto const& bk = back.const_array(0);
            auto const& sm = same.const_array(0);
            const amrex::Real omega_f = 1.0 / lf.tau;
            const amrex::Real r = lf.tau / (2.0 * lc.tau);

            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                const amrex::Real rho_s = rhoCell<LatticeTag>(sc, i, j, k);
                const amrex::Real rho_f = rhoCell<LatticeTag>(fp, i, j, k);
                const Vec u_s = guoVel<LatticeTag>(sc, i, j, k, rho_s, lc.F);
                const Vec u_f = guoVel<LatticeTag>(fp, i, j, k, rho_f, lf.F);
                checkClose(rho_f, rho_s, 1e-14, name, "density invariant", tau_c);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    checkClose(u_f[d], u_s[d], 1e-14, name, "Guo velocity invariant", tau_c);
                }
                for (int n = 0; n < Q; ++n) {
                    checkClose(bk(i, j, k, n), sc(i, j, k, n), 1e-14, name,
                               "coarse->fine->coarse identity", tau_c);
                    checkClose(sm(i, j, k, n), sc(i, j, k, n), 1e-14, name,
                               "same-level identity", tau_c);
                    const amrex::Real feq = feqCell<LatticeTag>(rho_s, u_s, n);
                    const amrex::Real post = fp(i, j, k, n) - omega_f * (fp(i, j, k, n) - feq)
                                           + guoForceTerm<LatticeTag>(n, u_s, lf.F, omega_f);
                    checkClose(fq(i, j, k, n), post, 1e-14, name, "post = collide(pre)", tau_c);
                    if (!forced) {
                        checkClose(fp(i, j, k, n) - feq, r * (sc(i, j, k, n) - feq), 1e-14,
                                   name, "unforced fneq scaling", tau_c);
                    }
                }
            });
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== AMR population conversion tests ===\n";
        std::mt19937 rng(2468);
#if (AMREX_SPACEDIM == 2)
        testLattice<D2Q9>("D2Q9", rng);
#else
        testLattice<D3Q15>("D3Q15", rng);
        testLattice<D3Q19>("D3Q19", rng);
        testLattice<D3Q27>("D3Q27", rng);
#endif
        if (g_failures == 0) {
            amrex::Print() << "All AMR conversion tests passed.\n";
        } else {
            amrex::Print() << g_failures << " AMR conversion check(s) FAILED.\n";
        }
    }
    amrex::Finalize();
    return g_failures == 0 ? 0 : 1;
}
