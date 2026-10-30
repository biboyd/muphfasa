// AmrInterface.cpp — AMR-only kernels (see AmrInterface.H).

#include "AmrInterface.H"
#include "Collision.H"
#include "Equilibrium.H"
#include "LatticeDescriptor.H"
#include "Macroscopic.H"
#include "Streaming.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>

#include <type_traits>

namespace muphfasa {

namespace {

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
guoVelocity(amrex::GpuArray<amrex::Real, LatticeTag::Q> const& f, amrex::Real rho,
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F) noexcept
{
#if (AMREX_SPACEDIM == 2)
    return computeVelocityCell2D<LatticeTag>(f, rho, F);
#else
    return computeVelocityCell3D<LatticeTag>(f, rho, F);
#endif
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real equilibrium(amrex::Real rho,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& u,
                        int n) noexcept
{
#if (AMREX_SPACEDIM == 2)
    return computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
    return computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
}

} // namespace

// ── pullCollideFusedPreMF ─────────────────────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedPreMF(const amrex::MultiFab& fin_mf,
                           amrex::MultiFab& fout_mf,
                           amrex::MultiFab& fpre_mf,
                           amrex::Real omega,
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F,
                           bool has_forcing)
{
    BL_PROFILE("pullCollideFusedPreMF()");
    static_assert(std::is_same_v<CollisionModel, BGK>,
                  "pullCollideFusedPreMF: only BGK is implemented");

    // With has_forcing == false F is zeroed, so the forced formulas reduce
    // exactly to the unforced ones (u = sum c f / rho, zero source term).
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Fk{};
    if (has_forcing) { Fk = F; }

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(fout_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        auto const& fin  = fin_mf.const_array(mfi);
        auto const& fout = fout_mf.array(mfi);
        auto const& fpre = fpre_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::GpuArray<amrex::Real, LatticeTag::Q> f;
            for (int n = 0; n < LatticeTag::Q; ++n) {
                f[n] = streamCell<LatticeTag>(fin, i, j, k, n);
                fpre(i, j, k, n) = f[n];
            }
            const amrex::Real rho = computeDensityCell<LatticeTag>(f);
            const auto u = guoVelocity<LatticeTag>(f, rho, Fk);
            for (int n = 0; n < LatticeTag::Q; ++n) {
                const amrex::Real feq_n = equilibrium<LatticeTag>(rho, u, n);
                // BGK relaxation (BGK::collide lives in Collision.cpp and
                // is not inlinable from here).
                fout(i, j, k, n) = (amrex::Real(1.0) - omega) * f[n] + omega * feq_n
                                 + guoForceTerm<LatticeTag>(n, u, Fk, omega);
            }
        });
    }
    amrex::Gpu::streamSynchronize();
}

// ── convertPopulationsMF ──────────────────────────────────────────────────
template <typename LatticeTag>
void convertPopulationsMF(const amrex::MultiFab& src_pre,
                          amrex::MultiFab* post_mf, int post_comp,
                          amrex::MultiFab* pre_mf, int pre_comp,
                          LevelParams const& src, LevelParams const& tgt,
                          int ngrow)
{
    BL_PROFILE("convertPopulationsMF()");

    const amrex::Real half    = amrex::Real(0.5);
    const amrex::Real r       = (tgt.tau * tgt.dt) / (src.tau * src.dt);
    const amrex::Real Pfac    = src.tau - half;
    const amrex::Real omega_t = amrex::Real(1.0) / tgt.tau;
    const auto Fs = src.F;
    const auto Ft = tgt.F;
    const bool do_post = (post_mf != nullptr);
    const bool do_pre  = (pre_mf  != nullptr);

    amrex::MultiFab& ref = do_post ? *post_mf : *pre_mf;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.growntilebox(ngrow) & src_pre[mfi].box();
        auto const& fs = src_pre.const_array(mfi);
        amrex::Array4<amrex::Real> post = do_post ? post_mf->array(mfi)
                                                  : amrex::Array4<amrex::Real>{};
        amrex::Array4<amrex::Real> pre  = do_pre  ? pre_mf->array(mfi)
                                                  : amrex::Array4<amrex::Real>{};

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::GpuArray<amrex::Real, LatticeTag::Q> f;
            for (int n = 0; n < LatticeTag::Q; ++n) { f[n] = fs(i, j, k, n); }
            const amrex::Real rho = computeDensityCell<LatticeTag>(f);
            const auto u = guoVelocity<LatticeTag>(f, rho, Fs);
            for (int n = 0; n < LatticeTag::Q; ++n) {
                const amrex::Real feq_n = equilibrium<LatticeTag>(rho, u, n);
                // guoForceTerm(n,u,F,0) = G_n(u).F (no (1 - omega/2) factor)
                const amrex::Real P = Pfac * guoForceTerm<LatticeTag>(n, u, Fs, amrex::Real(0.0));
                const amrex::Real fpre_t = feq_n + r * (f[n] - feq_n - P) + P;
                if (do_pre)  { pre(i, j, k, pre_comp + n) = fpre_t; }
                if (do_post) {
                    post(i, j, k, post_comp + n) =
                        fpre_t - omega_t * (fpre_t - feq_n)
                        + guoForceTerm<LatticeTag>(n, u, Ft, omega_t);
                }
            }
        });
    }
    amrex::Gpu::streamSynchronize();
}

// ── Explicit instantiations (dimension matched) ───────────────────────────
#if (AMREX_SPACEDIM == 2)
template void pullCollideFusedPreMF<D2Q9, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void convertPopulationsMF<D2Q9>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
#else
template void pullCollideFusedPreMF<D3Q15, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void pullCollideFusedPreMF<D3Q19, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void pullCollideFusedPreMF<D3Q27, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void convertPopulationsMF<D3Q15>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
template void convertPopulationsMF<D3Q19>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
template void convertPopulationsMF<D3Q27>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
#endif

} // namespace muphfasa
