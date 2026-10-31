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

// ── Average-consistent coarse -> fine interpolation ───────────────────────

namespace {

// First and second differences of crse component n along d at iv. Where a
// neighbour lies outside `ok` (non-periodic domain exterior, or beyond the
// coarse fab), they become second- or first-order one-sided differences.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void interpDifferences(amrex::Array4<amrex::Real const> const& c, int n, amrex::Box const& ok,
                       amrex::IntVect const& iv, int d, amrex::Real& s, amrex::Real& k) noexcept
{
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(d);
    const bool lo = ok.contains(iv - e);
    const bool hi = ok.contains(iv + e);
    const amrex::Real c0 = c(iv, n);
    s = amrex::Real(0.0);
    k = amrex::Real(0.0);
    if (lo && hi) {
        s = amrex::Real(0.5) * (c(iv + e, n) - c(iv - e, n));
        k = c(iv + e, n) - amrex::Real(2.0) * c0 + c(iv - e, n);
    } else if (hi) {
        if (ok.contains(iv + 2 * e)) {
            s = amrex::Real(0.5) * (-amrex::Real(3.0) * c0 + amrex::Real(4.0) * c(iv + e, n)
                                    - c(iv + 2 * e, n));
            k = c(iv + 2 * e, n) - amrex::Real(2.0) * c(iv + e, n) + c0;
        } else {
            s = c(iv + e, n) - c0;
        }
    } else if (lo) {
        if (ok.contains(iv - 2 * e)) {
            s = amrex::Real(0.5) * (amrex::Real(3.0) * c0 - amrex::Real(4.0) * c(iv - e, n)
                                    + c(iv - 2 * e, n));
            k = c0 - amrex::Real(2.0) * c(iv - e, n) + c(iv - 2 * e, n);
        } else {
            s = c0 - c(iv - e, n);
        }
    }
}

// Central (else first-order one-sided) first difference along d at iv.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real interpSlope(amrex::Array4<amrex::Real const> const& c, int n, amrex::Box const& ok,
                        amrex::IntVect const& iv, int d) noexcept
{
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(d);
    const bool lo = ok.contains(iv - e);
    const bool hi = ok.contains(iv + e);
    if (lo && hi) { return amrex::Real(0.5) * (c(iv + e, n) - c(iv - e, n)); }
    if (hi)       { return c(iv + e, n) - c(iv, n); }
    if (lo)       { return c(iv, n) - c(iv - e, n); }
    return amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void avgConsistentInterpCell(amrex::IntVect const& fiv, int n,
                             amrex::Array4<amrex::Real> const& fine, int fine_comp,
                             amrex::Array4<amrex::Real const> const& crse, int crse_comp,
                             amrex::IntVect const& ratio, amrex::Box const& ok) noexcept
{
    const amrex::IntVect civ = amrex::coarsen(fiv, ratio);
    const int nc = crse_comp + n;
    amrex::Real xi[AMREX_SPACEDIM];
    amrex::Real val = crse(civ, nc);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const amrex::Real r = amrex::Real(ratio[d]);
        xi[d] = (amrex::Real(fiv[d] - civ[d] * ratio[d]) + amrex::Real(0.5)) / r - amrex::Real(0.5);
        const amrex::Real xi2_avg = (r * r - amrex::Real(1.0)) / (amrex::Real(12.0) * r * r);
        amrex::Real s, k;
        interpDifferences(crse, nc, ok, civ, d, s, k);
        val += s * xi[d] + amrex::Real(0.5) * k * (xi[d] * xi[d] - xi2_avg);
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int e = d + 1; e < AMREX_SPACEDIM; ++e) {
            const amrex::IntVect ee = amrex::IntVect::TheDimensionVector(e);
            const bool lo = ok.contains(civ - ee);
            const bool hi = ok.contains(civ + ee);
            amrex::Real m = amrex::Real(0.0);
            if (lo && hi) {
                m = amrex::Real(0.5) * (interpSlope(crse, nc, ok, civ + ee, d)
                                      - interpSlope(crse, nc, ok, civ - ee, d));
            } else if (hi) {
                m = interpSlope(crse, nc, ok, civ + ee, d) - interpSlope(crse, nc, ok, civ, d);
            } else if (lo) {
                m = interpSlope(crse, nc, ok, civ, d) - interpSlope(crse, nc, ok, civ - ee, d);
            }
            val += m * xi[d] * xi[e];
        }
    }
    fine(fiv, fine_comp + n) = val;
}

} // namespace

amrex::Box AvgConsistentInterp::CoarseBox(const amrex::Box& fine, const amrex::IntVect& ratio)
{
    amrex::Box crse = amrex::coarsen(fine, ratio);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (ratio[d] > 1) { crse.grow(d, 1); }
    }
    return crse;
}

amrex::Box AvgConsistentInterp::CoarseBox(const amrex::Box& fine, int ratio)
{
    return CoarseBox(fine, amrex::IntVect(ratio));
}

void AvgConsistentInterp::interp(const amrex::FArrayBox& crse, int crse_comp,
                                 amrex::FArrayBox& fine, int fine_comp, int ncomp,
                                 const amrex::Box& fine_region, const amrex::IntVect& ratio,
                                 const amrex::Geometry& crse_geom,
                                 const amrex::Geometry& /*fine_geom*/,
                                 amrex::Vector<amrex::BCRec> const& /*bcr*/,
                                 int /*actual_comp*/, int /*actual_state*/, amrex::RunOn runon)
{
    BL_PROFILE("AvgConsistentInterp::interp()");

    // Usable coarse cells: inside the coarse fab and inside the domain,
    // except in periodic directions where the ghost cells hold valid data.
    amrex::Box ok = crse.box();
    const amrex::Box& dom = crse_geom.Domain();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!crse_geom.isPeriodic(d)) {
            ok.setSmall(d, amrex::max(ok.smallEnd(d), dom.smallEnd(d)));
            ok.setBig(d, amrex::min(ok.bigEnd(d), dom.bigEnd(d)));
        }
    }

    auto const& c = crse.const_array();
    auto const& f = fine.array();
    AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FLAG(runon, fine_region, ncomp, i, j, k, n,
    {
        amrex::ignore_unused(j, k);
        avgConsistentInterpCell(amrex::IntVect(AMREX_D_DECL(i, j, k)), n,
                                f, fine_comp, c, crse_comp, ratio, ok);
    });
}

AvgConsistentInterp& avgConsistentInterp()
{
    static AvgConsistentInterp interp;
    return interp;
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
