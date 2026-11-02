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

// ── Initial non-equilibrium populations ───────────────────────────────────

template <typename LatticeTag>
void addInitialNonEquilibriumMF(const amrex::MultiFab& rho, const amrex::MultiFab& vel,
                                amrex::MultiFab& fpre, amrex::MultiFab& fpost,
                                amrex::Real tau, const amrex::Geometry& geom)
{
    BL_PROFILE("addInitialNonEquilibriumMF()");
    AMREX_ALWAYS_ASSERT(vel.nGrow() >= 1);

    // Cells whose velocity may be used: the domain, extended across
    // periodic faces.
    amrex::Box ok = geom.Domain();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (geom.isPeriodic(d)) { ok.grow(d, 1); }
    }
    const amrex::Real omega = amrex::Real(1.0) / tau;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(fpre, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        auto const& r    = rho.const_array(mfi);
        auto const& u    = vel.const_array(mfi);
        auto const& pre  = fpre.array(mfi);
        auto const& post = fpost.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            // g[a][b] = d u_b / d x_a
            amrex::Real g[AMREX_SPACEDIM][AMREX_SPACEDIM];
            for (int a = 0; a < AMREX_SPACEDIM; ++a) {
                const amrex::IntVect e = amrex::IntVect::TheDimensionVector(a);
                const bool lo = ok.contains(iv - e);
                const bool hi = ok.contains(iv + e);
                for (int b = 0; b < AMREX_SPACEDIM; ++b) {
                    if (lo && hi) {
                        g[a][b] = amrex::Real(0.5) * (u(iv + e, b) - u(iv - e, b));
                    } else if (hi) {
                        g[a][b] = u(iv + e, b) - u(iv, b);
                    } else if (lo) {
                        g[a][b] = u(iv, b) - u(iv - e, b);
                    } else {
                        g[a][b] = amrex::Real(0.0);
                    }
                }
            }
            amrex::Real div = amrex::Real(0.0);
            for (int a = 0; a < AMREX_SPACEDIM; ++a) { div += g[a][a]; }
            const amrex::Real fac = -tau * r(i, j, k) * LatticeTag::cs2inv;
            for (int n = 0; n < LatticeTag::Q; ++n) {
                const int c[3] = {LatticeTag::cx(n), LatticeTag::cy(n),
#if (AMREX_SPACEDIM == 3)
                                  LatticeTag::cz(n)
#else
                                  0
#endif
                };
                amrex::Real qg = -LatticeTag::cs2 * div;   // Q_n : grad(u)
                for (int a = 0; a < AMREX_SPACEDIM; ++a) {
                    for (int b = 0; b < AMREX_SPACEDIM; ++b) {
                        qg += c[a] * c[b] * g[a][b];
                    }
                }
                const amrex::Real fneq = fac * LatticeTag::weights(n) * qg;
                pre(i, j, k, n)  += fneq;
                post(i, j, k, n) += (amrex::Real(1.0) - omega) * fneq;
            }
        });
    }
}

// ── Coarse/fine refluxing ─────────────────────────────────────────────────

namespace {

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::IntVect latticeVector(int n) noexcept
{
    return amrex::IntVect(AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), LatticeTag::cz(n)));
}

// Adds one fine coarse/fine crossing (the link x -> x + c, ratio 2, of
// signed weighted population a; pu = coarse parent of its ghost end) to the
// register. The crossing is matched to the coarse link X -> X + c whose
// midpoint is nearest to its own: along each dimension with c_d != 0 the
// fine midpoint lies either on a coarse channel or halfway between two, in
// which case it is split equally. Matching link by link makes the register
// vanish identically for a uniform state, including at corners of the fine
// region, where the coarse and fine levels cross the boundary along
// different diagonals. The contribution goes to the uncovered end of the
// coarse link (its source if both ends are uncovered, pu if neither is).
// Links with an end outside the domain have no coarse counterpart (the
// coarse population bounces back): their mass goes to the uncovered end or
// pu, but not their momentum, which is a difference in wall force between
// the levels rather than a conservation error.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void addFineCrossing(amrex::Array4<amrex::Real> const& r, amrex::Array4<int const> const& cm,
                     amrex::IntVect const& x, amrex::IntVect const& c, amrex::IntVect const& pu,
                     amrex::Real a) noexcept
{
    amrex::IntVect X0 = amrex::coarsen(x, 2);
    int odd = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (c[d] == 0) { continue; }
        // Twice the (real) coarse start index of a link with the same midpoint.
        const int t2 = (c[d] > 0) ? x[d] - 1 : x[d];
        if (t2 & 1) {
            odd |= (1 << d);
            X0[d] = (t2 - 1) >> 1;
        } else {
            X0[d] = t2 >> 1;
        }
    }
    int nodd = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { nodd += (odd >> d) & 1; }
    const amrex::Real w = a / amrex::Real(1 << nodd);

    for (int s = 0; s < (1 << AMREX_SPACEDIM); ++s) {
        if ((s & ~odd) != 0) { continue; }
        amrex::IntVect X = X0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { X[d] += (s >> d) & 1; }
        const int ms = cm(X);
        const int md = cm(X + c);
        amrex::IntVect U = pu;
        bool mom = true;
        if (ms == 0) {
            U = X;
        } else if (md == 0) {
            U = X + c;
        }
        if (ms == -1 || md == -1) { mom = false; }
        amrex::Gpu::Atomic::AddNoRet(&r(U, 0), w);
        if (mom) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::Gpu::Atomic::AddNoRet(&r(U, 1 + d), c[d] * w);
            }
        }
    }
}

} // namespace

template <typename LatticeTag>
void accumulateCoarseCrossingsMF(const amrex::MultiFab& fin, const amrex::iMultiFab& cmask,
                                 amrex::MultiFab& reg)
{
    BL_PROFILE("accumulateCoarseCrossingsMF()");
    AMREX_ALWAYS_ASSERT(fin.nGrow() >= 1 && cmask.nGrow() >= 1);

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(reg, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        auto const& f = fin.const_array(mfi);
        auto const& m = cmask.const_array(mfi);
        auto const& r = reg.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            if (m(iv) != 0) { return; }
            amrex::Real dm = amrex::Real(0.0);
            amrex::Real dj[AMREX_SPACEDIM] = {};
            for (int n = 1; n < LatticeTag::Q; ++n) {
                const amrex::IntVect c = latticeVector<LatticeTag>(n);
                amrex::Real a = amrex::Real(0.0);
                if (m(iv - c) == 1) { a += f(iv - c, n); }
                if (m(iv + c) == 1) { a -= f(iv, n); }
                dm += a;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { dj[d] += c[d] * a; }
            }
            r(iv, 0) += dm;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { r(iv, 1 + d) += dj[d]; }
        });
    }
}

template <typename LatticeTag>
void accumulateFineCrossingsMF(const amrex::MultiFab& fin, const amrex::iMultiFab& fmask,
                               const amrex::iMultiFab& cmask, amrex::MultiFab& reg,
                               amrex::Real weight)
{
    BL_PROFILE("accumulateFineCrossingsMF()");
    AMREX_ALWAYS_ASSERT(fin.nGrow() >= 1 && fmask.nGrow() >= 1);
    AMREX_ALWAYS_ASSERT(cmask.nGrow() >= 2 && reg.nGrow() >= 2);

    // reg and cmask share fin's DistributionMapping (they are defined on the
    // coarsened fine BoxArray), so the same MFIter index addresses all of
    // them. Several fine links share a coarse cell: accumulate atomically,
    // without tiling.
    for (amrex::MFIter mfi(fin); mfi.isValid(); ++mfi) {
        const amrex::Box vb = mfi.validbox();
        auto const& f  = fin.const_array(mfi);
        auto const& m  = fmask.const_array(mfi);
        auto const& cm = cmask.const_array(mfi);
        auto const& r  = reg.array(mfi);
        amrex::ParallelFor(vb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            for (int n = 1; n < LatticeTag::Q; ++n) {
                const amrex::IntVect c = latticeVector<LatticeTag>(n);
                if (m(iv - c) == 0) {        // pulled from a ghost
                    addFineCrossing(r, cm, iv - c, c, amrex::coarsen(iv - c, 2),
                                    weight * f(iv - c, n));
                }
                if (m(iv + c) == 0) {        // streamed into a ghost
                    addFineCrossing(r, cm, iv, c, amrex::coarsen(iv + c, 2),
                                    -weight * f(iv, n));
                }
            }
        });
    }
}

template <typename LatticeTag>
void applyRefluxMF(const amrex::MultiFab& reg, const amrex::iMultiFab& cmask,
                   amrex::MultiFab& fpost, amrex::MultiFab& fpre)
{
    BL_PROFILE("applyRefluxMF()");

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(reg, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        auto const& r    = reg.const_array(mfi);
        auto const& m    = cmask.const_array(mfi);
        auto const& post = fpost.array(mfi);
        auto const& pre  = fpre.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            if (m(iv) != 0) { return; }
            const amrex::Real drho = -r(iv, 0);
            for (int n = 0; n < LatticeTag::Q; ++n) {
                const amrex::IntVect c = latticeVector<LatticeTag>(n);
                amrex::Real cj = amrex::Real(0.0);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { cj -= c[d] * r(iv, 1 + d); }
                const amrex::Real df = LatticeTag::weights(n) * (drho + LatticeTag::cs2inv * cj);
                post(iv, n) += df;
                pre(iv, n)  += df;
            }
        });
    }
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
template void addInitialNonEquilibriumMF<D2Q9>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, const amrex::Geometry&);
template void accumulateCoarseCrossingsMF<D2Q9>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&);
template void accumulateFineCrossingsMF<D2Q9>(const amrex::MultiFab&, const amrex::iMultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::Real);
template void applyRefluxMF<D2Q9>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::MultiFab&);
#else
template void pullCollideFusedPreMF<D3Q15, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void pullCollideFusedPreMF<D3Q19, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void pullCollideFusedPreMF<D3Q27, BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&, bool);
template void convertPopulationsMF<D3Q15>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
template void convertPopulationsMF<D3Q19>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
template void convertPopulationsMF<D3Q27>(const amrex::MultiFab&, amrex::MultiFab*, int, amrex::MultiFab*, int, LevelParams const&, LevelParams const&, int);
template void addInitialNonEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, const amrex::Geometry&);
template void accumulateCoarseCrossingsMF<D3Q15>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&);
template void accumulateFineCrossingsMF<D3Q15>(const amrex::MultiFab&, const amrex::iMultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::Real);
template void applyRefluxMF<D3Q15>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void addInitialNonEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, const amrex::Geometry&);
template void accumulateCoarseCrossingsMF<D3Q19>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&);
template void accumulateFineCrossingsMF<D3Q19>(const amrex::MultiFab&, const amrex::iMultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::Real);
template void applyRefluxMF<D3Q19>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void addInitialNonEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real, const amrex::Geometry&);
template void accumulateCoarseCrossingsMF<D3Q27>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&);
template void accumulateFineCrossingsMF<D3Q27>(const amrex::MultiFab&, const amrex::iMultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::Real);
template void applyRefluxMF<D3Q27>(const amrex::MultiFab&, const amrex::iMultiFab&, amrex::MultiFab&, amrex::MultiFab&);
#endif

} // namespace muphfasa
