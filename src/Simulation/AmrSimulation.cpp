// AmrSimulation.cpp — Multi-level LBM solver (see AmrSimulation.H and
// amr_implementation.md for the scheme).

#include "AmrSimulation.H"

#include "BCSetup.H"
#include "InitData.H"
#include "IO/Checkpoint.H"
#include "IO/InputParser.H"
#include "IO/Plotfile.H"
#include "Geometry/GeometryHandler.H"
#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"
#include "LBM/Streaming.H"
#include "LBM/Units.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_TagBox.H>

#include <cmath>

namespace muphfasa {

namespace {

amrex::Geometry makeLevel0Geometry(const InputParser& params)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!params.useEB(),
        "AMR (amr.max_level > 0) does not support embedded boundaries yet");
    GeometryHandler gh(params);
    return gh.geometry();
}

amrex::AmrInfo makeAmrInfo(const InputParser& params)
{
    amrex::AmrInfo info;
    info.max_level       = params.maxLevel();
    info.ref_ratio       = {amrex::IntVect(2)};
    info.blocking_factor = {amrex::IntVect(params.blockingFactor())};
    info.max_grid_size   = {amrex::IntVect(params.maxGridSize())};
    info.n_error_buf     = {amrex::IntVect(params.nErrorBuf())};
    info.grid_eff        = params.gridEff();
    // Coarse/fine stencil: 1 fine ghost cell + 1 coarse cell of
    // interpolation slope -> keep two coarse cells of proper nesting.
    info.n_proper        = 2;
    return info;
}

} // namespace

template <typename LatticeTag, typename CollisionModel>
AmrSimulationImpl<LatticeTag, CollisionModel>::AmrSimulationImpl(const InputParser& params)
    : amrex::AmrCore(makeLevel0Geometry(params), makeAmrInfo(params)),
      m_params(params)
{
    auto Ncells = params.ncells();
    int max_N = Ncells[0];
    for (auto n : Ncells) { max_N = (n > max_N) ? n : max_N; }

    // Units are defined on level 0, exactly as in the single-level solver.
    m_units = std::make_unique<UnitConverter>(params.refLength(),
                                              params.refVelocity(),
                                              params.reynoldsNumber(),
                                              params.refDensity(),
                                              max_N,
                                              LatticeTag::cs2,
                                              params.latticeVelocity(),
                                              params.relaxationTime());

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> F0{};
    const auto accel_phys = params.forcingAccel();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        F0[d] = m_units->toLatticeAcceleration(accel_phys[d]);
        if (accel_phys[d] != amrex::Real(0.0)) { m_has_forcing = true; }
    }

    // Acoustic scaling per level.
    const int nlev = max_level + 1;
    m_lp.resize(nlev);
    for (int lev = 0; lev < nlev; ++lev) {
        const amrex::Real fac = amrex::Real(1 << lev);
        m_lp[lev].tau = fac * (m_units->relaxationTime() - amrex::Real(0.5)) + amrex::Real(0.5);
        m_lp[lev].dt  = amrex::Real(1.0) / fac;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { m_lp[lev].F[d] = F0[d] / fac; }
        amrex::Print() << "AMR level " << lev << ": tau = " << m_lp[lev].tau
                       << ", dt = " << m_lp[lev].dt << "\n";
    }

    m_regrid_int  = params.regridInterval();
    m_vort_thresh = params.vorticityThreshold();

    m_fin.resize(nlev);
    m_fout.resize(nlev);
    m_fpre_new.resize(nlev);
    m_fpre_old.resize(nlev);
    m_rho.resize(nlev);
    m_vel.resize(nlev);
    m_t_new.resize(nlev, amrex::Real(0.0));
    m_t_old.resize(nlev, amrex::Real(0.0));

    // Boundary conditions: one BCManager (lattice units are level
    // invariant), one PhysBCFunct per level geometry.
    const auto names = params.bcNames();
    BCManager<LatticeTag> bc_manager = makeBCManager<LatticeTag>(params, *m_units, Geom(0));
    m_bcs_lbm = makeBCRecVector(names, LatticeTag::Q);

    m_bcs_interp.resize(2 * LatticeTag::Q);
    for (auto& bcr : m_bcs_interp) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const bool per = Geom(0).isPeriodic(dir);
            bcr.setLo(dir, per ? amrex::BCType::int_dir : amrex::BCType::foextrap);
            bcr.setHi(dir, per ? amrex::BCType::int_dir : amrex::BCType::foextrap);
        }
    }

    m_phys_bc.resize(nlev);
    m_extrap_bc.resize(nlev);
    for (int lev = 0; lev < nlev; ++lev) {
        amrex::GpuBndryFuncFab<AmrBndryFunc<LatticeTag>> lbm_func(
            AmrBndryFunc<LatticeTag>{bc_manager, true});
        m_phys_bc[lev] = std::make_unique<PhysBCFunctType>(Geom(lev), m_bcs_lbm, lbm_func);
        amrex::GpuBndryFuncFab<AmrBndryFunc<LatticeTag>> extrap_func(
            AmrBndryFunc<LatticeTag>{bc_manager, false});
        m_extrap_bc[lev] = std::make_unique<PhysBCFunctType>(Geom(lev), m_bcs_interp, extrap_func);
    }
}

template <typename LatticeTag, typename CollisionModel>
AmrSimulationImpl<LatticeTag, CollisionModel>::~AmrSimulationImpl() = default;

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::initialize()
{
    InitFromScratch(amrex::Real(0.0));
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::Print() << "AMR level " << lev << " grids: " << grids[lev].size()
                       << ", cells: " << grids[lev].numPts() << "\n";
    }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::defineLevel(
    int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    m_fin[lev].define     (ba, dm, LatticeTag::Q, LatticeTag::nGhost);
    m_fout[lev].define    (ba, dm, LatticeTag::Q, LatticeTag::nGhost);
    m_fpre_new[lev].define(ba, dm, LatticeTag::Q, 0);
    m_fpre_old[lev].define(ba, dm, LatticeTag::Q, 0);
    m_rho[lev].define     (ba, dm, 1, 0);
    m_vel[lev].define     (ba, dm, AMREX_SPACEDIM, 0);
    m_fout[lev].setVal(amrex::Real(0.0));
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::initLevelZero()
{
    // Same initial condition as SimulationImpl::setInitialConditions().
    const auto& F = m_lp[0].F;
    initData(m_rho[0], m_vel[0], *m_units);
    if (!m_has_forcing) {
        computeEquilibriumMF<LatticeTag>(m_rho[0], m_vel[0], m_fin[0]);
    } else {
        initVelCorrection(m_rho[0], m_vel[0], F);
        computeEquilibriumMF<LatticeTag>(m_rho[0], m_vel[0], m_fin[0]);
        computeMacroscopic<LatticeTag>(m_fin[0], m_rho[0], m_vel[0], F);
    }
    // At equilibrium pre- and post-collision populations coincide.
    amrex::MultiFab::Copy(m_fpre_new[0], m_fin[0], 0, 0, LatticeTag::Q, 0);
    amrex::MultiFab::Copy(m_fpre_old[0], m_fin[0], 0, 0, LatticeTag::Q, 0);
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::MakeNewLevelFromScratch(
    int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    defineLevel(lev, ba, dm);
    m_t_new[lev] = time;
    m_t_old[lev] = time;

    if (lev == 0) {
        initLevelZero();
        fillGhosts(0, time);
    } else {
        // Fine levels start from the (converted, interpolated) coarse state,
        // so any initData() override only needs to handle level 0.
        fillFromLevels(lev, time, m_fin[lev], nullptr, true);
        fillFromLevels(lev, time, m_fpre_new[lev], nullptr, false);
        amrex::MultiFab::Copy(m_fpre_old[lev], m_fpre_new[lev], 0, 0, LatticeTag::Q, 0);
        fillGhosts(lev, time);
    }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::MakeNewLevelFromCoarse(
    int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    defineLevel(lev, ba, dm);
    m_t_new[lev] = time;
    m_t_old[lev] = time;
    fillFromLevels(lev, time, m_fin[lev], nullptr, true);
    fillFromLevels(lev, time, m_fpre_new[lev], nullptr, false);
    amrex::MultiFab::Copy(m_fpre_old[lev], m_fpre_new[lev], 0, 0, LatticeTag::Q, 0);
    fillGhosts(lev, time);
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::RemakeLevel(
    int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    amrex::MultiFab new_fin (ba, dm, LatticeTag::Q, LatticeTag::nGhost);
    amrex::MultiFab new_fpre(ba, dm, LatticeTag::Q, 0);
    if (lev == 0) {
        new_fin.ParallelCopy(m_fin[0], 0, 0, LatticeTag::Q, 0, 0, Geom(0).periodicity());
        new_fpre.ParallelCopy(m_fpre_new[0], 0, 0, LatticeTag::Q, 0, 0, Geom(0).periodicity());
    } else {
        fillFromLevels(lev, time, new_fin, &m_fin[lev], true);
        fillFromLevels(lev, time, new_fpre, &m_fpre_new[lev], false);
    }
    defineLevel(lev, ba, dm);
    std::swap(m_fin[lev], new_fin);
    std::swap(m_fpre_new[lev], new_fpre);
    amrex::MultiFab::Copy(m_fpre_old[lev], m_fpre_new[lev], 0, 0, LatticeTag::Q, 0);
    m_t_new[lev] = time;
    m_t_old[lev] = time;
    fillGhosts(lev, time);
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::ClearLevel(int lev)
{
    m_fin[lev].clear();
    m_fout[lev].clear();
    m_fpre_new[lev].clear();
    m_fpre_old[lev].clear();
    m_rho[lev].clear();
    m_vel[lev].clear();
}

// ── Coarse/fine coupling ──────────────────────────────────────────────────

template <typename LatticeTag, typename CollisionModel>
amrex::MultiFab AmrSimulationImpl<LatticeTag, CollisionModel>::convertToFine(
    int clev, const amrex::MultiFab& src_pre, bool post) const
{
    amrex::MultiFab out(src_pre.boxArray(), src_pre.DistributionMap(), LatticeTag::Q, 0);
    convertPopulationsMF<LatticeTag>(src_pre,
                                     post ? &out : nullptr, 0,
                                     post ? nullptr : &out, 0,
                                     m_lp[clev], m_lp[clev + 1], 0);
    return out;
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::fillGhosts(int lev, amrex::Real time)
{
    BL_PROFILE("AmrSimulationImpl::fillGhosts()");

    if (lev == 0) {
        m_fin[0].FillBoundary(Geom(0).periodicity());
        (*m_phys_bc[0])(m_fin[0], 0, LatticeTag::Q, m_fin[0].nGrowVect(), time, 0);
        return;
    }

    // Converted coarse post-collision data at the coarse time level(s)
    // bracketing `time` (linear interpolation in time inside FillPatch).
    const int clev = lev - 1;
    const amrex::Real told = m_t_old[clev];
    const amrex::Real tnew = m_t_new[clev];
    const amrex::Real eps  = amrex::Real(1.e-8) * m_lp[lev].dt;

    amrex::MultiFab c_old, c_new;
    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ct;
    if (std::abs(time - tnew) < eps || std::abs(tnew - told) < eps) {
        c_new = convertToFine(clev, m_fpre_new[clev], true);
        cmf = {&c_new};
        ct  = {time};
    } else if (std::abs(time - told) < eps) {
        c_old = convertToFine(clev, m_fpre_old[clev], true);
        cmf = {&c_old};
        ct  = {time};
    } else {
        AMREX_ALWAYS_ASSERT(time > told && time < tnew);
        c_old = convertToFine(clev, m_fpre_old[clev], true);
        c_new = convertToFine(clev, m_fpre_new[clev], true);
        cmf = {&c_old, &c_new};
        ct  = {told, tnew};
    }

    amrex::Vector<amrex::MultiFab*> fmf = {&m_fin[lev]};
    amrex::Vector<amrex::Real> ft = {time};
    amrex::FillPatchTwoLevels(m_fin[lev], time, cmf, ct, fmf, ft,
                              0, 0, LatticeTag::Q, Geom(clev), Geom(lev),
                              *m_extrap_bc[clev], 0, *m_phys_bc[lev], 0,
                              refRatio(clev), &amrex::lincc_interp, m_bcs_interp, 0);
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::fillFromLevels(
    int lev, amrex::Real time, amrex::MultiFab& dst, const amrex::MultiFab* old_fine, bool post)
{
    // Coarse data converted at the coarse level's current (new) time; used
    // only at synchronised times (initialisation and regridding).
    const int clev = lev - 1;
    amrex::MultiFab conv = convertToFine(clev, m_fpre_new[clev], post);
    amrex::Vector<amrex::MultiFab*> cmf = {&conv};
    amrex::Vector<amrex::Real> ct = {time};
    PhysBCFunctType& fbc = post ? *m_phys_bc[lev] : *m_extrap_bc[lev];

    if (old_fine == nullptr) {
        amrex::InterpFromCoarseLevel(dst, time, conv, 0, 0, LatticeTag::Q,
                                     Geom(clev), Geom(lev),
                                     *m_extrap_bc[clev], 0, fbc, 0,
                                     refRatio(clev), &amrex::lincc_interp, m_bcs_interp, 0);
    } else {
        amrex::Vector<amrex::MultiFab*> fmf = {const_cast<amrex::MultiFab*>(old_fine)};
        amrex::Vector<amrex::Real> ft = {time};
        amrex::FillPatchTwoLevels(dst, time, cmf, ct, fmf, ft,
                                  0, 0, LatticeTag::Q, Geom(clev), Geom(lev),
                                  *m_extrap_bc[clev], 0, fbc, 0,
                                  refRatio(clev), &amrex::lincc_interp, m_bcs_interp, 0);
    }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::averageDown(int crse_lev)
{
    BL_PROFILE("AmrSimulationImpl::averageDown()");

    // Fine pre-collision populations (at the synchronised time) are averaged
    // onto the coarsened fine grids, converted fine -> coarse, and copied
    // over the covered coarse cells (post- and pre-collision).
    const int flev = crse_lev + 1;
    const amrex::IntVect ratio = refRatio(crse_lev);
    const amrex::BoxArray cba = amrex::coarsen(grids[flev], ratio);
    const amrex::DistributionMapping& fdm = dmap[flev];

    amrex::MultiFab avg (cba, fdm, LatticeTag::Q, 0);
    amrex::MultiFab post(cba, fdm, LatticeTag::Q, 0);
    amrex::MultiFab pre (cba, fdm, LatticeTag::Q, 0);
    amrex::average_down(m_fpre_new[flev], avg, 0, LatticeTag::Q, ratio);
    convertPopulationsMF<LatticeTag>(avg, &post, 0, &pre, 0, m_lp[flev], m_lp[crse_lev], 0);

    m_fin[crse_lev].ParallelCopy(post, 0, 0, LatticeTag::Q);
    m_fpre_new[crse_lev].ParallelCopy(pre, 0, 0, LatticeTag::Q);
}

// ── Time stepping ─────────────────────────────────────────────────────────

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::timeStep(int lev)
{
    // Ghosts of fine levels are filled at the start of each substep, after
    // the coarse covered cells at the start time have been averaged down.
    if (lev > 0) { fillGhosts(lev, m_t_new[lev]); }

    std::swap(m_fpre_old[lev], m_fpre_new[lev]);
    const amrex::Real omega = amrex::Real(1.0) / m_lp[lev].tau;
    pullCollideFusedPreMF<LatticeTag, CollisionModel>(m_fin[lev], m_fout[lev], m_fpre_new[lev],
                                                      omega, m_lp[lev].F, m_has_forcing);
    std::swap(m_fin[lev], m_fout[lev]);
    m_t_old[lev] = m_t_new[lev];
    m_t_new[lev] += m_lp[lev].dt;

    if (lev < finest_level) {
        timeStep(lev + 1);
        timeStep(lev + 1);
        averageDown(lev);
    }

    if (lev == 0) { fillGhosts(0, m_t_new[0]); }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::advanceFused(int step)
{
    BL_PROFILE("AmrSimulationImpl::advanceFused()");

    timeStep(0);

    if (m_regrid_int > 0 && max_level > 0 && (step + 1) % m_regrid_int == 0) {
        regrid(0, m_t_new[0]);
    }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::advanceSep(int step)
{
    amrex::ignore_unused(step);
    amrex::Abort("AmrSimulationImpl: only the fused stream-collide path supports AMR");
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::updateMacroscopic()
{
    // As in SimulationImpl: macroscopic fields from the streamed populations.
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (lev > 0) { fillGhosts(lev, m_t_new[lev]); }
        stream<LatticeTag>(m_fin[lev], m_fout[lev], Geom(lev));
        if (!m_has_forcing) {
            computeMacroscopic<LatticeTag>(m_fout[lev], m_rho[lev], m_vel[lev]);
        } else {
            computeMacroscopic<LatticeTag>(m_fout[lev], m_rho[lev], m_vel[lev], m_lp[lev].F);
        }
    }
}

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::writeOutput(int step)
{
    BL_PROFILE("AmrSimulationImpl::writeOutput()");

    if (m_params.plotInterval() > 0 && step % m_params.plotInterval() == 0) {
        updateMacroscopic();

        const std::string fname = plotfileName(m_params.plotfilePrefix(), step);
        const amrex::Real t_phys = m_units->toPhysicalTime(m_t_new[0]);
        const int nlevels = finest_level + 1;

        amrex::Vector<amrex::MultiFab> rho_phys(nlevels), vel_phys(nlevels);
        amrex::Vector<amrex::Geometry> geoms(nlevels);
        amrex::Vector<amrex::IntVect> ratios(nlevels - 1);
        for (int lev = 0; lev < nlevels; ++lev) {
            rho_phys[lev].define(grids[lev], dmap[lev], 1, 0);
            vel_phys[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
            amrex::MultiFab::Copy(rho_phys[lev], m_rho[lev], 0, 0, 1, 0);
            amrex::MultiFab::Copy(vel_phys[lev], m_vel[lev], 0, 0, AMREX_SPACEDIM, 0);
            m_units->toPhysical(rho_phys[lev], vel_phys[lev]);
            geoms[lev] = Geom(lev);
            if (lev < nlevels - 1) { ratios[lev] = refRatio(lev); }
        }
        writeMacroPlotfileMultiLevel(fname, rho_phys, vel_phys, geoms, ratios, t_phys, step);
    }

    if (m_params.checkpointInterval() > 0 && step % m_params.checkpointInterval() == 0) {
        // Only level 0 for now (restart is not supported by either solver).
        const std::string cname = checkpointName(m_params.checkpointPrefix(), step);
        writeCheckpoint(cname, m_fin[0], Geom(0), step, m_t_new[0]);
    }
}

// ── Tagging ───────────────────────────────────────────────────────────────

template <typename LatticeTag, typename CollisionModel>
void AmrSimulationImpl<LatticeTag, CollisionModel>::ErrorEst(
    int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
    amrex::ignore_unused(time, ngrow);

    std::vector<amrex::Real> blo, bhi;
    const bool has_box = m_params.tagBox(lev, blo, bhi);
    const bool use_vort = (m_vort_thresh > amrex::Real(0.0));
    if (!has_box && !use_vort) { return; }

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo{}, hi{};
    if (has_box) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { lo[d] = blo[d]; hi[d] = bhi[d]; }
    }

    // Velocity (Guo) from the current pre-collision populations, with one
    // ghost layer for centred differences. Ghosts that FillBoundary cannot
    // fill (non-periodic domain faces, coarse/fine boundaries) keep a
    // sentinel and cells whose stencil touches one are not vorticity-tagged.
    constexpr amrex::Real kUnset = amrex::Real(1.e30);
    amrex::MultiFab rho, vel;
    if (use_vort) {
        rho.define(grids[lev], dmap[lev], 1, 0);
        vel.define(grids[lev], dmap[lev], AMREX_SPACEDIM, 1);
        vel.setVal(kUnset);
        computeMacroscopic<LatticeTag>(m_fpre_new[lev], rho, vel, m_lp[lev].F);
        vel.FillBoundary(Geom(lev).periodicity());
    }
    // Lattice vorticity of level lev in units of level-0 lattice spacing.
    const amrex::Real vscale = amrex::Real(1 << lev);
    const amrex::Real vthresh = m_vort_thresh;

    for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.validbox();
        auto const& tag = tags.array(mfi);
        amrex::Array4<amrex::Real const> u = use_vort ? vel.const_array(mfi)
                                                      : amrex::Array4<amrex::Real const>{};
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::ignore_unused(k);
            bool flag = false;
            if (has_box) {
                bool inside = true;
                const int iv[3] = {i, j, k};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    const amrex::Real x = problo[d] + (iv[d] + amrex::Real(0.5)) * dx[d];
                    inside = inside && (x >= lo[d]) && (x <= hi[d]);
                }
                flag = inside;
            }
            if (use_vort) {
                bool valid = true;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                        valid = valid && u(i+di, j+dj, k+dk, c) < kUnset
                                      && u(i-di, j-dj, k-dk, c) < kUnset;
                    }
                }
                if (valid) {
#if (AMREX_SPACEDIM == 2)
                    const amrex::Real w = amrex::Real(0.5) *
                        ((u(i+1,j,k,1) - u(i-1,j,k,1)) - (u(i,j+1,k,0) - u(i,j-1,k,0)));
                    const amrex::Real wmag = std::abs(w);
#else
                    const amrex::Real wx = amrex::Real(0.5) *
                        ((u(i,j+1,k,2) - u(i,j-1,k,2)) - (u(i,j,k+1,1) - u(i,j,k-1,1)));
                    const amrex::Real wy = amrex::Real(0.5) *
                        ((u(i,j,k+1,0) - u(i,j,k-1,0)) - (u(i+1,j,k,2) - u(i-1,j,k,2)));
                    const amrex::Real wz = amrex::Real(0.5) *
                        ((u(i+1,j,k,1) - u(i-1,j,k,1)) - (u(i,j+1,k,0) - u(i,j-1,k,0)));
                    const amrex::Real wmag = std::sqrt(wx*wx + wy*wy + wz*wz);
#endif
                    if (wmag * vscale > vthresh) { flag = true; }
                }
            }
            if (flag) { tag(i, j, k) = amrex::TagBox::SET; }
        });
    }
}

// ── Explicit instantiations ─────────────────────────────────────────────────
#if (AMREX_SPACEDIM == 2)
template class AmrSimulationImpl<D2Q9, BGK>;
#else
template class AmrSimulationImpl<D3Q15, BGK>;
template class AmrSimulationImpl<D3Q19, BGK>;
template class AmrSimulationImpl<D3Q27, BGK>;
#endif

} // namespace muphfasa
