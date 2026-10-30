// AmrSimulation.H — Multi-level (block-structured AMR) LBM solver.
//
// Selected by makeSimulation() when amr.max_level > 0; the single-level
// SimulationImpl is untouched otherwise. Scheme (see amr_implementation.md):
//  * refinement ratio 2, acoustic scaling (dx, dt halve per level),
//    tau_{l+1} - 1/2 = 2 (tau_l - 1/2), F_{l+1} = F_l / 2;
//  * recursive subcycling (2 fine steps per coarse step);
//  * coarse/fine coupling through pre-collision non-equilibrium rescaling
//    (convertPopulationsMF): fine ghost cells are filled from converted
//    coarse data (space/time interpolated, FillPatchTwoLevels), covered
//    coarse cells are overwritten by converted, averaged fine data.
//
// All times are in level-0 lattice units (level l has dt = 2^-l).

#ifndef MUPHFASA_AMR_SIMULATION_H
#define MUPHFASA_AMR_SIMULATION_H

#include "ISimulation.H"
#include "LBM/AmrInterface.H"
#include "LBM/BoundaryConditions.H"

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>

#include <memory>

namespace muphfasa {

class InputParser;
class UnitConverter;

// Boundary functor used for every AMR fill. FillPatchTwoLevels needs one
// functor type for coarse and fine fills, so the LBM ghost fill
// (BCManager, active = true, with sentinel BCRecs) and the pure
// first-order extrapolation used for converted/interpolated data
// (active = false; GpuBndryFuncFab applies the foextrap BCRecs itself)
// share this wrapper.
template <typename LatticeTag>
struct AmrBndryFunc
{
    BCManager<LatticeTag> bcm{};
    bool active = true;

    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& dest,
                    int dcomp, int numcomp, amrex::GeometryData const& geom,
                    amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const
    {
        if (active) { bcm(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp); }
    }
};

template <typename LatticeTag, typename CollisionModel>
class AmrSimulationImpl : public ISimulation, public amrex::AmrCore
{
public:
    explicit AmrSimulationImpl(const InputParser& params);
    ~AmrSimulationImpl() override;

    // ── ISimulation lifecycle ─────────────────────────────────────────────
    void initialize() override;
    void advanceFused(int step) override;
    void advanceSep(int step) override;     // AMR uses the fused path only
    void updateMacroscopic() override;
    void writeOutput(int step) override;

    // ── State accessors (un-suffixed = level 0) ───────────────────────────
    const amrex::MultiFab& f()   const noexcept override { return m_fin[0]; }
    const amrex::MultiFab& rho() const noexcept override { return m_rho[0]; }
    const amrex::MultiFab& vel() const noexcept override { return m_vel[0]; }
    const amrex::MultiFab& f(int lev)   const noexcept override { return m_fin[lev]; }
    const amrex::MultiFab& rho(int lev) const noexcept override { return m_rho[lev]; }
    const amrex::MultiFab& vel(int lev) const noexcept override { return m_vel[lev]; }

    amrex::Real time() const noexcept override { return m_t_new[0]; }
    amrex::Real omega() const noexcept override { return amrex::Real(1.0) / m_lp[0].tau; }
    const UnitConverter& units() const noexcept override { return *m_units; }
    const amrex::Geometry& geom() const noexcept override { return Geom(0); }
    const amrex::Geometry& geom(int lev) const noexcept override { return Geom(lev); }
    int finestLevel() const noexcept override { return finest_level; }

protected:
    // ── AmrCore interface ─────────────────────────────────────────────────
    void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;
    void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override;
    void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override;
    void ClearLevel(int lev) override;
    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

private:
    const InputParser&             m_params;
    std::unique_ptr<UnitConverter> m_units;

    // Per-level tau, dt, lattice force (sized max_level + 1).
    amrex::Vector<LevelParams> m_lp;
    bool m_has_forcing = false;
    int  m_regrid_int  = 0;
    amrex::Real m_vort_thresh = amrex::Real(0.0);

    // ── Per-level state ───────────────────────────────────────────────────
    amrex::Vector<amrex::MultiFab> m_fin;       // post-collision, nGhost ghosts
    amrex::Vector<amrex::MultiFab> m_fout;      // scratch for the fused step
    amrex::Vector<amrex::MultiFab> m_fpre_new;  // pre-collision at m_t_new
    amrex::Vector<amrex::MultiFab> m_fpre_old;  // pre-collision at m_t_old
    amrex::Vector<amrex::MultiFab> m_rho;
    amrex::Vector<amrex::MultiFab> m_vel;
    amrex::Vector<amrex::Real>     m_t_new;
    amrex::Vector<amrex::Real>     m_t_old;

    // ── Boundary conditions ───────────────────────────────────────────────
    // LBM ghost fill (halfway bounce-back etc.) per level, and a
    // first-order-extrapolation fill for converted/interpolated data.
    using PhysBCFunctType = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<AmrBndryFunc<LatticeTag>>>;
    amrex::Vector<amrex::BCRec> m_bcs_lbm;     // sentinel BCRecs (BCManager)
    amrex::Vector<amrex::BCRec> m_bcs_interp;  // int_dir / foextrap (2Q comps)
    amrex::Vector<std::unique_ptr<PhysBCFunctType>>   m_phys_bc;
    amrex::Vector<std::unique_ptr<PhysBCFunctType>> m_extrap_bc;

    // ── Helpers ───────────────────────────────────────────────────────────
    void defineLevel(int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
    void timeStep(int lev);
    void averageDown(int crse_lev);
    void fillGhosts(int lev, amrex::Real time);
    // Converted coarse populations (pre- or post-collision of level clev+1)
    // on the BoxArray of level clev, from coarse pre-collision src.
    amrex::MultiFab convertToFine(int clev, const amrex::MultiFab& src_pre, bool post) const;
    // Fill dst (level lev, Q comps) at time from old fine data (may be
    // nullptr) and converted coarse data at m_t_new[lev-1].
    void fillFromLevels(int lev, amrex::Real time, amrex::MultiFab& dst,
                        const amrex::MultiFab* old_fine, bool post);
    void initLevelZero();
};

} // namespace muphfasa

#endif // MUPHFASA_AMR_SIMULATION_H
