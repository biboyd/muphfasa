// BCSetup.cpp — see BCSetup.H.

#include "BCSetup.H"

#include "IO/InputParser.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Units.H"

namespace muphfasa {

template <typename LatticeTag>
BCManager<LatticeTag> makeBCManager(const InputParser& params,
                                    const UnitConverter& units,
                                    const amrex::Geometry& geom)
{
    const auto names = params.bcNames();

    // wall/inflow params in lattice units
    const auto rhoWall = units.toLatticeDensity(params.refDensity());
    auto Uwall = params.Uwall();
    auto Uin = params.Uinflow();
    const auto faceRho = params.faceDensity();
    for (int wall_idx=0; wall_idx < 2 * AMREX_SPACEDIM; ++wall_idx) {
        for (int dim=0; dim < AMREX_SPACEDIM; ++dim) {
            Uwall[wall_idx][dim] = units.toLatticeVelocity(Uwall[wall_idx][dim]);
            Uin[wall_idx][dim]   = units.toLatticeVelocity(Uin[wall_idx][dim]);
        }
    }
    BCManager<LatticeTag> bc_manager{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int bound : {-1, 1}) {
            const std::string& name = names[faceIndex(dir, bound)];
            if (name == "Periodic") {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    geom.isPeriodic(dir),
                    "buildBCManager: bc input requests \"Periodic\" on a "
                    "face whose Geometry is not periodic in that direction");
                continue; // no lbmBCtype to set; FillBoundary handles it
            }
            bc_manager.setFaceBC(dir, bound, parselbmBCtype(name));
            if (name == "MovingWall") {
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir+1]);
            } else if (name == "NoSlip") {
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero{AMREX_D_DECL(0.,0.,0.)};
                bc_manager.setFaceVel(dir, bound, zero);
            } else if (name == "Inflow") {
                // reuses the reference density as rho_in; see InputParser::Uinflow()
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir+1]);
            } else if (name == "PressureOutlet") {
                bc_manager.setFaceDensity(dir, bound,
                    units.toLatticeDensity(faceRho[faceIndex(dir, bound)]));
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero{AMREX_D_DECL(0.,0.,0.)};
                bc_manager.setFaceVel(dir, bound, zero);
            } else if (name == "Outflow") {
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero{AMREX_D_DECL(0.,0.,0.)};
                bc_manager.setFaceVel(dir, bound, zero);
            }
        }
    }
    bc_manager.finalize();
    return bc_manager;
}

#if (AMREX_SPACEDIM == 2)
template BCManager<D2Q9> makeBCManager<D2Q9>(const InputParser&, const UnitConverter&, const amrex::Geometry&);
#else
template BCManager<D3Q15> makeBCManager<D3Q15>(const InputParser&, const UnitConverter&, const amrex::Geometry&);
template BCManager<D3Q19> makeBCManager<D3Q19>(const InputParser&, const UnitConverter&, const amrex::Geometry&);
template BCManager<D3Q27> makeBCManager<D3Q27>(const InputParser&, const UnitConverter&, const amrex::Geometry&);
#endif

} // namespace muphfasa
