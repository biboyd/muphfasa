// BCSetup.H — Builds the BCManager<LatticeTag> from the input file.
// Shared by the single-level (SimulationImpl) and multi-level
// (AmrSimulationImpl) solvers. Wall velocities/densities are stored in
// lattice units, which are level invariant under acoustic scaling, so one
// BCManager serves every AMR level.

#ifndef MUPHFASA_BC_SETUP_H
#define MUPHFASA_BC_SETUP_H

#include "LBM/BoundaryConditions.H"

#include <AMReX_Geometry.H>

namespace muphfasa {

class InputParser;
class UnitConverter;

template <typename LatticeTag>
BCManager<LatticeTag> makeBCManager(const InputParser& params,
                                    const UnitConverter& units,
                                    const amrex::Geometry& geom);

} // namespace muphfasa

#endif // MUPHFASA_BC_SETUP_H
