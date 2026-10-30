// Plotfile.cpp — AMReX plotfile writing.
// Delegates to amrex::WriteSingleLevelPlotfile which produces output
// readable by VisIt and Paraview via the AMReX BoxLib I/O layer.

#include "Plotfile.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <iomanip>
#include <sstream>

namespace muphfasa {

void writePlotfile(const std::string&                          file_name,
                   const amrex::Vector<const amrex::MultiFab*>& mfs,
                   const amrex::Vector<std::string>&           var_names,
                   const amrex::Geometry&                      geom,
                   amrex::Real                                 time,
                   int                                         step)
{
    BL_PROFILE("writePlotfile()");

    if (mfs.empty()) {
        amrex::Print() << "Plotfile: nothing to write for " << file_name << "\n";
        return;
    }

    int ncomp_total = 0;
    for (const auto* mf : mfs) { ncomp_total += mf->nComp(); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ncomp_total == static_cast<int>(var_names.size()),
        "writePlotfile: var_names size does not match total component count");

    amrex::MultiFab plot_mf(mfs[0]->boxArray(), mfs[0]->DistributionMap(), ncomp_total, 0);

    int dcomp = 0;
    for (const auto* mf : mfs) {
        amrex::MultiFab::Copy(plot_mf, *mf, 0, dcomp, mf->nComp(), 0);
        dcomp += mf->nComp();
    }

    amrex::WriteSingleLevelPlotfile(file_name, plot_mf, var_names, geom, time, step);

    amrex::Print() << "Plotfile: wrote " << file_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void writeMacroPlotfile(const std::string&     file_name,
                        const amrex::MultiFab& rho_mf,
                        const amrex::MultiFab& vel_mf,
                        const amrex::Geometry& geom,
                        amrex::Real            time,
                        int                    step)
{
    BL_PROFILE("writeMacroPlotfile()");

    amrex::Vector<std::string> names = {"density"};
#if   (AMREX_SPACEDIM == 2)
    names.push_back("vx"); names.push_back("vy");
#elif (AMREX_SPACEDIM == 3)
    names.push_back("vx"); names.push_back("vy"); names.push_back("vz");
#endif

    const amrex::Vector<const amrex::MultiFab*> mfs = {&rho_mf, &vel_mf};
    writePlotfile(file_name, mfs, names, geom, time, step);
}

void writeMacroPlotfileMultiLevel(const std::string&                     file_name,
                                  const amrex::Vector<amrex::MultiFab>&  rho_mf,
                                  const amrex::Vector<amrex::MultiFab>&  vel_mf,
                                  const amrex::Vector<amrex::Geometry>&  geom,
                                  const amrex::Vector<amrex::IntVect>&   ref_ratio,
                                  amrex::Real                            time,
                                  int                                    step)
{
    BL_PROFILE("writeMacroPlotfileMultiLevel()");

    amrex::Vector<std::string> names = {"density"};
#if   (AMREX_SPACEDIM == 2)
    names.push_back("vx"); names.push_back("vy");
#elif (AMREX_SPACEDIM == 3)
    names.push_back("vx"); names.push_back("vy"); names.push_back("vz");
#endif

    const int nlevels = static_cast<int>(rho_mf.size());
    const int ncomp   = 1 + AMREX_SPACEDIM;
    amrex::Vector<amrex::MultiFab> plot_mf(nlevels);
    amrex::Vector<const amrex::MultiFab*> plot_ptr(nlevels);
    amrex::Vector<int> steps(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
        plot_mf[lev].define(rho_mf[lev].boxArray(), rho_mf[lev].DistributionMap(), ncomp, 0);
        amrex::MultiFab::Copy(plot_mf[lev], rho_mf[lev], 0, 0, 1, 0);
        amrex::MultiFab::Copy(plot_mf[lev], vel_mf[lev], 0, 1, AMREX_SPACEDIM, 0);
        plot_ptr[lev] = &plot_mf[lev];
        steps[lev] = step * (1 << lev);
    }

    amrex::WriteMultiLevelPlotfile(file_name, nlevels, plot_ptr, names, geom, time,
                                   steps, ref_ratio);

    amrex::Print() << "Plotfile: wrote " << file_name << "  levels=" << nlevels
                   << "  step=" << step << "  time=" << time << "\n";
}

std::string plotfileName(const std::string& prefix, int step)
{
    std::ostringstream ss;
    ss << prefix << std::setw(7) << std::setfill('0') << step;
    return ss.str();
}

} // namespace muphfasa
