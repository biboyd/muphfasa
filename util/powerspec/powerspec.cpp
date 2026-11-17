// powerspec.cpp — Shell-averaged kinetic energy spectrum E(k) of a muphfasa plotfile.
//
// Reads the velocity (vx, vy[, vz]) from a single-level plotfile, transforms it
// with amrex::FFT::R2C, and sums |u_hat|^2 / (2 N^2) into spherical shells of
// wavenumber. Based on amrex-tutorials FFT/EnergySpectrum. The field is
// assumed periodic; velocities are the physical (SI) values muphfasa writes,
// so E is per unit mass.
//
// Wavenumber shells: dk = 2 pi / L_max and shell n holds the modes with
// |k| / dk in [n - 1/2, n + 1/2), where k_i = 2 pi m_i / L_i. For a cubic
// domain n is just the integer |m|. For a box with unequal sides the shells
// still use L_max, so the finer sides resolve the spectrum more finely than dk.
//
// Usage:  muphfasa_powerspec <plotfile> [outfile=spectrum.txt]
// The executable is built for one dimensionality (AMREX_SPACEDIM, 3 by
// default): it must match the plotfile.

#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using amrex::Real;
constexpr int D = AMREX_SPACEDIM;

// Dimensionality stored in <plotfile>/Header (line 1: version, 2: ncomp,
// then ncomp variable names, then the space dimension), or -1 if unreadable.
int plotfileDim(const std::string& plotfile)
{
    std::ifstream hdr(plotfile + "/Header");
    std::string line;
    int ncomp = 0, dim = -1;
    if (!std::getline(hdr, line) || !(hdr >> ncomp)) { return -1; }
    std::getline(hdr, line);
    for (int i = 0; i < ncomp; ++i) { std::getline(hdr, line); }
    return (hdr >> dim) ? dim : -1;
}

struct Spectrum {
    std::vector<Real> e_shell;   // sum of |u_hat|^2/(2 N^2) in each shell
    Real dk = 0;                 // shell width [rad/m]
    Real ke_phys = 0;            // 1/2 <|u|^2> in physical space
};

Spectrum computeSpectrum(const amrex::MultiFab& vel, const amrex::Box& domain,
                         const std::array<Real, D>& length)
{
    using namespace amrex;

    // Batched forward transform: all velocity components in one call.
    FFT::Info info{};
    info.setBatchSize(D);
    FFT::R2C<Real, FFT::Direction::forward> r2c(domain, info);
    auto const& [cba, cdm] = r2c.getSpectralDataLayout();
    cMultiFab vel_hat(cba, cdm, D, 0);
    r2c.forward(vel, vel_hat);

    // Mode index m_i -> wavenumber in units of dk: scale_i * m_i.
    const Real lmax = *std::max_element(length.begin(), length.end());
    GpuArray<Real, 3> scale{Real(1), Real(1), Real(1)};
    GpuArray<int, 3> n{1, 1, 1};
    for (int d = 0; d < D; ++d) {
        scale[d] = lmax / length[d];
        n[d] = domain.length(d);
    }

    Real kmax2 = 0;
    for (int d = 0; d < D; ++d) {
        const Real kd = Real(n[d] / 2) * scale[d];
        kmax2 += kd * kd;
    }
    const int nshells = int(std::sqrt(kmax2)) + 2;

    const Real npts = domain.d_numPts();
    const Real fac = Real(0.5) / (npts * npts);

    Gpu::DeviceVector<Real> ek_d(nshells, Real(0));
    Real* pek = ek_d.data();
    auto const& vh = vel_hat.const_arrays();
    ParallelFor(vel_hat, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
    {
        // Only kx >= 0 is stored. Indices above n/2 are negative wavenumbers.
        const int ky = (j <= n[1] / 2) ? j : j - n[1];
        const int kz = (k <= n[2] / 2) ? k : k - n[2];
        const Real kx_s = Real(i) * scale[0];
        const Real ky_s = Real(ky) * scale[1];
        const Real kz_s = Real(kz) * scale[2];
        const int shell = int(std::round(std::sqrt(kx_s * kx_s + ky_s * ky_s + kz_s * kz_s)));
        Real e = 0;
        for (int c = 0; c < D; ++c) { e += amrex::norm(vh[b](i, j, k, c)); }
        // Hermitian symmetry u_hat(-k) = conj(u_hat(k)): except for kx = 0 and
        // the Nyquist mode, each stored point stands for two modes.
        if (i > 0 && 2 * i != n[0]) { e *= Real(2); }
        HostDevice::Atomic::Add(pek + shell, e * fac);
    });

    Spectrum s;
    s.e_shell.resize(nshells);
    Gpu::copy(Gpu::deviceToHost, ek_d.begin(), ek_d.end(), s.e_shell.begin());
    ParallelDescriptor::ReduceRealSum(s.e_shell.data(), nshells);
    s.dk = Real(2) * Math::pi<Real>() / lmax;
    s.ke_phys = Real(0.5) * MultiFab::Dot(vel, 0, vel, 0, D, 0) / npts;
    return s;
}

int run(const std::string& plotfile, const std::string& outfile)
{
    amrex::PlotFileData pf(plotfile);
    if (pf.finestLevel() > 0) {
        amrex::Print() << "WARNING: " << plotfile << " has " << pf.finestLevel() + 1
                       << " levels; using level 0 only.\n";
    }

    // Velocity components by name (muphfasa writes density, vx, vy[, vz]).
    static const char* names[] = {"vx", "vy", "vz"};
    const auto& vars = pf.varNames();
    int icomp[D];
    for (int d = 0; d < D; ++d) {
        const auto it = std::find(vars.begin(), vars.end(), std::string(names[d]));
        if (it == vars.end()) {
            throw std::runtime_error(std::string("variable \"") + names[d] + "\" not found in " + plotfile);
        }
        icomp[d] = int(it - vars.begin());
    }

    const amrex::MultiFab all = pf.get(0);
    amrex::MultiFab vel(all.boxArray(), all.DistributionMap(), D, 0);
    for (int d = 0; d < D; ++d) { amrex::MultiFab::Copy(vel, all, icomp[d], d, 1, 0); }

    const amrex::Box domain = pf.probDomain(0);
    std::array<Real, D> length{};
    for (int d = 0; d < D; ++d) { length[d] = pf.probSize()[d]; }

    const Spectrum s = computeSpectrum(vel, domain, length);
    const Real ke_spec = std::accumulate(s.e_shell.begin(), s.e_shell.end(), Real(0));
    const Real parseval = std::abs(ke_spec - s.ke_phys) / std::max(s.ke_phys, Real(1e-300));

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream ofs(outfile);
        if (!ofs) { throw std::runtime_error("cannot write " + outfile); }
        ofs.precision(10);
        ofs << "# kinetic energy spectrum of " << plotfile << "\n"
            << "# time = " << pf.time() << " s, step = " << pf.levelStep(0) << ", domain = " << domain << "\n"
            << "# dk = 2 pi / L_max = " << s.dk << " rad/m;  E per unit mass\n"
            << "# shell  k[rad/m]  E_shell[m^2/s^2]  E(k)=E_shell/dk[m^3/s^2]\n";
        for (std::size_t n = 0; n < s.e_shell.size(); ++n) {
            ofs << n << " " << Real(n) * s.dk << " " << s.e_shell[n] << " " << s.e_shell[n] / s.dk << "\n";
        }
    }

    const std::size_t peak = std::max_element(s.e_shell.begin() + 1, s.e_shell.end()) - s.e_shell.begin();
    amrex::Print() << "powerspec: " << plotfile << "  (" << D << "-D, time = " << pf.time() << " s)\n"
                   << "  domain          " << domain << "\n"
                   << "  shell width dk  " << s.dk << " rad/m\n"
                   << "  kinetic energy  physical space " << s.ke_phys << ", spectral space " << ke_spec
                   << "  (Parseval rel. error " << parseval << ")\n"
                   << "  mean flow       " << s.e_shell[0] << "  (shell 0)\n"
                   << "  peak (k > 0)    shell " << peak << ", k = " << Real(peak) * s.dk << " rad/m\n"
                   << "  written to      " << outfile << "\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    const std::string first = argc > 1 ? argv[1] : "";
    if (argc < 2 || first == "-h" || first == "--help" || first.find('=') != std::string::npos) {
        std::cout << "Usage: " << argv[0] << " <plotfile> [outfile=spectrum.txt]\n"
                  << "Computes the shell-averaged kinetic energy spectrum E(k) of a muphfasa plotfile\n"
                  << "(periodic field assumed). Build dimension (" << D << "-D) must match the plotfile.\n";
        return argc < 2 ? 1 : 0;
    }
    const std::string plotfile = first;
    const int dim = plotfileDim(plotfile);
    if (dim < 0) {
        std::cerr << "ERROR: cannot read a plotfile Header in \"" << plotfile << "\"\n";
        return 1;
    }
    if (dim != D) {
        std::cerr << "ERROR: " << plotfile << " is " << dim << "-D but this executable is built for " << D
                  << "-D (rebuild with DIM=" << dim << " / -Dmuphfasa_DIM=" << dim << ").\n";
        return 1;
    }

    // AMReX would treat a bare argv[1] as an inputs file: hand it the key=value args only.
    std::vector<char*> amrex_args{argv[0]};
    for (int i = 2; i < argc; ++i) { amrex_args.push_back(argv[i]); }
    amrex_args.push_back(nullptr);   // argv must be null-terminated (MPI_Init reads argv[argc])

    int rc = 0;
    int amrex_argc = int(amrex_args.size()) - 1;
    char** amrex_argv = amrex_args.data();
    amrex::Initialize(amrex_argc, amrex_argv, std::function<void()>([] {
        amrex::ParmParse pp("amrex");
        pp.add("verbose", 0);
        pp.add("throw_exception", 1);
    }));
    {
        std::string outfile = "spectrum.txt";
        amrex::ParmParse().query("outfile", outfile);
        try {
            rc = run(plotfile, outfile);
        } catch (const std::exception& e) {
            amrex::Print() << "\nERROR: " << e.what() << "\n";
            rc = 1;
        }
    }
    amrex::Finalize();
    return rc;
}
