// LatticeDescriptor.H — Compile-time descriptors for LBM velocity sets.
// Provides D2Q9, D3Q15, D3Q19, and D3Q27 lattices: discrete velocities,
// quadrature weights, opposite-direction indices, and the lattice sound speed.
// All data is accessible from both host and GPU device code.

#ifndef MUPHFASA_LATTICE_DESCRIPTOR_H
#define MUPHFASA_LATTICE_DESCRIPTOR_H

#include <AMReX_GpuArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Primary template (specialisations below) ───────────────────────────────
template <int NDIM, int NQ>
struct LatticeDescriptor;

// ── D2Q9 ──────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<2, 9>
{
    static constexpr int D = 2;
    static constexpr int Q = 9;
    static constexpr amrex::Real cs2 = amrex::Real(1.0) / amrex::Real(3.0);

    // cx[q], cy[q]: integer components of the q-th discrete velocity
    AMREX_GPU_HOST_DEVICE
    static constexpr amrex::GpuArray<int, 9> cx() noexcept;

    AMREX_GPU_HOST_DEVICE
    static constexpr amrex::GpuArray<int, 9> cy() noexcept;

    // w[q]: lattice quadrature weights (sum to 1)
    AMREX_GPU_HOST_DEVICE
    static constexpr amrex::GpuArray<amrex::Real, 9> weights() noexcept;

    // opposite[q]: index r such that c_r = -c_q  (used in bounce-back BCs)
    AMREX_GPU_HOST_DEVICE
    static constexpr amrex::GpuArray<int, 9> opposite() noexcept;
};

// ── D3Q15 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 15>
{
    static constexpr int D = 3;
    static constexpr int Q = 15;
    static constexpr amrex::Real cs2 = amrex::Real(1.0) / amrex::Real(3.0);

    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 15> cx() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 15> cy() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 15> cz() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<amrex::Real, 15> weights() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 15> opposite() noexcept;
};

// ── D3Q19 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 19>
{
    static constexpr int D = 3;
    static constexpr int Q = 19;
    static constexpr amrex::Real cs2 = amrex::Real(1.0) / amrex::Real(3.0);

    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 19> cx() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 19> cy() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 19> cz() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<amrex::Real, 19> weights() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 19> opposite() noexcept;
};

// ── D3Q27 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 27>
{
    static constexpr int D = 3;
    static constexpr int Q = 27;
    static constexpr amrex::Real cs2 = amrex::Real(1.0) / amrex::Real(3.0);

    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 27> cx() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 27> cy() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 27> cz() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<amrex::Real, 27> weights() noexcept;
    AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<int, 27> opposite() noexcept;
};

// ── Convenience aliases ────────────────────────────────────────────────────
using D2Q9  = LatticeDescriptor<2,  9>;
using D3Q15 = LatticeDescriptor<3, 15>;
using D3Q19 = LatticeDescriptor<3, 19>;
using D3Q27 = LatticeDescriptor<3, 27>;

} // namespace muphfasa

#endif // MUPHFASA_LATTICE_DESCRIPTOR_H
