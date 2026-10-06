// LatticeDescriptor.cpp — Definitions of lattice velocity sets.
// Each specialisation returns compile-time constant arrays for the discrete
// velocities, quadrature weights, and opposite-direction indices.
// All values are standard, well-documented choices from the LBM literature.

#include "LatticeDescriptor.H"

namespace muphfasa {

// ── D2Q9 ──────────────────────────────────────────────────────────────────
//   q:  0   1   2   3   4   5   6   7   8
// (cx): 0   1   0  -1   0   1  -1  -1   1
// (cy): 0   0   1   0  -1   1   1  -1  -1

constexpr amrex::GpuArray<int, 9> LatticeDescriptor<2,9>::cx() noexcept
{
    return { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
}

constexpr amrex::GpuArray<int, 9> LatticeDescriptor<2,9>::cy() noexcept
{
    return { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
}

constexpr amrex::GpuArray<amrex::Real, 9> LatticeDescriptor<2,9>::weights() noexcept
{
    return {
        amrex::Real(4.0/9.0),
        amrex::Real(1.0/9.0), amrex::Real(1.0/9.0),
        amrex::Real(1.0/9.0), amrex::Real(1.0/9.0),
        amrex::Real(1.0/36.0), amrex::Real(1.0/36.0),
        amrex::Real(1.0/36.0), amrex::Real(1.0/36.0)
    };
}

constexpr amrex::GpuArray<int, 9> LatticeDescriptor<2,9>::opposite() noexcept
{
    return { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
}

// ── D3Q15 ─────────────────────────────────────────────────────────────────
// TODO: fill in D3Q15 velocity vectors, weights, and opposites.

constexpr amrex::GpuArray<int, 15> LatticeDescriptor<3,15>::cx() noexcept
{
    return {};
}

constexpr amrex::GpuArray<int, 15> LatticeDescriptor<3,15>::cy() noexcept
{
    return {};
}

constexpr amrex::GpuArray<int, 15> LatticeDescriptor<3,15>::cz() noexcept
{
    return {};
}

constexpr amrex::GpuArray<amrex::Real, 15> LatticeDescriptor<3,15>::weights() noexcept
{
    return {};
}

constexpr amrex::GpuArray<int, 15> LatticeDescriptor<3,15>::opposite() noexcept
{
    return {};
}

// ── D3Q19 ─────────────────────────────────────────────────────────────────
// TODO: fill in D3Q19 velocity vectors, weights, and opposites.

constexpr amrex::GpuArray<int, 19> LatticeDescriptor<3,19>::cx() noexcept { return {}; }
constexpr amrex::GpuArray<int, 19> LatticeDescriptor<3,19>::cy() noexcept { return {}; }
constexpr amrex::GpuArray<int, 19> LatticeDescriptor<3,19>::cz() noexcept { return {}; }
constexpr amrex::GpuArray<amrex::Real, 19> LatticeDescriptor<3,19>::weights() noexcept { return {}; }
constexpr amrex::GpuArray<int, 19> LatticeDescriptor<3,19>::opposite() noexcept { return {}; }

// ── D3Q27 ─────────────────────────────────────────────────────────────────
// TODO: fill in D3Q27 velocity vectors, weights, and opposites.

constexpr amrex::GpuArray<int, 27> LatticeDescriptor<3,27>::cx() noexcept { return {}; }
constexpr amrex::GpuArray<int, 27> LatticeDescriptor<3,27>::cy() noexcept { return {}; }
constexpr amrex::GpuArray<int, 27> LatticeDescriptor<3,27>::cz() noexcept { return {}; }
constexpr amrex::GpuArray<amrex::Real, 27> LatticeDescriptor<3,27>::weights() noexcept { return {}; }
constexpr amrex::GpuArray<int, 27> LatticeDescriptor<3,27>::opposite() noexcept { return {}; }

} // namespace muphfasa
