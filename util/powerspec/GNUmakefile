# GNUmakefile for powerspec (util/powerspec/), the standalone kinetic energy
# spectrum tool. Builds only AMReX Base + FFT and powerspec.cpp; it compiles
# no solver code.
#
# Run it from this directory:
#   make                       3-D build (default), MPI
#   make DIM=2                 2-D build (for 2-D plotfiles)
#   make USE_MPI=FALSE         serial build
#   make DEBUG=TRUE            debug build
#   make DIM=2 realclean       remove build products for this configuration
#
# The result is powerspec<dim>d.<comp>[.<opts>].ex, e.g. powerspec3d.gnu.MPI.ex.
# The dimension must match the plotfile. Usage: ./powerspec3d.gnu.MPI.ex <plotfile>
# Needs FFTW (set FFTW_HOME if it is not in the default search path) or the
# GPU vendor FFT library. AMREX_HOME is taken from the environment if set,
# else external/amrex.

TOP := ../..

AMREX_HOME ?= $(TOP)/external/amrex

COMP ?= gnu
DIM  ?= 3

DEBUG    ?= FALSE
USE_MPI  ?= TRUE
USE_OMP  ?= FALSE
USE_CUDA ?= FALSE
USE_HIP  ?= FALSE
USE_SYCL ?= FALSE

# FFTW on CPU, cuFFT/rocFFT/oneMKL on GPU
USE_FFT = TRUE

EBASE = powerspec

# No Fortran sources in this project.
BL_NO_FORT = TRUE

CEXE_sources += powerspec.cpp

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
