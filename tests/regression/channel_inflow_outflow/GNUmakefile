# GNUmakefile for the channel inflow/outflow regression test
# (tests/regression/channel_inflow_outflow/). Builds test_channel directly,
# without going through CMake. Modeled on tests/regression/couette/GNUmakefile.
#
# Usage:
#   make                      # build test_channel (2-D, no MPI, by default)
#   ./test_channel2d.gnu.ex inputs

AMREX_HOME ?= $(CURDIR)/../../../external/amrex

COMP ?= gnu
DIM  ?= 2
USE_EB ?= TRUE

BL_NO_FORT = TRUE

EBASE = test_channel

SRC_DIR := ../../../src

VPATH_LOCATIONS   += $(SRC_DIR)/LBM $(SRC_DIR)/Geometry $(SRC_DIR)/IO $(SRC_DIR)/Simulation
INCLUDE_LOCATIONS += $(SRC_DIR) $(SRC_DIR)/LBM $(SRC_DIR)/Geometry $(SRC_DIR)/IO $(SRC_DIR)/Simulation

CEXE_sources += test_channel.cpp

# Solver sources, excluding main.cpp (this test defines its own main()).
CEXE_sources += LatticeDescriptor.cpp
CEXE_sources += Macroscopic.cpp
CEXE_sources += Equilibrium.cpp
CEXE_sources += Streaming.cpp
CEXE_sources += Collision.cpp
CEXE_sources += BoundaryConditions.cpp
CEXE_sources += Units.cpp
CEXE_sources += GeometryHandler.cpp
CEXE_sources += InputParser.cpp
CEXE_sources += Checkpoint.cpp
CEXE_sources += Plotfile.cpp
CEXE_sources += SimulationSetup.cpp
CEXE_sources += SimulationFactory.cpp

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

Pdirs := Base Boundary AmrCore EB
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
