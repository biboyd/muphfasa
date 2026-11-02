# GNUmakefile for the AMR Taylor-Green vortex regression test
# (tests/regression/tgv2d_amr/).
#
# Usage:
#   make                      # builds test_tgv2d_amr2d.gnu.ex (2-D only)
#   ./test_tgv2d_amr2d.gnu.ex inputs
#
# The Taylor-Green initial condition (InitData.cpp) and analytic solution
# (TgvAnalytic.H) are shared with ../tgv2d: listing that directory ahead of
# the solver sources makes its InitData.cpp win over the default quiescent
# src/Simulation/InitData.cpp (vpath searches directories in order).
#
# All the shared build logic lives in ../../../Make.muphfasa.

TOP := ../../..

DIM ?= 2

EBASE = test_tgv2d_amr

CEXE_sources += test_tgv2d_amr.cpp

VPATH_LOCATIONS   += ../tgv2d
INCLUDE_LOCATIONS += ../tgv2d

include $(TOP)/Make.muphfasa
