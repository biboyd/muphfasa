# GNUmakefile for the shearwave AMR regression test
# (tests/regression/shearwave_amr/).
#
# Usage:
#   make DIM=2                    # builds test_shearwave_amr2d.gnu.ex
#   make DIM=3                    # builds test_shearwave_amr3d.gnu.ex
#   ./test_shearwave_amr2d.gnu.ex inputs
#
# The shear-wave InitData.cpp lives in this directory; listing it (via VPATH
# and the local file) ahead of the solver sources makes it win over the
# default src/Simulation/InitData.cpp (vpath searches directories in order).
# The Make.muphfasa shared logic lives in ../../../Make.muphfasa.

TOP := ../../..

DIM ?= 2

EBASE = test_shearwave_amr

CEXE_sources += test_shearwave_amr.cpp

# Local InitData.cpp must win over src/Simulation/InitData.cpp.
# VPATH_LOCATIONS is searched in order by Make.rules; putting the test's own
# directory first ensures its InitData.cpp is found first.
VPATH_LOCATIONS   += .
INCLUDE_LOCATIONS += .

include $(TOP)/Make.muphfasa
