# GNUmakefile for the AMR Couette regression test (tests/regression/couette_amr/).
#
# Usage:
#   make                      # builds test_couette_amr2d.gnu.ex
#   ./test_couette_amr2d.gnu.ex inputs           # (or inputs_band, inputs_regrid,
#                                                #  inputs_poiseuille)
#   make DIM=3                # builds test_couette_amr3d.gnu.ex
#   ./test_couette_amr3d.gnu.ex inputs3d
#
# All the shared build logic lives in ../../../Make.muphfasa.

TOP := ../../..

DIM ?= 2

EBASE = test_couette_amr

CEXE_sources += test_couette_amr.cpp

include $(TOP)/Make.muphfasa
