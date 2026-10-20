.. MuPhFASa documentation master file.

MuPhFASa Documentation
=======================

The **MultiPHase Flow Adaptive Simulator** (MuPhFASa) is a Lattice-Boltzmann
solver built on top of `AMReX <https://amrex-codes.github.io/amrex/>`_, the
block-structured adaptive mesh refinement (AMR) framework. MuPhFASa currently
targets single-phase flow, with multiphase support planned.

This documentation is organized into a user guide (building the code,
writing input files) and a developer / API reference guide (the LBM
algorithm, module-by-module source documentation, and the test suite).

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   input_files
   testing

.. toctree::
   :maxdepth: 2
   :caption: LBM Theory & Algorithm

   theory

.. toctree::
   :maxdepth: 2
   :caption: Developer / API Reference

   api/lattice_descriptor
   api/core_kernels
   api/boundary_conditions
   api/geometry_io
   api/simulation

Indices and tables
===================

* :ref:`genindex`
* :ref:`search`
