// ----------------------------------------------------------------------
// Copyright (2019) Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000
// with Sandia Corporation, the U.S. Government
// retains certain rights in this software. This
// software is distributed under the Zero Clause
// BSD License
//
// TestSNAP - A prototype for the SNAP force kernel
// Version 0.0.3
// Main changes: GPU AoSoA data layout, optimized recursive polynomial evaluation
//
// Original author: Aidan P. Thompson, athomps@sandia.gov
// http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
//
// Additional authors:
// Sarah Anderson
// Rahul Gayatri
// Steve Plimpton
// Christian Trott
// Evan Weinberg
//
// Collaborators:
// Stan Moore
// Nick Lubbers
// Mitch Wood
//
// ----------------------------------------------------------------------
#ifndef LMP_TESTSNAP_H
#define LMP_TESTSNAP_H

#include "memory.h"
#include "sna.h"

Memory* memory = NULL;
// MD data

int nnbor;  // num neighbors per atom
int natoms;
int nghost;      // number of ghost atoms
int ntotal;      // number of total atoms
int nsteps = 1;  // num of force evaluations
double** f;
int ncoeff;  // number of beta coefficients

// SNAP data

SNA* snaptr = NULL;
double rcutfac;          // SNAP parameter, set by refdata
int twojmax;             // SNAP parameter, set by refdata
double rfac0 = 0.99363;  // SNAP parameter
double rmin0 = 0.0;      // SNAP parameter
int switchflag = 1;      // SNAP parameter
int bzeroflag = 1;       // SNAP parameter
int quadraticflag = 0;   // SNAP parameter

// function declarations

void options(int, char*[]);
void init();
void compute();

// timer classes

typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::duration<float> myduration;

// math stuff

// error tally

double sumsqferr;

#endif
