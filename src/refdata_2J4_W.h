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
// test data generated from 2J2_W.SNAP benchmark
// increased 2J from 2 to 4

#define REFDATA_NLOCAL 2
#define REFDATA_NGHOST 33
#define REFDATA_NTOTAL (REFDATA_NLOCAL + REFDATA_NGHOST)
#define REFDATA_NINSIDE 8
#define REFDATA_NCOEFF 14

struct REFDATA {
  int ninside, twojmax, ncoeff,nlocal,nghost;
  double rcutfac;
  double coeff[REFDATA_NCOEFF+1];
  int idlist[REFDATA_NLOCAL];
  int jlist[REFDATA_NLOCAL*REFDATA_NINSIDE];
  double rij[REFDATA_NLOCAL*3*REFDATA_NINSIDE];
  double fj[REFDATA_NTOTAL*3];
};

REFDATA refdata = {
  .ninside = REFDATA_NINSIDE,
  .twojmax = 4,
  .ncoeff = REFDATA_NCOEFF,
  .nlocal = REFDATA_NLOCAL,
  .nghost = REFDATA_NGHOST,
  .rcutfac = 2.8,
  .coeff = {
                       0,
          0.015793096489,
         -0.016849827611,
          0.273384755806,
         -0.086264586934,
         -0.056964378338,
                       6.0,
                       7.0,
                       8.0,
                       9.0,
                       10.0,
                       11.0,
                       12.0,
                       13.0,
                       14.0,
  },
  .idlist = {
    1,
    2,
  },
  .jlist = {
        34,
        32,
        28,
        26,
        12,
        10,
         4,
         1,
         0,
         2,
         5,
         7,
        13,
        15,
        18,
        20,
  },
  .rij = {
       -1.59110051241338,    -1.59141213164593,    -1.58879657317784,
        1.58919948758662,    -1.59141213164593,    -1.58879657317784,
       -1.59110051241338,     1.58888786835407,    -1.58879657317784,
        1.58919948758662,     1.58888786835407,    -1.58879657317784,
       -1.59110051241338,    -1.59141213164593,     1.59150342682216,
        1.58919948758662,    -1.59141213164593,     1.59150342682216,
       -1.59110051241338,     1.58888786835407,     1.59150342682216,
        1.58919948758662,     1.58888786835407,     1.59150342682216,
       -1.58919948758662,    -1.58888786835407,    -1.59150342682216,
        1.59110051241338,    -1.58888786835407,    -1.59150342682216,
       -1.58919948758662,     1.59141213164593,    -1.59150342682216,
        1.59110051241338,     1.59141213164593,    -1.59150342682216,
       -1.58919948758662,    -1.58888786835407,     1.58879657317784,
        1.59110051241338,    -1.58888786835407,     1.58879657317784,
       -1.58919948758662,     1.59141213164593,     1.58879657317784,
        1.59110051241338,     1.59141213164593,     1.58879657317784,
  },
  .fj = {
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
                       0,                    0,                    0,
  },
};

