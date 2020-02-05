// ----------------------------------------------------------------------
// Copyright (2019) Sandia Corporation. 
// Under the terms of Contract DE-AC04-94AL85000 
// with Sandia Corporation, the U.S. Government 
// retains certain rights in this software. This 
// software is distributed under the Zero Clause 
// BSD License
//
// TestSNAP - A prototype for the SNAP force kernel
// Version 0.0.2
// Main changes: Y array trick, memory compaction 
//
// Original author: Aidan P. Thompson, athomps@sandia.gov
// http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
//
// Additional authors: 
// Sarah Anderson
// Rahul Gayatri
// Steve Plimpton
// Christian Trott
//
// Collaborators:
// Stan Moore
// Evan Weinberg
// Nick Lubbers
// Mitch Wood
//
// ----------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono> 
#include <cmath>
#include "sna.h"
#include "memory.h"
#include "test_snap.h"

#if REFDATA_TWOJ==8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ==14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ==2
#include "refdata_2J2_W.h"
#elif REFDATA_TWOJ==4
#include "refdata_2J4_W.h"
#endif

/* ---------------------------------------------------------------------- */

int main(int argc, char* argv[]){

  // process command line options

  options(argc, argv);

  // initialize data structures

  init();

    // loop over steps

  auto start = myclock::now();
  for (int istep = 0; istep < nsteps; istep++) {
    
    // evaluate force kernel

    compute();

  }
  auto stop = myclock::now();
  myduration elapsed = stop - start;

  printf("-----------------------\n");
  printf("Summary of TestSNAP run\n");
  printf("-----------------------\n");
  printf("natoms = %d \n",nlocal);
  printf("nghostatoms = %d \n",nghost);
  printf("nsteps = %d \n",nsteps);
  printf("nneighs = %d \n",ninside);
  printf("twojmax = %d \n",twojmax);
  printf("duration = %g [sec]\n",elapsed.count());
  printf("step time = %g [sec/step]\n",elapsed.count()/nsteps);
  printf("grind time = %g [msec/atom-step]\n",1000.0*elapsed.count()/(nlocal*nsteps));
  printf("RMS |Fj| deviation %g [eV/A]\n",sqrt(sumsqferr/(ntotal*nsteps)));
}

/* ----------------------------------------------------------------------
   Allocate memory and initialize data structures
------------------------------------------------------------------------- */

void options(int argc, char* argv[]) {

  for (int i = 1; i < argc; i++) {

    if ( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0) ) {
      printf("TestSNAP 1.0 (stand-alone SNAP force kernel)\n\n");
      printf("The following optional command-line switches override default values\n");
      printf("-ns, --nsteps <val>: set the number of force calls to val (default 1)\n");
      exit(0);
    } else if ( (strcmp(argv[i], "-ns") == 0) || (strcmp(argv[i], "--nsteps") == 0) ) {
      nsteps = atoi(argv[++i]);
    } else {
      printf("ERROR: Unknown command line argument: %s\n",argv[i]);
      exit(1);
    }
  }
}

/* ----------------------------------------------------------------------
   Allocate memory and initialize data structures
------------------------------------------------------------------------- */

void init() {

  // initialize SNAP model using reference data

  ninside = refdata.ninside;
  ncoeff = refdata.ncoeff;
  nlocal = refdata.nlocal;
  nghost = refdata.nghost;
  ntotal = nlocal+nghost;
  twojmax = refdata.twojmax;
  rcutfac = refdata.rcutfac;
  coeffi = memory->grow(coeffi,ncoeff+1,"coeffi");
  for (int icoeff = 0; icoeff < ncoeff+1; icoeff++)
    coeffi[icoeff] = refdata.coeff[icoeff];

  // allocate SNA object

  memory = new Memory();

  // omit beta0 from beta vector

  SNADOUBLE* beta = coeffi+1; 
  snaptr = new SNA(memory,rfac0,twojmax,
                   rmin0,switchflag,bzeroflag,beta);
  int tmp = snaptr->ncoeff;
  if (tmp != ncoeff) {
    printf("ERROR: ncoeff from SNA does not match reference data\n");
    exit(1);
  }

  f = memory->grow(f,ntotal,3,"f");
  snaptr->grow_rij(ninside);

  // initialize SNA object

  snaptr->init();

  // initialize error tally

  sumsqferr = 0.0;
}

/* ----------------------------------------------------------------------
   Calculate forces on all local atoms 
------------------------------------------------------------------------- */

void compute() {
  int jt, jjt;

  // initialize all forces to zero

  for (int j = 0; j < ntotal; j++) {
    f[j][0] = 0.0;
    f[j][1] = 0.0;
    f[j][2] = 0.0;
  }

  // loop over atoms
 
  int jneigh = 0;

  jt = 0;
  jjt = 0;
  for (int i = 0; i < nlocal; i++) {
      
    // generate neighbors, dummy values

    for (int jj = 0; jj < ninside; jj++) {
      snaptr->rij[jj][0] = refdata.rij[jt++];
      snaptr->rij[jj][1] = refdata.rij[jt++];
      snaptr->rij[jj][2] = refdata.rij[jt++];
      snaptr->inside[jj] = refdata.jlist[jjt++];
      snaptr->wj[jj] = 1.0;
      snaptr->rcutij[jj] = rcutfac;
    }

    // compute Ui, Yi for atom I

    snaptr->compute_ui(ninside);

    SNADOUBLE* beta = coeffi+1; 
    snaptr->compute_yi(beta);

    // loop over neighbors
    // for neighbors of I within cutoff:
    // compute dUi/drj and dBi/drj
    // Fij = dEi/dRj = -dEi/dRi => add to Fi, subtract from Fj

    SNADOUBLE fij[3];

    for (int jj = 0; jj < ninside; jj++) {
      int j = snaptr->inside[jj];
      snaptr->compute_duidrj(snaptr->rij[jj],
                             snaptr->wj[jj],snaptr->rcutij[jj]);
      snaptr->compute_deidrj(fij);
        
      f[i][0] += fij[0];
      f[i][1] += fij[1];
      f[i][2] += fij[2];
      f[j][0] -= fij[0];
      f[j][1] -= fij[1];
      f[j][2] -= fij[2];

    } // loop over neighbor forces

  } // loop over atoms

  jt = 0;
  for (int j = 0; j < ntotal; j++) {
    double ferrx = f[j][0]-refdata.fj[jt++];
    double ferry = f[j][1]-refdata.fj[jt++];
    double ferrz = f[j][2]-refdata.fj[jt++];
    sumsqferr += ferrx*ferrx + ferry*ferry + ferrz*ferrz;
  }
}

