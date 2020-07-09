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
#include "test_snap.h"
#include "memory.h"
#include "sna.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>

#if REFDATA_TWOJ == 8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 2
#include "refdata_2J2_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#endif

using namespace std::chrono;
/* ----------------------------------------------------------------------
  Vars to record timings of individual routines
------------------------------------------------------------------------- */
static double elapsed_ui = 0.0, elapsed_yi = 0.0, elapsed_duidrj = 0.0,
              elapsed_deidrj = 0.0;

/* ----------------------------------------------------------------------
  Elapsed Time
------------------------------------------------------------------------- */
inline double
elapsedTime(timeval start_time, timeval end_time)
{
  return ((end_time.tv_sec - start_time.tv_sec) +
          1e-6 * (end_time.tv_usec - start_time.tv_usec));
}

/* ----------------------------------------------------------------------
  Init forces
------------------------------------------------------------------------- */
void inline init_forces()
{
  for (int j = 0; j < ntotal; j++) {
    f(j, 0) = 0.0;
    f(j, 1) = 0.0;
    f(j, 2) = 0.0;
  }
}

/* ----------------------------------------------------------------------
  Compute forces
------------------------------------------------------------------------- */
inline void
compute_forces(SNA* snaptr)
{
  const int num_atoms = snaptr->num_atoms;
  const int num_nbor = snaptr->num_nbor;
  for (int natom = 0; natom < num_atoms; natom++) {
    for (int nbor = 0; nbor < num_nbor; nbor++) {
      int j = snaptr->inside(natom, nbor);
      f(natom, 0) += snaptr->dedr(natom, nbor, 0);
      f(natom, 1) += snaptr->dedr(natom, nbor, 1);
      f(natom, 2) += snaptr->dedr(natom, nbor, 2);
      f(j, 0) -= snaptr->dedr(natom, nbor, 0);
      f(j, 1) -= snaptr->dedr(natom, nbor, 1);
      f(j, 2) -= snaptr->dedr(natom, nbor, 2);

    } // loop over neighbor forces
  }   // loop over atoms
}

/* ----------------------------------------------------------------------
  Compute error
------------------------------------------------------------------------- */
inline void
compute_error(SNA* snaptr)
{
  int jt = 0;
  for (int j = 0; j < ntotal; j++) {
    double ferrx = f(j, 0) - refdata.fj[jt++];
    double ferry = f(j, 1) - refdata.fj[jt++];
    double ferrz = f(j, 2) - refdata.fj[jt++];
    sumsqferr += ferrx * ferrx + ferry * ferry + ferrz * ferrz;
  }
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char* argv[])
{
#if defined(_OPENMP)
#if defined(OPENMP_TARGET)
  std::cout << "******* OpenMP4.5 version *******"
            << " \n";
#else
  std::cout << "******* OpenMP3.0 version *******"
            << " \n";
#endif
#else
  std::cout << "******* Sequential version *******"
            << " \n";
#endif

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
  printf("natoms = %d \n", nlocal);
  printf("nghostatoms = %d \n", nghost);
  printf("nsteps = %d \n", nsteps);
  printf("nneighs = %d \n", ninside);
  printf("twojmax = %d \n", twojmax);
  printf("duration = %g [sec]\n", elapsed.count());
  printf("step time = %g [sec/step]\n", elapsed.count() / nsteps);
  printf("grind time = %g [msec/atom-step]\n",
         1000.0 * elapsed.count() / (nlocal * nsteps));
  printf("RMS |Fj| deviation %g [eV/A]\n", sqrt(sumsqferr / (ntotal * nsteps)));

  printf("\n Individual routine timings\n");
  printf("compute_ui = %f\n", elapsed_ui);
  printf("compute_yi = %f\n", elapsed_yi);
  printf("compute_duidrj = %f\n", elapsed_duidrj);
  printf("compute_deidrj = %f\n", elapsed_deidrj);
}

/* ----------------------------------------------------------------------
   Allocate memory and initialize data structures
------------------------------------------------------------------------- */

void
options(int argc, char* argv[])
{

  for (int i = 1; i < argc; i++) {

    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      printf("TestSNAP 1.0 (stand-alone SNAP force kernel)\n\n");
      printf("The following optional command-line switches override default "
             "values\n");
      printf("-ns, --nsteps <val>: set the number of force calls to val "
             "(default 1)\n");
      exit(0);
    } else if ((strcmp(argv[i], "-ns") == 0) ||
               (strcmp(argv[i], "--nsteps") == 0)) {
      nsteps = atoi(argv[++i]);
    } else {
      printf("ERROR: Unknown command line argument: %s\n", argv[i]);
      exit(1);
    }
  }
}

/* ----------------------------------------------------------------------
   Allocate memory and initialize data structures
------------------------------------------------------------------------- */

void
init()
{

  // initialize SNAP model using reference data

  ninside = refdata.ninside;
  ncoeff = refdata.ncoeff;
  nlocal = refdata.nlocal;
  nghost = refdata.nghost;
  ntotal = nlocal + nghost;
  twojmax = refdata.twojmax;
  rcutfac = refdata.rcutfac;
  coeffi = memory->grow(coeffi, ncoeff + 1, "coeffi");
  for (int icoeff = 0; icoeff < ncoeff + 1; icoeff++)
    coeffi[icoeff] = refdata.coeff[icoeff];

  // allocate SNA object

  memory = new Memory();
  f.resize(ntotal, 3);

  // omit beta0 from beta vector

  SNADOUBLE* beta = coeffi + 1;
  snaptr = new SNA(ninside,
                   nlocal,
                   memory,
                   rfac0,
                   twojmax,
                   rmin0,
                   switchflag,
                   bzeroflag,
                   beta);
  int tmp = snaptr->ncoeff;
  if (tmp != ncoeff) {
    printf("ERROR: ncoeff from SNA does not match reference data\n");
    exit(1);
  }

  snaptr->grow_rij(ninside);

  // initialize SNA object

  snaptr->init();

  // initialize error tally

  sumsqferr = 0.0;
}

/* ----------------------------------------------------------------------
   Calculate forces on all local atoms
------------------------------------------------------------------------- */

void
compute()
{
  time_point<system_clock> start, end;
  duration<double> elapsed;
  // initialize all forces to zero
  init_forces();

  int jt = 0, jjt = 0;
  const int num_atoms = snaptr->num_atoms;
  const int num_nbor = snaptr->num_nbor;

  // loop over atoms and generate neighbors, dummy values
  for (int natom = 0; natom < num_atoms; natom++) {
    for (int nbor = 0; nbor < num_nbor; nbor++) {
      snaptr->rij(natom, nbor, 0) = refdata.rij[jt++];
      snaptr->rij(natom, nbor, 1) = refdata.rij[jt++];
      snaptr->rij(natom, nbor, 2) = refdata.rij[jt++];
      snaptr->inside(natom, nbor) = refdata.jlist[jjt++];
      snaptr->wj(natom, nbor) = 1.0;
      snaptr->rcutij(natom, nbor) = rcutfac;
    }
  }

#if _OPENMP
#if (OPENMP_TARGET)
  snaptr->omp_offload_update();
#endif
#endif

  // compute_ui
  start = system_clock::now();
  snaptr->compute_ui();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_ui += elapsed.count();

  // compute_yi
  start = system_clock::now();
  SNADOUBLE* beta = coeffi + 1;
  snaptr->compute_yi(beta);
  end = system_clock::now();
  elapsed = end - start;
  elapsed_yi += elapsed.count();

  // compute_duidrj
  start = system_clock::now();
  snaptr->compute_duidrj();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_duidrj += elapsed.count();

  start = system_clock::now();
  snaptr->compute_deidrj();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_deidrj += elapsed.count();

  // Compute forces and error
  compute_forces(snaptr);
  compute_error(snaptr);
}
