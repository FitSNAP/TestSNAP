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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

using namespace std;
using namespace std::chrono;

#if REFDATA_TWOJ == 8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 2
#include "refdata_2J2_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#endif

/* ----------------------------------------------------------------------
  Vars to record timings of individual routines
------------------------------------------------------------------------- */
static double elapsed_ui = 0.0, elapsed_yi = 0.0, elapsed_duidrj = 0.0,
              elapsed_deidrj = 0.0;

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
  Elapsed Time
------------------------------------------------------------------------- */
inline double
elapsedTime(timeval start_time, timeval end_time)
{
  return ((end_time.tv_sec - start_time.tv_sec) +
          1e-6 * (end_time.tv_usec - start_time.tv_usec));
}

inline void
compute_error()
{
  for (int j = 0, jt = 0; j < ntotal; j++) {
    double ferrx = f[j][0] - refdata.fj[jt++];
    double ferry = f[j][1] - refdata.fj[jt++];
    double ferrz = f[j][2] - refdata.fj[jt++];
    sumsqferr += ferrx * ferrx + ferry * ferry + ferrz * ferrz;
  }
}

/* ----------------------------------------------------------------------
  Compute forces
------------------------------------------------------------------------- */
inline void
compute_forces(SNA* snaptr)
{
  for (int i = 0; i < natoms; i++) {
    for (int jj = 0; jj < nnbor; jj++) {
      int j = snaptr->h_inside(i, jj);
      f[i][0] += snaptr->h_dedr(i, jj, 0);
      f[i][1] += snaptr->h_dedr(i, jj, 1);
      f[i][2] += snaptr->h_dedr(i, jj, 2);
      f[j][0] -= snaptr->h_dedr(i, jj, 0);
      f[j][1] -= snaptr->h_dedr(i, jj, 1);
      f[j][2] -= snaptr->h_dedr(i, jj, 2);
    } // loop over neighbor forces
  }   // loop over atoms
}

/* ----------------------------------------------------------------------
  Init forces
------------------------------------------------------------------------- */
void inline init_forces()
{
  for (int j = 0; j < ntotal; j++) {
    f[j][0] = 0.0;
    f[j][1] = 0.0;
    f[j][2] = 0.0;
  }
}

int
main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);
  {
#if defined(KOKKOS_ENABLE_OPENMP)
    cout << "****** OpenMP  execution space ****** " << endl;
#elif defined(KOKKOS_ENABLE_CUDA)
    cout << "****** CUDA execution space ****** " << endl;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    cout << "****** OpenMPTarget execution space ****** " << endl;
#else
    cout << "****** Serial_space execution ****** " << endl;
#endif

    // process command line options
    options(argc, argv);

    // initialize data structures

    init();

    // Print memory usage
    //    printf("Memory footprint = %f GBs\n",
    //    (snaptr->memory_usage()/(1024*1024*1024)));

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
    printf("natoms = %d \n", natoms);
    printf("nghostatoms = %d \n", nghost);
    printf("nsteps = %d \n", nsteps);
    printf("nneighs = %d \n", nnbor);
    printf("twojmax = %d \n", twojmax);
    printf("duration = %g [sec]\n", elapsed.count());
    printf("step time = %g [sec/step]\n", elapsed.count() / nsteps);
    printf("grind time = %g [msec/atom-step]\n",
           1000.0 * elapsed.count() / (natoms * nsteps));
    printf("RMS |Fj| deviation %g [eV/A]\n",
           sqrt(sumsqferr / (ntotal * nsteps)));

    printf("\n Individual routine timings\n");
    printf("compute_ui = %f\n", elapsed_ui);
    printf("compute_yi = %f\n", elapsed_yi);
    printf("compute_duidrj = %f\n", elapsed_duidrj);
    printf("compute_deidrj = %f\n", elapsed_deidrj);
  }
  Kokkos::finalize();

  return 0;
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

  nnbor = refdata.ninside;
  ncoeff = refdata.ncoeff;
  natoms = refdata.nlocal;
  nghost = refdata.nghost;
  ntotal = natoms + nghost;
  twojmax = refdata.twojmax;
  rcutfac = refdata.rcutfac;
  // allocate SNA object

  snaptr = new SNA(rfac0, twojmax, rmin0, switchflag, bzeroflag, natoms, nnbor);
  int tmp = snaptr->ncoeff;
  if (tmp != ncoeff) {
    printf("ERROR: ncoeff from SNA does not match reference data\n");
    exit(1);
  }

  //  Kokkos::resize(f, ntotal,3);
  //  f.resize(ntotal, 3);

  snaptr->coeffi = new SNADOUBLE[ncoeff + 1];

  for (int icoeff = 0; icoeff < ncoeff + 1; icoeff++)
    snaptr->coeffi[icoeff] = refdata.coeff[icoeff];

  memory = new Memory();

  f = memory->grow(f, ntotal, 3, "f");

  // initialize SNA object
  snaptr->init();

  // Deep copies from host to device
  deep_copy(snaptr->idxcg_block, snaptr->h_idxcg_block);
  deep_copy(snaptr->idxu_block, snaptr->h_idxu_block);
  deep_copy(snaptr->idxz, snaptr->h_idxz);
  deep_copy(snaptr->betaj_index, snaptr->h_betaj_index);
  deep_copy(snaptr->idxb_block, snaptr->h_idxb_block);
  deep_copy(snaptr->cglist, snaptr->h_cglist);
  deep_copy(snaptr->rootpqarray, snaptr->h_rootpqarray);
  deep_copy(snaptr->ulist_parity, snaptr->h_ulist_parity);
  deep_copy(snaptr->idxdu_block, snaptr->h_idxdu_block);
  deep_copy(snaptr->idxzbeta, snaptr->h_idxzbeta);
  deep_copy(snaptr->rootpqparityarray, snaptr->h_rootpqparityarray);

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
  // generate neighbors, dummy values
  for (int i = 0; i < natoms; i++) {
    for (int jj = 0; jj < nnbor; jj++) {
      snaptr->h_rij(i, jj, 0) = refdata.rij[jt++];
      snaptr->h_rij(i, jj, 1) = refdata.rij[jt++];
      snaptr->h_rij(i, jj, 2) = refdata.rij[jt++];
      snaptr->h_inside(i, jj) = refdata.jlist[jjt++];
      snaptr->h_wj(i, jj) = 1.0;
      snaptr->h_rcutij(i, jj) = rcutfac;
    }
  }

  deep_copy(snaptr->rij, snaptr->h_rij);
  deep_copy(snaptr->wj, snaptr->h_wj);
  deep_copy(snaptr->rcutij, snaptr->h_rcutij);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  start = system_clock::now();
  snaptr->compute_ui();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_ui += elapsed.count();

  // compute_yi
  start = system_clock::now();
  snaptr->compute_yi();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_yi += elapsed.count();

  start = system_clock::now();
  snaptr->compute_fused_deidrj();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_duidrj += elapsed.count();
#else
  start = system_clock::now();
  snaptr->compute_ui_cpu();
  end = system_clock::now();
  elapsed = end - start;
  elapsed_ui += elapsed.count();

  // compute_yi
  start = system_clock::now();
  snaptr->compute_yi_cpu();
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
#endif

  deep_copy(snaptr->h_dedr, snaptr->dedr);
  compute_forces(snaptr);

  compute_error();
}
