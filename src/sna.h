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

#ifndef LMP_SNA_H
#define LMP_SNA_H

#if _OPENMP
#include <omp.h>
#endif

// Use the column-major format of ArrayMD for GPUs
#if (OPENMP_TARGET)
#include "arrayMDgpu.h"
#else
#include "arrayMDcpu.h"
#endif

typedef double SNADOUBLE;
typedef float SNAFLOAT;

// Complex type structure
using SNAcomplex = struct
{
  SNADOUBLE re, im;
};

struct SNA_BINDICES
{
  int j1, j2, j;
};

class SNA
{

public:
  SNA(int natoms,
      int nbors,
      class Memory*,
      SNADOUBLE,
      int,
      SNADOUBLE,
      int,
      int,
      const double*);
  ~SNA();
  void build_indexlist(const double*);
  void init();
  SNADOUBLE memory_usage();

  int ncoeff;

  // functions for bispectrum coefficients

  void compute_ui();
  void compute_yi(SNADOUBLE*);

  // functions for derivatives

  void compute_duidrj();
  void compute_deidrj();

  // per sna class instance for OMP use

  Array3D<SNADOUBLE> rij;
  Array2D<int> inside;
  Array2D<SNADOUBLE> wj;
  Array2D<SNADOUBLE> rcutij;

  // Final output from deidrj into dedr;
  Array3D<SNADOUBLE> dedr;

  int nmax;

  void grow_rij(int);

  // Public vars for number of atoms and neighbors
  int num_nbor, num_atoms;

#if (OPENMP_TARGET)
  void omp_offload_init();
  void omp_offload_update();
#endif

private:
  Memory* memory;
  SNADOUBLE rmin0, rfac0;

  // use indexlist instead of loops, constructor generates these

  Array2D<int> idxz;
  Array1D<SNADOUBLE> idxzbeta;
  SNA_BINDICES* idxb;
  int idxcg_max, idxu_max, idxdu_max, idxz_max, idxb_max;

  // data for bispectrum coefficients

  int twojmax;

  Array2D<SNADOUBLE> rootpqarray;
  Array1D<SNADOUBLE> cglist;
  Array3D<int> idxcg_block;

  // Main data structures
  Array3D<SNAcomplex> ulist;
  Array2D<SNAcomplex> ulisttot;
  Array1D<int> ulist_parity;
  Array1D<int> idxdu_block;
  Array1D<int> idxu_block;

  int*** idxz_block;
  int*** idxb_block;

  // derivatives of data

  Array4D<SNAcomplex> dulist;
  Array2D<SNAcomplex> ylist;

  static const int nmaxfactorial = 167;
  static const SNADOUBLE nfac_table[];
  SNADOUBLE factorial(int);

  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void zero_uarraytot();
  void addself_uarraytot(SNADOUBLE);
  void add_uarraytot(int, int, SNADOUBLE, SNADOUBLE, SNADOUBLE);
  void compute_uarray(int,
                      int,
                      SNADOUBLE,
                      SNADOUBLE,
                      SNADOUBLE,
                      SNADOUBLE,
                      SNADOUBLE);
  SNADOUBLE deltacg(int, int, int);
  int compute_ncoeff();
  void compute_duarray(int,
                       int,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE,
                       SNADOUBLE);
  SNADOUBLE compute_sfac(SNADOUBLE, SNADOUBLE);
  SNADOUBLE compute_dsfac(SNADOUBLE, SNADOUBLE);

  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine

  int switch_flag;

  // self-weight

  SNADOUBLE wself;
};

#endif
