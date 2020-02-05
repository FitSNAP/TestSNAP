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

/* ----------------------------------------------------------------------
   Contributing authors: Aidan Thompson, Christian Trott, SNL
------------------------------------------------------------------------- */

#ifndef LMP_SNA_H
#define LMP_SNA_H

typedef double SNADOUBLE;
typedef float SNAFLOAT;

//struct SNA_ZINDICES {
//  int j1, j2, j, ma1min, ma2max, mb1min, mb2max, na, nb, jju;
//  SNADOUBLE betaj;
//};

struct SNA_BINDICES {
  int j1, j2, j;
};

class SNA {

public:
  SNA(class Memory*, SNADOUBLE, int, SNADOUBLE, int, int, const double*);
  ~SNA();
  void build_indexlist(const double*);
  void init();
  SNADOUBLE memory_usage();

  int ncoeff;

  // functions for bispectrum coefficients

  void compute_ui(int);

  void compute_yi(SNADOUBLE*);

  // functions for derivatives

  void compute_duidrj(SNADOUBLE*, SNADOUBLE, SNADOUBLE);
  void compute_deidrj(SNADOUBLE*);

  // per sna class instance for OMP use

  SNADOUBLE** rij;
  int* inside;
  SNADOUBLE* wj;
  SNADOUBLE* rcutij;
  int nmax;

  void grow_rij(int);

private:
  Memory* memory;
  SNADOUBLE rmin0, rfac0;

  // use indexlist instead of loops, constructor generates these

  int** idxz_j1j2j;
  int** idxz_ma;
  int** idxz_mb;
  SNA_BINDICES* idxb;
  int idxcg_max, idxu_max, idxz_max, idxb_max;
  int idxz_j1j2j_max, idxz_ma_max, idxz_mb_max;

  // data for bispectrum coefficients

  int twojmax;
  SNADOUBLE** rootpqarray;
  SNADOUBLE* cglist;
  int*** idxcg_block;

  SNADOUBLE* ulisttot_r, * ulisttot_i;
  SNADOUBLE* ulist_r, * ulist_i;
  int* idxu_block;

  int*** idxz_block;

  int*** idxb_block;

  // derivatives of data

  SNADOUBLE** dulist_r, ** dulist_i;
  SNADOUBLE* ylist_r, * ylist_i;

  static const int nmaxfactorial = 167;
  static const SNADOUBLE nfac_table[];
  SNADOUBLE factorial(int);

  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void zero_uarraytot();
  void addself_uarraytot(SNADOUBLE);
  void add_uarraytot(SNADOUBLE, SNADOUBLE, SNADOUBLE);
  void compute_uarray(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                      SNADOUBLE, SNADOUBLE);
  void compute_uarray_inlineinversion(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                      SNADOUBLE, SNADOUBLE);
  void compute_uarray_2J2(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                      SNADOUBLE, SNADOUBLE);
  void compute_uarray_2J2_byhand(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                      SNADOUBLE, SNADOUBLE);
  SNADOUBLE deltacg(int, int, int);
  int compute_ncoeff();
  void compute_duarray(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                       SNADOUBLE, SNADOUBLE, SNADOUBLE, SNADOUBLE, SNADOUBLE);
  void compute_duarray_inlineinversion(SNADOUBLE, SNADOUBLE, SNADOUBLE,
                       SNADOUBLE, SNADOUBLE, SNADOUBLE, SNADOUBLE, SNADOUBLE);
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

