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

#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include "Kokkos_Defines.h"

KOKKOS_INLINE_FUNCTION
double compute_dsfac_loc(double r, double rcut, bool switch_flag, double rmin0);
struct SNA_ZINDICES {
  int j1, j2, j, ma1min, ma2max, mb1min, mb2max, na, nb;
};

struct SNA_BINDICES {
  int j1, j2, j;
};

class SNA {

public:
  SNA(double, int, double, int, int, int, int);
  ~SNA();
  void build_indexlist();
  void init();
  double memory_usage();

  int ncoeff;

  // functions for bispectrum coefficients

  void compute_ui();
  void compute_ui_cpu();
  void ui_launch();
  void update_ulisttot_launch();

  void compute_yi();
  void compute_yi_cpu();
  void compute_yi_reduce_idxz();
  void ulisttot_transpose_launch();
  void yi_zero_launch();
  void yi_launch();
#if defined(KOKKOS_ENABLE_CUDA)
  void compute_fused_deidrj();
#endif

  // functions for derivatives

  void compute_duidrj();
  void compute_deidrj();
  void compute_all();

  // per sna class instance for OMP use
  int nmax;
  int num_nbor, num_atoms;
  int idxcg_max, idxu_max, idxz_max, idxb_max;
  double *coeffi;

  int_View1D ulist_parity = int_View1D("ulist_parity",0);
  int_View1D idxdu_block = int_View1D("idxdu_block",0);
  HostInt_View1D h_ulist_parity, h_idxdu_block;
  int idxdu_max;
  double_View1D idxzbeta = double_View1D("idxzbeta",0);
  HostDouble_View1D h_idxzbeta;
  double_View2D rootpqparityarray = double_View2D("rootpqarray",0,0);
  HostDouble_View2D h_rootpqparityarray;


  int_View1D idxu_block = int_View1D("idxu_block",0);
  int_View2D inside = int_View2D("inside",0,0);
  int_View2D idxz = int_View2D("idxz",0,0);


  //These data structures replace idxz
  int_View2D idxz_j1j2j = int_View2D("idxz_j1j2j",0,0);
  int_View2D idxz_ma = int_View2D("idxz_ma",0,0);
  int_View2D idxz_mb = int_View2D("idxz_mb",0,0);

  int_View2D idxb = int_View2D("idxb",0,0);
  int_View3D idxcg_block = int_View3D("idxcg_block",0,0,0);
  int_View3D idxz_block = int_View3D("idxz_block",0,0,0);
  int_View3D idxb_block = int_View3D("idxb_block",0,0,0);
  HostInt_View3D h_idxcg_block, h_idxb_block;
  HostInt_View2D h_idxz, h_inside;
  HostInt_View1D h_idxu_block;
  HostInt_View2D h_idxz_j1j2j, h_idxz_ma, h_idxz_mb;

  double_View1D betaj_index = double_View1D("betaj_index",0);
  double_View1D cglist = double_View1D("cglist",0);
  double_View2D rootpqarray = double_View2D("rootpqarray",0,0);
  double_View2D wj = double_View2D("wj",0,0);
  double_View2D rcutij = double_View2D("rcutij",0,0);
  double_View3D dedr = double_View3D("dedr",0,0,0);
  double_View3D rij = double_View3D("rij",0,0,0);
  HostDouble_View3D h_rij, h_dedr;
  HostDouble_View2D h_wj, h_rcutij, h_rootpqarray;
  HostDouble_View1D h_betaj_index, h_cglist;

  SNAcomplex_View2DR ylist = SNAcomplex_View2DR("ylist",0,0);
  SNAcomplex_View2D ulisttot = SNAcomplex_View2D("ulisttot",0,0);
  SNAcomplex_View2DR ulisttot_r = SNAcomplex_View2DR("ulisttot_r",0,0);

  SNAcomplex_View3D ulist_ij = SNAcomplex_View3D("ulist_ij",0,0,0);
  SNAcomplex_View3D ulist = SNAcomplex_View3D("ulist",0,0,0);
  SNAcomplex_View2DR ulist_2d = SNAcomplex_View2DR("ulist_2d",0,0);

  SNAcomplex_View4D dulist = SNAcomplex_View4D("dulist",0,0,0,0);

  void grow_rij(int);
  int twojmax = 0;
  int idxz_j1j2j_max = 0, idxz_ma_max = 0, idxz_mb_max = 0;

//private:
  double rmin0, rfac0;

//  SNA_ZINDICES* idxz;

  // data for bispectrum coefficients

  // derivatives of data

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  double factorial(int);

  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void zero_uarraytot();
  void addself_uarraytot(double);
  void add_uarraytot(int natom, int nbor, double r, double wj, double rcut);
  void compute_uarray(int natom, int nbor, double x, double y, double z,
                         double z0, double r);
  double deltacg(int, int, int);
  int compute_ncoeff();
  void compute_duarray();
  KOKKOS_INLINE_FUNCTION
  double compute_sfac(double, double);
  double compute_dsfac(double, double);


  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine

  int switch_flag;

  // self-weight

  double wself;

};

#endif

