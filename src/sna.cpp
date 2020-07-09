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

#include "sna.h"
#include "memory.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

/* ----------------------------------------------------------------------

   this implementation is based on the method outlined
   in Bartok[1], using formulae from VMK[2].

   for the Clebsch-Gordan coefficients, we
   convert the VMK half-integral labels
   a, b, c, alpha, beta, gamma
   to array offsets j1, j2, j, m1, m2, m
   using the following relations:

   j1 = 2*a
   j2 = 2*b
   j =  2*c

   m1 = alpha+a      2*alpha = 2*m1 - j1
   m2 = beta+b    or 2*beta = 2*m2 - j2
   m =  gamma+c      2*gamma = 2*m - j

   in this way:

   -a <= alpha <= a
   -b <= beta <= b
   -c <= gamma <= c

   becomes:

   0 <= m1 <= j1
   0 <= m2 <= j2
   0 <= m <= j

   and the requirement that
   a+b+c be integral implies that
   j1+j2+j must be even.
   The requirement that:

   gamma = alpha+beta

   becomes:

   2*m - j = 2*m1 - j1 + 2*m2 - j2

   Similarly, for the Wigner U-functions U(J,m,m') we
   convert the half-integral labels J,m,m' to
   array offsets j,ma,mb:

   j = 2*J
   ma = J+m
   mb = J+m'

   so that:

   0 <= j <= 2*Jmax
   0 <= ma, mb <= j.

   For the bispectrum components B(J1,J2,J) we convert to:

   j1 = 2*J1
   j2 = 2*J2
   j = 2*J

   and the requirement:

   |J1-J2| <= J <= J1+J2, for j1+j2+j integral

   becomes:

   |j1-j2| <= j <= j1+j2, for j1+j2+j even integer

   or

   j = |j1-j2|, |j1-j2|+2,...,j1+j2-2,j1+j2

   [1] Albert Bartok-Partay, "Gaussian Approximation..."
   Doctoral Thesis, Cambrindge University, (2009)

   [2] D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii,
   "Quantum Theory of Angular Momentum," World Scientific (1988)

------------------------------------------------------------------------- */

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))
static const SNADOUBLE MY_PI = 3.14159265358979323846; // pi

SNA::SNA(int natoms,
         int nbors,
         Memory* memory_in,
         SNADOUBLE rfac0_in,
         int twojmax_in,
         SNADOUBLE rmin0_in,
         int switch_flag_in,
         int /*bzero_flag_in*/,
         const double* beta)
{
  wself = 1.0;

  num_nbor = natoms;
  num_atoms = nbors;
  memory = memory_in;
  rfac0 = rfac0_in;
  rmin0 = rmin0_in;
  switch_flag = switch_flag_in;

  twojmax = twojmax_in;

  ncoeff = compute_ncoeff();

  nmax = 0;
  idxb = NULL;

  build_indexlist(beta);
  create_twojmax_arrays();
}

/* ---------------------------------------------------------------------- */

SNA::~SNA()
{
  destroy_twojmax_arrays();
}

void
SNA::build_indexlist(const double* beta)
{
  int jdim = twojmax + 1;

  // index list for cglist

  idxcg_block.resize(jdim, jdim, jdim);

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxcg_block(j1, j2, j) = idxcg_count;
        for (int m1 = 0; m1 <= j1; m1++)
          for (int m2 = 0; m2 <= j2; m2++)
            idxcg_count++;
      }
  idxcg_max = idxcg_count;

  // index list for uarray
  // need to include both halves
  // **** only place rightside is used is in compute_yi() ***

  idxu_block.resize(jdim);
  int idxu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxu_block(j) = idxu_count;
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxu_count++;
  }
  idxu_max = idxu_count;

  // parity list for uarray inversion symmetry
  // parity +1: u[ma-j][mb-j] = +Conj([u[ma][mb])
  // parity -1: u[ma-j][mb-j] = -Conj([u[ma][mb])

  ulist_parity.resize(idxu_max);
  idxu_count = 0;
  for (int j = 0; j <= twojmax; j++) {
    int mbpar = 1;
    for (int mb = 0; mb <= j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        ulist_parity(idxu_count) = mapar;
        mapar = -mapar;
        idxu_count++;
      }
      mbpar = -mbpar;
    }
  }

  // index list for duarray, yarray
  // only include left half
  // NOTE: idxdu indicates lefthalf only
  //       idxu indicates both halves

  idxdu_block.resize(jdim);
  int idxdu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxdu_block(j) = idxdu_count;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxdu_count++;
  }
  idxdu_max = idxdu_count;

  // index list for beta and B

  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          idxb_count++;

  idxb_max = idxb_count;
  idxb = new SNA_BINDICES[idxb_max];

  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) {
          idxb[idxb_count].j1 = j1;
          idxb[idxb_count].j2 = j2;
          idxb[idxb_count].j = j;
          idxb_count++;
        }

  // reverse index list for beta and b

  memory->create(idxb_block, jdim, jdim, jdim, "sna:idxb_block");
  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        if (j < j1)
          continue;
        idxb_block[j1][j2][j] = idxb_count;
        idxb_count++;
      }

  // index list for zlist

  int idxz_count = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;

  idxz_max = idxz_count;
  idxz.resize(idxz_max, 9);

  idxzbeta.resize(idxz_max);
  memory->create(idxz_block, jdim, jdim, jdim, "sna:idxz_block");

  idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxz_block[j1][j2][j] = idxz_count;

        // find right beta[jjb] entries
        // multiply and divide by j+1 factors
        // account for multiplicity of 1, 2, or 3
        // this should not be computed here

        SNADOUBLE betaj;
        if (j >= j1) {
          const int jjb = idxb_block[j1][j2][j];
          if (j1 == j) {
            if (j2 == j)
              betaj = 3 * beta[jjb];
            else
              betaj = 2 * beta[jjb];
          } else
            betaj = beta[jjb];
        } else if (j >= j2) {
          const int jjb = idxb_block[j][j2][j1];
          if (j2 == j)
            betaj = 2 * beta[jjb] * (j1 + 1) / (j + 1.0);
          else
            betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        } else {
          const int jjb = idxb_block[j2][j][j1];
          betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        }

        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++) {

            idxz(idxz_count, 0) = j1;
            idxz(idxz_count, 1) = j2;
            idxz(idxz_count, 2) = j;

            int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
            idxz(idxz_count, 3) = ma1min;
            idxz(idxz_count, 4) = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
            idxz(idxz_count, 5) =
              MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

            int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
            idxz(idxz_count, 6) = mb1min;
            idxz(idxz_count, 7) = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
            idxz(idxz_count, 8) =
              MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

            idxzbeta(idxz_count) = betaj;

            idxz_count++;
          }
      }
}

/* ---------------------------------------------------------------------- */
#if (OPENMP_TARGET)
void
SNA::omp_offload_init()
{
#pragma omp target enter data map(to : this [0:1])
#pragma omp target enter data map(                                             \
  to                                                                           \
  : this->idxu_block.dptr [0:this->idxu_block.size],                           \
    this->ulist_parity.dptr [0:this->ulist_parity.size],                       \
    this->rootpqarray.dptr [0:this->rootpqarray.size],                         \
    this->idxz.dptr [0:this->idxz.size])
#pragma omp target enter data map(                                             \
  to                                                                           \
  : this->idxzbeta.dptr [0:this->idxzbeta.size],                               \
    this->idxcg_block.dptr [0:this->idxcg_block.size],                         \
    this->idxdu_block.dptr [0:this->idxdu_block.size],                         \
    this->cglist.dptr [0:this->cglist.size])

#pragma omp target enter data map(to                                           \
                                  : this->ulist.dptr [0:this->ulist.size],     \
                                    this->ulisttot.dptr                        \
                                    [0:this->ulisttot.size],                   \
                                    this->dulist.dptr [0:this->dulist.size],   \
                                    this->ylist.dptr [0:this->ylist.size],     \
                                    this->dedr.dptr [0:this->dedr.size])
}

void
SNA::omp_offload_update()
{
#pragma omp target enter data map(to                                           \
                                  : this->rij.dptr [0:this->rij.size],         \
                                    this->rcutij.dptr [0:this->rcutij.size],   \
                                    this->wj.dptr [0:this->wj.size])
}
#endif
/* ---------------------------------------------------------------------- */

void
SNA::init()
{
  init_clebsch_gordan();
  init_rootpqarray();
#if _OPENMP
#if (OPENMP_TARGET)
  omp_offload_init();
#endif
#endif
}

void
SNA::grow_rij(int newnmax)
{
  if (newnmax <= nmax)
    return;

  nmax = newnmax;

  // Allocate memory for rij - 2D array
  rij.resize(num_atoms, num_nbor, 3);
  inside.resize(num_atoms, num_nbor);
  wj.resize(num_atoms, num_nbor);
  rcutij.resize(num_atoms, num_nbor);
}

/* ----------------------------------------------------------------------
   compute Ui by summing over neighbors j
------------------------------------------------------------------------- */

void
SNA::compute_ui()
{
  // utot(j,ma,mb) = 0 for all j,ma,ma
  // utot(j,ma,ma) = 1 for all j,ma
  // for j in neighbors of i:
  //   compute r0 = (x,y,z,z0)
  //   utot(j,ma,mb) += u(r0;j,ma,mb) for all j,ma,mb

  zero_uarraytot();
  addself_uarraytot(wself);

#if _OPENMP
#if (OPENMP_TARGET)
#pragma omp target teams distribute parallel for collapse(2)
#else
#pragma omp parallel for collapse(2) default(none)                             \
  shared(rootpqarray, ulist_parity, idxu_block, ulist, MY_PI)
#endif
#endif
  for (int nbor = 0; nbor < num_nbor; nbor++) {
    for (int natom = 0; natom < num_atoms; natom++) {
      SNADOUBLE x = rij(natom, nbor, 0);
      SNADOUBLE y = rij(natom, nbor, 1);
      SNADOUBLE z = rij(natom, nbor, 2);
      SNADOUBLE rsq = x * x + y * y + z * z;
      SNADOUBLE r = sqrt(rsq);

      SNADOUBLE theta0 =
        (r - rmin0) * rfac0 * MY_PI / (rcutij(natom, nbor) - rmin0);
      SNADOUBLE z0 = r / tan(theta0);

      compute_uarray(natom, nbor, x, y, z, z0, r);
      add_uarraytot(natom, nbor, r, wj(natom, nbor), rcutij(natom, nbor));
    }
  }
}

/* ----------------------------------------------------------------------
   compute Yi from Ui without storing Zi, looping over zlist
------------------------------------------------------------------------- */

void
SNA::compute_yi(SNADOUBLE* beta)
{

  // Initialize ylist elements to zeros
#if _OPENMP
#if defined(OPENMP_TARGET)
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for collapse(2) default(none)
#endif
#endif
  for (int natom = 0; natom < num_atoms; natom++)
    for (int jjdu = 0; jjdu < idxdu_max; jjdu++)
      ylist(natom, jjdu) = { 0.0, 0.0 };

#if _OPENMP
#if defined(OPENMP_TARGET)
#pragma omp target teams distribute parallel for collapse(2)
#else
#pragma omp parallel for collapse(2) default(none) shared(idxz,                \
                                                          idxzbeta,            \
                                                          idxcg_block,         \
                                                          idxdu_block,         \
                                                          idxu_block,          \
                                                          cglist,              \
                                                          ulisttot,            \
                                                          ylist)
#endif
#endif
#if defined(_OPENMP) && defined(OPENMP_TARGET)
  for (int jjz = 0; jjz < idxz_max; jjz++)
    for (int natom = 0; natom < num_atoms; natom++)
#else
  for (int natom = 0; natom < num_atoms; natom++)
    for (int jjz = 0; jjz < idxz_max; jjz++)
#endif
    {
      const int j1 = idxz(jjz, 0);
      const int j2 = idxz(jjz, 1);
      const int j = idxz(jjz, 2);
      const int ma1min = idxz(jjz, 3);
      const int ma2max = idxz(jjz, 4);
      const int na = idxz(jjz, 5);
      const int mb1min = idxz(jjz, 6);
      const int mb2max = idxz(jjz, 7);
      const int nb = idxz(jjz, 8);

      const SNADOUBLE betaj = idxzbeta(jjz);

      const SNADOUBLE* cgblock = cglist.dptr + idxcg_block(j1, j2, j);
      int mb = (2 * (mb1min + mb2max) - j1 - j2 + j) / 2;
      int ma = (2 * (ma1min + ma2max) - j1 - j2 + j) / 2;
      const int jjdu = idxdu_block(j) + (j + 1) * mb + ma;

      int jju1 = idxu_block(j1) + (j1 + 1) * mb1min;
      int jju2 = idxu_block(j2) + (j2 + 1) * mb2max;
      int icgb = mb1min * (j2 + 1) + mb2max;

      SNADOUBLE ztmp_r = 0.0;
      SNADOUBLE ztmp_i = 0.0;

      // loop over columns of u1 and corresponding
      // columns of u2 satisfying Clebsch-Gordan constraint
      //      2*mb-j = 2*mb1-j1 + 2*mb2-j2

      for (int ib = 0; ib < nb; ib++) {

        SNADOUBLE suma1_r = 0.0;
        SNADOUBLE suma1_i = 0.0;

        int ma1 = ma1min;
        int ma2 = ma2max;
        int icga = ma1min * (j2 + 1) + ma2max;

        // loop over elements of row u1[mb1] and corresponding elements
        // of row u2[mb2] satisfying Clebsch-Gordan constraint
        //      2*ma-j = 2*ma1-j1 + 2*ma2-j2

        for (int ia = 0; ia < na; ia++) {
          suma1_r +=
            cgblock[icga] *
            (ulisttot(natom, jju1 + ma1).re * ulisttot(natom, jju2 + ma2).re -
             ulisttot(natom, jju1 + ma1).im * ulisttot(natom, jju2 + ma2).im);
          suma1_i +=
            cgblock[icga] *
            (ulisttot(natom, jju1 + ma1).re * ulisttot(natom, jju2 + ma2).im +
             ulisttot(natom, jju1 + ma1).im * ulisttot(natom, jju2 + ma2).re);

          ma1++;
          ma2--;
          icga += j2;
        } // end loop over ia

        ztmp_r += cgblock[icgb] * suma1_r;
        ztmp_i += cgblock[icgb] * suma1_i;
        jju1 += j1 + 1;
        jju2 -= j2 + 1;
        icgb += j2;
      } // end loop over ib

      // apply z(j1,j2,j,ma,mb) to unique element of y(j)

#if _OPENMP
#pragma omp atomic
#endif
      ylist(natom, jjdu).re += betaj * ztmp_r;
#if _OPENMP
#pragma omp atomic
#endif
      ylist(natom, jjdu).im += betaj * ztmp_i;

    } // end jjz and natom loop
}

/* ----------------------------------------------------------------------
   compute dEidRj
------------------------------------------------------------------------- */

void
SNA::compute_deidrj()
{
#if _OPENMP
#if (OPENMP_TARGET)
#pragma omp target teams distribute parallel for collapse(2)
#else
#pragma omp parallel for collapse(2)
#endif
#endif
  for (int nbor = 0; nbor < num_nbor; nbor++) {
    for (int natom = 0; natom < num_atoms; natom++) {
      for (int k = 0; k < 3; k++)
        dedr(natom, nbor, k) = 0.0;

      for (int j = 0; j <= twojmax; j++) {
        int jjdu = idxdu_block(j);

        for (int mb = 0; 2 * mb < j; mb++)
          for (int ma = 0; ma <= j; ma++) {

            SNADOUBLE jjjmambyarray_r = ylist(natom, jjdu).re;
            SNADOUBLE jjjmambyarray_i = ylist(natom, jjdu).im;

            for (int k = 0; k < 3; k++)
              dedr(natom, nbor, k) +=
                dulist(natom, nbor, jjdu, k).re * jjjmambyarray_r +
                dulist(natom, nbor, jjdu, k).im * jjjmambyarray_i;
            jjdu++;
          } // end loop over ma mb

        // For j even, handle middle column

        if (j % 2 == 0) {

          int mb = j / 2;
          for (int ma = 0; ma < mb; ma++) {
            SNADOUBLE jjjmambyarray_r = ylist(natom, jjdu).re;
            SNADOUBLE jjjmambyarray_i = ylist(natom, jjdu).im;

            for (int k = 0; k < 3; k++)
              dedr(natom, nbor, k) +=
                dulist(natom, nbor, jjdu, k).re * jjjmambyarray_r +
                dulist(natom, nbor, jjdu, k).im * jjjmambyarray_i;
            jjdu++;
          }

          SNADOUBLE jjjmambyarray_r = ylist(natom, jjdu).re;
          SNADOUBLE jjjmambyarray_i = ylist(natom, jjdu).im;

          for (int k = 0; k < 3; k++)
            dedr(natom, nbor, k) +=
              (dulist(natom, nbor, jjdu, k).re * jjjmambyarray_r +
               dulist(natom, nbor, jjdu, k).im * jjjmambyarray_i) *
              0.5;
          jjdu++;

        } // end if jeven

      } // end loop over j

      for (int k = 0; k < 3; k++)
        dedr(natom, nbor, k) *= 2.0;

    } // nbor
  }   // natom
#if defined(_OPENMP) && defined(OPENMP_TARGET)
#pragma omp target update from(this->dedr.dptr [0:this->dedr.size])
#endif
}

/* ----------------------------------------------------------------------
   calculate derivative of Ui w.r.t. atom j
------------------------------------------------------------------------- */

void
SNA::compute_duidrj()
{
#if _OPENMP
#if defined(OPENMP_TARGET)
#pragma omp target teams distribute parallel for collapse(2)
#else
#pragma omp parallel default(none) shared(rij, wj, rcutij, rootpqarray, dulist)
#pragma omp for collapse(2)
#endif
#endif
  for (int nbor = 0; nbor < num_nbor; nbor++) {
    for (int natom = 0; natom < num_atoms; natom++) {
      SNADOUBLE wj_in = wj(natom, nbor);
      SNADOUBLE rcut = rcutij(natom, nbor);

      SNADOUBLE x = rij(natom, nbor, 0);
      SNADOUBLE y = rij(natom, nbor, 1);
      SNADOUBLE z = rij(natom, nbor, 2);
      SNADOUBLE rsq = x * x + y * y + z * z;
      SNADOUBLE r = sqrt(rsq);
      SNADOUBLE rscale0 = rfac0 * MY_PI / (rcut - rmin0);
      SNADOUBLE theta0 = (r - rmin0) * rscale0;
      SNADOUBLE cs = cos(theta0);
      SNADOUBLE sn = sin(theta0);
      SNADOUBLE z0 = r * cs / sn;
      SNADOUBLE dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

      compute_duarray(natom, nbor, x, y, z, z0, r, dz0dr, wj_in, rcut);
    }
  }
}

/* ---------------------------------------------------------------------- */

void
SNA::zero_uarraytot()
{
#if _OPENMP
#if defined(OPENMP_TARGET)
#pragma omp target teams distribute parallel for collapse(2)
#else
#pragma omp parallel for collapse(2) default(none) shared(ulisttot)
#endif
#endif
  for (int natom = 0; natom < num_atoms; ++natom)
    for (int j = 0; j < idxu_max; j++)
      ulisttot(natom, j) = { 0.0, 0.0 };
}

/* ---------------------------------------------------------------------- */

void
SNA::addself_uarraytot(SNADOUBLE wself_in)
{
#if _OPENMP
#if (OPENMP_TARGET)
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for default(none) shared(ulisttot, wself_in, idxu_block)
#endif
#endif
  for (int natom = 0; natom < num_atoms; natom++) {
    for (int j = 0; j <= twojmax; j++) {
      int jju = idxu_block(j);
      for (int ma = 0; ma <= j; ma++) {
        ulisttot(natom, jju) = { wself_in, 0.0 };
        jju += j + 2;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   add Wigner U-functions for one neighbor to the total
------------------------------------------------------------------------- */

void
SNA::add_uarraytot(int natom,
                   int nbor,
                   SNADOUBLE r,
                   SNADOUBLE wj_in,
                   SNADOUBLE rcut)
{
  SNADOUBLE sfac;

  sfac = compute_sfac(r, rcut);
  sfac *= wj_in;

  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block(j);
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
#if _OPENMP
#pragma omp atomic
#endif
        ulisttot(natom, jju).re += sfac * ulist(natom, nbor, jju).re;
#if _OPENMP
#pragma omp atomic
#endif
        ulisttot(natom, jju).im += sfac * ulist(natom, nbor, jju).im;
        jju++;
      }
  }
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for one neighbor
------------------------------------------------------------------------- */

void
SNA::compute_uarray(int natom,
                    int nbor,
                    SNADOUBLE x,
                    SNADOUBLE y,
                    SNADOUBLE z,
                    SNADOUBLE z0,
                    SNADOUBLE r)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, b_r, a_i, b_i;
  SNADOUBLE rootpq;
  int jju, jjup;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  SNADOUBLE sfac;

  sfac = compute_sfac(r, rcutij(natom, nbor));
  sfac *= wj(natom, nbor);
  // Recursion relations
  // VMK Section 4.8.2

  //   u[j,ma,mb] = Sqrt((j-ma)/(j-mb)) a* u[j-1,ma,mb]
  //               -Sqrt((ma)/(j-mb)) b* u[j-1,ma-1,mb]

  //   u[j,ma,mb] = Sqrt((j-ma)/(mb)) b u[j-1,ma,mb-1]
  //                Sqrt((ma)/(mb)) a u[j-1,ma-1,mb-1]

  // initialize first entry
  // initialize top row of each layer to zero
  ulist(natom, nbor, 0).re = 1.0;
  ulist(natom, nbor, 0).im = 0.0;

  // skip over right half of each uarray
  jju = 1;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++) {
      ulist(natom, nbor, jju).re = 0.0;
      ulist(natom, nbor, jju).im = 0.0;
      jju += deljju;
    }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }

  jju = 1;
  jjup = 0;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    int deljjup = j;
    int mb_max = (j + 1) / 2;
    int ma_max = j;
    int m_max = ma_max * mb_max;

    // fill in left side of matrix layer from previous layer
    for (int m_iter = 0; m_iter < m_max; ++m_iter) {
      int mb = m_iter / ma_max;
      int ma = m_iter % ma_max;
      double up_r = ulist(natom, nbor, jjup).re;
      double up_i = ulist(natom, nbor, jjup).im;

      rootpq = rootpqarray(j - ma, j - mb);
      ulist(natom, nbor, jju).re += rootpq * (a_r * up_r + a_i * up_i);
      ulist(natom, nbor, jju).im += rootpq * (a_r * up_i - a_i * up_r);

      rootpq = rootpqarray(ma + 1, j - mb);
      ulist(natom, nbor, jju + 1).re = -rootpq * (b_r * up_r + b_i * up_i);
      ulist(natom, nbor, jju + 1).im = -rootpq * (b_r * up_i - b_i * up_r);

      // assign middle column i.e. mb+1

      if (2 * (mb + 1) == j) {
        rootpq = rootpqarray(j - ma, mb + 1);
        ulist(natom, nbor, jju + deljju).re +=
          rootpq * (b_r * up_r - b_i * up_i);
        ulist(natom, nbor, jju + deljju).im +=
          rootpq * (b_r * up_i + b_i * up_r);

        rootpq = rootpqarray(ma + 1, mb + 1);
        ulist(natom, nbor, jju + 1 + deljju).re =
          rootpq * (a_r * up_r - a_i * up_i);
        ulist(natom, nbor, jju + 1 + deljju).im =
          rootpq * (a_r * up_i + a_i * up_r);
      }

      jju++;
      jjup++;

      if (ma == ma_max - 1)
        jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2)
    // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])
    // dependence on idxu_block could be removed
    // renamed counters b/c can not modify jju, jjup
    int jjui = idxu_block(j);
    int jjuip = jjui + (j + 1) * (j + 1) - 1;
    for (int mb = 0; 2 * mb < j; mb++) {
      for (int ma = 0; ma <= j; ma++) {
        ulist(natom, nbor, jjuip).re =
          ulist_parity(jjui) * ulist(natom, nbor, jjui).re;
        ulist(natom, nbor, jjuip).im =
          ulist_parity(jjui) * -ulist(natom, nbor, jjui).im;
        jjui++;
        jjuip--;
      }
    }

    // skip middle and right half cols
    // b/c no longer using idxu_block
    if (j % 2 == 0)
      jju += deljju;
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
    int ncolhalfp = deljjup / 2;
    jjup += deljjup * ncolhalfp;
  }
}

/* ----------------------------------------------------------------------
   Compute derivatives of Wigner U-functions for one neighbor
   see comments in compute_uarray()
------------------------------------------------------------------------- */

void
SNA::compute_duarray(int natom,
                     int nbor,
                     SNADOUBLE x,
                     SNADOUBLE y,
                     SNADOUBLE z,
                     SNADOUBLE z0,
                     SNADOUBLE r,
                     SNADOUBLE dz0dr,
                     SNADOUBLE wj_in,
                     SNADOUBLE rcut)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, a_i, b_r, b_i;
  SNADOUBLE da_r[3], da_i[3], db_r[3], db_i[3];
  SNADOUBLE dz0[3], dr0inv[3], dr0invdr;
  SNADOUBLE rootpq;
  int jju, jjup, jjdu, jjdup;

  SNADOUBLE rinv = 1.0 / r;
  SNADOUBLE ux = x * rinv;
  SNADOUBLE uy = y * rinv;
  SNADOUBLE uz = z * rinv;

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = z0 * r0inv;
  a_i = -z * r0inv;
  b_r = y * r0inv;
  b_i = -x * r0inv;

  dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; k++) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }

  da_i[2] += -r0inv;

  for (int k = 0; k < 3; k++) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }

  db_i[0] += -r0inv;
  db_r[1] += r0inv;

  for (int k = 0; k < 3; ++k)
    dulist(natom, nbor, 0, k) = { 0.0, 0.0 };

  jju = 1;
  jjdu = 1;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++) {

      for (int k = 0; k < 3; ++k)
        dulist(natom, nbor, jjdu, k) = { 0.0, 0.0 };

      jju += deljju;
      jjdu += deljju;
    }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }

  jju = 1;
  jjdu = 1;
  jjup = 0;
  jjdup = 0;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    int deljjup = j;

    for (int mb = 0; 2 * mb < j; mb++) {

      for (int ma = 0; ma < j; ma++) {

        double up_r = ulist(natom, nbor, jjup).re;
        double up_i = ulist(natom, nbor, jjup).im;

        rootpq = rootpqarray(j - ma, j - mb);
        for (int k = 0; k < 3; k++) {
          dulist(natom, nbor, jjdu, k).re +=
            rootpq * (da_r[k] * up_r + da_i[k] * up_i +
                      a_r * dulist(natom, nbor, jjdup, k).re +
                      a_i * dulist(natom, nbor, jjdup, k).im);
          dulist(natom, nbor, jjdu, k).im +=
            rootpq * (da_r[k] * up_i - da_i[k] * up_r +
                      a_r * dulist(natom, nbor, jjdup, k).im -
                      a_i * dulist(natom, nbor, jjdup, k).re);
        }

        rootpq = rootpqarray(ma + 1, j - mb);
        for (int k = 0; k < 3; k++) {
          dulist(natom, nbor, jjdu + 1, k).re =
            -rootpq * (db_r[k] * up_r + db_i[k] * up_i +
                       b_r * dulist(natom, nbor, jjdup, k).re +
                       b_i * dulist(natom, nbor, jjdup, k).im);
          dulist(natom, nbor, jjdu + 1, k).im =
            -rootpq * (db_r[k] * up_i - db_i[k] * up_r +
                       b_r * dulist(natom, nbor, jjdup, k).im -
                       b_i * dulist(natom, nbor, jjdup, k).re);
        }

        // assign middle column i.e. mb+1

        if (2 * (mb + 1) == j) {
          rootpq = rootpqarray(j - ma, mb + 1);
          for (int k = 0; k < 3; k++) {
            dulist(natom, nbor, jjdu + deljju, k).re +=
              rootpq * (db_r[k] * up_r - db_i[k] * up_i +
                        b_r * dulist(natom, nbor, jjdup, k).re -
                        b_i * dulist(natom, nbor, jjdup, k).im);
            dulist(natom, nbor, jjdu + deljju, k).im +=
              rootpq * (db_r[k] * up_i + db_i[k] * up_r +
                        b_r * dulist(natom, nbor, jjdup, k).im +
                        b_i * dulist(natom, nbor, jjdup, k).re);
          }

          rootpq = rootpqarray(ma + 1, mb + 1);
          for (int k = 0; k < 3; k++) {
            dulist(natom, nbor, jjdu + 1 + deljju, k).re =
              rootpq * (da_r[k] * up_r - da_i[k] * up_i +
                        a_r * dulist(natom, nbor, jjdup, k).re -
                        a_i * dulist(natom, nbor, jjdup, k).im);
            dulist(natom, nbor, jjdu + 1 + deljju, k).im =
              rootpq * (da_r[k] * up_i + da_i[k] * up_r +
                        a_r * dulist(natom, nbor, jjdup, k).im +
                        a_i * dulist(natom, nbor, jjdup, k).re);
          }
        }

        jju++;
        jjup++;
        jjdu++;
        jjdup++;
      }
      jju++;
      jjdu++;
    }
    if (j % 2 == 0) {
      jju += deljju;
      jjdu += deljju;
    }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
    int ncolhalfp = deljjup / 2;
    jjup += deljjup * ncolhalfp;
  }

  SNADOUBLE sfac = compute_sfac(r, rcut);
  SNADOUBLE dsfac = compute_dsfac(r, rcut);

  sfac *= wj_in;
  dsfac *= wj_in;
  jju = 0;
  jjdu = 0;
  for (int j = 0; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        dulist(natom, nbor, jjdu, 0).re =
          dsfac * ulist(natom, nbor, jju).re * ux +
          sfac * dulist(natom, nbor, jjdu, 0).re;
        dulist(natom, nbor, jjdu, 0).im =
          dsfac * ulist(natom, nbor, jju).im * ux +
          sfac * dulist(natom, nbor, jjdu, 0).im;
        dulist(natom, nbor, jjdu, 1).re =
          dsfac * ulist(natom, nbor, jju).re * uy +
          sfac * dulist(natom, nbor, jjdu, 1).re;
        dulist(natom, nbor, jjdu, 1).im =
          dsfac * ulist(natom, nbor, jju).im * uy +
          sfac * dulist(natom, nbor, jjdu, 1).im;
        dulist(natom, nbor, jjdu, 2).re =
          dsfac * ulist(natom, nbor, jju).re * uz +
          sfac * dulist(natom, nbor, jjdu, 2).re;
        dulist(natom, nbor, jjdu, 2).im =
          dsfac * ulist(natom, nbor, jju).im * uz +
          sfac * dulist(natom, nbor, jjdu, 2).im;
        jju++;
        jjdu++;
      }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }
}

/* ----------------------------------------------------------------------
   memory usage of arrays
------------------------------------------------------------------------- */

SNADOUBLE
SNA::memory_usage()
{
  int jdimpq = twojmax + 2;
  int jdim = twojmax + 1;
  SNADOUBLE bytes;
  bytes = ncoeff * sizeof(SNADOUBLE); // coeff

  bytes += jdimpq * jdimpq * sizeof(SNADOUBLE); // pqarray
  bytes += idxcg_max * sizeof(SNADOUBLE);       // cglist
  bytes += jdim * jdim * jdim * sizeof(int);    // idxcg_block

  bytes += idxu_max * sizeof(SNADOUBLE) * 2;      // ulist
  bytes += idxu_max * sizeof(SNADOUBLE) * 2;      // ulisttot
  bytes += idxdu_max * 3 * sizeof(SNADOUBLE) * 2; // dulist
  bytes += idxu_max * sizeof(int);                // ulist_parity
  bytes += jdim * sizeof(int);                    // idxu_block
  bytes += jdim * sizeof(int);                    // idxdu_block

  bytes += idxz_max * 9 * sizeof(int);       // idxz
  bytes += idxz_max * sizeof(SNADOUBLE);     // idxzbeta
  bytes += jdim * jdim * jdim * sizeof(int); // idxz_block

  bytes += idxdu_max * sizeof(SNADOUBLE) * 2; // ylist
  bytes += idxb_max * 3 * sizeof(int);        // idxb

  bytes += jdim * jdim * jdim * sizeof(int); // idxb_block

  return bytes;
}

/* ---------------------------------------------------------------------- */

void
SNA::create_twojmax_arrays()
{
  int jdimpq = twojmax + 2;

  rootpqarray.resize(jdimpq, jdimpq);

  cglist.resize(idxcg_max);
  // ALlocate memory for ulist
  ulist.resize(num_atoms, num_nbor, idxu_max);
  dedr.resize(num_atoms, num_nbor, 3);
  ylist.resize(num_atoms, idxdu_max);
  ulisttot.resize(num_atoms, idxu_max);
  dulist.resize(num_atoms, num_nbor, idxdu_max, 3);
}

/* ---------------------------------------------------------------------- */

void
SNA::destroy_twojmax_arrays()
{
  //  memory->destroy(cglist);
  //
  //  memory->destroy(idxz_block);
  //
  //  memory->destroy(idxb);
  //  memory->destroy(idxb_block);
}

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

SNADOUBLE
SNA::factorial(int n)
{
  if (n < 0 || n > nmaxfactorial) {
    // printf("Invalid argument to factorial %d", n);
    exit(1);
  }

  return nfac_table[n];
}

/* ----------------------------------------------------------------------
   factorial n table, size SNA::nmaxfactorial+1
------------------------------------------------------------------------- */

const SNADOUBLE SNA::nfac_table[] = {
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e+15,
  1.21645100408832e+17,
  2.43290200817664e+18,
  5.10909421717094e+19,
  1.12400072777761e+21,
  2.5852016738885e+22,
  6.20448401733239e+23,
  1.5511210043331e+25,
  4.03291461126606e+26,
  1.08888694504184e+28,
  3.04888344611714e+29,
  8.8417619937397e+30,
  2.65252859812191e+32,
  8.22283865417792e+33,
  2.63130836933694e+35,
  8.68331761881189e+36,
  2.95232799039604e+38,
  1.03331479663861e+40,
  3.71993326789901e+41,
  1.37637530912263e+43,
  5.23022617466601e+44,
  2.03978820811974e+46, // nmaxfactorial = 39
};

/* ----------------------------------------------------------------------
   the function delta given by VMK Eq. 8.2(1)
------------------------------------------------------------------------- */

SNADOUBLE
SNA::deltacg(int j1, int j2, int j)
{
  SNADOUBLE sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2) *
              factorial((-j1 + j2 + j) / 2) / sfaccg);
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients using
   the quasi-binomial formula VMK 8.2.1(3)
------------------------------------------------------------------------- */

void
SNA::init_clebsch_gordan()
{
  SNADOUBLE sum, dcg, sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; m1++) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2++) {

            // -c <= cc <= c

            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if (m < 0 || m > j) {
              cglist(idxcg_count) = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int z = MAX(0, MAX(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                 z <=
                 MIN((j1 + j2 - j) / 2, MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                 z++) {
              ifac = z % 2 ? -1 : 1;
              sum += ifac / (factorial(z) * factorial((j1 + j2 - j) / 2 - z) *
                             factorial((j1 - aa2) / 2 - z) *
                             factorial((j2 + bb2) / 2 - z) *
                             factorial((j - j2 + aa2) / 2 + z) *
                             factorial((j - j1 - bb2) / 2 + z));
            }

            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = sqrt(
              factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2) *
              factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2) *
              factorial((j + cc2) / 2) * factorial((j - cc2) / 2) * (j + 1));

            cglist(idxcg_count) = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
}

/* ----------------------------------------------------------------------
   pre-compute table of sqrt[p/m2], p, q = 1,twojmax
   the p = 0, q = 0 entries are allocated and skipped for convenience.
   a second table is computed with +/-1 parity factor
------------------------------------------------------------------------- */

void
SNA::init_rootpqarray()
{
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray(p, q) = sqrt(static_cast<SNADOUBLE>(p) / q);
}

/* ---------------------------------------------------------------------- */

int
SNA::compute_ncoeff()
{
  int ncount;

  ncount = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          ncount++;

  return ncount;
}

/* ---------------------------------------------------------------------- */

SNADOUBLE
SNA::compute_sfac(SNADOUBLE r, SNADOUBLE rcut)
{
  if (switch_flag == 0)
    return 1.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 1.0;
    else if (r > rcut)
      return 0.0;
    else {
      SNADOUBLE rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

SNADOUBLE
SNA::compute_dsfac(SNADOUBLE r, SNADOUBLE rcut)
{
  if (switch_flag == 0)
    return 0.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 0.0;
    else if (r > rcut)
      return 0.0;
    else {
      SNADOUBLE rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}
