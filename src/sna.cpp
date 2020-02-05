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

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "sna.h"
#include "memory.h"

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

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))
static const SNADOUBLE MY_PI  = 3.14159265358979323846; // pi

SNA::SNA(Memory* memory_in, SNADOUBLE rfac0_in,
         int twojmax_in,
         SNADOUBLE rmin0_in, int switch_flag_in, int /*bzero_flag_in*/,
         const double* beta)
{
  wself = 1.0;

  memory = memory_in;
  rfac0 = rfac0_in;
  rmin0 = rmin0_in;
  switch_flag = switch_flag_in;

  twojmax = twojmax_in;

  ncoeff = compute_ncoeff();

  rij = NULL;
  inside = NULL;
  wj = NULL;
  rcutij = NULL;
  nmax = 0;
  idxz_j1j2j = NULL;
  idxz_ma = NULL;
  idxz_mb = NULL;
  idxb = NULL;

  build_indexlist(beta);
  create_twojmax_arrays();

}

/* ---------------------------------------------------------------------- */

SNA::~SNA()
{
  destroy_twojmax_arrays();
  memory->destroy(rij);
  memory->destroy(inside);
  memory->destroy(wj);
  memory->destroy(rcutij);
  memory->destroy(idxz_j1j2j);
  memory->destroy(idxz_ma);
  memory->destroy(idxz_mb);
  memory->destroy(idxb);
}

void SNA::build_indexlist(const double* beta)
{
  int jdim = twojmax + 1;
  
  // index list for cglist

  memory->create(idxcg_block, jdim, jdim, jdim,
                 "sna:idxcg_block");

  int idxcg_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxcg_block[j1][j2][j] = idxcg_count; 
        for (int m1 = 0; m1 <= j1; m1++)
          for (int m2 = 0; m2 <= j2; m2++)
            idxcg_count++;
      }
  idxcg_max = idxcg_count;

  // index list for uarray
  // need to include both halves
  // **** idxu_block could be moved into idxb ****

  memory->create(idxu_block, jdim,
                 "sna:idxu_block");

  int idxu_count = 0;
  
  for(int j = 0; j <= twojmax; j++) {
    idxu_block[j] = idxu_count; 
    for(int mb = 0; mb <= j; mb++)
      for(int ma = 0; ma <= j; ma++)
        idxu_count++;
  }
  idxu_max = idxu_count;

  // index list for beta and B

  int idxb_count = 0;  
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) idxb_count++;
  
  idxb_max = idxb_count;
  idxb = new SNA_BINDICES[idxb_max];
  
  idxb_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) {
          idxb[idxb_count].j1 = j1;
          idxb[idxb_count].j2 = j2;
          idxb[idxb_count].j = j;
          idxb_count++;
        }

  // reverse index list for beta and b

  memory->create(idxb_block, jdim, jdim, jdim,
                 "sna:idxb_block");
  idxb_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        if (j < j1) continue;
        idxb_block[j1][j2][j] = idxb_count; 
        idxb_count++;
      }

  // index list for zlist

  int idxz_count = 0;
  int idxz_j1j2j_count = 0;
  int idxz_ma_count = 0;
  int idxz_mb_count = 0;

  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxz_j1j2j_count++;
        for (int ma = 0; ma <= j; ma++)
          idxz_ma_count++;
        for (int mb = 0; 2*mb <= j; mb++) 
          idxz_mb_count++;
        for (int mb = 0; 2*mb <= j; mb++) 
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;
      }

  idxz_max = idxz_count;
  idxz_j1j2j_max = idxz_j1j2j_count;
  idxz_ma_max = idxz_ma_count;
  idxz_mb_max = idxz_mb_count;
  memory->create(idxz_j1j2j,idxz_j1j2j_max,3,
                 "sna:idxz_j1j2j");
  memory->create(idxz_ma,idxz_ma_max,3,
                 "sna:idxz_ma");
  memory->create(idxz_mb,idxz_mb_max,3,
                 "sna:idxz_mb");

  // **** idxz_block could be moved into idxb ****
  memory->create(idxz_block, jdim, jdim, jdim,
                 "sna:idxz_block");
  
  idxz_count = 0;
  idxz_j1j2j_count = 0;
  idxz_ma_count = 0;
  idxz_mb_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {

        idxz_j1j2j[idxz_j1j2j_count][0] = j1;
        idxz_j1j2j[idxz_j1j2j_count][1] = j2;
        idxz_j1j2j[idxz_j1j2j_count][2] = j;
        idxz_j1j2j_count++;

        for (int ma = 0; ma <= j; ma++) {
          int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
          idxz_ma[idxz_ma_count][0] = ma1min;
          idxz_ma[idxz_ma_count][1] = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
          idxz_ma[idxz_ma_count][2] = MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;
          idxz_ma_count++;
        }

        for (int mb = 0; 2*mb <= j; mb++) {
          int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
          idxz_mb[idxz_mb_count][0] = mb1min;
          idxz_mb[idxz_mb_count][1] = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
          idxz_mb[idxz_mb_count][2] = MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;
          idxz_mb_count++;
        }

        idxz_block[j1][j2][j] = idxz_count; 
        for (int mb = 0; 2*mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;

      }
}

/* ---------------------------------------------------------------------- */

void SNA::init()
{
  init_clebsch_gordan();
  init_rootpqarray();
}


void SNA::grow_rij(int newnmax)
{
  if(newnmax <= nmax) return;

  nmax = newnmax;

  memory->destroy(rij);
  memory->destroy(inside);
  memory->destroy(wj);
  memory->destroy(rcutij);
  memory->create(rij, nmax, 3, "pair:rij");
  memory->create(inside, nmax, "pair:inside");
  memory->create(wj, nmax, "pair:wj");
  memory->create(rcutij, nmax, "pair:rcutij");
}

/* ----------------------------------------------------------------------
   compute Ui by summing over neighbors j
------------------------------------------------------------------------- */

void SNA::compute_ui(int jnum)
{
  SNADOUBLE rsq, r, x, y, z, z0, theta0;

  // utot(j,ma,mb) = 0 for all j,ma,ma
  // utot(j,ma,ma) = 1 for all j,ma
  // for j in neighbors of i:
  //   compute r0 = (x,y,z,z0)
  //   utot(j,ma,mb) += u(r0;j,ma,mb) for all j,ma,mb

  zero_uarraytot();
  addself_uarraytot(wself);

  for(int j = 0; j < jnum; j++) {
    x = rij[j][0];
    y = rij[j][1];
    z = rij[j][2];
    rsq = x * x + y * y + z * z;
    r = sqrt(rsq);

    theta0 = (r - rmin0) * rfac0 * MY_PI / (rcutij[j] - rmin0);
    //    theta0 = (r - rmin0) * rscale0;
    z0 = r / tan(theta0);

    compute_uarray(x, y, z, z0, r);

    add_uarraytot(r, wj[j], rcutij[j]);
  }

}

/* ----------------------------------------------------------------------
   compute Yi from Ui without storing Zi, looping over zlist
------------------------------------------------------------------------- */

void SNA::compute_yi(SNADOUBLE* beta)
{
  for(int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for(int mb = 0; 2*mb <= j; mb++)
      for(int ma = 0; ma <= j; ma++) {
        ylist_r[jju] = 0.0;
        ylist_i[jju] = 0.0;
        jju++;
      } // end loop over ma, mb
  } // end loop over j

  int idxz_j1j2j_count = 0;
  int idxz_ma_count = 0;
  int idxz_mb_count = 0;
  for(int jjz = 0; jjz < idxz_j1j2j_max; jjz++) {
    const int j1 = idxz_j1j2j[idxz_j1j2j_count][0];
    const int j2 = idxz_j1j2j[idxz_j1j2j_count][1];
    const int j = idxz_j1j2j[idxz_j1j2j_count][2];
    idxz_j1j2j_count++;
    int jju = idxu_block[j];

    int* idxz_mb_ptr = idxz_mb[idxz_mb_count];
    for (int mb = 0; 2*mb <= j; mb++) {
      const int mb1min = idxz_mb_ptr[0];
      const int mb2max = idxz_mb_ptr[1];
      const int nb = idxz_mb_ptr[2];
      idxz_mb_ptr += 3;

      int* idxz_ma_ptr = idxz_ma[idxz_ma_count];
      for (int ma = 0; ma <= j; ma++) {
        const int ma1min = idxz_ma_ptr[0];
        const int ma2max = idxz_ma_ptr[1];
        const int na = idxz_ma_ptr[2];
        idxz_ma_ptr += 3;

        const SNADOUBLE* cgblock = cglist + idxcg_block[j1][j2][j];

        SNADOUBLE ztmp_r = 0.0;
        SNADOUBLE ztmp_i = 0.0;

        int jju1 = idxu_block[j1] + (j1+1)*mb1min;
        int jju2 = idxu_block[j2] + (j2+1)*mb2max;
        int icgb = mb1min*(j2+1) + mb2max;
        for(int ib = 0; ib < nb; ib++) {

          SNADOUBLE suma1_r = 0.0;
          SNADOUBLE suma1_i = 0.0;

          const SNADOUBLE* u1_r = &ulisttot_r[jju1];
          const SNADOUBLE* u1_i = &ulisttot_i[jju1];
          const SNADOUBLE* u2_r = &ulisttot_r[jju2];
          const SNADOUBLE* u2_i = &ulisttot_i[jju2];

          int ma1 = ma1min;
          int ma2 = ma2max;
          int icga = ma1min*(j2+1) + ma2max;

          for(int ia = 0; ia < na; ia++) {
            suma1_r += cgblock[icga] * (u1_r[ma1] * u2_r[ma2] - u1_i[ma1] * u2_i[ma2]);
            suma1_i += cgblock[icga] * (u1_r[ma1] * u2_i[ma2] + u1_i[ma1] * u2_r[ma2]);
            ma1++;
            ma2--;
            icga += j2;
          } // end loop over ia

          ztmp_r += cgblock[icgb] * suma1_r;
          ztmp_i += cgblock[icgb] * suma1_i;
          jju1 += j1+1;
          jju2 -= j2+1;
          icgb += j2;
        } // end loop over ib

        // apply z(j1,j2,j,ma,mb) to unique element of y(j)
        // find right y_list[jju] and beta[jjb] entries
        // multiply and divide by j+1 factors
        // account for multiplicity of 1, 2, or 3
        
        SNADOUBLE betaj; 
        if (j >= j1) {
          const int jjb = idxb_block[j1][j2][j];
          if (j1 == j) {
            if (j2 == j) betaj = 3*beta[jjb];
            else betaj = 2*beta[jjb];
          } else betaj = beta[jjb]; 
        } else if (j >= j2) {
          const int jjb = idxb_block[j][j2][j1];
          if (j2 == j) betaj = 2*beta[jjb]*(j1+1)/(j+1.0);
          else betaj = beta[jjb]*(j1+1)/(j+1.0);
        } else {
          const int jjb = idxb_block[j2][j][j1];
          betaj = beta[jjb]*(j1+1)/(j+1.0); 
        }

        ylist_r[jju] += betaj*ztmp_r;
        ylist_i[jju] += betaj*ztmp_i;
        jju++;

        // printf("jju betaj ztmp zlist %d %g %g %d %d %d %d %d\n",jju,betaj,ztmp_r,j1,j2,j,ma,mb);
      } // end ma loop
    } // end mb loop
    idxz_mb_count += j/2+1;
    idxz_ma_count += j+1;
  } // end j1j2j loop
}

/* ----------------------------------------------------------------------
   compute dEidRj
------------------------------------------------------------------------- */

void SNA::compute_deidrj(SNADOUBLE* dedr)
{

  for(int k = 0; k < 3; k++)
    dedr[k] = 0.0;

  for(int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];

    for(int mb = 0; 2*mb < j; mb++)
      for(int ma = 0; ma <= j; ma++) {

        SNADOUBLE* dudr_r = dulist_r[jju];
        SNADOUBLE* dudr_i = dulist_i[jju];
        SNADOUBLE jjjmambyarray_r = ylist_r[jju];
        SNADOUBLE jjjmambyarray_i = ylist_i[jju];

        for(int k = 0; k < 3; k++)
          dedr[k] +=
            dudr_r[k] * jjjmambyarray_r +
            dudr_i[k] * jjjmambyarray_i;
        jju++;
      } //end loop over ma mb

    // For j even, handle middle column

    if (j%2 == 0) {

      int mb = j/2;
      for(int ma = 0; ma < mb; ma++) {
        SNADOUBLE* dudr_r = dulist_r[jju];
        SNADOUBLE* dudr_i = dulist_i[jju];
        SNADOUBLE jjjmambyarray_r = ylist_r[jju];
        SNADOUBLE jjjmambyarray_i = ylist_i[jju];

        for(int k = 0; k < 3; k++)
          dedr[k] +=
            dudr_r[k] * jjjmambyarray_r +
            dudr_i[k] * jjjmambyarray_i;
        jju++;
      }

      int ma = mb;
      SNADOUBLE* dudr_r = dulist_r[jju];
      SNADOUBLE* dudr_i = dulist_i[jju];
      SNADOUBLE jjjmambyarray_r = ylist_r[jju];
      SNADOUBLE jjjmambyarray_i = ylist_i[jju];

      for(int k = 0; k < 3; k++)
        dedr[k] += 
          (dudr_r[k] * jjjmambyarray_r +
           dudr_i[k] * jjjmambyarray_i)*0.5;
      jju++;

    } // end if jeven

  } // end loop over j

  for(int k = 0; k < 3; k++)
    dedr[k] *= 2.0;

}


/* ----------------------------------------------------------------------
   calculate derivative of Ui w.r.t. atom j
------------------------------------------------------------------------- */

void SNA::compute_duidrj(SNADOUBLE* rij, SNADOUBLE wj_in, SNADOUBLE rcut)
{
  SNADOUBLE rsq, r, x, y, z, z0, theta0, cs, sn;
  SNADOUBLE dz0dr;
  SNADOUBLE wj = wj_in;

  x = rij[0];
  y = rij[1];
  z = rij[2];
  rsq = x * x + y * y + z * z;
  r = sqrt(rsq);
  SNADOUBLE rscale0 = rfac0 * MY_PI / (rcut - rmin0);
  theta0 = (r - rmin0) * rscale0;
  cs = cos(theta0);
  sn = sin(theta0);
  z0 = r * cs / sn;
  dz0dr = z0 / r - (r*rscale0) * (rsq + z0 * z0) / rsq;

  compute_duarray(x, y, z, z0, r, dz0dr, wj, rcut);
}

/* ---------------------------------------------------------------------- */

void SNA::zero_uarraytot()
{
  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        ulisttot_r[jju] = 0.0;
        ulisttot_i[jju] = 0.0;
        jju++;
      }
  }
}

/* ---------------------------------------------------------------------- */

void SNA::addself_uarraytot(SNADOUBLE wself_in)
{
  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for (int ma = 0; ma <= j; ma++) {
      ulisttot_r[jju] = wself_in;
      ulisttot_i[jju] = 0.0;
      jju += j+2;
    }
  }
}

/* ----------------------------------------------------------------------
   add Wigner U-functions for one neighbor to the total
------------------------------------------------------------------------- */

void SNA::add_uarraytot(SNADOUBLE r, SNADOUBLE wj, SNADOUBLE rcut)
{
  SNADOUBLE sfac;

  sfac = compute_sfac(r, rcut);

  sfac *= wj;

  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        ulisttot_r[jju] +=
          sfac * ulist_r[jju];
        ulisttot_i[jju] +=
          sfac * ulist_i[jju];
        jju++;
      }
  }
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for one neighbor
------------------------------------------------------------------------- */

void SNA::compute_uarray(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                         SNADOUBLE z0, SNADOUBLE r)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, b_r, a_i, b_i;
  SNADOUBLE rootpq;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  // VMK Section 4.8.2

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  for (int j = 1; j <= twojmax; j++) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j-1];

    // fill in left side of matrix layer from previous layer

    for (int mb = 0; 2*mb <= j; mb++) {
      ulist_r[jju] = 0.0;
      ulist_i[jju] = 0.0;

      for (int ma = 0; ma < j; ma++) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] +=
          rootpq *
          (a_r * ulist_r[jjup] +
           a_i * ulist_i[jjup]);
        ulist_i[jju] +=
          rootpq *
          (a_r * ulist_i[jjup] -
           a_i * ulist_r[jjup]);

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq *
          (b_r * ulist_r[jjup] +
           b_i * ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq *
          (b_r * ulist_i[jjup] -
           b_i * ulist_r[jjup]);
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2)
    // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])

    jju = idxu_block[j];
    jjup = jju+(j+1)*(j+1)-1;
    int mbpar = 1;
    for (int mb = 0; 2*mb < j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
    
    // For j even, handle middle column

    if (j%2 == 0) {
      int mb = j/2;
      int mapar = mbpar;
      for (int ma = 0; 2*ma < j; ma++) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
    } // end if jeven
  }
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for one neighbor, inversion inlined
------------------------------------------------------------------------- */

void SNA::compute_uarray_inlineinversion(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                         SNADOUBLE z0, SNADOUBLE r)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, b_r, a_i, b_i;
  SNADOUBLE rootpq;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  // VMK Section 4.8.2

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  for (int j = 1; j <= twojmax; j++) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j-1];
    int jjui = jju+(j+1)*(j+1)-1;

    // fill in left side of matrix layer from previous layer

    int mbpar = 1;
    for (int mb = 0; 2*mb < j; mb++) {
      ulist_r[jju] = 0.0;
      ulist_i[jju] = 0.0;

      int mapar = mbpar;
      for (int ma = 0; ma < j; ma++) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] +=
          rootpq *
          (a_r * ulist_r[jjup] +
           a_i * ulist_i[jjup]);
        ulist_i[jju] +=
          rootpq *
          (a_r * ulist_i[jjup] -
           a_i * ulist_r[jjup]);

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq *
          (b_r * ulist_r[jjup] +
           b_i * ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq *
          (b_r * ulist_i[jjup] -
           b_i * ulist_r[jjup]);

        // copy left side to right side with inversion symmetry VMK 4.4(2)
        // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])

        if (mapar == 1) {
          ulist_r[jjui] = ulist_r[jju];
          ulist_i[jjui] = -ulist_i[jju];
        } else {
          ulist_r[jjui] = -ulist_r[jju];
          ulist_i[jjui] = ulist_i[jju];
        }

        jju++;
        jjup++;
        mapar = -mapar;
        jjui--;

      } // ma loop

      // copy the last entry in the column (ma == j)
      
      if (mapar == 1) {
        ulist_r[jjui] = ulist_r[jju];
        ulist_i[jjui] = -ulist_i[jju];
      } else {
        ulist_r[jjui] = -ulist_r[jju];
        ulist_i[jjui] = ulist_i[jju];
      }

      jju++;
      mbpar = -mbpar;
      jjui--;

    } // mb loop

    // For j even, handle middle column

    if (j%2 == 0) {

      int mb = j/2;
      ulist_r[jju] = 0.0;
      ulist_i[jju] = 0.0;
      int mapar = mbpar;
      for(int ma = 0; ma < mb; ma++) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] +=
          rootpq *
          (a_r * ulist_r[jjup] +
           a_i * ulist_i[jjup]);
        ulist_i[jju] +=
          rootpq *
          (a_r * ulist_i[jjup] -
           a_i * ulist_r[jjup]);
        
        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq *
          (b_r * ulist_r[jjup] +
           b_i * ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq *
          (b_r * ulist_i[jjup] -
           b_i * ulist_r[jjup]);
        
        // copy left side to right side with inversion symmetry VMK 4.4(2)
        // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])
        
        if (mapar == 1) {
          ulist_r[jjui] = ulist_r[jju];
          ulist_i[jjui] = -ulist_i[jju];
        } else {
          ulist_r[jjui] = -ulist_r[jju];
          ulist_i[jjui] = ulist_i[jju];
        }
          
        jju++;
        jjup++;
        mapar = -mapar;
        jjui--;

      }

      int ma = mb;

      rootpq = rootpqarray[j - ma][j - mb];
      ulist_r[jju] +=
        rootpq *
        (a_r * ulist_r[jjup] +
         a_i * ulist_i[jjup]);
      ulist_i[jju] +=
        rootpq *
        (a_r * ulist_i[jjup] -
         a_i * ulist_r[jjup]);
        
    } // end if jeven

  } // j loop
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for twojmax = 2
   based on explicit expansion of recursion:

     U(j,ma,mb) = Sqrt((j-ma)/(j-mb))Conj(a)U(j-1,ma,mb)
                 -Sqrt((ma)/(j-mb))Conj(b)U(j-1,ma-1,mb)

    ignore first term for ma=j, ignore second term for ma=0
    only use for 2*mb<=j, then use inversion symmettry for 2*mb>j
    j = 0,..,2jmax; ma,mb = 0,...,j, 
    so U is a set of (j+1)*(j+1) square matrices
    Ulist orders the entries sequentially (j,ma,mb), with ma changing
    fastest, then mb, then j.

------------------------------------------------------------------------- */

void SNA::compute_uarray_2J2(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                         SNADOUBLE z0, SNADOUBLE r)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, b_r, a_i, b_i;
  SNADOUBLE rootpq;
  int j,ma,mb,jju,jjup,mbpar,mapar;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  // j = 0

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  // j = 1

  jju = 1;
  ulist_r[jju] = rootpqarray[1][1] * a_r;
  ulist_i[jju] = -rootpqarray[1][1] * a_i;

  jju = 2;
  ulist_r[jju] = -rootpqarray[1][1] * b_r;
  ulist_i[jju] = rootpqarray[1][1] * b_i;

  // j = 2

  jju = 5;
  ulist_r[jju] =
    rootpqarray[2][2] *
    (a_r * (rootpqarray[1][1] * a_r) +
     a_i * (-rootpqarray[1][1] * a_i));
  ulist_i[jju] =
    rootpqarray[2][2] *
    (a_r * (-rootpqarray[1][1] * a_i) -
     a_i * (rootpqarray[1][1] * a_r));

  jju = 6;
  ulist_r[jju] =
    -rootpqarray[1][2] *
    (b_r * (rootpqarray[1][1] * a_r) +
     b_i * (-rootpqarray[1][1] * a_i))
    + 
    rootpqarray[1][2] *
    (a_r * (-rootpqarray[1][1] * b_r) +
     a_i * (rootpqarray[1][1] * b_i));
  ulist_i[jju] =
    -rootpqarray[1][2] *
    (b_r * (-rootpqarray[1][1] * a_i) -
     b_i * (rootpqarray[1][1] * a_r))
    + 
    rootpqarray[1][2] *
    (a_r * (rootpqarray[1][1] * b_i) -
     a_i * (-rootpqarray[1][1] * b_r));

  jju = 7;
  ulist_r[jju] =
    -rootpqarray[2][2] *
    (b_r * (-rootpqarray[1][1] * b_r) +
     b_i * (rootpqarray[1][1] * b_i));
  ulist_i[jju] =
    -rootpqarray[2][2] *
    (b_r * (rootpqarray[1][1] * b_i) -
     b_i * (-rootpqarray[1][1] * b_r));

  jju = 8;
  ulist_r[jju] =
    rootpqarray[2][1] *
    (a_r * (-(-rootpqarray[1][1] * b_r)) +
     a_i * (rootpqarray[1][1] * b_i));
  ulist_i[jju] =
    rootpqarray[2][1] *
    (a_r * (rootpqarray[1][1] * b_i) -
     a_i * (-(-rootpqarray[1][1] * b_r)));

  jju = 9;
  ulist_r[jju] =
    -rootpqarray[1][1] *
    (b_r * (-(-rootpqarray[1][1] * b_r)) +
     b_i * (rootpqarray[1][1] * b_i))
    +
    rootpqarray[1][1] *
    (a_r * (rootpqarray[1][1] * a_r) +
     a_i * (-(-rootpqarray[1][1] * a_i)));
  ulist_i[jju] =
    -rootpqarray[1][1] *
    (b_r * (rootpqarray[1][1] * b_i) -
     b_i * (-(-rootpqarray[1][1] * b_r)))
    +
    rootpqarray[1][1] *
    (a_r * (-(-rootpqarray[1][1] * a_i)) -
     a_i * (rootpqarray[1][1] * a_r));
  
  jju = 10;
  ulist_r[jju] =
    -rootpqarray[2][1] *
    (b_r * (rootpqarray[1][1] * a_r) +
     b_i * (-(-rootpqarray[1][1] * a_i)));
  ulist_i[jju] =
    -rootpqarray[2][1] *
    (b_r * (-(-rootpqarray[1][1] * a_i)) -
     b_i * (rootpqarray[1][1] * a_r));

  // copy left side to right side with inversion symmetry VMK 4.4(2)
  // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])

  for (int j = 1; j <= twojmax; j++) {
    jju = idxu_block[j];
    jjup = jju+(j+1)*(j+1)-1;
    mbpar = 1;
    for (int mb = 0; 2*mb <= j; mb++) {
      mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for twojmax = 2
   based on hand-conversion of VMK Table 4.23, 4.24
------------------------------------------------------------------------- */

void SNA::compute_uarray_2J2_byhand(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                         SNADOUBLE z0, SNADOUBLE r)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, b_r, a_i, b_i;
  SNADOUBLE rootpq;
  int j,ma,mb,jju,jjup,mbpar,mapar;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  // j = 0

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  // j = 1

  ulist_r[1] = a_r;
  ulist_i[1] = -a_i;

  ulist_r[2] = -b_r;
  ulist_i[2] = b_i;

  ulist_r[3] = b_r;
  ulist_i[3] = b_i;

  ulist_r[4] = a_r;
  ulist_i[4] = a_i;

  // j = 2

  const double sqrt2 = sqrt(2.0);
  const double invsqrt2 = 1.0/sqrt2;

  // Conj(a)^2

  ulist_r[5] = a_r * a_r - a_i * a_i;
  ulist_i[5] = - a_r * a_i - a_i * a_r;

  // -Sqrt(2).Conj(a).Conj(b)

  ulist_r[6] = -sqrt2 * (a_r * b_r - a_i * b_i);
  ulist_i[6] = sqrt2 * (a_i * b_r + b_i * a_r);

  // Conj(b)^2

  ulist_r[7] = b_r * b_r - b_i * b_i;
  ulist_i[7] = -2.0 * b_r * b_i;

  // Sqrt(2).Conj(a).b

  ulist_r[8] = sqrt2 * (a_r * b_r + a_i * b_i);
  ulist_i[8] = sqrt2 * (a_r * b_i - a_i * b_r);

  // a.Conj(a)-b.Conj(b)

  ulist_r[9] = a_r * a_r + a_i * a_i - (b_r * b_r + b_i * b_i);
  ulist_i[9] = 0.0;
  
  // -Sqrt(2).a.Conj(b)

  ulist_r[10] = -sqrt2 * (a_r * b_r + a_i * b_i);
  ulist_i[10] = -sqrt2 * (a_i * b_r - a_r * b_i);

  // b^2

  ulist_r[11] = b_r * b_r - b_i * b_i;
  ulist_i[11] = 2.0 * b_r * b_i;

  // Sqrt(2).a.b

  ulist_r[12] = sqrt2 * (a_r * b_r - a_i * b_i);
  ulist_i[12] = sqrt2 * (a_i * b_r + b_i * a_r);

  // a^2

  ulist_r[13] = a_r * a_r - a_i * a_i;
  ulist_i[13] = 2.0 * a_r * a_i;

}

/* ----------------------------------------------------------------------
   Compute derivatives of Wigner U-functions for one neighbor
   see comments in compute_uarray()
------------------------------------------------------------------------- */

void SNA::compute_duarray(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                          SNADOUBLE z0, SNADOUBLE r, SNADOUBLE dz0dr,
                          SNADOUBLE wj, SNADOUBLE rcut)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, a_i, b_r, b_i;
  SNADOUBLE da_r[3], da_i[3], db_r[3], db_i[3];
  SNADOUBLE dz0[3], dr0inv[3], dr0invdr;
  SNADOUBLE rootpq;

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

  ulist_r[0] = 1.0;
  dulist_r[0][0] = 0.0;
  dulist_r[0][1] = 0.0;
  dulist_r[0][2] = 0.0;
  ulist_i[0] = 0.0;
  dulist_i[0][0] = 0.0;
  dulist_i[0][1] = 0.0;
  dulist_i[0][2] = 0.0;

  for (int j = 1; j <= twojmax; j++) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j-1];
    for (int mb = 0; 2*mb <= j; mb++) {
      ulist_r[jju] = 0.0;
      dulist_r[jju][0] = 0.0;
      dulist_r[jju][1] = 0.0;
      dulist_r[jju][2] = 0.0;
      ulist_i[jju] = 0.0;
      dulist_i[jju][0] = 0.0;
      dulist_i[jju][1] = 0.0;
      dulist_i[jju][2] = 0.0;

      for (int ma = 0; ma < j; ma++) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] += rootpq *
                               (a_r *  ulist_r[jjup] +
                                a_i *  ulist_i[jjup]);
        ulist_i[jju] += rootpq *
                               (a_r *  ulist_i[jjup] -
                                a_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju][k] +=
            rootpq * (da_r[k] * ulist_r[jjup] +
                      da_i[k] * ulist_i[jjup] +
                      a_r * dulist_r[jjup][k] +
                      a_i * dulist_i[jjup][k]);
          dulist_i[jju][k] +=
            rootpq * (da_r[k] * ulist_i[jjup] -
                      da_i[k] * ulist_r[jjup] +
                      a_r * dulist_i[jjup][k] -
                      a_i * dulist_r[jjup][k]);
        }

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq * (b_r *  ulist_r[jjup] +
                     b_i *  ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq * (b_r *  ulist_i[jjup] -
                     b_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju+1][k] =
            -rootpq * (db_r[k] * ulist_r[jjup] +
                       db_i[k] * ulist_i[jjup] +
                       b_r * dulist_r[jjup][k] +
                       b_i * dulist_i[jjup][k]);
          dulist_i[jju+1][k] =
            -rootpq * (db_r[k] * ulist_i[jjup] -
                       db_i[k] * ulist_r[jjup] +
                       b_r * dulist_i[jjup][k] -
                       b_i * dulist_r[jjup][k]);
        }
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2)
    // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])

    jju = idxu_block[j];
    jjup = jju+(j+1)*(j+1)-1;
    int mbpar = 1;
    for (int mb = 0; 2*mb <= j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjup][k] = dulist_r[jju][k];
            dulist_i[jjup][k] = -dulist_i[jju][k];
          }
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjup][k] = -dulist_r[jju][k];
            dulist_i[jjup][k] = dulist_i[jju][k];
          }
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }

  SNADOUBLE sfac = compute_sfac(r, rcut);
  SNADOUBLE dsfac = compute_dsfac(r, rcut);

  sfac *= wj;
  dsfac *= wj;
  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for (int mb = 0; 2*mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        dulist_r[jju][0] = dsfac * ulist_r[jju] * ux +
                                  sfac * dulist_r[jju][0];
        dulist_i[jju][0] = dsfac * ulist_i[jju] * ux +
                                  sfac * dulist_i[jju][0];
        dulist_r[jju][1] = dsfac * ulist_r[jju] * uy +
                                  sfac * dulist_r[jju][1];
        dulist_i[jju][1] = dsfac * ulist_i[jju] * uy +
                                  sfac * dulist_i[jju][1];
        dulist_r[jju][2] = dsfac * ulist_r[jju] * uz +
                                  sfac * dulist_r[jju][2];
        dulist_i[jju][2] = dsfac * ulist_i[jju] * uz +
                                  sfac * dulist_i[jju][2];
        jju++;
      }
  }
}

/* ----------------------------------------------------------------------
   Compute derivatives of Wigner U-functions for one neighbor
   see comments in compute_uarray()
------------------------------------------------------------------------- */

void SNA::compute_duarray_inlineinversion(SNADOUBLE x, SNADOUBLE y, SNADOUBLE z,
                          SNADOUBLE z0, SNADOUBLE r, SNADOUBLE dz0dr,
                          SNADOUBLE wj, SNADOUBLE rcut)
{
  SNADOUBLE r0inv;
  SNADOUBLE a_r, a_i, b_r, b_i;
  SNADOUBLE da_r[3], da_i[3], db_r[3], db_i[3];
  SNADOUBLE dz0[3], dr0inv[3], dr0invdr;
  SNADOUBLE rootpq;

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

  ulist_r[0] = 1.0;
  dulist_r[0][0] = 0.0;
  dulist_r[0][1] = 0.0;
  dulist_r[0][2] = 0.0;
  ulist_i[0] = 0.0;
  dulist_i[0][0] = 0.0;
  dulist_i[0][1] = 0.0;
  dulist_i[0][2] = 0.0;

  for (int j = 1; j <= twojmax; j++) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j-1];
    int jjui = jju+(j+1)*(j+1)-1;
    int mbpar = 1;
    for (int mb = 0; 2*mb < j; mb++) {
      ulist_r[jju] = 0.0;
      dulist_r[jju][0] = 0.0;
      dulist_r[jju][1] = 0.0;
      dulist_r[jju][2] = 0.0;
      ulist_i[jju] = 0.0;
      dulist_i[jju][0] = 0.0;
      dulist_i[jju][1] = 0.0;
      dulist_i[jju][2] = 0.0;

      int mapar = mbpar;
      for (int ma = 0; ma < j; ma++) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] += rootpq *
                               (a_r *  ulist_r[jjup] +
                                a_i *  ulist_i[jjup]);
        ulist_i[jju] += rootpq *
                               (a_r *  ulist_i[jjup] -
                                a_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju][k] +=
            rootpq * (da_r[k] * ulist_r[jjup] +
                      da_i[k] * ulist_i[jjup] +
                      a_r * dulist_r[jjup][k] +
                      a_i * dulist_i[jjup][k]);
          dulist_i[jju][k] +=
            rootpq * (da_r[k] * ulist_i[jjup] -
                      da_i[k] * ulist_r[jjup] +
                      a_r * dulist_i[jjup][k] -
                      a_i * dulist_r[jjup][k]);
        }

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq * (b_r *  ulist_r[jjup] +
                     b_i *  ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq * (b_r *  ulist_i[jjup] -
                     b_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju+1][k] =
            -rootpq * (db_r[k] * ulist_r[jjup] +
                       db_i[k] * ulist_i[jjup] +
                       b_r * dulist_r[jjup][k] +
                       b_i * dulist_i[jjup][k]);
          dulist_i[jju+1][k] =
            -rootpq * (db_r[k] * ulist_i[jjup] -
                       db_i[k] * ulist_r[jjup] +
                       b_r * dulist_i[jjup][k] -
                       b_i * dulist_r[jjup][k]);
        }

        if (mapar == 1) {
          ulist_r[jjui] = ulist_r[jju];
          ulist_i[jjui] = -ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjui][k] = dulist_r[jju][k];
            dulist_i[jjui][k] = -dulist_i[jju][k];
          }
        } else {
          ulist_r[jjui] = -ulist_r[jju];
          ulist_i[jjui] = ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjui][k] = -dulist_r[jju][k];
            dulist_i[jjui][k] = dulist_i[jju][k];
          }
        }

        jju++;
        jjup++;
        mapar = -mapar;
        jjui--;

      } // ma loop

      // copy the last entry in the column (ma == j)
      
      if (mapar == 1) {
        ulist_r[jjui] = ulist_r[jju];
        ulist_i[jjui] = -ulist_i[jju];
        for (int k = 0; k < 3; k++) {
          dulist_r[jjui][k] = dulist_r[jju][k];
          dulist_i[jjui][k] = -dulist_i[jju][k];
        }
      } else {
        ulist_r[jjui] = -ulist_r[jju];
        ulist_i[jjui] = ulist_i[jju];
        for (int k = 0; k < 3; k++) {
          dulist_r[jjui][k] = -dulist_r[jju][k];
          dulist_i[jjui][k] = dulist_i[jju][k];
        }
      }

      jju++;
      mbpar = -mbpar;
      jjui--;

    } // mb loop

    // For j even, handle middle column

    if (j%2 == 0) {

      int mb = j/2;
      ulist_r[jju] = 0.0;
      dulist_r[jju][0] = 0.0;
      dulist_r[jju][1] = 0.0;
      dulist_r[jju][2] = 0.0;
      ulist_i[jju] = 0.0;
      dulist_i[jju][0] = 0.0;
      dulist_i[jju][1] = 0.0;
      dulist_i[jju][2] = 0.0;
      int mapar = mbpar;
      for(int ma = 0; ma < mb; ma++) {

        // this block copied straight from above
          
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] += rootpq *
                               (a_r *  ulist_r[jjup] +
                                a_i *  ulist_i[jjup]);
        ulist_i[jju] += rootpq *
                               (a_r *  ulist_i[jjup] -
                                a_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju][k] +=
            rootpq * (da_r[k] * ulist_r[jjup] +
                      da_i[k] * ulist_i[jjup] +
                      a_r * dulist_r[jjup][k] +
                      a_i * dulist_i[jjup][k]);
          dulist_i[jju][k] +=
            rootpq * (da_r[k] * ulist_i[jjup] -
                      da_i[k] * ulist_r[jjup] +
                      a_r * dulist_i[jjup][k] -
                      a_i * dulist_r[jjup][k]);
        }

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju+1] =
          -rootpq * (b_r *  ulist_r[jjup] +
                     b_i *  ulist_i[jjup]);
        ulist_i[jju+1] =
          -rootpq * (b_r *  ulist_i[jjup] -
                     b_i *  ulist_r[jjup]);

        for (int k = 0; k < 3; k++) {
          dulist_r[jju+1][k] =
            -rootpq * (db_r[k] * ulist_r[jjup] +
                       db_i[k] * ulist_i[jjup] +
                       b_r * dulist_r[jjup][k] +
                       b_i * dulist_i[jjup][k]);
          dulist_i[jju+1][k] =
            -rootpq * (db_r[k] * ulist_i[jjup] -
                       db_i[k] * ulist_r[jjup] +
                       b_r * dulist_i[jjup][k] -
                       b_i * dulist_r[jjup][k]);
        }

        if (mapar == 1) {
          ulist_r[jjui] = ulist_r[jju];
          ulist_i[jjui] = -ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjui][k] = dulist_r[jju][k];
            dulist_i[jjui][k] = -dulist_i[jju][k];
          }
        } else {
          ulist_r[jjui] = -ulist_r[jju];
          ulist_i[jjui] = ulist_i[jju];
          for (int k = 0; k < 3; k++) {
            dulist_r[jjui][k] = -dulist_r[jju][k];
            dulist_i[jjui][k] = dulist_i[jju][k];
          }
        }

        jju++;
        jjup++;
        mapar = -mapar;
        jjui--;

      }

      int ma = mb;

      rootpq = rootpqarray[j - ma][j - mb];
      ulist_r[jju] += rootpq *
        (a_r *  ulist_r[jjup] +
         a_i *  ulist_i[jjup]);
      ulist_i[jju] += rootpq *
        (a_r *  ulist_i[jjup] -
         a_i *  ulist_r[jjup]);
      
      for (int k = 0; k < 3; k++) {
        dulist_r[jju][k] +=
          rootpq * (da_r[k] * ulist_r[jjup] +
                    da_i[k] * ulist_i[jjup] +
                    a_r * dulist_r[jjup][k] +
                    a_i * dulist_i[jjup][k]);
        dulist_i[jju][k] +=
          rootpq * (da_r[k] * ulist_i[jjup] -
                    da_i[k] * ulist_r[jjup] +
                    a_r * dulist_i[jjup][k] -
                    a_i * dulist_r[jjup][k]);
      }

    } // end if jeven

  } // j loop

  SNADOUBLE sfac = compute_sfac(r, rcut);
  SNADOUBLE dsfac = compute_dsfac(r, rcut);

  sfac *= wj;
  dsfac *= wj;
  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block[j];
    for (int mb = 0; 2*mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        dulist_r[jju][0] = dsfac * ulist_r[jju] * ux +
                                  sfac * dulist_r[jju][0];
        dulist_i[jju][0] = dsfac * ulist_i[jju] * ux +
                                  sfac * dulist_i[jju][0];
        dulist_r[jju][1] = dsfac * ulist_r[jju] * uy +
                                  sfac * dulist_r[jju][1];
        dulist_i[jju][1] = dsfac * ulist_i[jju] * uy +
                                  sfac * dulist_i[jju][1];
        dulist_r[jju][2] = dsfac * ulist_r[jju] * uz +
                                  sfac * dulist_r[jju][2];
        dulist_i[jju][2] = dsfac * ulist_i[jju] * uz +
                                  sfac * dulist_i[jju][2];
        jju++;
      }
  }
}

/* ----------------------------------------------------------------------
   memory usage of arrays
------------------------------------------------------------------------- */

SNADOUBLE SNA::memory_usage()
{
  int jdimpq = twojmax + 2;
  int jdim = twojmax + 1;
  SNADOUBLE bytes;
  bytes = ncoeff * sizeof(SNADOUBLE);                       // coeff

  bytes += jdimpq*jdimpq * sizeof(SNADOUBLE);               // pqarray
  bytes += idxcg_max * sizeof(SNADOUBLE);                   // cglist
  bytes += jdim * jdim * jdim * sizeof(int);                // idxcg_block

  bytes += idxu_max * sizeof(SNADOUBLE) * 2;                // ulist
  bytes += idxu_max * sizeof(SNADOUBLE) * 2;                // ulisttot
  bytes += idxu_max * 3 * sizeof(SNADOUBLE) * 2;            // dulist
  bytes += jdim * sizeof(int);                              // idxu_block

  bytes += idxz_j1j2j_max * 3 * sizeof(int);                // idxz_j1j2j
  bytes += idxz_ma_max * 3 * sizeof(int);                   // idxz_ma
  bytes += idxz_mb_max * 3 * sizeof(int);                   // idxz_mb
  bytes += jdim * jdim * jdim * sizeof(int);                // idxz_block

  bytes += idxu_max * sizeof(SNADOUBLE) * 2;                // ylist
  bytes += idxb_max * 3 * sizeof(int);                      // idxb

  bytes += jdim * jdim * jdim * sizeof(int);                // idxb_block

  return bytes;
}

/* ---------------------------------------------------------------------- */

void SNA::create_twojmax_arrays()
{
  int jdimpq = twojmax + 2;
  memory->create(rootpqarray, jdimpq, jdimpq,
                 "sna:rootpqarray");
  memory->create(cglist, idxcg_max, "sna:cglist");
  memory->create(ulist_r, idxu_max, "sna:ulist");
  memory->create(ulist_i, idxu_max, "sna:ulist");
  memory->create(ulisttot_r, idxu_max, "sna:ulisttot");
  memory->create(ulisttot_i, idxu_max, "sna:ulisttot");
  memory->create(dulist_r, idxu_max, 3, "sna:dulist");
  memory->create(dulist_i, idxu_max, 3, "sna:dulist");
  memory->create(ylist_r, idxu_max, "sna:ylist");
  memory->create(ylist_i, idxu_max, "sna:ylist");
}

/* ---------------------------------------------------------------------- */

void SNA::destroy_twojmax_arrays()
{
  memory->destroy(rootpqarray);
  memory->destroy(cglist);
  memory->destroy(idxcg_block);

  memory->destroy(ulist_r);
  memory->destroy(ulist_i);
  memory->destroy(ulisttot_r);
  memory->destroy(ulisttot_i);
  memory->destroy(dulist_r);
  memory->destroy(dulist_i);
  memory->destroy(idxu_block);

  memory->destroy(idxz_block);

  memory->destroy(ylist_r);
  memory->destroy(ylist_i);

  memory->destroy(idxb_block);
}

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

SNADOUBLE SNA::factorial(int n)
{
  if (n < 0 || n > nmaxfactorial) {
    char str[128];
    //printf("Invalid argument to factorial %d", n);
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
  2.03978820811974e+46,  // nmaxfactorial = 39
};

/* ----------------------------------------------------------------------
   the function delta given by VMK Eq. 8.2(1)
------------------------------------------------------------------------- */

SNADOUBLE SNA::deltacg(int j1, int j2, int j)
{
  SNADOUBLE sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return sqrt(factorial((j1 + j2 - j) / 2) *
              factorial((j1 - j2 + j) / 2) *
              factorial((-j1 + j2 + j) / 2) / sfaccg);
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients using
   the quasi-binomial formula VMK 8.2.1(3)
------------------------------------------------------------------------- */

void SNA::init_clebsch_gordan()
{
  SNADOUBLE sum,dcg,sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  int idxcg_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; m1++) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2++) {

            // -c <= cc <= c

            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if(m < 0 || m > j) {
              cglist[idxcg_count] = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int z = MAX(0, MAX(-(j - j2 + aa2)
                                    / 2, -(j - j1 - bb2) / 2));
                 z <= MIN((j1 + j2 - j) / 2,
                          MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                 z++) {
              ifac = z % 2 ? -1 : 1;
              sum += ifac /
                (factorial(z) *
                 factorial((j1 + j2 - j) / 2 - z) *
                 factorial((j1 - aa2) / 2 - z) *
                 factorial((j2 + bb2) / 2 - z) *
                 factorial((j - j2 + aa2) / 2 + z) *
                 factorial((j - j1 - bb2) / 2 + z));
            }
            
            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = sqrt(factorial((j1 + aa2) / 2) *
                          factorial((j1 - aa2) / 2) *
                          factorial((j2 + bb2) / 2) *
                          factorial((j2 - bb2) / 2) *
                          factorial((j  + cc2) / 2) *
                          factorial((j  - cc2) / 2) *
                          (j + 1));
            
            cglist[idxcg_count] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
}

/* ----------------------------------------------------------------------
   pre-compute table of sqrt[p/m2], p, q = 1,twojmax
   the p = 0, q = 0 entries are allocated and skipped for convenience.
------------------------------------------------------------------------- */

void SNA::init_rootpqarray()
{
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray[p][q] = sqrt(static_cast<SNADOUBLE>(p)/q);
}

/* ---------------------------------------------------------------------- */

int SNA::compute_ncoeff()
{
  int ncount;

  ncount = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2);
           j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) ncount++;

  return ncount;
}

/* ---------------------------------------------------------------------- */

SNADOUBLE SNA::compute_sfac(SNADOUBLE r, SNADOUBLE rcut)
{
  if (switch_flag == 0) return 1.0;
  if (switch_flag == 1) {
    if(r <= rmin0) return 1.0;
    else if(r > rcut) return 0.0;
    else {
      SNADOUBLE rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

SNADOUBLE SNA::compute_dsfac(SNADOUBLE r, SNADOUBLE rcut)
{
  if (switch_flag == 0) return 0.0;
  if (switch_flag == 1) {
    if(r <= rmin0) return 0.0;
    else if(r > rcut) return 0.0;
    else {
      SNADOUBLE rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}

