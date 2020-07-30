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
static const double MY_PI = 3.14159265358979323846; // pi

SNA::SNA(double rfac0_in,
         int twojmax_in,
         double rmin0_in,
         int switch_flag_in,
         int /*bzero_flag_in*/,
         int natoms,
         int nnbor)
{
  wself = 1.0;

  rfac0 = rfac0_in;
  rmin0 = rmin0_in;
  switch_flag = switch_flag_in;
  num_atoms = natoms;
  num_nbor = nnbor;

  twojmax = twojmax_in;

  ncoeff = compute_ncoeff();

  nmax = 0;

  grow_rij(num_nbor);
  create_twojmax_arrays();

  //  build_indexlist();
}

/* ---------------------------------------------------------------------- */

SNA::~SNA()
{
  destroy_twojmax_arrays();
}

void
SNA::build_indexlist()
{

  double* beta = coeffi + 1;
  int jdim = twojmax + 1;
  idxb_block = int_View3D("idxb_block", jdim, jdim, jdim);
  h_idxb_block = create_mirror_view(idxb_block);

  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        if (j < j1)
          continue;
        h_idxb_block(j1, j2, j) = idxb_count;
        idxb_count++;
      }

  int idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {

        // find right beta[jjb] entries
        // multiply and divide by j+1 factors
        // account for multiplicity of 1, 2, or 3
        // this should not be computed here

        SNADOUBLE betaj;
        if (j >= j1) {
          const int jjb = h_idxb_block(j1, j2, j);
          if (j1 == j) {
            if (j2 == j)
              betaj = 3 * beta[jjb];
            else
              betaj = 2 * beta[jjb];
          } else
            betaj = beta[jjb];
        } else if (j >= j2) {
          const int jjb = h_idxb_block(j, j2, j1);
          if (j2 == j)
            betaj = 2 * beta[jjb] * (j1 + 1) / (j + 1.0);
          else
            betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        } else {
          const int jjb = h_idxb_block(j2, j, j1);
          betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        }

        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++) {

            h_idxz(idxz_count, 0) = j1;
            h_idxz(idxz_count, 1) = j2;
            h_idxz(idxz_count, 2) = j;

            int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
            h_idxz(idxz_count, 3) = ma1min;
            h_idxz(idxz_count, 4) = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
            h_idxz(idxz_count, 5) =
              MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

            int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
            h_idxz(idxz_count, 6) = mb1min;
            h_idxz(idxz_count, 7) = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
            h_idxz(idxz_count, 8) =
              MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

            h_idxzbeta(idxz_count) = betaj;

            idxz_count++;
          }
      }
}

/* ---------------------------------------------------------------------- */

void
SNA::init()
{
  init_clebsch_gordan();
  init_rootpqarray();
  build_indexlist();
}

void
SNA::grow_rij(int newnmax)
{
  nmax = num_nbor;

  Kokkos::resize(inside, num_atoms, nmax);
  Kokkos::resize(wj, num_atoms, nmax);
  Kokkos::resize(rcutij, num_atoms, nmax);
  Kokkos::resize(rij, num_atoms, nmax, 3);

  h_rij = create_mirror_view(rij);
  h_wj = create_mirror_view(wj);
  h_rcutij = create_mirror_view(rcutij);
  h_inside = create_mirror_view(inside);
}

/* ----------------------------------------------------------------------
   compute Ui by summing over neighbors j
------------------------------------------------------------------------- */

void
SNA::compute_ui()
{
  zero_uarraytot();
  addself_uarraytot(wself);

  int nTotal = num_atoms * num_nbor;

  int ui_team_size = 4;
  int vector_size = 32;
  int numBlocks = nTotal / ui_team_size + 1;

  team_policy policy_ui(numBlocks, ui_team_size, vector_size);

  const int scratch_level = 0;

  // scratch memory views for SNAcomplex
  using ScratchViewType =
    View<SNAcomplex*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const int PerTeamScratch = (twojmax + 1) * (twojmax / 2 + 1);
  int scratch_size =
    ScratchViewType::shmem_size(ui_team_size * 2 * PerTeamScratch);
  policy_ui = policy_ui.set_scratch_size(scratch_level, PerTeam(scratch_size));

  parallel_for(
    policy_ui, KOKKOS_CLASS_LAMBDA(const member_type team_member) {
      parallel_for(
        TeamThreadRange(team_member, ui_team_size), [&](const int not_using) {
          int iter = team_member.league_rank() * team_member.team_size() +
                     team_member.team_rank();
          if (iter < nTotal) {
            int nbor = iter / num_atoms;
            int natom = iter % num_atoms;

            ScratchViewType buf1(team_member.team_scratch(scratch_level),
                                 ui_team_size * PerTeamScratch);

            ScratchViewType buf2(team_member.team_scratch(scratch_level),
                                 ui_team_size * PerTeamScratch);

            // thread id's and the index for the start of the buffer for
            // each thread.
            const int tid = team_member.team_rank();
            const int tid_index = tid * PerTeamScratch;

            parallel_for(ThreadVectorRange(team_member, PerTeamScratch),
                         [&](const int i) {
                           buf1[tid_index + i] = { 0.0, 0.0 };
                           //               buf2[tid_index + i] = { 0.0, 0.0 };
                         });
            buf2[tid_index] = { 1.0, 0.0 };

            SNADOUBLE x = rij(natom, nbor, 0);
            SNADOUBLE y = rij(natom, nbor, 1);
            SNADOUBLE z = rij(natom, nbor, 2);
            SNADOUBLE rsq = x * x + y * y + z * z;
            SNADOUBLE r = sqrt(rsq);
            SNADOUBLE theta0 = (r - rmin0) * rfac0 * MY_PI /
                             (rcutij(natom, nbor) - rmin0);
            SNADOUBLE z0 = r / tan(theta0);

            //  //This part is compute_uarray
            SNADOUBLE r0inv;
            SNADOUBLE a_r, b_r, a_i, b_i;
            SNADOUBLE rootpq;

            r0inv = 1.0 / sqrt(r * r + z0 * z0);
            a_r = r0inv * z0;
            a_i = -r0inv * z;
            b_r = r0inv * y;
            b_i = -r0inv * x;

            SNADOUBLE rcut = rcutij(natom,nbor);
            SNADOUBLE sfac  = compute_sfac(r, rcut);

            sfac *= wj(natom, nbor);

            Kokkos::single(Kokkos::PerThread(team_member), [=]() {
              Kokkos::atomic_add(&(ulisttot(0, natom).re),
                                 sfac * buf2[tid_index].re);
            });

            for (int j = 1; j <= twojmax; j++) {
              int jju = idxu_block(j);
              int jju0 = jju;
              int jjup = idxu_block(j - 1);

              // Collapse ma and mb loops
              int mb_max = (j + 1) / 2;
              int ma_max = j;
              int m_max = mb_max * ma_max;

              // fill in left side of matrix layer from previous layer
              // jju
              parallel_for(
                ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                  const int mb = m_iter / ma_max;
                  const int ma = m_iter % ma_max;

                  const int buf1_index = tid_index + mb * (ma_max + 1) + ma;
                  const int buf2_index = tid_index + m_iter;

                  rootpq = rootpqarray(j - ma, j - mb);
                  buf1[buf1_index].re += rootpq * (a_r * buf2[buf2_index].re +
                                                   a_i * buf2[buf2_index].im);
                  buf1[buf1_index].im += rootpq * (a_r * buf2[buf2_index].im -
                                                   a_i * buf2[buf2_index].re);
                });

              // jju+1
              parallel_for(
                ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                  const int mb = m_iter / ma_max;
                  const int ma = m_iter % ma_max;

                  const int buf1_index = tid_index + mb * (ma_max + 1) + ma + 1;
                  const int buf2_index = tid_index + m_iter;

                  rootpq = rootpqarray(ma + 1, j - mb);
                  buf1[buf1_index].re += -rootpq * (b_r * buf2[buf2_index].re +
                                                    b_i * buf2[buf2_index].im);
                  buf1[buf1_index].im += -rootpq * (b_r * buf2[buf2_index].im -
                                                    b_i * buf2[buf2_index].re);
                });

              jjup += m_max - 1;
              jju += mb_max * (ma_max + 1);

              // handle middle column using inversion symmetry of previous
              // layer
              if (j % 2 == 0) {
                int mb = j / 2;
                int buf_jju = tid_index + mb_max * (ma_max + 1);
                buf1[buf_jju] = { 0.0, 0.0 };
                int buf2_index = tid_index + m_max - 1;

                parallel_for(ThreadVectorRange(team_member, j),
                             [&](const int ma) {
                               rootpq = ulist_parity(jjup - ma) *
                                        rootpqarray(j - ma, j - mb);
                               buf1[buf_jju + ma].re +=
                                 rootpq * (a_r * buf2[buf2_index - ma].re +
                                           a_i * -buf2[buf2_index - ma].im);
                               buf1[buf_jju + ma].im +=
                                 rootpq * (a_r * -buf2[buf2_index - ma].im -
                                           a_i * buf2[buf2_index - ma].re);
                             });

                parallel_for(ThreadVectorRange(team_member, j),
                             [&](const int ma) {
                               rootpq = ulist_parity(jjup - ma) *
                                        rootpqarray(ma + 1, j - mb);
                               buf1[buf_jju + ma + 1].re +=
                                 -rootpq * (b_r * buf2[buf2_index - ma].re +
                                            b_i * -buf2[buf2_index - ma].im);
                               buf1[buf_jju + ma + 1].im +=
                                 -rootpq * (b_r * -buf2[buf2_index - ma].im -
                                            b_i * buf2[buf2_index - ma].re);
                             });
              } // middle loop end

              // Add uarraytot for left half.
              mb_max = j / 2 + 1;
              ma_max = j + 1;
              m_max = mb_max * ma_max;
              parallel_for(
                ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                  Kokkos::atomic_add(&(ulisttot(jju0 + m_iter, natom).re),
                                     sfac * buf1[tid_index + m_iter].re);
                  Kokkos::atomic_add(&(ulisttot(jju0 + m_iter, natom).im),
                                     sfac * buf1[tid_index + m_iter].im);
                });

              // Add uarraytot for right half.
              jjup = jju0 + (j + 1) * (j + 1) - 1;
              mb_max = (j + 1) / 2;
              m_max = mb_max * ma_max;

              parallel_for(
                ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                  Kokkos::atomic_add(&(ulisttot(jjup - m_iter, natom).re),
                                     sfac * ulist_parity(jju0 + m_iter) *
                                       buf1[tid_index + m_iter].re);
                  Kokkos::atomic_add(&(ulisttot(jjup - m_iter, natom).im),
                                     sfac * ulist_parity(jju0 + m_iter) *
                                       -buf1[tid_index + m_iter].im);
                });

              // Swap buffers and initialize buf1 to zeros
              auto tmp = buf2;
              buf2 = buf1;
              buf1 = tmp;

              parallel_for(ThreadVectorRange(team_member, PerTeamScratch),
                           [&](const int i) {
                             buf1[tid_index + i] = { 0.0, 0.0 };
                           });

            } // twojmax
          }   // iter
        });   // TeamThreadRange
    });       // League

  Kokkos::fence();
}

void
SNA::compute_ui_cpu()
{
  zero_uarraytot();
  addself_uarraytot(wself);

  int nTotal = num_atoms * num_nbor;

  int numThreads,numBlocks;
  numThreads = 1;
  numBlocks = nTotal;

  team_policy policy(numBlocks, numThreads);

  parallel_for(
    policy, KOKKOS_CLASS_LAMBDA(const member_type team_member) {
      int iter = team_member.league_rank() * team_member.team_size() +
                 team_member.team_rank();
      if (iter < nTotal) {
        int nbor = iter / num_atoms;
        int natom = iter % num_atoms;

        SNADOUBLE x = rij(natom, nbor, 0);
        SNADOUBLE y = rij(natom, nbor, 1);
        SNADOUBLE z = rij(natom, nbor, 2);
        SNADOUBLE rsq = x * x + y * y + z * z;
        SNADOUBLE r = sqrt(rsq);
        SNADOUBLE theta0 = (r - rmin0) * rfac0 * MY_PI /
                           (rcutij(natom, nbor) - rmin0);
        SNADOUBLE z0 = r / tan(theta0);

        //  //This part is compute_uarray
        SNADOUBLE r0inv;
        SNADOUBLE a_r, b_r, a_i, b_i;
        SNADOUBLE rootpq;

        r0inv = 1.0 / sqrt(r * r + z0 * z0);
        a_r = r0inv * z0;
        a_i = -r0inv * z;
        b_r = r0inv * y;
        b_i = -r0inv * x;

        double sfac = compute_sfac(
          r, rcutij(natom, nbor));

        sfac *= wj(natom, nbor);
        Kokkos::atomic_add(&(ulisttot(natom, 0).re),
                           sfac * ulist(natom, nbor, 0).re);
        Kokkos::atomic_add(&(ulisttot(natom, 0).im),
                           sfac * ulist(natom, nbor, 0).im);

        for (int j = 1; j <= twojmax; j++) {
          int jju = idxu_block(j);
          int jju0 = jju;
          int jjup = idxu_block(j - 1);

          // Collapse ma and mb loops
          int mb_max = (j + 1) / 2;
          int ma_max = j;
          int m_max = mb_max * ma_max;
          int u_index = 0;

          SNAcomplex ulist_up = ulist(natom, nbor, jjup);

          SNAcomplex ulist_jju1 = ulist(natom, nbor, jju);
          SNAcomplex ulist_jju2 = ulist(natom, nbor, jju + 1);

          // fill in left side of matrix layer from previous layer
          for (int m_iter = 0; m_iter < m_max; ++m_iter) {
            const int mb = m_iter / ma_max;
            const int ma = m_iter % ma_max;

            rootpq = rootpqarray(j - ma, j - mb);

            // jju
            ulist_jju1.re += rootpq * (a_r * ulist_up.re + a_i * ulist_up.im);
            ulist_jju1.im += rootpq * (a_r * ulist_up.im - a_i * ulist_up.re);

            // jju+1
            rootpq = rootpqarray(ma + 1, j - mb);
            ulist_jju2.re += -rootpq * (b_r * ulist_up.re + b_i * ulist_up.im);
            ulist_jju2.im += -rootpq * (b_r * ulist_up.im - b_i * ulist_up.re);

            ulist(natom, nbor, jju) = ulist_jju1;
            ulist(natom, nbor, jju + 1) = ulist_jju2;

            ulist_jju1 = ulist_jju2;

            jju++;
            u_index++;
            jjup++;
            if (ma == ma_max - 1) {
              jju++;
              ulist_jju1 = ulist(natom, nbor, jju);
            }

            ulist_up = ulist(natom, nbor, jjup);
            ulist_jju2 = ulist(natom, nbor, jju + 1);
          }

          // handle middle column using inversion symmetry of previous layer

          if (j % 2 == 0) {
            int mb = j / 2;
            jjup--;
            ulist(natom, nbor, jju) = { 0.0, 0.0 };
            for (int ma = 0; ma < j; ma++) {
              rootpq = ulist_parity(jjup) * rootpqarray(j - ma, j - mb);
              ulist(natom, nbor, jju).re +=
                rootpq * (a_r * ulist(natom, nbor, jjup).re +
                          a_i * -ulist(natom, nbor, jjup).im);
              ulist(natom, nbor, jju).im +=
                rootpq * (a_r * -ulist(natom, nbor, jjup).im -
                          a_i * ulist(natom, nbor, jjup).re);

              rootpq = ulist_parity(jjup) * rootpqarray(ma + 1, j - mb);
              ulist(natom, nbor, jju + 1).re +=
                -rootpq * (b_r * ulist(natom, nbor, jjup).re +
                           b_i * -ulist(natom, nbor, jjup).im);
              ulist(natom, nbor, jju + 1).im +=
                -rootpq * (b_r * -ulist(natom, nbor, jjup).im -
                           b_i * ulist(natom, nbor, jjup).re);

              jju++;
              jjup--;
              u_index++;
            }
            jju++;
          }
          // This part is add_uarraytot
          jju = jju0;
          mb_max = j / 2 + 1;
          ma_max = j + 1;
          m_max = mb_max * ma_max;
          for (int m_iter = 0; m_iter < m_max; ++m_iter) {
            Kokkos::atomic_add(&(ulisttot(natom, jju).re),
                               sfac * ulist(natom, nbor, jju).re);
            Kokkos::atomic_add(&(ulisttot(natom, jju).im),
                               sfac * ulist(natom, nbor, jju).im);
            jju++;
          }

          // This part is add_uarraytot
          jju = jju0;
          jjup = jju0 + (j + 1) * (j + 1) - 1;
          mb_max = (j + 1) / 2;
          ma_max = j + 1;
          m_max = mb_max * ma_max;

          for (int m_iter = 0; m_iter < m_max; ++m_iter) {
            Kokkos::atomic_add(&(ulisttot(natom, jjup).re),
                               sfac * ulist_parity(jju) *
                                 ulist(natom, nbor, jju).re);
            Kokkos::atomic_add(&(ulisttot(natom, jjup).im),
                               sfac * ulist_parity(jju) *
                                 -ulist(natom, nbor, jju).im);
            jju++;
            jjup--;
          }
        } // twojmax
      }
    }); // natom && nbor

  Kokkos::fence();
}

void
SNA::compute_yi()
{
  team_policy policy(num_atoms, Kokkos::AUTO);
  parallel_for(
    policy, KOKKOS_CLASS_LAMBDA(const member_type& thread) {
      const int natom = thread.league_rank();
      parallel_for(Kokkos::TeamThreadRange(thread, idxdu_max),
                   [&](const int j) {
                     ylist(natom, j) = { 0.0, 0.0 };
                   });
    });

  parallel_for(
    num_atoms * idxz_max, KOKKOS_CLASS_LAMBDA(const int iter) {
      const int natom = iter / idxz_max;
      const int jjz = iter % idxz_max;

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

      const SNADOUBLE* cgblock = cglist.data() + idxcg_block(j1, j2, j);
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

        for (int ia = 0; ia < na; ia++) {
          suma1_r += cgblock[icga] * (ulisttot(jju1 + ma1, natom).re *
                                        ulisttot(jju2 + ma2, natom).re -
                                      ulisttot(jju1 + ma1, natom).im *
                                        ulisttot(jju2 + ma2, natom).im);
          suma1_i += cgblock[icga] * (ulisttot(jju1 + ma1, natom).re *
                                        ulisttot(jju2 + ma2, natom).im +
                                      ulisttot(jju1 + ma1, natom).im *
                                        ulisttot(jju2 + ma2, natom).re);
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

      //    Atomic updates to ulist due to parallelization over the bispectrum coeffients.
      Kokkos::atomic_add(&(ylist(natom, jjdu).re), betaj * ztmp_r);
      Kokkos::atomic_add(&(ylist(natom, jjdu).im), betaj * ztmp_i);
    }); // end jjz and atom loop
  Kokkos::fence();
}

void
SNA::compute_yi_cpu()
{
  team_policy policy(num_atoms, Kokkos::AUTO);

  parallel_for(
    policy, KOKKOS_CLASS_LAMBDA(const member_type& thread) {
      const int natom = thread.league_rank();
      parallel_for(Kokkos::TeamThreadRange(thread, idxdu_max),
                   [&](const int j) {
                     ylist(natom, j) = { 0.0, 0.0 };
                   });
    });

  parallel_for(
    num_atoms * idxz_max, KOKKOS_CLASS_LAMBDA(const int iter) {
      const int natom = iter / idxz_max;
      const int jjz = iter % idxz_max;

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

      const SNADOUBLE* cgblock = cglist.data() + idxcg_block(j1, j2, j);
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

        for (int ia = 0; ia < na; ia++) {
          suma1_r += cgblock[icga] * (ulisttot(natom, jju1 + ma1).re *
                                        ulisttot(natom, jju2 + ma2).re -
                                      ulisttot(natom, jju1 + ma1).im *
                                        ulisttot(natom, jju2 + ma2).im);
          suma1_i += cgblock[icga] * (ulisttot(natom, jju1 + ma1).re *
                                        ulisttot(natom, jju2 + ma2).im +
                                      ulisttot(natom, jju1 + ma1).im *
                                        ulisttot(natom, jju2 + ma2).re);
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

      //    Atomic updates to ulist due to parallelization over the bispectrum coeffients.
      Kokkos::atomic_add(&(ylist(natom, jjdu).re), betaj * ztmp_r);
      Kokkos::atomic_add(&(ylist(natom, jjdu).im), betaj * ztmp_i);
    }); // end jjz and atom loop
  Kokkos::fence();
}

void
SNA::compute_duarray()
{
  parallel_for(
    num_atoms * num_nbor, KOKKOS_CLASS_LAMBDA(const int iter) {
      int nbor = iter / num_atoms;
      int natom = iter % num_atoms;
      SNADOUBLE rsq, r, x, y, z, z0, theta0, cs, sn;
      SNADOUBLE dz0dr;

      x = rij(natom, nbor, 0);
      y = rij(natom, nbor, 1);
      z = rij(natom, nbor, 2);
      rsq = x * x + y * y + z * z;
      r = sqrt(rsq);
      SNADOUBLE rscale0 =
        rfac0 * MY_PI / (rcutij(natom, nbor) - rmin0);
      theta0 = (r - rmin0) * rscale0;
      cs = cos(theta0);
      sn = sin(theta0);
      z0 = r * cs / sn;
      dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;
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

      for (int k = 0; k < 3; ++k)
        dulist(natom, nbor, 0, k) = { 0.0, 0.0 };

      for (int j = 1; j <= twojmax; j++) {
        int jjup = idxu_block(j - 1);
        int jju = idxu_block(j);
        int jjdup = idxdu_block(j - 1);
        int jjdu = idxdu_block(j);

        //      printf("twojamx = %d\t jjdup = %d\t jjup = %d\n", j, jjdup,
        //      jjup);

        for (int mb = 0; 2 * mb < j; mb++) {
          for (int k = 0; k < 3; ++k)
            dulist(natom, nbor, jjdu, k) = { 0.0, 0.0 };

          for (int ma = 0; ma < j; ma++) {
            rootpq = rootpqarray(j - ma, j - mb);

            for (int k = 0; k < 3; k++) {
              dulist(natom, nbor, jjdu, k).re +=
                rootpq * (da_r[k] * ulist(natom, nbor, jjup).re +
                          da_i[k] * ulist(natom, nbor, jjup).im +
                          a_r * dulist(natom, nbor, jjdup, k).re +
                          a_i * dulist(natom, nbor, jjdup, k).im);
              dulist(natom, nbor, jjdu, k).im +=
                rootpq * (da_r[k] * ulist(natom, nbor, jjup).im -
                          da_i[k] * ulist(natom, nbor, jjup).re +
                          a_r * dulist(natom, nbor, jjdup, k).im -
                          a_i * dulist(natom, nbor, jjdup, k).re);
            }

            rootpq = rootpqarray(ma + 1, j - mb);

            for (int k = 0; k < 3; k++) {
              dulist(natom, nbor, jjdu + 1, k).re =
                -rootpq * (db_r[k] * ulist(natom, nbor, jjup).re +
                           db_i[k] * ulist(natom, nbor, jjup).im +
                           b_r * dulist(natom, nbor, jjdup, k).re +
                           b_i * dulist(natom, nbor, jjdup, k).im);
              dulist(natom, nbor, jjdu + 1, k).im =
                -rootpq * (db_r[k] * ulist(natom, nbor, jjup).im -
                           db_i[k] * ulist(natom, nbor, jjup).re +
                           b_r * dulist(natom, nbor, jjdup, k).im -
                           b_i * dulist(natom, nbor, jjdup, k).re);
            }

            jju++;
            jjup++;
            jjdu++;
            jjdup++;
          }
          jju++;
          jjdu++;
        }

        // handle middle column using inversion symmetry of previous layer

        if (j % 2 == 0) {
          int mb = j / 2;
          jjup--;
          jjdup--;
          for (int k = 0; k < 3; ++k)
            dulist(natom, nbor, jjdu, k) = { 0.0, 0.0 };

          for (int ma = 0; ma < j; ma++) {
            rootpq = rootpqparityarray(j - ma, j - mb);
            for (int k = 0; k < 3; k++) {
              dulist(natom, nbor, jjdu, k).re +=
                rootpq * (da_r[k] * ulist(natom, nbor, jjup).re +
                          da_i[k] * -ulist(natom, nbor, jjup).im +
                          a_r * dulist(natom, nbor, jjdup, k).re +
                          a_i * -dulist(natom, nbor, jjdup, k).im);
              dulist(natom, nbor, jjdu, k).im +=
                rootpq * (da_r[k] * -ulist(natom, nbor, jjup).im -
                          da_i[k] * ulist(natom, nbor, jjup).re +
                          a_r * -dulist(natom, nbor, jjdup, k).im -
                          a_i * dulist(natom, nbor, jjdup, k).re);
            }

            rootpq = -rootpqparityarray(ma + 1, j - mb);

            for (int k = 0; k < 3; k++) {
              dulist(natom, nbor, jjdu + 1, k).re =
                -rootpq * (db_r[k] * ulist(natom, nbor, jjup).re +
                           db_i[k] * -ulist(natom, nbor, jjup).im +
                           b_r * dulist(natom, nbor, jjdup, k).re +
                           b_i * -dulist(natom, nbor, jjdup, k).im);
              dulist(natom, nbor, jjdu + 1, k).im =
                -rootpq * (db_r[k] * -ulist(natom, nbor, jjup).im -
                           db_i[k] * ulist(natom, nbor, jjup).re +
                           b_r * -dulist(natom, nbor, jjdup, k).im -
                           b_i * dulist(natom, nbor, jjdup, k).re);
            }
            jju++;
            jjup--;
            jjdu++;
            jjdup--;
          }
          jju++;
          jjdu++;
        } // middle column
      }   // twojmax

      double rcut = rcutij(natom, nbor);
      SNADOUBLE sfac = compute_sfac(
        r, rcutij(natom, nbor));
      SNADOUBLE dsfac = compute_dsfac(r, rcutij(natom,nbor));

      sfac *= wj(natom, nbor);
      dsfac *= wj(natom, nbor);
      for (int j = 0; j <= twojmax; j++) {
        int jju = idxu_block(j);
        int jjdu = idxdu_block(j);

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
      }
    });

  Kokkos::fence();
}

void
SNA::compute_deidrj()
{
  parallel_for(
    num_atoms * num_nbor, KOKKOS_CLASS_LAMBDA(const int iter) {
      int nbor = iter / num_atoms;
      int natom = iter % num_atoms;

      for (int k = 0; k < 3; ++k)
        dedr(natom, nbor, k) = 0.0;

      for (int j = 0; j <= twojmax; j++) {
        int jjdu = idxdu_block(j);

        for (int mb = 0; 2 * mb < j; mb++)
          for (int ma = 0; ma <= j; ma++) {
            for (int k = 0; k < 3; k++)
              dedr(natom, nbor, k) +=
                dulist(natom, nbor, jjdu, k).re *
                  ylist(natom, jjdu).re +
                dulist(natom, nbor, jjdu, k).im * ylist(natom, jjdu).im;
            jjdu++;
          } // end loop over ma mb

        // For j even, handle middle column

        if (j % 2 == 0) {
          int mb = j / 2;
          for (int ma = 0; ma < mb; ma++) {
            for (int k = 0; k < 3; k++)
              dedr(natom, nbor, k) +=
                dulist(natom, nbor, jjdu, k).re *
                  ylist(natom, jjdu).re +
                dulist(natom, nbor, jjdu, k).im * ylist(natom, jjdu).im;
            jjdu++;
          }

          for (int k = 0; k < 3; k++)
            dedr(natom, nbor, k) +=
              (dulist(natom, nbor, jjdu, k).re * ylist(natom, jjdu).re +
               dulist(natom, nbor, jjdu, k).im *
                 ylist(natom, jjdu).im) *
              0.5;
          jjdu++;
        } // end if jeven

      } // end loop over j

      for (int k = 0; k < 3; k++)
        dedr(natom, nbor, k) *= 2.0;
    }); // nbor and atom loop collapsed
  Kokkos::fence();
}

#if defined(KOKKOS_ENABLE_CUDA)
void
SNA::compute_fused_deidrj()
{
  int nTotal = num_atoms * num_nbor;

  int du_team_size = 4;
  int vector_size = 32;
  int numBlocks = nTotal / du_team_size + 1;
  team_policy policy_du(numBlocks, du_team_size, vector_size);

  const int scratch_level = 0;

  // scratch memory views for SNAcomplex
  using ScratchViewType =
    View<SNAcomplex*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const int PerTeamScratch = (twojmax + 1) * (twojmax / 2 + 1);
  int scratch_size =
    ScratchViewType::shmem_size(du_team_size * PerTeamScratch * 4);
  policy_du = policy_du.set_scratch_size(scratch_level, PerTeam(scratch_size));

  parallel_for(
    policy_du, KOKKOS_CLASS_LAMBDA(const member_type team_member) {
      parallel_for(
        TeamThreadRange(team_member, du_team_size), [&](const int not_using) {
          int iter = team_member.league_rank() * team_member.team_size() +
                     team_member.team_rank();
          if (iter < nTotal) {
            int nbor = iter / num_atoms;
            int natom = iter % num_atoms;

            ScratchViewType buf1(team_member.team_scratch(scratch_level),
                                 du_team_size * PerTeamScratch);

            ScratchViewType buf2(team_member.team_scratch(scratch_level),
                                 du_team_size * PerTeamScratch);

            ScratchViewType ubuf1(team_member.team_scratch(scratch_level),
                                  du_team_size * PerTeamScratch);

            ScratchViewType ubuf2(team_member.team_scratch(scratch_level),
                                  du_team_size * PerTeamScratch);

            // thread id's and the index for the start of the buffer for
            // each thread.
            const int tid = team_member.team_rank();
            const int tid_index = tid * PerTeamScratch;

            parallel_for(ThreadVectorRange(team_member, PerTeamScratch),
                         [&](const int i) {
                           buf1[tid_index + i] = { 0., 0. };
                           ubuf1[tid_index + i] = { 0., 0. };
                         });

            buf2[tid_index] = { 0., 0. };

            SNADOUBLE rsq, r, x, y, z, z0, theta0, cs, sn;
            SNADOUBLE dz0dr;

            x = rij(natom, nbor, 0);
            y = rij(natom, nbor, 1);
            z = rij(natom, nbor, 2);
            rsq = x * x + y * y + z * z;
            r = sqrt(rsq);
            SNADOUBLE rscale0 =
              rfac0 * MY_PI / (rcutij(natom, nbor) - rmin0);
            theta0 = (r - rmin0) * rscale0;
            cs = cos(theta0);
            sn = sin(theta0);
            z0 = r * cs / sn;
            dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;
            SNADOUBLE r0inv;
            SNADOUBLE a_r, a_i, b_r, b_i;
            SNADOUBLE da_r[3], da_i[3], db_r[3], db_i[3];
            SNADOUBLE dz0[3], dr0inv[3], dr0invdr;

            SNADOUBLE rinv = 1.0 / r;

            SNADOUBLE u[3] = { x * rinv, y * rinv, z * rinv };

            r0inv = 1.0 / sqrt(r * r + z0 * z0);
            a_r = z0 * r0inv;
            a_i = -z * r0inv;
            b_r = y * r0inv;
            b_i = -x * r0inv;
            SNADOUBLE sfac = compute_sfac(
              r, rcutij(natom, nbor));
            SNADOUBLE dsfac = compute_dsfac(
              r, rcutij(natom, nbor));

            sfac *= wj(natom, nbor);
            dsfac *= wj(natom, nbor);

            dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

            dr0inv[0] = dr0invdr * u[0];
            dr0inv[1] = dr0invdr * u[1];
            dr0inv[2] = dr0invdr * u[2];

            dz0[0] = dz0dr * u[0];
            dz0[1] = dz0dr * u[1];
            dz0[2] = dz0dr * u[2];

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
            SNADOUBLE dedrTmp;

            // compute ulist values and store it in buf3 for later use
            for (int k = 0; k < 3; ++k) {
              dedr(natom, nbor, k) = 0.0;

              buf2[tid_index] = { 0., 0. };
              ubuf2[tid_index] = { 1.0, 0.0 };
              for (int j = 1; j <= twojmax; j++) {

                int jju = idxu_block(j);
                int jjup = idxu_block(j - 1);
                int jjdu = idxdu_block(j);

                // Collapse ma and mb loops
                int mb_max = (j + 1) / 2;
                int ma_max = j;
                int m_max = mb_max * ma_max;

                // 0 initialization for all ma == 0
                parallel_for(ThreadVectorRange(team_member, m_max),
                             [&](const int m_iter) {
                               // Initialize all jjdu to zeros'
                               const int mb = m_iter / ma_max;
                               const int ma = m_iter % ma_max;
                               int buf1_index = tid_index + (mb * (ma_max + 1));
                               if (ma == 0)
                                 buf1[buf1_index] = { 0., 0. };
                             });

                // fill in left side of matrix layer from previous layer

                // jju+1
                parallel_for(
                  ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                    const int mb = m_iter / ma_max;
                    const int ma = m_iter % ma_max;

                    int buf1_index = tid_index + (mb * (ma_max + 1) + ma) + 1;
                    int buf2_index = tid_index + m_iter;

                    SNADOUBLE rootpq = rootpqarray(ma + 1, j - mb);
                    ubuf1[buf1_index].re +=
                      -rootpq *
                      (b_r * ubuf2[buf2_index].re + b_i * ubuf2[buf2_index].im);
                    ubuf1[buf1_index].im +=
                      -rootpq *
                      (b_r * ubuf2[buf2_index].im - b_i * ubuf2[buf2_index].re);

                    // Assign to jju+1
                    buf1[buf1_index].re =
                      -rootpq *
                      (db_r[k] * ubuf2[buf2_index].re +
                       db_i[k] * ubuf2[buf2_index].im +
                       b_r * buf2[buf2_index].re + b_i * buf2[buf2_index].im);
                    buf1[buf1_index].im =
                      -rootpq *
                      (db_r[k] * ubuf2[buf2_index].im -
                       db_i[k] * ubuf2[buf2_index].re +
                       b_r * buf2[buf2_index].im - b_i * buf2[buf2_index].re);
                  });

                // jju
                parallel_for(
                  ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                    const int mb = m_iter / ma_max;
                    const int ma = m_iter % ma_max;

                    const int buf1_index = tid_index + mb * (ma_max + 1) + ma;
                    const int buf2_index = tid_index + m_iter;

                    SNADOUBLE rootpq = rootpqarray(j - ma, j - mb);
                    ubuf1[buf1_index].re +=
                      rootpq *
                      (a_r * ubuf2[buf2_index].re + a_i * ubuf2[buf2_index].im);
                    ubuf1[buf1_index].im +=
                      rootpq *
                      (a_r * ubuf2[buf2_index].im - a_i * ubuf2[buf2_index].re);

                    buf1[buf1_index].re +=
                      rootpq *
                      (da_r[k] * ubuf2[buf2_index].re +
                       da_i[k] * ubuf2[buf2_index].im +
                       a_r * buf2[buf2_index].re + a_i * buf2[buf2_index].im);
                    buf1[buf1_index].im +=
                      rootpq *
                      (da_r[k] * ubuf2[buf2_index].im -
                       da_i[k] * ubuf2[buf2_index].re +
                       a_r * buf2[buf2_index].im - a_i * buf2[buf2_index].re);
                  });

                // handle middle column using inversion symmetry of previous
                // layer
                if (j % 2 == 0) {
                  jjup += m_max - 1;
                  jju += mb_max * (ma_max + 1);

                  int mb = j / 2;
                  int ubuf2_index = tid_index + m_max - 1;
                  int buf1_index0 = tid_index + (mb_max * (ma_max + 1));
                  buf1[buf1_index0] = { 0., 0. };
                  ubuf1[buf1_index0] = { 0.0, 0.0 };

                  parallel_for(
                    ThreadVectorRange(team_member, j), [&](const int ma) {
                      int buf1_index = buf1_index0 + ma;
                      int buf2_index = tid_index + (m_max - ma - 1);
                      SNADOUBLE rootpq = ulist_parity(jjup - ma) *
                                       rootpqarray(j - ma, j - mb);

                      ubuf1[buf1_index].re +=
                        rootpq * (a_r * ubuf2[ubuf2_index - ma].re +
                                  a_i * -ubuf2[ubuf2_index - ma].im);
                      ubuf1[buf1_index].im +=
                        rootpq * (a_r * -ubuf2[ubuf2_index - ma].im -
                                  a_i * ubuf2[ubuf2_index - ma].re);

                      rootpq = -rootpqparityarray(ma + 1, j - mb);
                      buf1[buf1_index + 1].re =
                        -rootpq * (db_r[k] * ubuf2[buf2_index].re +
                                   db_i[k] * -ubuf2[buf2_index].im +
                                   b_r * buf2[buf2_index].re +
                                   b_i * -buf2[buf2_index].im);
                      buf1[buf1_index + 1].im =
                        -rootpq * (db_r[k] * -ubuf2[buf2_index].im -
                                   db_i[k] * ubuf2[buf2_index].re +
                                   b_r * -buf2[buf2_index].im -
                                   b_i * buf2[buf2_index].re);
                    });

                  parallel_for(
                    ThreadVectorRange(team_member, j), [&](const int ma) {
                      int buf1_index = buf1_index0 + ma;
                      int buf2_index = tid_index + (m_max - ma - 1);

                      SNADOUBLE rootpq = ulist_parity(jjup - ma) *
                                       rootpqarray(ma + 1, j - mb);
                      ubuf1[buf1_index + 1].re +=
                        -rootpq * (b_r * ubuf2[ubuf2_index - ma].re +
                                   b_i * -ubuf2[ubuf2_index - ma].im);
                      ubuf1[buf1_index + 1].im +=
                        -rootpq * (b_r * -ubuf2[ubuf2_index - ma].im -
                                   b_i * ubuf2[ubuf2_index - ma].re);

                      rootpq = rootpqparityarray(j - ma, j - mb);
                      buf1[buf1_index].re +=
                        rootpq * (da_r[k] * ubuf2[buf2_index].re +
                                  da_i[k] * -ubuf2[buf2_index].im +
                                  a_r * buf2[buf2_index].re +
                                  a_i * -buf2[buf2_index].im);
                      buf1[buf1_index].im +=
                        rootpq * (da_r[k] * -ubuf2[buf2_index].im -
                                  da_i[k] * ubuf2[buf2_index].re +
                                  a_r * -buf2[buf2_index].im -
                                  a_i * buf2[buf2_index].re);
                    });
                } // middle loop end

                // Update dedr with the first element of ulist and ylist only
                // once with j==1 .
                if (j == 1) {

                  Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                    dedr(natom, nbor, k) +=
                      (dsfac * ubuf2[tid_index].re * u[k] *
                       ylist(natom, 0).re) *
                      0.5;
                  });
                }

                // swap dulist buffers
                auto tmpu = buf2;
                buf2 = buf1;
                auto tmpdu = ubuf2;
                ubuf2 = ubuf1;
                buf1 = tmpdu;
                ubuf1 = tmpu;

                mb_max = j / 2 + 1;
                ma_max = j + 1;
                m_max = ma_max * mb_max;

                parallel_for(
                  ThreadVectorRange(team_member, m_max), [&](const int m_iter) {
                    const int ubuf_index = tid_index + m_iter;
                    const int buf_index = tid_index + m_iter;

                    SNAcomplex tmp1(0.0, 0.0);

                    buf1[buf_index].re = dsfac * ubuf2[ubuf_index].re * u[k] +
                                         sfac * buf2[buf_index].re;
                    buf1[buf_index].im = dsfac * ubuf2[ubuf_index].im * u[k] +
                                         sfac * buf2[buf_index].im;
                  });

                mb_max = (j + 1) / 2;
                m_max = ma_max * mb_max;

                parallel_reduce(
                  ThreadVectorRange(team_member, m_max),
                  [&](const int m_iter, SNADOUBLE& updateDedr) {
                    const int jjdu_index = jjdu + m_iter;
                    int buf_index = tid_index + m_iter;

                    updateDedr +=
                      buf1[buf_index].re * ylist(natom, jjdu_index).re +
                      buf1[buf_index].im * ylist(natom, jjdu_index).im;
                  },
                  dedrTmp); // end loop over ma mb

                dedr(natom, nbor, k) += dedrTmp;

                jjdu += m_max;

                // For j even, handle middle column
                if (j % 2 == 0) {
                  int mb = j / 2;
                  dedrTmp = 0.;
                  parallel_reduce(
                    ThreadVectorRange(team_member, mb),
                    [&](const int ma, SNADOUBLE& updateDedr) {
                      int jjdu_index = jjdu + ma;
                      int buf_index = tid_index + m_max + ma;
                      updateDedr +=
                        buf1[buf_index].re * ylist(natom, jjdu_index).re +
                        buf1[buf_index].im * ylist(natom, jjdu_index).im;
                    },
                    dedrTmp);

                  jjdu += mb;
                  int buf_index = tid_index + m_max + mb;

                  Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                    dedrTmp +=
                      (buf1[buf_index].re * ylist(natom, jjdu).re +
                       buf1[buf_index].im * ylist(natom, jjdu).im) *
                      0.5;

                    dedr(natom, nbor, k) += dedrTmp;
                  });
                } // end if jeven

                parallel_for(ThreadVectorRange(team_member, PerTeamScratch),
                             [&](const int i) {
                               buf1[tid_index + i] = { 0., 0. };
                               ubuf1[tid_index + i] = { 0.0, 0.0 };
                             });
              } // twojmax

              Kokkos::single(Kokkos::PerThread(team_member),
                             [&]() { dedr(natom, nbor, k) *= 2.0; });
            } // k
          }   // iter
        });   // TeamThreadRange
    });       // League

  Kokkos::fence();
}
#endif

/* ----------------------------------------------------------------------
   calculate derivative of Ui w.r.t. atom j
------------------------------------------------------------------------- */

void
SNA::compute_duidrj()
{
  compute_duarray();
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void
SNA::zero_uarraytot()
{
#if defined(KOKKOS_ENABLE_CUDA)
  parallel_for(
    num_atoms * idxu_max, KOKKOS_CLASS_LAMBDA(const int iter) {
      int natom = iter / idxu_max;
      int jju = iter % idxu_max;
      ;
      ulisttot(jju, natom) = { 0.0, 0.0 };
    });

#else
  auto nTotal = num_atoms * num_nbor;

  parallel_for(
    num_atoms * idxu_max, KOKKOS_CLASS_LAMBDA(const int iter) {
      int natom = iter / idxu_max;
      int jju = iter % idxu_max;
      ;
      ulisttot(natom, jju) = { 0.0, 0.0 };
    });

  parallel_for(
    nTotal, KOKKOS_CLASS_LAMBDA(const int iter) {
      int nbor = iter / num_atoms;
      int natom = iter % num_atoms;

      ulist(natom, nbor, 0) = { 1.0, 0.0 };
      for (int jju = 1; jju < idxu_max; ++jju)
        ulist(natom, nbor, jju) = { 0.0, 0.0 };
    });
#endif
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void
SNA::addself_uarraytot(double wself_in)
{

  const int jdim = twojmax+1;
#if defined(KOKKOS_ENABLE_CUDA)
  parallel_for(
    num_atoms * jdim, KOKKOS_CLASS_LAMBDA(const int iter) {
      int natom = iter / jdim;
      int j = iter % jdim;
      ;
      int jju = idxu_block(j);
      for (int ma = 0; ma <= j; ma++) {
        ulisttot(jju, natom) = { wself_in, 0.0 };
        jju += j + 2;
      }
    });
#else
  parallel_for(
    num_atoms * jdim, KOKKOS_CLASS_LAMBDA(const int iter) {
      int natom = iter / jdim;
      int j = iter % jdim;
      ;
      int jju = idxu_block(j);
      for (int ma = 0; ma <= j; ma++) {
        ulisttot(natom, jju) = { wself_in, 0.0 };
        jju += j + 2;
      }
    });
#endif
}

/* ----------------------------------------------------------------------
   add Wigner U-functions for one neighbor to the total
------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
void
SNA::add_uarraytot(int natom, int nbor, double r, double wj, double rcut)
{
  double sfac = compute_sfac(r, rcut);
  sfac *= wj;

  for (int j = 0; j <= twojmax; j++) {
    int jju = idxu_block(j);
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        ulisttot(natom, jju).re += sfac * ulist(natom, nbor, jju).re;
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
                    double x,
                    double y,
                    double z,
                    double z0,
                    double r)
{
  // compute Cayley-Klein parameters for unit quaternion

  //  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  //  a_r = r0inv * z0;
  //  a_i = -r0inv * z;
  //  b_r = r0inv * y;
  //  b_i = -r0inv * x;

  // VMK Section 4.8.2

  ulist(natom, nbor, 0) = { 1.0, 0.0 };

  for (int j = 1; j <= twojmax; j++) {
    //    int jju = idxu_block(j);
    //    int jjup = idxu_block(j-1);
    //
    //    // fill in left side of matrix layer from previous layer
    //
    //    for (int mb = 0; 2*mb <= j; mb++) {
    //      ulist(natom,nbor,jju) = {0.0,0.0};
    //
    //      for (int ma = 0; ma < j; ma++) {
    //        rootpq = rootpqarray(j - ma,j - mb);
    //        ulist(natom,nbor,jju).re +=
    //          rootpq *
    //          (a_r * ulist(natom,nbor,jjup).re +
    //           a_i * ulist(natom,nbor,jjup).im);
    //        ulist(natom,nbor,jju).im +=
    //          rootpq *
    //          (a_r * ulist(natom,nbor,jjup).im -
    //           a_i * ulist(natom,nbor,jjup).re);
    //
    //        rootpq = rootpqarray(ma + 1,j - mb);
    //        ulist(natom,nbor,jju+1).re =
    //          -rootpq *
    //          (b_r * ulist(natom,nbor,jjup).re +
    //           b_i * ulist(natom,nbor,jjup).im);
    //        ulist(natom,nbor,jju+1).im =
    //          -rootpq *
    //          (b_r * ulist(natom,nbor,jjup).im -
    //           b_i * ulist(natom,nbor,jjup).re);
    //        jju++;
    //        jjup++;
    //      }
    //      jju++;
    //    }
    //
    //    // copy left side to right side with inversion symmetry VMK 4.4(2)
    //    // u[ma-j][mb-j] = (-1)^(ma-mb)*Conj([u[ma][mb])
    //
    //    jju = idxu_block(j);
    //    jjup = jju+(j+1)*(j+1)-1;
    //    int mbpar = 1;
    //    for (int mb = 0; 2*mb <= j; mb++) {
    //      int mapar = mbpar;
    //      for (int ma = 0; ma <= j; ma++) {
    //        if (mapar == 1) {
    //          ulist(natom,nbor,jjup).re = ulist(natom,nbor,jju).re;
    //          ulist(natom,nbor,jjup).im = -ulist(natom,nbor,jju).im;
    //        } else {
    //          ulist(natom,nbor,jjup).re = -ulist(natom,nbor,jju).re;
    //          ulist(natom,nbor,jjup).im = ulist(natom,nbor,jju).im;
    //        }
    //        mapar = -mapar;
    //        jju++;
    //        jjup--;
    //      }
    //      mbpar = -mbpar;
    //    }
  }
}

/* ----------------------------------------------------------------------
   compute derivatives of Wigner U-functions for one neighbor
   see comments in compute_uarray()
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   memory usage of arrays
------------------------------------------------------------------------- */

double
SNA::memory_usage()
{
  size_t bytes;
  bytes = ncoeff * sizeof(double); // coeff

  bytes += ulist.span() * sizeof(SNAcomplex);      // ulist
  bytes += cglist.span() * sizeof(double);         // cglist
  bytes += idxcg_block.span() * sizeof(int);       // idxcg_block
  bytes += ulisttot.span() * sizeof(SNAcomplex);   // ulisttot
  bytes += ulisttot_r.span() * sizeof(SNAcomplex); // ulisttot
  bytes += dulist.span() * sizeof(SNAcomplex);     // dulist
  bytes += idxu_block.span() * sizeof(int);        // idxu_block
  bytes += idxz.span() * sizeof(int);              // idxz
  bytes += idxz_block.span() * sizeof(int);        // idxz_block
  bytes += ylist.span() * sizeof(SNAcomplex);      // ylist
  bytes += dedr.span() * sizeof(double);           // dedr
  bytes += rootpqarray.span() * sizeof(double);    // rootpqarray

  bytes += rij.span() * sizeof(double);
  bytes += inside.span() * sizeof(int);
  bytes += wj.span() * sizeof(double);
  bytes += rcutij.span() * sizeof(double);

  return bytes;
}

/* ---------------------------------------------------------------------- */

void
SNA::create_twojmax_arrays()
{
  int jdimpq = twojmax + 2;
  int jdim = twojmax + 1;

  // Building index lists-counts for resizing
  idxcg_block = int_View3D("idxcg_block", jdim, jdim, jdim);
  idxu_block = int_View1D("idxu_block", jdim);
  h_idxcg_block = create_mirror_view(idxcg_block);
  h_idxu_block = create_mirror_view(idxu_block);

  // Resize z-list
  int idxz_count = 0;
  int idxz_j1j2j_count = 0;
  int idxz_ma_count = 0;
  int idxz_mb_count = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxz_j1j2j_count++;
        for (int ma = 0; ma <= j; ma++)
          idxz_ma_count++;
        for (int mb = 0; 2 * mb <= j; mb++)
          idxz_mb_count++;
        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;
      }

  idxz_max = idxz_count;

  idxz = int_View2D("idxz", idxz_max, 9);
  idxzbeta = double_View1D("idxzbeta", idxz_max);

  h_idxz = create_mirror_view(idxz);
  h_idxzbeta = create_mirror_view(idxzbeta);

  betaj_index = double_View1D("betaj_index", idxz_max);
  h_betaj_index = create_mirror_view(betaj_index);

  // Resize B
  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          idxb_count++;

  idxb_max = idxb_count;
  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        h_idxcg_block(j1, j2, j) = idxcg_count;
        for (int m1 = 0; m1 <= j1; m1++)
          for (int m2 = 0; m2 <= j2; m2++)
            idxcg_count++;
      }
  idxcg_max = idxcg_count;

  // index list for uarray
  // need to include both halves
  int idxu_count = 0;
  for (int j = 0; j <= twojmax; j++) {
    h_idxu_block(j) = idxu_count;
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxu_count++;
  }
  idxu_max = idxu_count;

  // Ulist parity data structure resizing
  ulist_parity = int_View1D("ulist_parity", idxu_max);
  h_ulist_parity = create_mirror_view(ulist_parity);

  idxu_count = 0;
  for (int j = 0; j <= twojmax; j++) {
    int mbpar = 1;
    for (int mb = 0; mb <= j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        h_ulist_parity(idxu_count) = mapar;
        mapar = -mapar;
        idxu_count++;
      }
      mbpar = -mbpar;
    }
  }

  idxdu_block = int_View1D("idxdu_block", jdim);
  h_idxdu_block = create_mirror_view(idxdu_block);
  int idxdu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    h_idxdu_block(j) = idxdu_count;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxdu_count++;
  }
  idxdu_max = idxdu_count;

  // resizing data-structures
  cglist = double_View1D("cglist", idxcg_max);
  rootpqarray = double_View2D("rootpqarray", jdimpq, jdimpq);
  dedr = double_View3D("dedr", num_atoms, num_nbor, 3);
  rootpqparityarray = double_View2D("rootpqparityarray", jdimpq, jdimpq);
  ylist = SNAcomplex_View2DR("ylist", num_atoms, idxdu_max);
  dulist = SNAcomplex_View4D("dulist", num_atoms, num_nbor, idxdu_max, 3);
#if defined(KOKKOS_ENABLE_CUDA)
  ulisttot = SNAcomplex_View2D("ulisttot", idxu_max, num_atoms);
#else
  ulisttot = SNAcomplex_View2D("ulisttot", num_atoms, idxu_max);
#endif
  ulist = SNAcomplex_View3D("ulist", num_atoms, num_nbor, idxu_max);

  h_dedr = create_mirror_view(dedr);
  h_cglist = create_mirror_view(cglist);
  h_rootpqarray = create_mirror_view(rootpqarray);
  h_rootpqparityarray = create_mirror_view(rootpqparityarray);
}

/* ---------------------------------------------------------------------- */

void
SNA::destroy_twojmax_arrays()
{}

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

double
SNA::factorial(int n)
{
  if (n < 0 || n > nmaxfactorial) {
    exit(1);
  }

  return nfac_table[n];
}

/* ----------------------------------------------------------------------
   factorial n table, size SNA::nmaxfactorial+1
------------------------------------------------------------------------- */

const double SNA::nfac_table[] = {
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

double
SNA::deltacg(int j1, int j2, int j)
{
  double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
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
  double sum, dcg, sfaccg;
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
              h_cglist(idxcg_count) = 0.0;
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

            h_cglist(idxcg_count) = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
}

/* ----------------------------------------------------------------------
   pre-compute table of sqrt[p/m2], p, q = 1,twojmax
   the p = 0, q = 0 entries are allocated and skipped for convenience.
------------------------------------------------------------------------- */

void
SNA::init_rootpqarray()
{
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      h_rootpqarray(p, q) = sqrt(static_cast<SNADOUBLE>(p) / q);

  int ppar = 1;
  for (int p = 1; p <= twojmax; p++) {
    int qpar = ppar;
    for (int q = 1; q <= twojmax; q++) {
      h_rootpqparityarray(p, q) = qpar * sqrt(static_cast<SNADOUBLE>(p) / q);
      qpar = -qpar;
    }
    ppar = -ppar;
  }
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

KOKKOS_INLINE_FUNCTION
double
SNA::compute_sfac(double r, double rcut) const
{
  if (switch_flag == 0)
    return 1.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 1.0;
    else if (r > rcut)
      return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double
SNA::compute_dsfac(double r, double rcut) const
{
  if (switch_flag == 0)
    return 0.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 0.0;
    else if (r > rcut)
      return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}
