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
// Main changes: GPU AoSoA data layout, optimized recursive polynomial
// evaluation
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
void SNA::build_indexlist() {
    double *beta = coeffi + 1;
    idxb_block = int_View3D("idxb_block", jdim, jdim, jdim);
    h_idxb_block = create_mirror_view(idxb_block);

    int idxb_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
        for (int j2 = 0; j2 <= j1; j2++)
            for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
                if (j < j1) continue;
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
                        h_idxz(idxz_count, 4) =
                            (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
                        h_idxz(idxz_count, 5) =
                            MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

                        int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
                        h_idxz(idxz_count, 6) = mb1min;
                        h_idxz(idxz_count, 7) =
                            (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
                        h_idxz(idxz_count, 8) =
                            MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

                        h_idxzbeta(idxz_count) = betaj;

                        idxz_count++;
                    }
            }
}

void SNA::init() {
    init_clebsch_gordan();
    init_rootpqarray();
    build_indexlist();
}

void SNA::grow_rij() {
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
   memory usage of arrays
------------------------------------------------------------------------- */

double SNA::memory_usage() {
    size_t bytes;
    bytes = ncoeff * sizeof(double);  // coeff

    bytes += ulist.span() * sizeof(SNAcomplex);     // ulist
    bytes += cglist.span() * sizeof(double);        // cglist
    bytes += idxcg_block.span() * sizeof(int);      // idxcg_block
    bytes += ulisttot.span() * sizeof(SNAcomplex);  // ulisttot
    bytes += dulist.span() * sizeof(SNAcomplex);    // dulist
    bytes += idxu_block.span() * sizeof(int);       // idxu_block
    bytes += idxz.span() * sizeof(int);             // idxz
    bytes += idxz_block.span() * sizeof(int);       // idxz_block
    bytes += ylist.span() * sizeof(SNAcomplex);     // ylist
    bytes += dedr.span() * sizeof(double);          // dedr
    bytes += rootpqarray.span() * sizeof(double);   // rootpqarray

    bytes += rij.span() * sizeof(double);
    bytes += wj.span() * sizeof(double);
    bytes += rcutij.span() * sizeof(double);

    return bytes;
}

/* ---------------------------------------------------------------------- */

void SNA::create_twojmax_arrays() {
    int jdimpq = twojmax + 2;

    // Building index lists-counts for resizing
    idxcg_block = int_View3D("idxcg_block", jdim, jdim, jdim);
    idxu_block = int_View1D("idxu_block", jdim);
    idxu_half_block = int_View1D("idxu_half_block", jdim);
    h_idxcg_block = create_mirror_view(idxcg_block);
    h_idxu_block = create_mirror_view(idxu_block);
    h_idxu_half_block = create_mirror_view(idxu_half_block);

    // Resize z-list
    int idxz_count = 0;
    int idxz_j1j2j_count = 0;
    int idxz_ma_count = 0;
    int idxz_mb_count = 0;

    for (int j1 = 0; j1 <= twojmax; j1++)
        for (int j2 = 0; j2 <= j1; j2++)
            for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
                idxz_j1j2j_count++;
                for (int ma = 0; ma <= j; ma++) idxz_ma_count++;
                for (int mb = 0; 2 * mb <= j; mb++) idxz_mb_count++;
                for (int mb = 0; 2 * mb <= j; mb++)
                    for (int ma = 0; ma <= j; ma++) idxz_count++;
            }

    idxz_max = idxz_count;

    idxz = int_View2D("idxz", idxz_max, 9);
    idxzbeta = double_View1D("idxzbeta", idxz_max);

    h_idxz = create_mirror_view(idxz);
    h_idxzbeta = create_mirror_view(idxzbeta);

    // Resize B
    int idxb_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
        for (int j2 = 0; j2 <= j1; j2++)
            for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
                if (j >= j1) idxb_count++;

    idxb_max = idxb_count;
    int idxcg_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
        for (int j2 = 0; j2 <= j1; j2++)
            for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
                h_idxcg_block(j1, j2, j) = idxcg_count;
                for (int m1 = 0; m1 <= j1; m1++)
                    for (int m2 = 0; m2 <= j2; m2++) idxcg_count++;
            }
    idxcg_max = idxcg_count;

    // index list for uarray
    // need to include both halves
    int idxu_count = 0;
    for (int j = 0; j <= twojmax; j++) {
        h_idxu_block(j) = idxu_count;
        for (int mb = 0; mb <= j; mb++)
            for (int ma = 0; ma <= j; ma++) idxu_count++;
    }
    idxu_max = idxu_count;

    // index list for uarray, compressed
    int idxu_half_count = 0;
    for (int j = 0; j <= twojmax; j++) {
        h_idxu_half_block(j) = idxu_half_count;
        for (int mb = 0; 2 * mb <= j; mb++)
            for (int ma = 0; ma <= j; ma++) idxu_half_count++;
    }
    idxu_half_max = idxu_half_count;

    // mapping from full -> half indexing, encoding sign flips
    idxu_full_half = FullHalfMap_View1D("idxu_full_half", idxu_max);
    h_idxu_full_half = create_mirror_view(idxu_full_half);
    idxu_count = 0;
    for (int j = 0; j <= twojmax; j++) {
        int jju_half = h_idxu_half_block[j];
        for (int mb = 0; mb <= j; mb++) {
            for (int ma = 0; ma <= j; ma++) {
                FullHalfMap map;
                if (2 * mb <= j) {
                    map.idxu_half = jju_half + mb * (j + 1) + ma;
                    map.flip_sign = 0;
                } else {
                    map.idxu_half =
                        jju_half + (j + 1 - mb) * (j + 1) - (ma + 1);
                    map.flip_sign = (((ma + mb) % 2 == 0) ? 1 : -1);
                }
                h_idxu_full_half[idxu_count] = map;
                idxu_count++;
            }
        }
    }

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
            for (int ma = 0; ma <= j; ma++) idxdu_count++;
    }
    idxdu_max = idxdu_count;

    // resizing data-structures
    cglist = double_View1D("cglist", idxcg_max);
    rootpqarray = double_View2D("rootpqarray", jdimpq, jdimpq);
    dedr = double_View3D("dedr", num_atoms, num_nbor, 3);
    rootpqparityarray = double_View2D("rootpqparityarray", jdimpq, jdimpq);
#ifdef SNAP_ENABLE_GPU
    alist_gpu =
        SNAcomplex_View3DL("alist_gpu", vector_length, num_nbor, num_atoms_div);
    blist_gpu =
        SNAcomplex_View3DL("blist_gpu", vector_length, num_nbor, num_atoms_div);
    dalist_gpu = SNAcomplex_View4DL("dalist_gpu", vector_length, num_nbor,
                                    num_atoms_div, 3);
    dblist_gpu = SNAcomplex_View4DL("dblist_gpu", vector_length, num_nbor,
                                    num_atoms_div, 3);
    sfaclist_gpu = double_View4DL("sfaclist_gpu", vector_length, num_nbor,
                                  num_atoms_div, 4);

    ulisttot_re_gpu = double_View3DL("ulisttot_re_gpu", vector_length,
                                     idxu_half_max, num_atoms_div);
    ulisttot_im_gpu = double_View3DL("ulisttot_im_gpu", vector_length,
                                     idxu_half_max, num_atoms_div);
    ulisttot_gpu = SNAcomplex_View3DL("ulisttot_gpu", vector_length, idxu_max,
                                      num_atoms_div);

    ylist_re_gpu = double_View3DL("ylist_re_gpu", vector_length, idxu_half_max,
                                  num_atoms_div);
    ylist_im_gpu = double_View3DL("ylist_im_gpu", vector_length, idxu_half_max,
                                  num_atoms_div);

#else
    ylist = SNAcomplex_View2DR("ylist", num_atoms, idxdu_max);
    dulist = SNAcomplex_View4D("dulist", num_atoms, num_nbor, idxdu_max, 3);
    ulisttot = SNAcomplex_View2D("ulisttot", num_atoms, idxu_max);
    ulist = SNAcomplex_View3D("ulist", num_atoms, num_nbor, idxu_max);

#endif

    h_dedr = create_mirror_view(dedr);
    h_cglist = create_mirror_view(cglist);
    h_rootpqarray = create_mirror_view(rootpqarray);
    h_rootpqparityarray = create_mirror_view(rootpqparityarray);
}

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

double SNA::factorial(int n) {
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
    2.03978820811974e+46,  // nmaxfactorial = 39
};

/* ----------------------------------------------------------------------
   the function delta given by VMK Eq. 8.2(1)
------------------------------------------------------------------------- */

double SNA::deltacg(int j1, int j2, int j) {
    double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
    return sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2) *
                factorial((-j1 + j2 + j) / 2) / sfaccg);
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients using
   the quasi-binomial formula VMK 8.2.1(3)
------------------------------------------------------------------------- */

void SNA::init_clebsch_gordan() {
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

                        for (int z = MAX(0, MAX(-(j - j2 + aa2) / 2,
                                                -(j - j1 - bb2) / 2));
                             z <= MIN((j1 + j2 - j) / 2,
                                      MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                             z++) {
                            ifac = z % 2 ? -1 : 1;
                            sum += ifac / (factorial(z) *
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
                                      factorial((j + cc2) / 2) *
                                      factorial((j - cc2) / 2) * (j + 1));

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

void SNA::init_rootpqarray() {
    for (int p = 1; p <= twojmax; p++)
        for (int q = 1; q <= twojmax; q++)
            h_rootpqarray(p, q) = sqrt(static_cast<SNADOUBLE>(p) / q);

    int ppar = 1;
    for (int p = 1; p <= twojmax; p++) {
        int qpar = ppar;
        for (int q = 1; q <= twojmax; q++) {
            h_rootpqparityarray(p, q) =
                qpar * sqrt(static_cast<SNADOUBLE>(p) / q);
            qpar = -qpar;
        }
        ppar = -ppar;
    }
}

/* ---------------------------------------------------------------------- */

int SNA::compute_ncoeff() {
    int ncount = 0;

    for (int j1 = 0; j1 <= twojmax; j1++)
        for (int j2 = 0; j2 <= j1; j2++)
            for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
                if (j >= j1) ncount++;

    return ncount;
}

/* ---------------------------------------------------------------------- */

