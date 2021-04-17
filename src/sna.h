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

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "Kokkos_Defines.h"

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))
static const double MY_PI = 3.14159265358979323846;  // pi

struct InitUlisttotTag {};
struct InitUlistTag {};
struct AddSelfUarraytotTag {};
struct ComputeUiTag {};
struct ComputeYiInitTag {};
struct ComputeYiTag {};
struct ComputeDUiTag {};
struct ComputeDEiTag {};

class SNA {
   private:
    double rmin0, rfac0;

    static const int nmaxfactorial = 167;
    static const double nfac_table[];
    double factorial(int);

    void create_twojmax_arrays();
    void init_clebsch_gordan();
    void init_rootpqarray();
    void compute_uarray(int natom, int nbor, double x, double y, double z,
                        double z0, double r);
    double deltacg(int, int, int);
    int compute_ncoeff();
    // Sets the style for the switching function
    // 0 = none
    // 1 = cosine

    int switch_flag;

    // self-weight

    double wself;

   public:
    SNA(double, int, double, int, int, int, int);
    ~SNA();
    void build_indexlist();
    void init();
    double memory_usage();

    int ncoeff;

    // functions for bispectrum coefficients

    void compute_ui();
    void ui_launch();
    void update_ulisttot_launch();

    void compute_yi();
    void compute_yi_reduce_idxz();
    void ulisttot_transpose_launch();
    void yi_zero_launch();
    void yi_launch();
    void compute_duarray();
    void zero_uarraytot();
    void addself_uarraytot();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    void compute_fused_deidrj();
#endif

    // functions for derivatives

    void compute_deidrj();
    void compute_all();

    // per sna class instance for OMP use
    int nmax;
    int num_nbor, num_atoms, nTotal;
    int idxcg_max, idxu_max, idxz_max, idxb_max;
    double *coeffi;

    int_View1D ulist_parity = int_View1D("ulist_parity", 0);
    int_View1D idxdu_block = int_View1D("idxdu_block", 0);
    HostInt_View1D h_ulist_parity, h_idxdu_block;
    int idxdu_max;
    double_View1D idxzbeta = double_View1D("idxzbeta", 0);
    HostDouble_View1D h_idxzbeta;
    double_View2D rootpqparityarray = double_View2D("rootpqarray", 0, 0);
    HostDouble_View2D h_rootpqparityarray;

    int_View1D idxu_block = int_View1D("idxu_block", 0);
    int_View2D inside = int_View2D("inside", 0, 0);
    int_View2D idxz = int_View2D("idxz", 0, 0);

    // These data structures replace idxz
    int_View2D idxz_j1j2j = int_View2D("idxz_j1j2j", 0, 0);
    int_View2D idxz_ma = int_View2D("idxz_ma", 0, 0);
    int_View2D idxz_mb = int_View2D("idxz_mb", 0, 0);

    int_View2D idxb = int_View2D("idxb", 0, 0);
    int_View3D idxcg_block = int_View3D("idxcg_block", 0, 0, 0);
    int_View3D idxz_block = int_View3D("idxz_block", 0, 0, 0);
    int_View3D idxb_block = int_View3D("idxb_block", 0, 0, 0);
    HostInt_View3D h_idxcg_block, h_idxb_block;
    HostInt_View2D h_idxz, h_inside;
    HostInt_View1D h_idxu_block;
    HostInt_View2D h_idxz_j1j2j, h_idxz_ma, h_idxz_mb;

    double_View1D betaj_index = double_View1D("betaj_index", 0);
    double_View1D cglist = double_View1D("cglist", 0);
    double_View2D rootpqarray = double_View2D("rootpqarray", 0, 0);
    double_View2D wj = double_View2D("wj", 0, 0);
    double_View2D rcutij = double_View2D("rcutij", 0, 0);
    double_View3D dedr = double_View3D("dedr", 0, 0, 0);
    double_View3D rij = double_View3D("rij", 0, 0, 0);
    HostDouble_View3D h_rij, h_dedr;
    HostDouble_View2D h_wj, h_rcutij, h_rootpqarray;
    HostDouble_View1D h_betaj_index, h_cglist;

    SNAcomplex_View2DR ylist = SNAcomplex_View2DR("ylist", 0, 0);
    SNAcomplex_View2D ulisttot = SNAcomplex_View2D("ulisttot", 0, 0);
    SNAcomplex_View2DR ulisttot_r = SNAcomplex_View2DR("ulisttot_r", 0, 0);

    SNAcomplex_View3D ulist_ij = SNAcomplex_View3D("ulist_ij", 0, 0, 0);
    SNAcomplex_View3D ulist = SNAcomplex_View3D("ulist", 0, 0, 0);
    SNAcomplex_View2DR ulist_2d = SNAcomplex_View2DR("ulist_2d", 0, 0);

    SNAcomplex_View4D dulist = SNAcomplex_View4D("dulist", 0, 0, 0, 0);

    void grow_rij();
    int twojmax = 0, jdim = 0;
    int idxz_j1j2j_max = 0, idxz_ma_max = 0, idxz_mb_max = 0;

    KOKKOS_INLINE_FUNCTION
    double compute_sfac(double r, double rcut) const {
        if (switch_flag == 0) return 1.0;
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
    double compute_dsfac(double r, double rcut) const {
        if (switch_flag == 0) return 0.0;
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

    KOKKOS_INLINE_FUNCTION
    void operator()(const InitUlisttotTag, const int iter) const {
        int natom = iter / idxu_max;
        int jju = iter % idxu_max;
        ;
        ulisttot(natom, jju) = {0.0, 0.0};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const InitUlistTag, const int iter) const {
        int nbor = iter / num_atoms;
        int natom = iter % num_atoms;

        ulist(natom, nbor, 0) = {1.0, 0.0};
        for (int jju = 1; jju < idxu_max; ++jju)
            ulist(natom, nbor, jju) = {0.0, 0.0};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const AddSelfUarraytotTag, const int iter) const {
        int natom = iter / jdim;
        int j = iter % jdim;
        ;
        int jju = idxu_block(j);
        for (int ma = 0; ma <= j; ma++) {
            ulisttot(natom, jju) = {wself, 0.0};
            jju += j + 2;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeUiTag, const member_type team_member) const {
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
            SNADOUBLE theta0 =
                (r - rmin0) * rfac0 * MY_PI / (rcutij(natom, nbor) - rmin0);
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

            double sfac = compute_sfac(r, rcutij(natom, nbor));

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
                    ulist_jju1.re +=
                        rootpq * (a_r * ulist_up.re + a_i * ulist_up.im);
                    ulist_jju1.im +=
                        rootpq * (a_r * ulist_up.im - a_i * ulist_up.re);

                    // jju+1
                    rootpq = rootpqarray(ma + 1, j - mb);
                    ulist_jju2.re +=
                        -rootpq * (b_r * ulist_up.re + b_i * ulist_up.im);
                    ulist_jju2.im +=
                        -rootpq * (b_r * ulist_up.im - b_i * ulist_up.re);

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

                // handle middle column using inversion symmetry of previous
                // layer

                if (j % 2 == 0) {
                    int mb = j / 2;
                    jjup--;
                    ulist(natom, nbor, jju) = {0.0, 0.0};
                    for (int ma = 0; ma < j; ma++) {
                        rootpq =
                            ulist_parity(jjup) * rootpqarray(j - ma, j - mb);
                        ulist(natom, nbor, jju).re +=
                            rootpq * (a_r * ulist(natom, nbor, jjup).re +
                                      a_i * -ulist(natom, nbor, jjup).im);
                        ulist(natom, nbor, jju).im +=
                            rootpq * (a_r * -ulist(natom, nbor, jjup).im -
                                      a_i * ulist(natom, nbor, jjup).re);

                        rootpq =
                            ulist_parity(jjup) * rootpqarray(ma + 1, j - mb);
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
                    Kokkos::atomic_add(
                        &(ulisttot(natom, jjup).re),
                        sfac * ulist_parity(jju) * ulist(natom, nbor, jju).re);
                    Kokkos::atomic_add(
                        &(ulisttot(natom, jjup).im),
                        sfac * ulist_parity(jju) * -ulist(natom, nbor, jju).im);
                    jju++;
                    jjup--;
                }
            }  // twojmax
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeYiInitTag, const int iter) const {
        const int natom = iter / idxdu_max;
        const int j = iter % idxdu_max;
        ylist(natom, j) = {0.0, 0.0};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeYiTag, const int iter) const {
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

        const SNADOUBLE *cgblock = cglist.data() + idxcg_block(j1, j2, j);
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
            }  // end loop over ia

            ztmp_r += cgblock[icgb] * suma1_r;
            ztmp_i += cgblock[icgb] * suma1_i;
            jju1 += j1 + 1;
            jju2 -= j2 + 1;
            icgb += j2;
        }  // end loop over ib

        //    Atomic updates to ulist due to parallelization over the
        //    bispectrum coeffients.
        Kokkos::atomic_add(&(ylist(natom, jjdu).re), betaj * ztmp_r);
        Kokkos::atomic_add(&(ylist(natom, jjdu).im), betaj * ztmp_i);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeDUiTag, const int iter) const {
        int nbor = iter / num_atoms;
        int natom = iter % num_atoms;
        SNADOUBLE rsq, r, x, y, z, z0, theta0, cs, sn;
        SNADOUBLE dz0dr;

        x = rij(natom, nbor, 0);
        y = rij(natom, nbor, 1);
        z = rij(natom, nbor, 2);
        rsq = x * x + y * y + z * z;
        r = sqrt(rsq);
        SNADOUBLE rscale0 = rfac0 * MY_PI / (rcutij(natom, nbor) - rmin0);
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

        for (int k = 0; k < 3; ++k) dulist(natom, nbor, 0, k) = {0.0, 0.0};

        for (int j = 1; j <= twojmax; j++) {
            int jjup = idxu_block(j - 1);
            int jju = idxu_block(j);
            int jjdup = idxdu_block(j - 1);
            int jjdu = idxdu_block(j);

            //      printf("twojamx = %d\t jjdup = %d\t jjup = %d\n", j,
            //      jjdup, jjup);

            for (int mb = 0; 2 * mb < j; mb++) {
                for (int k = 0; k < 3; ++k)
                    dulist(natom, nbor, jjdu, k) = {0.0, 0.0};

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

            // handle middle column using inversion symmetry of previous
            // layer

            if (j % 2 == 0) {
                int mb = j / 2;
                jjup--;
                jjdup--;
                for (int k = 0; k < 3; ++k)
                    dulist(natom, nbor, jjdu, k) = {0.0, 0.0};

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
            }  // middle column
        }      // twojmax

        double rcut = rcutij(natom, nbor);
        SNADOUBLE sfac = compute_sfac(r, rcutij(natom, nbor));
        SNADOUBLE dsfac = compute_dsfac(r, rcutij(natom, nbor));

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
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeDEiTag, const int iter) const {
        int nbor = iter / num_atoms;
        int natom = iter % num_atoms;

        for (int k = 0; k < 3; ++k) dedr(natom, nbor, k) = 0.0;

        for (int j = 0; j <= twojmax; j++) {
            int jjdu = idxdu_block(j);

            for (int mb = 0; 2 * mb < j; mb++)
                for (int ma = 0; ma <= j; ma++) {
                    for (int k = 0; k < 3; k++)
                        dedr(natom, nbor, k) +=
                            dulist(natom, nbor, jjdu, k).re *
                                ylist(natom, jjdu).re +
                            dulist(natom, nbor, jjdu, k).im *
                                ylist(natom, jjdu).im;
                    jjdu++;
                }  // end loop over ma mb

            // For j even, handle middle column

            if (j % 2 == 0) {
                int mb = j / 2;
                for (int ma = 0; ma < mb; ma++) {
                    for (int k = 0; k < 3; k++)
                        dedr(natom, nbor, k) +=
                            dulist(natom, nbor, jjdu, k).re *
                                ylist(natom, jjdu).re +
                            dulist(natom, nbor, jjdu, k).im *
                                ylist(natom, jjdu).im;
                    jjdu++;
                }

                for (int k = 0; k < 3; k++)
                    dedr(natom, nbor, k) += (dulist(natom, nbor, jjdu, k).re *
                                                 ylist(natom, jjdu).re +
                                             dulist(natom, nbor, jjdu, k).im *
                                                 ylist(natom, jjdu).im) *
                                            0.5;
                jjdu++;
            }  // end if jeven

        }  // end loop over j

        for (int k = 0; k < 3; k++) dedr(natom, nbor, k) *= 2.0;
    }
};

#endif

