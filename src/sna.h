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

// CPU

struct InitUlisttotTag {};
struct InitUlistTag {};
struct AddSelfUarraytotTag {};
struct ComputeUiTag {};
struct ComputeYiInitTag {};
struct ComputeYiTag {};
struct ComputeDUiTag {};
struct ComputeDEiTag {};

// GPU
struct ComputeCayleyKleinTagGPU {};
struct PreUiTagGPU {};
struct ComputeUiTagGPU {};
struct TransformUiTagGPU {};
struct ComputeYiTagGPU {};
template <int dir>
struct ComputeFusedDeiDrjTagGPU {};

class SNA {
    // Modify as appropriate
#if defined(SNAP_ENABLE_GPU) && !defined(KOKKOS_ENABLE_OPENMPTARGET)
    static constexpr int vector_length = 32;
#else
    static constexpr int vector_length = 1;
#endif

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
    SNA(double rfac0_in, int twojmax_in, double rmin0_in, int switch_flag_in,
        int /*bzero_flag_in*/, int natoms, int nnbor) {
        wself = 1.0;

        rfac0 = rfac0_in;
        rmin0 = rmin0_in;
        switch_flag = switch_flag_in;
        num_atoms = natoms;
        num_atoms_div = (natoms + vector_length - 1) / vector_length;
        num_nbor = nnbor;
        nTotal = num_atoms * num_nbor;

        twojmax = twojmax_in;
        jdim = twojmax + 1;

        ncoeff = compute_ncoeff();

        nmax = 0;

        grow_rij();
        create_twojmax_arrays();

        //  build_indexlist();
    }

    /* ---------------------------------------------------------------------- */

    ~SNA() {}

    void build_indexlist();
    void init();
    double memory_usage();

    int ncoeff;

    int nmax;
    int num_nbor, num_atoms, num_atoms_div, nTotal;
    int idxcg_max, idxu_max, idxu_half_max, idxz_max, idxb_max;
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
    int_View1D idxu_half_block = int_View1D("idxu_half_block", 0);
    int_View2D inside = int_View2D("inside", 0, 0);
    int_View2D idxz = int_View2D("idxz", 0, 0);

    // These data structures replace idxz
    int_View2D idxz_ma = int_View2D("idxz_ma", 0, 0);
    int_View2D idxz_mb = int_View2D("idxz_mb", 0, 0);

    // mapping from compressed ulisttot to full
    FullHalfMap_View1D idxu_full_half = FullHalfMap_View1D("idxu_full_half", 0);
    HostFullHalfMap_View1D h_idxu_full_half;

    int_View2D idxb = int_View2D("idxb", 0, 0);
    int_View3D idxcg_block = int_View3D("idxcg_block", 0, 0, 0);
    int_View3D idxz_block = int_View3D("idxz_block", 0, 0, 0);
    int_View3D idxb_block = int_View3D("idxb_block", 0, 0, 0);
    HostInt_View3D h_idxcg_block, h_idxb_block;
    HostInt_View2D h_idxz, h_inside;
    HostInt_View1D h_idxu_block, h_idxu_half_block;

    double_View1D cglist = double_View1D("cglist", 0);
    double_View2D rootpqarray = double_View2D("rootpqarray", 0, 0);
    double_View2D wj = double_View2D("wj", 0, 0);
    double_View2D rcutij = double_View2D("rcutij", 0, 0);
    double_View3D dedr = double_View3D("dedr", 0, 0, 0);
    double_View3D rij = double_View3D("rij", 0, 0, 0);
    HostDouble_View3D h_rij, h_dedr;
    HostDouble_View2D h_wj, h_rcutij, h_rootpqarray;
    HostDouble_View1D h_cglist;

    // for "CPU" code
    SNAcomplex_View2DR ylist = SNAcomplex_View2DR("ylist", 0, 0);
    SNAcomplex_View2D ulisttot = SNAcomplex_View2D("ulisttot", 0, 0);
    SNAcomplex_View3D ulist = SNAcomplex_View3D("ulist", 0, 0, 0);
    SNAcomplex_View4D dulist = SNAcomplex_View4D("dulist", 0, 0, 0, 0);

    // for "GPU" code
    SNAcomplex_View3DL alist_gpu = SNAcomplex_View3DL("alist_gpu", 0, 0, 0);
    SNAcomplex_View3DL blist_gpu = SNAcomplex_View3DL("blist_gpu", 0, 0, 0);
    SNAcomplex_View4DL dalist_gpu =
        SNAcomplex_View4DL("dalist_gpu", 0, 0, 0, 0);
    SNAcomplex_View4DL dblist_gpu =
        SNAcomplex_View4DL("dblist_gpu", 0, 0, 0, 0);
    double_View4DL sfaclist_gpu = double_View4DL("sfaclist_gpu", 0, 0, 0, 0);

    double_View3DL ulisttot_re_gpu =
        double_View3DL("ulisttot_re_gpu", 0, 0, 0);  // compressed data layout
    double_View3DL ulisttot_im_gpu =
        double_View3DL("ulisttot_im_gpu", 0, 0, 0);  // compressed data layout
    SNAcomplex_View3DL ulisttot_gpu =
        SNAcomplex_View3DL("ulisttot_gpu", 0, 0, 0);  // compressed data layout

    double_View3DL ylist_re_gpu =
        double_View3DL("ylist_re_gpu", 0, 0, 0);  // compressed data layout
    double_View3DL ylist_im_gpu =
        double_View3DL("ylist_im_gpu", 0, 0, 0);  // compressed data layout

    void grow_rij();
    int twojmax = 0, jdim = 0;

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
#ifdef SNAP_ENABLE_GPU
                suma1_r += cgblock[icga] * (ulisttot(jju1 + ma1, natom).re *
                                                ulisttot(jju2 + ma2, natom).re -
                                            ulisttot(jju1 + ma1, natom).im *
                                                ulisttot(jju2 + ma2, natom).im);
                suma1_i += cgblock[icga] * (ulisttot(jju1 + ma1, natom).re *
                                                ulisttot(jju2 + ma2, natom).im +
                                            ulisttot(jju1 + ma1, natom).im *
                                                ulisttot(jju2 + ma2, natom).re);
#else
                suma1_r += cgblock[icga] * (ulisttot(natom, jju1 + ma1).re *
                                                ulisttot(natom, jju2 + ma2).re -
                                            ulisttot(natom, jju1 + ma1).im *
                                                ulisttot(natom, jju2 + ma2).im);
                suma1_i += cgblock[icga] * (ulisttot(natom, jju1 + ma1).re *
                                                ulisttot(natom, jju2 + ma2).im +
                                            ulisttot(natom, jju1 + ma1).im *
                                                ulisttot(natom, jju2 + ma2).re);
#endif
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

    /* GPU codepath */

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeCayleyKleinTagGPU, const int natom_mod,
                    const int nbor, const int natom_div) const {
        const int natom = natom_mod + natom_div * vector_length;
        if (natom >= num_atoms) return;
        if (nbor >= num_nbor) return;

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
        SNADOUBLE dsfacu[3];

        SNADOUBLE rinv = 1.0 / r;

        SNADOUBLE u[3] = {x * rinv, y * rinv, z * rinv};

        r0inv = 1.0 / sqrt(r * r + z0 * z0);
        a_r = z0 * r0inv;
        a_i = -z * r0inv;
        b_r = y * r0inv;
        b_i = -x * r0inv;
        SNADOUBLE sfac = compute_sfac(r, rcutij(natom, nbor));
        SNADOUBLE dsfac = compute_dsfac(r, rcutij(natom, nbor));

        sfac *= wj(natom, nbor);
        dsfac *= wj(natom, nbor);

        dr0invdr = -r0inv * r0inv * r0inv * (r + z0 * dz0dr);

        dr0inv[0] = dr0invdr * u[0];
        dr0inv[1] = dr0invdr * u[1];
        dr0inv[2] = dr0invdr * u[2];

        dz0[0] = dz0dr * u[0];
        dz0[1] = dz0dr * u[1];
        dz0[2] = dz0dr * u[2];

        dsfacu[0] = dsfac * u[0];
        dsfacu[1] = dsfac * u[1];
        dsfacu[2] = dsfac * u[2];

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

        alist_gpu(natom_mod, nbor, natom_div) = {a_r, a_i};
        blist_gpu(natom_mod, nbor, natom_div) = {b_r, b_i};

        dalist_gpu(natom_mod, nbor, natom_div, 0) = {da_r[0], da_i[0]};
        dblist_gpu(natom_mod, nbor, natom_div, 0) = {db_r[0], db_i[0]};

        dalist_gpu(natom_mod, nbor, natom_div, 1) = {da_r[1], da_i[1]};
        dblist_gpu(natom_mod, nbor, natom_div, 1) = {db_r[1], db_i[1]};

        dalist_gpu(natom_mod, nbor, natom_div, 2) = {da_r[2], da_i[2]};
        dblist_gpu(natom_mod, nbor, natom_div, 2) = {db_r[2], db_i[2]};

        sfaclist_gpu(natom_mod, nbor, natom_div, 0) = sfac;
        sfaclist_gpu(natom_mod, nbor, natom_div, 1) = dsfacu[0];
        sfaclist_gpu(natom_mod, nbor, natom_div, 2) = dsfacu[1];
        sfaclist_gpu(natom_mod, nbor, natom_div, 3) = dsfacu[2];

        // need to zero this here b/c we atomic add into this in
        // ComputeFusedDeiDrjTagGPU
        dedr(natom, nbor, 0) = 0;
        dedr(natom, nbor, 1) = 0;
        dedr(natom, nbor, 2) = 0;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const PreUiTagGPU, const int natom_mod, const int j,
                    const int natom_div) const {
        const int natom = natom_mod + natom_div * vector_length;
        if (natom >= num_atoms) return;
        if (j > twojmax) return;

        int jju_half = idxu_half_block(j);

        // Initialize diagonal elements, symmetric half of matrix only
        for (int mb = 0; 2 * mb <= j; mb++) {
            for (int ma = 0; ma <= j; ma++) {
                ulisttot_re_gpu(natom_mod, jju_half, natom_div) =
                    (ma == mb) ? wself : 0;
                ulisttot_im_gpu(natom_mod, jju_half, natom_div) = 0;
                jju_half++;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeUiTagGPU,
                    const member_type team_member) const {
        int iter = team_member.league_rank() * team_member.team_size() +
                   team_member.team_rank();

// Need this indirection because OMPT backend fails to compile with scalar
// member variables if scratch memory is being used. No idea why :facepalm:
#ifdef KOKKOS_ENABLE_OPENMPTARGET
        int vector_length_loc = 1;
#else
        int vector_length_loc = vector_length;
#endif
        // Unflatten bend location (fastest), neighbor (middle), natom mod
        // vector_length (slowest)
        int natom_div = iter / (num_nbor * (twojmax + 1));
        int nbor_jbend = iter - natom_div * (num_nbor * (twojmax + 1));
        int jbend = nbor_jbend / num_nbor;
        int nbor = nbor_jbend - jbend * num_nbor;
        if (jbend >= (twojmax + 1)) return;
        if (nbor >= num_nbor) return;

        const int PerTeamScratch = vector_length_loc * (twojmax + 1);
        ScratchViewType ulist_scratch(team_member.team_scratch(0),
                                      team_member.team_size() * PerTeamScratch);

        // thread id's and the index for the start of the buffer
        // for each thread.
        const int tid = team_member.team_rank();
        const int tid_index = tid * PerTeamScratch;

        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, vector_length_loc),
            [&](const int natom_mod) {
                int natom = natom_mod + natom_div * vector_length_loc;
                if (natom >= num_atoms) return;

                const SNAcomplex a = alist_gpu(natom_mod, nbor, natom_div);
                const SNAcomplex b = blist_gpu(natom_mod, nbor, natom_div);
                const SNADOUBLE sfac =
                    sfaclist_gpu(natom_mod, nbor, natom_div, 0);

                // initialize with 1
                ulist_scratch[tid_index + natom_mod] = {1., 0.};

                // build up row 0 of the Wigner U matrices, this is redundant
                // work
                for (int j = 1; j <= jbend; j++) {
                    SNAcomplex ulist_tmp = {0., 0.};

                    int ma;
                    for (ma = 0; ma < j; ma++) {
                        // grab value from previous level
                        const SNAcomplex ulist_prev =
                            ulist_scratch[tid_index + ma * vector_length_loc +
                                          natom_mod];

                        // ulist_tmp += rootpq * a.conj * ulist_prev
                        SNADOUBLE rootpq = rootpqarray(j - ma, j);  // mb = 0
                        ulist_tmp.re += rootpq * (a.re * ulist_prev.re +
                                                  a.im * ulist_prev.im);
                        ulist_tmp.im += rootpq * (a.re * ulist_prev.im -
                                                  a.im * ulist_prev.re);

                        // store ulist tmp for the next level
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod] = ulist_tmp;

                        // compute next value
                        // ulist_tmp = -rootpq * b.conj() * ulist_prev
                        rootpq = rootpqarray(ma + 1, j);  // mb = 0
                        ulist_tmp.re = -rootpq * (b.re * ulist_prev.re +
                                                  b.im * ulist_prev.im);
                        ulist_tmp.im = -rootpq * (b.re * ulist_prev.im -
                                                  b.im * ulist_prev.re);
                    }

                    // store final ulist_tmp for the next level

                    ulist_scratch[tid_index + ma * vector_length_loc +
                                  natom_mod] = ulist_tmp;
                }

                // unique work for each jbend location
                const int j_half_way = MIN(2 * jbend, twojmax);

                int mb = 1;
                int j;
                for (j = jbend + 1; j <= j_half_way; j++) {
                    // get jjup value for atomic adds into ulisttot
                    const int jjup = idxu_half_block(j - 1) + (mb - 1) * j;

                    SNAcomplex ulist_tmp = {0., 0.};

                    int ma;
                    for (ma = 0; ma < j; ma++) {
                        // grab value from previous level
                        const SNAcomplex ulist_prev =
                            ulist_scratch[tid_index + ma * vector_length_loc +
                                          natom_mod];

                        // atomic add previous level into ulisttot
                        Kokkos::atomic_add(
                            &(ulisttot_re_gpu(natom_mod, jjup + ma, natom_div)),
                            ulist_prev.re * sfac);
                        Kokkos::atomic_add(
                            &(ulisttot_im_gpu(natom_mod, jjup + ma, natom_div)),
                            ulist_prev.im * sfac);

                        // compute next level of U matrices
                        // ulist_tmp += rootpq * b * ulist_prev
                        SNADOUBLE rootpq = rootpqarray(j - ma, mb);
                        ulist_tmp.re += rootpq * (b.re * ulist_prev.re -
                                                  b.im * ulist_prev.im);
                        ulist_tmp.im += rootpq * (b.re * ulist_prev.im +
                                                  b.im * ulist_prev.re);

                        // store ulist tmp for the next level
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod] = ulist_tmp;

                        // compute next value
                        // ulist_tmp = rootpq * a * ulist_prev
                        rootpq = rootpqarray(ma + 1, mb);
                        ulist_tmp.re = rootpq * (a.re * ulist_prev.re -
                                                 a.im * ulist_prev.im);
                        ulist_tmp.im = rootpq * (a.re * ulist_prev.im +
                                                 a.im * ulist_prev.re);
                    }

                    // store final ulist_tmp for the next level
                    ulist_scratch[tid_index + ma * vector_length_loc +
                                  natom_mod] = ulist_tmp;

                    mb++;
                }

                // atomic add last set of values into ulisttot
                const int jjup = idxu_half_block(j - 1) + (mb - 1) * j;
                for (int ma = 0; ma < j; ma++) {
                    const SNAcomplex ulist_prev =
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod];
                    Kokkos::atomic_add(
                        &(ulisttot_re_gpu(natom_mod, jjup + ma, natom_div)),
                        ulist_prev.re * sfac);
                    Kokkos::atomic_add(
                        &(ulisttot_im_gpu(natom_mod, jjup + ma, natom_div)),
                        ulist_prev.im * sfac);
                }
            });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const TransformUiTagGPU, const int natom_mod,
                    const int idxu, const int natom_div) const {
        const int natom = natom_mod + natom_div * vector_length;
        if (natom >= num_atoms) return;
        if (idxu >= idxu_max) return;

        const FullHalfMap map = idxu_full_half[idxu];

        SNAcomplex utot = {
            ulisttot_re_gpu(natom_mod, map.idxu_half, natom_div),
            ulisttot_im_gpu(natom_mod, map.idxu_half, natom_div)};

        if (map.flip_sign == 1) {
            utot.im = -utot.im;
        } else if (map.flip_sign == -1) {
            utot.re = -utot.re;
        }

        ulisttot_gpu(natom_mod, idxu, natom_div) = utot;

        if (map.flip_sign == 0) {
            ylist_re_gpu(natom_mod, map.idxu_half, natom_div) = 0.;
            ylist_im_gpu(natom_mod, map.idxu_half, natom_div) = 0.;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ComputeYiTagGPU, const int natom_mod, const int jjz,
                    const int natom_div) const {
        const int natom = natom_mod + natom_div * vector_length;
        if (natom >= num_atoms) return;
        if (jjz >= idxz_max) return;

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
        const int jju_half = idxu_half_block(j) + (j + 1) * mb + ma;

        int jju1 = idxu_block(j1) + (j1 + 1) * mb1min;
        int jju2 = idxu_block(j2) + (j2 + 1) * mb2max;
        int icgb = mb1min * (j2 + 1) + mb2max;

        SNADOUBLE ztmp_r = 0.0;
        SNADOUBLE ztmp_i = 0.0;

        // loop over columns of u1 and corresponding
        // columns of u2 satisfying Clebsch-Gordan constraint
        //      2*mb-j = 2*mb1-j1 + 2*mb2-j2

#pragma unroll
        for (int ib = 0; ib < nb; ib++) {
            SNADOUBLE suma1_r = 0.0;
            SNADOUBLE suma1_i = 0.0;

            int ma1 = ma1min;
            int ma2 = ma2max;
            int icga = ma1min * (j2 + 1) + ma2max;

#pragma unroll
            for (int ia = 0; ia < na; ia++) {
                const SNAcomplex utot_1 =
                    ulisttot_gpu(natom_mod, jju1 + ma1, natom_div);
                const SNAcomplex utot_2 =
                    ulisttot_gpu(natom_mod, jju2 + ma2, natom_div);
                suma1_r += cgblock[icga] *
                           (utot_1.re * utot_2.re - utot_1.im * utot_2.im);
                suma1_i += cgblock[icga] *
                           (utot_1.re * utot_2.im + utot_1.im * utot_2.re);
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

        //    Atomic updates to ylist due to parallelization over the
        //    bispectrum coeffients.
        Kokkos::atomic_add(&(ylist_re_gpu(natom_mod, jju_half, natom_div)),
                           betaj * ztmp_r);
        Kokkos::atomic_add(&(ylist_im_gpu(natom_mod, jju_half, natom_div)),
                           betaj * ztmp_i);
    }

    template <int dir>
    KOKKOS_INLINE_FUNCTION void operator()(
        const ComputeFusedDeiDrjTagGPU<dir>,
        const member_type team_member) const {
        int iter = team_member.league_rank() * team_member.team_size() +
                   team_member.team_rank();

// Need this indirection because OMPT backend fails to compile with scalar
// member variables if scratch memory is being used. No idea why :facepalm:
#ifdef KOKKOS_ENABLE_OPENMPTARGET
        int vector_length_loc = 1;
#else
        int vector_length_loc = vector_length;
#endif

        // Unflatten bend location (fastest), neighbor (middle), natom mod
        // vector_length (slowest)
        int natom_div = iter / (num_nbor * (twojmax + 1));
        int nbor_jbend = iter - natom_div * (num_nbor * (twojmax + 1));
        int jbend = nbor_jbend / num_nbor;
        int nbor = nbor_jbend - jbend * num_nbor;

        if (jbend >= (twojmax + 1)) return;
        if (nbor >= num_nbor) return;

        // caching buffers for ulist, dulist
        const int PerTeamScratch = vector_length_loc * (twojmax + 1);
        ScratchViewType ulist_scratch(team_member.team_scratch(0),
                                      team_member.team_size() * PerTeamScratch);
        ScratchViewType dulist_scratch(
            team_member.team_scratch(0),
            team_member.team_size() * PerTeamScratch);

        // thread id's and the index for the start of the buffer
        // for each thread.
        const int tid = team_member.team_rank();
        const int tid_index = tid * PerTeamScratch;

        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, vector_length_loc),
            [&](const int natom_mod) {
                int natom = natom_mod + natom_div * vector_length_loc;
                if (natom >= num_atoms) return;

                const SNAcomplex a = alist_gpu(natom_mod, nbor, natom_div);
                const SNAcomplex b = blist_gpu(natom_mod, nbor, natom_div);
                const SNAcomplex da =
                    dalist_gpu(natom_mod, nbor, natom_div, dir);
                const SNAcomplex db =
                    dblist_gpu(natom_mod, nbor, natom_div, dir);
                const SNADOUBLE sfac =
                    sfaclist_gpu(natom_mod, nbor, natom_div, 0);
                const SNADOUBLE dsfacu = sfaclist_gpu(
                    natom_mod, nbor, natom_div, dir + 1);  // dsfac * u

                // accumulate dedr as we go
                SNADOUBLE dedr_sum = 0.;

                // initialize with 1, 0, respectively
                ulist_scratch[tid_index + natom_mod] = {1., 0.};
                dulist_scratch[tid_index + natom_mod] = {0., 0.};

                // build up row 0 of the Wigner U matrices and derivative, this
                // is redundant work
                for (int j = 1; j <= jbend; j++) {
                    SNAcomplex ulist_tmp = {0., 0.};
                    SNAcomplex dulist_tmp = {0., 0.};

                    int ma;
                    for (ma = 0; ma < j; ma++) {
                        // grab value from previous level
                        const SNAcomplex ulist_prev =
                            ulist_scratch[tid_index + ma * vector_length_loc +
                                          natom_mod];
                        const SNAcomplex dulist_prev =
                            dulist_scratch[tid_index + ma * vector_length_loc +
                                           natom_mod];

                        // ulist_tmp += rootpq * a.conj * ulist_prev
                        SNADOUBLE rootpq = rootpqarray(j - ma, j);  // mb = 0
                        ulist_tmp.re += rootpq * (a.re * ulist_prev.re +
                                                  a.im * ulist_prev.im);
                        ulist_tmp.im += rootpq * (a.re * ulist_prev.im -
                                                  a.im * ulist_prev.re);

                        // product rule of above
                        dulist_tmp.re +=
                            rootpq *
                            (da.re * ulist_prev.re + da.im * ulist_prev.im +
                             a.re * dulist_prev.re + a.im * dulist_prev.im);
                        dulist_tmp.im +=
                            rootpq *
                            (da.re * ulist_prev.im - da.im * ulist_prev.re +
                             a.re * dulist_prev.im - a.im * dulist_prev.re);

                        // store ulist, dulist tmp for the next level
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod] = ulist_tmp;
                        dulist_scratch[tid_index + ma * vector_length_loc +
                                       natom_mod] = dulist_tmp;

                        // compute next value
                        // ulist_tmp = -rootpq * b.conj() * ulist_prev
                        rootpq = rootpqarray(ma + 1, j);  // mb = 0
                        ulist_tmp.re = -rootpq * (b.re * ulist_prev.re +
                                                  b.im * ulist_prev.im);
                        ulist_tmp.im = -rootpq * (b.re * ulist_prev.im -
                                                  b.im * ulist_prev.re);

                        // product rule of above
                        dulist_tmp.re =
                            -rootpq *
                            (db.re * ulist_prev.re + db.im * ulist_prev.im +
                             b.re * dulist_prev.re + b.im * dulist_prev.im);
                        dulist_tmp.im =
                            -rootpq *
                            (db.re * ulist_prev.im - db.im * ulist_prev.re +
                             b.re * dulist_prev.im - b.im * dulist_prev.re);
                    }

                    // store final ulist_tmp, dulist_tmp for the next level
                    ulist_scratch[tid_index + ma * vector_length_loc +
                                  natom_mod] = ulist_tmp;
                    dulist_scratch[tid_index + ma * vector_length_loc +
                                   natom_mod] = dulist_tmp;
                }

                // unique work for each jbend location
                const int j_half_way = MIN(2 * jbend, twojmax);

                int mb = 1;
                int j;
                for (j = jbend + 1; j <= j_half_way; j++) {
                    // get jjup value for atomic adds into ulisttot
                    const int jjup = idxu_half_block(j - 1) + (mb - 1) * j;

                    SNAcomplex ulist_tmp = {0., 0.};
                    SNAcomplex dulist_tmp = {0., 0.};

                    int ma;
                    for (ma = 0; ma < j; ma++) {
                        // grab y_local; never a need to rescale here
                        const SNAcomplex ylist_value = SNAcomplex(
                            ylist_re_gpu(natom_mod, jjup + ma, natom_div),
                            ylist_im_gpu(natom_mod, jjup + ma, natom_div));

                        // grab value from previous level
                        const SNAcomplex ulist_prev =
                            ulist_scratch[tid_index + ma * vector_length_loc +
                                          natom_mod];
                        const SNAcomplex dulist_prev =
                            dulist_scratch[tid_index + ma * vector_length_loc +
                                           natom_mod];

                        // compute next level of U matrices
                        // ulist_tmp += rootpq * b * ulist_prev
                        SNADOUBLE rootpq = rootpqarray(j - ma, mb);
                        ulist_tmp.re += rootpq * (b.re * ulist_prev.re -
                                                  b.im * ulist_prev.im);
                        ulist_tmp.im += rootpq * (b.re * ulist_prev.im +
                                                  b.im * ulist_prev.re);

                        // product rule of above
                        dulist_tmp.re +=
                            rootpq *
                            (db.re * ulist_prev.re - db.im * ulist_prev.im +
                             b.re * dulist_prev.re - b.im * dulist_prev.im);
                        dulist_tmp.im +=
                            rootpq *
                            (db.re * ulist_prev.im + db.im * ulist_prev.re +
                             b.re * dulist_prev.im + b.im * dulist_prev.re);

                        // store ulist tmp for the next level
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod] = ulist_tmp;
                        dulist_scratch[tid_index + ma * vector_length_loc +
                                       natom_mod] = dulist_tmp;

                        // accumulate dedr
                        const SNAcomplex du_product = {
                            dsfacu * ulist_prev.re + sfac * dulist_prev.re,
                            dsfacu * ulist_prev.im + sfac * dulist_prev.im};
                        dedr_sum += du_product.re * ylist_value.re +
                                    du_product.im * ylist_value.im;

                        // compute next value
                        // ulist_tmp = rootpq * a * ulist_prev
                        rootpq = rootpqarray(ma + 1, mb);
                        ulist_tmp.re = rootpq * (a.re * ulist_prev.re -
                                                 a.im * ulist_prev.im);
                        ulist_tmp.im = rootpq * (a.re * ulist_prev.im +
                                                 a.im * ulist_prev.re);

                        // product rule of above
                        dulist_tmp.re =
                            rootpq *
                            (da.re * ulist_prev.re - da.im * ulist_prev.im +
                             a.re * dulist_prev.re - a.im * dulist_prev.im);
                        dulist_tmp.im =
                            rootpq *
                            (da.re * ulist_prev.im + da.im * ulist_prev.re +
                             a.re * dulist_prev.im + a.im * dulist_prev.re);
                    }

                    // store final ulist_tmp for the next level
                    ulist_scratch[tid_index + ma * vector_length_loc +
                                  natom_mod] = ulist_tmp;
                    dulist_scratch[tid_index + ma * vector_length_loc +
                                   natom_mod] = dulist_tmp;

                    mb++;
                }

                // atomic add last set of values into ulisttot
                const int jjup = idxu_half_block(j - 1) + (mb - 1) * j;
                for (int ma = 0; ma < j; ma++) {
                    // grab y_local early
                    SNAcomplex ylist_value = SNAcomplex(
                        ylist_re_gpu(natom_mod, jjup + ma, natom_div),
                        ylist_im_gpu(natom_mod, jjup + ma, natom_div));
                    if (j % 2 == 1 && 2 * (mb - 1) == j - 1) {
                        if (ma == (mb - 1)) {
                            ylist_value.re *= 0.5;
                            ylist_value.im *= 0.5;
                        } else if (ma > (mb - 1))
                            ylist_value = {0., 0.};
                        // else the ma < mb gets "double counted", cancelling
                        // the 0.5.
                    }

                    const SNAcomplex ulist_prev =
                        ulist_scratch[tid_index + ma * vector_length_loc +
                                      natom_mod];
                    const SNAcomplex dulist_prev =
                        dulist_scratch[tid_index + ma * vector_length_loc +
                                       natom_mod];

                    // Directly accumulate deidrj
                    const SNAcomplex du_product = {
                        dsfacu * ulist_prev.re + sfac * dulist_prev.re,
                        dsfacu * ulist_prev.im + sfac * dulist_prev.im};
                    dedr_sum += du_product.re * ylist_value.re +
                                du_product.im * ylist_value.im;
                }

                Kokkos::atomic_add(
                    &(dedr(natom_mod + vector_length_loc * natom_div, nbor,
                           dir)),
                    2. * dedr_sum);
            });
    }

    /* CPU CODEPATH */

    /* ---------------------------------------------------------------------- */
    void zero_uarraytot() {
        auto policy_InitUlisttot =
            Kokkos::RangePolicy<InitUlisttotTag>(0, num_atoms * idxu_max);
        Kokkos::parallel_for("InitUlisttot", policy_InitUlisttot, *this);

        auto policy_InitUlist = Kokkos::RangePolicy<InitUlistTag>(0, nTotal);
        Kokkos::parallel_for("InitUlist", policy_InitUlist, *this);
    }

    /* ---------------------------------------------------------------------- */
    void addself_uarraytot() {
        auto policy_AddSelfUarraytot =
            Kokkos::RangePolicy<AddSelfUarraytotTag>(0, num_atoms * jdim);
        Kokkos::parallel_for("AddSelfUarraytot", policy_AddSelfUarraytot,
                             *this);
    }

    /* ----------------------------------------------------------------------
       compute Ui by summing over neighbors j
    ------------------------------------------------------------------------- */
    void compute_ui() {
        zero_uarraytot();
        addself_uarraytot();

        int numThreads, numBlocks;
        numThreads = 32;
        numBlocks = nTotal / numThreads + 1;
        auto policy_Ui =
            Kokkos::TeamPolicy<ComputeUiTag>(numBlocks, numThreads);

        Kokkos::parallel_for("ComputeUi", policy_Ui, *this);

        Kokkos::fence();
    }

    /* ----------------------------------------------------------------------
       compute Yi by
    ------------------------------------------------------------------------- */
    void compute_yi() {
        Kokkos::parallel_for(
            "ComputeYiInit",
            Kokkos::RangePolicy<ComputeYiInitTag>(0, num_atoms * idxdu_max),
            *this);

        auto policy_Yi =
            Kokkos::RangePolicy<ComputeYiTag>(0, num_atoms * idxz_max);
        Kokkos::parallel_for("ComputeYi", policy_Yi, *this);

        Kokkos::fence();
    }

    /* ----------------------------------------------------------------------
      compute_duarray
    ------------------------------------------------------------------------- */
    void compute_duarray() {
        auto policy_DUi = Kokkos::RangePolicy<ComputeDUiTag>(0, nTotal);
        Kokkos::parallel_for("ComputeDUi", policy_DUi, *this);

        Kokkos::fence();
    }

    /* ----------------------------------------------------------------------
      compute_deidrj
    ------------------------------------------------------------------------- */
    void compute_deidrj() {
        auto policy_DEi = Kokkos::RangePolicy<ComputeDEiTag>(0, nTotal);
        Kokkos::parallel_for("ComputeDEi", policy_DEi, *this);
        Kokkos::fence();
    }

    /* GPU Codepath */

    /* ----------------------------------------------------------------------
       pre-compute Cayley-Klein parameters (a, b, da, db) and sfac
    ------------------------------------------------------------------------- */
    void compute_cayley_klein_gpu() {
        int neighbors_per_tile = 4;

        using Policy3D = Kokkos::MDRangePolicy<
            Kokkos::IndexType<int>,
            Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
            ComputeCayleyKleinTagGPU>;
        auto policy_ck =
            Policy3D({0, 0, 0}, {vector_length, num_nbor, num_atoms_div},
                     {vector_length, neighbors_per_tile, 1});
        Kokkos::parallel_for("ComputeCayleyKleinGPU", policy_ck, *this);
        Kokkos::fence();
    }

    /* ----------------------------------------------------------------------
       compute Ui by summing over neighbors j
    ------------------------------------------------------------------------- */
    void compute_ui_gpu() {
        {
            using Policy3D = Kokkos::MDRangePolicy<
                Kokkos::IndexType<int>,
                Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                PreUiTagGPU>;
            int jvalues_per_tile = 4;
            auto policy_pre_ui =
                Policy3D({0, 0, 0}, {vector_length, twojmax + 1, num_atoms_div},
                         {vector_length, jvalues_per_tile, 1});
            Kokkos::parallel_for("PreUiGPU", policy_pre_ui, *this);
            Kokkos::fence();
        }

        {
#ifdef KOKKOS_ENABLE_OPENMPTARGET
            int team_size = 32;
#else
            int team_size = 4;
#endif
            int num_teams = num_atoms_div * num_nbor * (twojmax + 1);
            int num_teams_div = (num_teams + team_size - 1) / team_size;

            auto policy_Ui = Kokkos::TeamPolicy<ComputeUiTagGPU>(
                num_teams_div, team_size, vector_length);

            // scratch size = vector_length * (twojmax + 1) for cache
            const int scratch_level = 0;
            const int PerTeamScratch = vector_length * (twojmax + 1);
            int scratch_size =
                ScratchViewType::shmem_size(team_size * PerTeamScratch);
            policy_Ui = policy_Ui.set_scratch_size(scratch_level,
                                                   PerTeam(scratch_size));

            Kokkos::parallel_for("ComputeUiGPU", policy_Ui, *this);
            Kokkos::fence();
        }

        {
            using Policy3D = Kokkos::MDRangePolicy<
                Kokkos::IndexType<int>,
                Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                TransformUiTagGPU>;
            int idxu_per_tile = 4;
            auto policy_transform_ui =
                Policy3D({0, 0, 0}, {vector_length, idxu_max, num_atoms_div},
                         {vector_length, idxu_per_tile, 1});
            Kokkos::parallel_for("TransformUiGPU", policy_transform_ui, *this);
            Kokkos::fence();
        }
    }

    /* ----------------------------------------------------------------------
       compute Yi by
    ------------------------------------------------------------------------- */
    void compute_yi_gpu() {
        int jjz_per_tile = 8;

        using Policy3D = Kokkos::MDRangePolicy<
            Kokkos::IndexType<int>,
            Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
            ComputeYiTagGPU>;
        auto policy_Yi =
            Policy3D({0, 0, 0}, {vector_length, idxz_max, num_atoms_div},
                     {vector_length, jjz_per_tile, 1});
        Kokkos::parallel_for("ComputeYiGPU", policy_Yi, *this);

        Kokkos::fence();
    }

    /* ----------------------------------------------------------------------
      fused compute_duarray, compute_deidrj
    ------------------------------------------------------------------------- */
    void compute_fused_deidrj_gpu() {
#ifdef KOKKOS_ENABLE_OPENMPTARGET
        int team_size = 32;
#else
        int team_size = 2;
#endif
        int num_teams = num_atoms_div * num_nbor * (twojmax + 1);
        int num_teams_div = (num_teams + team_size - 1) / team_size;

        // scratch size = 2 * vector_length * (twojmax + 1) for cache
        const int scratch_level = 0;
        const int PerTeamScratch = 2 * vector_length * (twojmax + 1);
        int scratch_size =
            ScratchViewType::shmem_size(team_size * PerTeamScratch);

        {
            auto policy_fused_deidrj =
                Kokkos::TeamPolicy<ComputeFusedDeiDrjTagGPU<0> >(
                    num_teams_div, team_size, vector_length);
            policy_fused_deidrj = policy_fused_deidrj.set_scratch_size(
                scratch_level, PerTeam(scratch_size));

            Kokkos::parallel_for("ComputeFusedDeiDrjGPU<0>",
                                 policy_fused_deidrj, *this);
        }

        {
            auto policy_fused_deidrj =
                Kokkos::TeamPolicy<ComputeFusedDeiDrjTagGPU<1> >(
                    num_teams_div, team_size, vector_length);
            policy_fused_deidrj = policy_fused_deidrj.set_scratch_size(
                scratch_level, PerTeam(scratch_size));

            Kokkos::parallel_for("ComputeFusedDeiDrjGPU<1>",
                                 policy_fused_deidrj, *this);
        }

        {
            auto policy_fused_deidrj =
                Kokkos::TeamPolicy<ComputeFusedDeiDrjTagGPU<2> >(
                    num_teams_div, team_size, vector_length);
            policy_fused_deidrj = policy_fused_deidrj.set_scratch_size(
                scratch_level, PerTeam(scratch_size));

            Kokkos::parallel_for("ComputeFusedDeiDrjGPU<2>",
                                 policy_fused_deidrj, *this);
        }

        Kokkos::fence();
    }
};

#endif

