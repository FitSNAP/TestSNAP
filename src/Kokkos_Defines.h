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
// Main changes: GPU AoSoA data layout, optimized recursive polynomial evaluation
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

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
#define SNAP_ENABLE_GPU
#endif

typedef double SNADOUBLE;
struct alignas(2 * sizeof(SNADOUBLE)) SNAcomplex
{
  SNADOUBLE re, im;

  KOKKOS_INLINE_FUNCTION
  SNAcomplex()
    : re(0)
    , im(0)
  {
    ;
  }

  KOKKOS_INLINE_FUNCTION
  SNAcomplex(SNADOUBLE real_in, SNADOUBLE imag_in)
    : re(real_in)
    , im(imag_in)
  {
    ;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const SNAcomplex src)
  {
    re = src.re;
    im = src.im;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const SNAcomplex src)
  {
    re += src.re;
    im += src.im;
  }

  KOKKOS_INLINE_FUNCTION
  void operator*=(const SNAcomplex src)
  {
    re += src.re;
    im += src.im;
  }
};

// Struct used to "unfold" ulisttot on the gpu
struct alignas(8) FullHalfMap {
  int idxu_half;
  int flip_sign; // 0 -> isn't flipped, 1 -> flip sign of imaginary, -1 -> flip sign of real
};

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using Layout = ExecSpace::array_layout;
using MemSpace = ExecSpace::memory_space;

using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::View;
using team_policy = Kokkos::TeamPolicy<ExecSpace>;
using member_type = team_policy::member_type;
using Kokkos::PerTeam;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;

// 1D,2D,3D views for int's
using int_View1D = View<int*, Layout, MemSpace>;
using int_View2D = View<int**, Layout, MemSpace>;
using int_View3D = View<int***, Layout, MemSpace>;
using int_View2DR = View<int**, Kokkos::LayoutRight, MemSpace>;
// Host-View mirrors for int's
using HostInt_View1D = int_View1D::HostMirror;
using HostInt_View2D = int_View2D::HostMirror;
using HostInt_View3D = int_View3D::HostMirror;
using HostInt_View2DR = int_View2DR::HostMirror;

// 1D,2D,3D views for double's
using double_View1D = View<double*, Layout, MemSpace>;
using double_View2D = View<double**, Layout, MemSpace>;
using double_View3D = View<double***, Layout, MemSpace>;
using double_View1DL = View<double*, Kokkos::LayoutLeft, MemSpace>;
using double_View2DL = View<double**, Kokkos::LayoutLeft, MemSpace>;
using double_View3DL = View<double***, Kokkos::LayoutLeft, MemSpace>;
using double_View4DL = View<double****, Kokkos::LayoutLeft, MemSpace>;

// Host-View mirrors for double's
using HostDouble_View1D = double_View1D::HostMirror;
using HostDouble_View2D = double_View2D::HostMirror;
using HostDouble_View3D = double_View3D::HostMirror;

// 1D,2D,3D views for SNAcomplex's
using SNAcomplex_View2D = View<SNAcomplex**, Layout, MemSpace>;
using SNAcomplex_View3D = View<SNAcomplex***, Layout, MemSpace>;
using SNAcomplex_View4D = View<SNAcomplex****, Layout, MemSpace>;
using SNAcomplex_View2DR = View<SNAcomplex**, Kokkos::LayoutRight, MemSpace>;
using SNAcomplex_View3DR = View<SNAcomplex***, Kokkos::LayoutRight, MemSpace>;
using SNAcomplex_View4DR = View<SNAcomplex****, Kokkos::LayoutRight, MemSpace>;
using SNAcomplex_View2DL = View<SNAcomplex**, Kokkos::LayoutLeft, MemSpace>;
using SNAcomplex_View3DL = View<SNAcomplex***, Kokkos::LayoutLeft, MemSpace>;
using SNAcomplex_View4DL = View<SNAcomplex****, Kokkos::LayoutLeft, MemSpace>;

// 1D view for mapping from compressed ulisttot -> full
using FullHalfMap_View1D = View<FullHalfMap*, Layout, MemSpace>;

// Host-View mirrors
using HostFullHalfMap_View1D = FullHalfMap_View1D::HostMirror;

// scratch memory views for SNAcomplex
using ScratchViewType = View<SNAcomplex *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
