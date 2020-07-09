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
/* ----------------------------------------------------------------
   Class library implementing simple array structures similar to Fortran,
   except since this is c++ the fast running index are the rightmost
   ( For this "GPU" version the array elements are accessed in column-major
   format. )

   These are the necessary set for the snap demo code, and may not be
   completely defined for all combinations of sub-dimensional references
   and copy constructors.

   2D through 4D arrays are supported,
       ArrayxD<type> newarray;
   The default constructor initializes bounds, but not memory

   The '()' operator is overlaaded here to access array elements.

   Copy constructors are provided for openmp private/firstprivate.
   In this case, the data pointer must be allocated with resize or
   assignment as above.

   When using the GPU, the data pointers must be set up another way,
   presumably with "#pragma enter data map(alloc..." and a deep copy
   of the pointer.
   ----------------------------------------------------------------*/
#ifndef LMP_ARRAYMD_H
#define LMP_ARRAYMD_H

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

template<typename T>
class Array1D
{
public:
  unsigned n1;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1) { return dptr[i1]; }

  Array1D()
  {
    n1 = size = 0;
    dptr = NULL;
  }

  Array1D(const Array1D& p)
  {
    n1 = p.n1;
    size = 0;
    dptr = p.dptr;
  }

  Array1D(int in1)
  {
    n1 = in1;
    size = n1;
    dptr = new T[size];
  }

  ~Array1D()
  {
    if (size && dptr)
      delete[] dptr;
  }

  void resize(unsigned in1)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    size = n1;
    dptr = new T[size];
  }
};

template<typename T>
struct Array2D
{
  unsigned n1, n2, b1;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2)
  {
    return dptr[i1 + (n1 * i2)];
  }

  Array2D()
  {
    n1 = n2 = size = 0;
    dptr = NULL;
  }

  Array2D(const Array2D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }

  Array2D(int in1, int in2)
  {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = new T[size];
  }

  ~Array2D()
  {
    if (size && dptr)
      delete[] dptr;
  }

  void resize(unsigned in1, unsigned in2)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = new T[size];
  }
};

template<typename T>
struct Array3D
{
  unsigned n1, n2, n3;
  unsigned size;
  T* dptr;
  inline T& operator()(unsigned i1, unsigned i2, unsigned i3)
  {
    return dptr[i1 + i2 * n1 + i3 * n2 * n1];
  }

  Array3D()
  {
    n1 = n2 = n3 = size = f2 = f3 = 0;
    dptr = NULL;
  }

  Array3D(const Array3D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    f2 = n1;
    f3 = f2 * n2;
    dptr = p.dptr;
  }

  Array3D(unsigned in1, unsigned in2, unsigned in3)
  {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n1;
    f3 = f2 * n2;

    dptr = new T[size];
  }
  ~Array3D()
  {
    if (size && dptr)
      delete[] dptr;
  }

  void resize(unsigned in1, unsigned in2, unsigned in3)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n1;
    f3 = f2 * n2;

    dptr = new T[size];
  }

private:
  unsigned f2, f3;
};

template<typename T>
struct Array4D
{
  unsigned n1, n2, n3, n4;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2, unsigned i3, unsigned i4)
  {
    return dptr[i4 * n1 * n2 * n3 + i3 * n1 * n2 + i2 * n1 + i1];
  }

  Array4D()
  {
    n1 = n2 = n3 = n4 = size = f2 = f3 = f4 = 0;
    dptr = NULL;
  }


  Array4D(const Array4D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    n4 = p.n4;
    size = 0;
    dptr = p.dptr;
    f2 = n1;
    f3 = f2 * n2;
    f4 = f3 * n3;
  }

  Array4D(unsigned in1, unsigned in2, unsigned in3, unsigned in4)
  {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    n4 = in4;
    size = n1 * n2 * n3 * n4;
    f2 = n1;
    f3 = f2 * n2;
    f4 = f3 * n3;
    dptr = new T[size];
  }

  ~Array4D()
  {
    if (size && dptr)
      delete[] dptr;
  }

  void resize(unsigned in1, unsigned in2, unsigned in3, unsigned in4)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    n2 = in2;
    n3 = in3;
    n4 = in4;
    size = n1 * n2 * n3 * n4;
    f2 = n1;
    f3 = f2 * n2;
    f4 = f3 * n3;

    dptr = new T[size];
  }

private:
  unsigned f2, f3, f4;
};

#endif
