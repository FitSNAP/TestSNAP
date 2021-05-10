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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "memory.h"

#if defined(LMP_USER_INTEL) && defined(__INTEL_COMPILER)
#ifndef LMP_INTEL_NO_TBB
#define LMP_USE_TBB_ALLOCATOR
#include "tbb/scalable_allocator.h"
#else
#include <malloc.h>
#endif
#endif

#if defined(LMP_USER_INTEL) && !defined(LAMMPS_MEMALIGN)
#define LAMMPS_MEMALIGN 64
#endif

/* ---------------------------------------------------------------------- */

Memory::Memory() {}

/* ----------------------------------------------------------------------
   safe malloc
------------------------------------------------------------------------- */

void *Memory::smalloc(bigint nbytes, const char *name)
{
  if (nbytes == 0) return NULL;

#if defined(LAMMPS_MEMALIGN)
  void *ptr;

#if defined(LMP_USE_TBB_ALLOCATOR)
  ptr = scalable_aligned_malloc(nbytes, LAMMPS_MEMALIGN);
#else
  int retval = posix_memalign(&ptr, LAMMPS_MEMALIGN, nbytes);
  if (retval) ptr = NULL;
#endif

#else
  void *ptr = malloc(nbytes);
#endif
  if (ptr == NULL) {
    printf("Failed to allocate " BIGINT_FORMAT " bytes for array %s",
            nbytes,name);
    exit(1);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe realloc
------------------------------------------------------------------------- */

void *Memory::srealloc(void *ptr, bigint nbytes, const char *name)
{
  if (nbytes == 0) {
    destroy(ptr);
    return NULL;
  }

#if defined(LMP_USE_TBB_ALLOCATOR)
  ptr = scalable_aligned_realloc(ptr, nbytes, LAMMPS_MEMALIGN);
#elif defined(LMP_INTEL_NO_TBB) && defined(LAMMPS_MEMALIGN)
  ptr = realloc(ptr, nbytes);
  uintptr_t offset = ((uintptr_t)(const void *)(ptr)) % LAMMPS_MEMALIGN;
  if (offset) {
    void *optr = ptr;
    ptr = smalloc(nbytes, name);
    memcpy(ptr, optr, MIN(nbytes,malloc_usable_size(optr)));
    free(optr);
  }
#else
  ptr = realloc(ptr,nbytes);
#endif
  if (ptr == NULL) {
    printf("Failed to reallocate " BIGINT_FORMAT " bytes for array %s",
            nbytes,name);
    exit(1);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe free
------------------------------------------------------------------------- */

void Memory::sfree(void *ptr)
{
  if (ptr == NULL) return;
  #if defined(LMP_USE_TBB_ALLOCATOR)
  scalable_aligned_free(ptr);
  #else
  free(ptr);
  #endif
}

/* ----------------------------------------------------------------------
   erroneous usage of templated create/grow functions
------------------------------------------------------------------------- */

void Memory::fail(const char *name)
{
  printf("Cannot create/grow a vector/array of pointers for %s",name);
  exit(1);
}
