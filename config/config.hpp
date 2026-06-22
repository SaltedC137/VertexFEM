#pragma once

#include "cconfig.hpp"

namespace vfem
{

#if defined VFEM_USE_DOUBLE && defined VFEM_USE_SINGLE
#error "Cannot define both VFEM_USE_DOUBLE and VFEM_USE_SINGLE"
#endif
#ifdef VFEM_USE_DOUBLE
using real_t = double;
#elif defined VFEM_USE_SINGLE
using real_t = float;
#else
#error "Must define either VFEM_USE_DOUBLE or VFEM_USE_SINGLE"
#endif

VFEM_HOST_DEVICE consteval real_t
operator""_rt (long double val)
{
  return static_cast<real_t> (val);
}

VFEM_HOST_DEVICE consteval real_t
operator""_rt (unsigned long long val)
{
  return static_cast<real_t> (val);
}

inline constexpr int VFEM_SKIP_RETURN_VALUE = 242;

} // namespace vfem

#define VFEM_THREAD_LOCAL thread_local

#define VFEM_DEPRECATED                                                       \
  [[deprecated (                                                              \
      "This function is deprecated and may be removed in future versions.")]]

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))                  \
    || defined(__clang__)

#define VFEM_HAVE_GCC_PRAGMA_DIGNOSTIC
#endif

#if defined(_WIN32) && !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

#if defined(_MSC_VER) && defined(VFEM_SHARED_LIB)
#ifdef Vfem_EXPORTS
#define VFEM_EXPORT __declspec (dllexport)
#else
#define VFEM_EXPORT __declspec (dllimport)
#endif
#else
#define VFEM_EXPORT
#endif

#ifdef VFEM_USE_MPI
#ifdef VFEM_USE_SINGLE
#define VFEM_MPI_REAL_T MPI_FLOAT
#elif defined VFEM_USE_DOUBLE
#define VFEM_MPI_REAL_T MPI_DOUBLE
#endif
#endif

// Check for incompatible options

#ifndef VFEM_USE_MPI
#ifdef VFEM_USE_SUPERLU
#error Building with SuperLU_DIST (VFEM_USE_SUPERLU=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#ifdef VFEM_USE_MUMPS
#error Building with MUMPS (VFEM_USE_MUMPS=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#ifdef VFEM_USE_CPARPACK
#error Building with CPARPACK (VFEM_USE_CPARPACK=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#ifdef VFEM_USE_PETSC
#error Building with PETSc (VFEM_USE_PETSC=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#ifdef VFEM_USE_SLEPC
#error Building with SLEPc (VFEM_USE_SLEPC=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#ifdef VFEM_USE_PUMI
#error Building with PUMI (VFEM_USE_PUMI=YES) requires MPI (VFEM_USE_MPI=YES)
#endif
#endif