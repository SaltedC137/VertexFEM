#ifndef VFEM_CUDA_HPP
#define VFEM_CUDA_HPP

#include "config.hpp"
#include "error.hpp"

#if defined(__CUDACC__)
#define VFEM_DEVICE __device__
#define VFEM_HOST __host__

#endif

#define VFEM_GPU_CHECK(X)                                                     \
  do                                                                          \
    {                                                                         \
      cudaError_t err = (X);                                                  \
      if (err != cudaSuccess)                                                 \
        {                                                                     \
          vfemError ("CUDA error: %s (%d) at %s:%d",                          \
                     cudaGetErrorString (err), err, __FILE__, __LINE__);      \
        }                                                                     \
    }                                                                         \
  while (0)

#endif