#ifndef VFEM_INTERNAL_CONFIG_HPP
#define VFEM_INTERNAL_CONFIG_HPP

#if defined(VFEM_USE_CUDA) || defined(VFEM_USE_HIP)
#define VFEM_USE_CUDA_OR_HIP
#endif

#if (defined(VFEM_USE_CUDA) && defined(__CUDACC__)) ||                         \
    (defined(VFEM_USE_HIP) && (defined(__HIPCC__) || defined(__HIP__)))
#define VFEM_USE_CUDA_OR_HIP_LANG
#define VFEM_HOST_DEVICE __host__ __device__
#define VFEM_DEVICE __device__
#define VFEM_HOST __host__
#define VFEM_LAMBDA __host__ __device__
#define VFEM_LAUNCH_BOUNDS(...) __launch_bounds__(__VA_ARGS__)
#else
#define VFEM_HOST_DEVICE
#define VFEM_DEVICE
#define VFEM_HOST
#define VFEM_LAMBDA
#define VFEM_LAUNCH_BOUNDS(...)
#endif

#endif // VFEM_INTERNAL_CONFIG_HPP