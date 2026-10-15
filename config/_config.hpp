// MIX
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define VFEM_HOST_DEVICE __host__ __device__
#define VFEM_DEVICE __device__
#define VFEM_HOST __host__
#else
#define VFEM_HOST_DEVICE
#define VFEM_DEVICE
#define VFEM_HOST
#endif

// SYCL
#if defined(__SYCL_DEVICE_ONLY__)
#define VFEM_HOST_DEVICE [[sycl::device]]
#define VFEM_DEVICE [[sycl::device]]
#define VFEM_HOST [[sycl::host]]

// CPU
#else
#define VFEM_HOST_DEVICE
#define VFEM_DEVICE
#define VFEM_HOST
#endif
