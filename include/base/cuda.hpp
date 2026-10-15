#ifndef VFEM_CUDA_HPP
#define VFEM_CUDA_HPP

#include "config.hpp" // IWYU pragma: keep

#ifdef VFEM_USE_CUDA
#include <cuda_runtime_api.h>
#include <memory>
#include <source_location>

#if defined(__CUDACC__)
#define VFEM_DEVICE __device__
#define VFEM_HOST __host__
#endif

#define VFEM_CUDA_BLOCKS 256

#define VFEM_GPU_CHECK_(x)                                                    \
  do                                                                          \
    {                                                                         \
      cudaError_t vfem_err_internal_var_name = (x);                           \
      if (vfem_err_internal_var_name != cudaSuccess)                          \
        {                                                                     \
          ::vfem::vfem_cuda_error (vfem_err_internal_var_name, #x,            \
                                   _VFEM_FUNC_NAME, __FILE__, __LINE__);      \
        }                                                                     \
    }                                                                         \
  while (0)

namespace vfem
{

VFEM_EXPORT bool cudaReportError (cudaError_t status, const char *operation,
                                  const std::source_location &location);

[[nodiscard]] inline bool
cudaCheck (cudaError_t status, const char *operation,
           const std::source_location &location
           = std::source_location::current ())
{
  return status == cudaSuccess
         || cudaReportError (status, operation, location);
}

[[nodiscard]] VFEM_EXPORT void *
cudaAllocDev (std::size_t bytes, std::size_t alignment = VFEM_CUDA_BLOCKS);

[[nodiscard]] VFEM_EXPORT void *
cudaAllocMan (std::size_t bytes, std::size_t alignment = VFEM_CUDA_BLOCKS);

[[nodiscard]] VFEM_EXPORT void *cudaAllocPin (std::size_t bytes,
                                              std::size_t alignment = 64);

// Deallocate functions for CUDA memory types
VFEM_EXPORT void cudaFreeDevice (void *ptr,
                                 std::size_t alignment = 0) noexcept;
VFEM_EXPORT void cudaFreePinned (void *ptr,
                                 std::size_t alignment = 0) noexcept;

[[nodiscard]] VFEM_EXPORT bool cudaCopyHtoD (void *dst, const void *src,
                                             std::size_t bytes);
[[nodiscard]] VFEM_EXPORT bool cudaCopyDtoH (void *dst, const void *src,
                                             std::size_t bytes);
[[nodiscard]] VFEM_EXPORT bool cudaCopyDtoD (void *dst, const void *src,
                                             std::size_t bytes);

[[nodiscard]] VFEM_EXPORT bool cudaStreamSync (cudaStream_t stream = nullptr);
[[nodiscard]] VFEM_EXPORT bool cudaDeviceSync ();
[[nodiscard]] VFEM_EXPORT bool cudaDeviceCount (int &count);
[[nodiscard]] VFEM_EXPORT bool cudaSetDevice (int device);

struct CudaDeviceDeleter
{
  void
  operator() (void *ptr) const noexcept
  {
    cudaFreeDevice (ptr);
  }
};

struct CudaPinnedDeleter
{
  void
  operator() (void *ptr) const noexcept
  {
    cudaFreePinned (ptr);
  }
};

using cudaDeviceMemory = std::unique_ptr<void, CudaDeviceDeleter>;
using cudaPinnedMemory = std::unique_ptr<void, CudaPinnedDeleter>;

[[nodiscard]] inline cudaDeviceMemory
makeCudaDeviceMemory (std::size_t bytes, std::size_t alignment = 256)
{
  return cudaDeviceMemory (cudaAllocDev (bytes, alignment));
}

[[nodiscard]] inline cudaDeviceMemory
makeCudaManagedMemory (std::size_t bytes, std::size_t alignment = 256)
{
  return cudaDeviceMemory (cudaAllocMan (bytes, alignment));
}

[[nodiscard]] inline cudaPinnedMemory
makeCudaPinnedMemory (std::size_t bytes, std::size_t alignment = 64)
{
  return cudaPinnedMemory (cudaAllocPin (bytes, alignment));
}
}

#endif

#define VFEM_GPU_CHECK(expr)                                                  \
  (::vfem::cudaCheck ((expr), #expr, std::source_location::current ()))

#endif