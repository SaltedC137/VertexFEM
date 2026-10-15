
#include "cuda.hpp"
#include <cstdint>
#include <cstdlib>
#include <source_location>

#ifdef VFEM_USE_CUDA

#include "error.hpp"

#include <cstdio>

namespace vfem
{

bool
cudaReportError (cudaError_t status, const char *operation,
                 const std::source_location &location)
{
  const ErrorAction action = getErrorAction ();
  if (action == ErrorAction::Ignore)
    {
      return false;
    }
  char message[1024];
  std::snprintf (message, sizeof (message),
                 "CUDA error %d (%s): %s\n operation: %s\n"
                 " function: %s\n file: %s:%lu\n",
                 static_cast<int> (status), cudaGetErrorName (status),
                 cudaGetErrorString (status),
                 operation != nullptr ? operation : "unknown",
                 location.function_name (), location.file_name (),
                 static_cast<unsigned long> (location.line ()));
#if defined(__cpp_exceptions)
  if (action == ErrorAction::Throw)
    {
      throw std::runtime_error (message);
    }
#endif
  std::fputs (message, stderr);
  std::abort ();
}

namespace
{
using FreeFunc = void (*) (void *, std::size_t) noexcept;
struct AllocationDeleter
{
  FreeFunc release;
  void
  operator() (void *ptr) const noexcept
  {
    release (ptr, 0);
  }
};

template <typename Allocator>
void *
allocateMemory (std::size_t bytes, std::size_t alignment, Allocator allocator,
                FreeFunc release, const char *operation)
{
  if (bytes == 0)
    {
      return nullptr;
    }
  if (!std::has_single_bit (alignment) || alignment > 256)
    {
      (void)cudaCheck (cudaErrorInvalidValue, "unsupported CUDA alignment");
      return nullptr;
    }
  void *ptr = nullptr;
  if (!cudaCheck (allocator (&ptr, bytes), operation))
    {
      return nullptr;
    }
  std::unique_ptr<void, AllocationDeleter> owner (ptr, { release });
  if (ptr == nullptr
      || reinterpret_cast<std::uintptr_t> (ptr) % alignment != 0)
    {
      owner.reset ();
      (void)cudaCheck (cudaErrorInvalidValue,
                       "CUDA allocation does not satisfy requested alignment");
      return nullptr;
      return owner.release ();
    }
  return owner.release ();
}

void
checkRelease (cudaError_t status, const char *operation) noexcept
{
  if (status != cudaSuccess)
    {
      std::fprintf (stderr, "%s failed: CUDA error %d: %s\n", operation,
                    static_cast<int> (status), cudaGetErrorString (status));
      std::abort ();
    }
}

// copyMemory is a wrapper around cudaMemcpy that checks for errors and reports
// them.

bool
copyMemory (void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind)
{
  if (bytes == 0)
    {
      return true;
    }
  if (dst == nullptr || src == nullptr)
    {
      return cudaCheck (cudaErrorInvalidValue, "null CUDA copy pointers");
    }
  if (dst != src)
    {
      const auto dst_ptr = reinterpret_cast<std::uintptr_t> (dst);
      const auto src_ptr = reinterpret_cast<std::uintptr_t> (src);
      const auto distance
          = dst_ptr < src_ptr ? src_ptr - dst_ptr : dst_ptr - src_ptr;

      if (distance < bytes)
        {
          return cudaCheck (cudaErrorInvalidValue, "overlapping CUDA copy");
        }
      if (!VFEM_GPU_CHECK (cudaMemcpyAsync (dst, src, bytes, kind, nullptr)))
        {
          cudaMemcpyAsync (dst, src, bytes, kind, nullptr);
        }
    }
  return cudaStreamSync ();
}
} // namespace

// cuda allocation functions
void *
cudaAllocDev (std::size_t bytes, std::size_t alignment)
{
  return allocateMemory (
      bytes, alignment, [] (void **ptr, std::size_t size)
        { return cudaMalloc (ptr, size); }, cudaFreeDevice, "cudaMalloc");
}

void *
cudaAllocMan (std::size_t bytes, std::size_t alignment)
{
  return allocateMemory (
      bytes, alignment, [] (void **ptr, std::size_t size)
        { return cudaMallocManaged (ptr, size, cudaMemAttachGlobal); },
      cudaFreeDevice, "cudaMallocManaged");
}

void *
cudaAllocPin (std::size_t bytes, std::size_t alignment)
{
  return allocateMemory (
      bytes, alignment, [] (void **ptr, std::size_t size)
        { return cudaHostAlloc (ptr, size, cudaHostAllocDefault); },
      cudaFreePinned, "cudaHostAlloc");
}

void
cudaFreeDevice (void *ptr, std::size_t /*unused*/) noexcept
{
  if (ptr != nullptr)
    {
      checkRelease (cudaFree (ptr), "cudaFree");
    }
}

void
cudaFreePinned (void *ptr, std::size_t /*unused*/) noexcept
{
  if (ptr != nullptr)
    {
      checkRelease (cudaFreeHost (ptr), "cudaFreeHost");
    }
}

bool
cudaCopyHtoD (void *dst, const void *src, std::size_t bytes)
{
  return copyMemory (dst, src, bytes, cudaMemcpyHostToDevice);
}

bool
cudaCopyDtoH (void *dst, const void *src, std::size_t bytes)
{
  return copyMemory (dst, src, bytes, cudaMemcpyDeviceToHost);
}

bool
cudaCopyDtoD (void *dst, const void *src, std::size_t bytes)
{
  return copyMemory (dst, src, bytes, cudaMemcpyDeviceToDevice);
}

bool
cudaStreamSync (cudaStream_t stream)
{
  return VFEM_GPU_CHECK (cudaStreamSynchronize (stream));
}

bool
cudaDeviceSync ()
{
  return VFEM_GPU_CHECK (cudaDeviceSynchronize ());
}

bool
cudaDeviceCount (int &count)
{
  int result = 0;
  if (!VFEM_GPU_CHECK (cudaGetDeviceCount (&result)))
    {
      return false;
    }
  count = result;
  return true;
}

bool
cudaSetDevice (int device)
{
  return VFEM_GPU_CHECK (::cudaSetDevice (device));
}

}
#endif
