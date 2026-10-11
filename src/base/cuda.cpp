
#include "cuda.hpp"
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




}
}

#endif
