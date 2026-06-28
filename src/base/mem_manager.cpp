
#include "base/mem_manager.hpp"

#ifdef _WIN32
#include <signal.h>
#include <unistd.h>

// Windows provides _aligned_malloc and _aligned_free, which are similar to
// posix_memalign and free, respectively. We can define our own macros to use
// these functions for aligned memory allocation and deallocation on Windows.
#define vfem_memalign(p, a, s) posix_memalign (p, a, s)
#define vfem_aligned_free free
#else
#define vfem_memalign(p, a, s)                                                \
  (((*(p)) = _aligned_malloc ((s), (a))), *(p) ? 0 : errno)
#define vfem_aligned_free _aligned_free
#endif

namespace vfem
{

static MemType
GetMemType (MemoryClass mc)
{
  switch (mc)
    {
    case MemoryClass::HOST:
      return MemType::HOST;

    case MemoryClass::HOST_32:
      return MemType::HOST_32;

    case MemoryClass::HOST_64:
      return MemType::HOST_64;

    case MemoryClass::DEVICE:
      return MemType::DEVICE;

    case MemoryClass::MANAGED:
      return MemType::MANAGED;
    }




}
}
