
#include "base/cuda.hpp"
#include "base/error.hpp"
#include "utils/check.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__SANITIZE_ADDRESS__)                                             \
    || (defined(__has_feature) && __has_feature(address_sanitizer))
extern "C" const char *
// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
__lsan_default_suppressions ()
{
  return "leak:libcuda.so\n"
         "leak:libcudart.so\n";
}
#endif

using namespace vfem;

namespace
{

class ErrorActionGuard
{
  ErrorAction previous{ getErrorAction () };

public:
  explicit ErrorActionGuard (ErrorAction action) { setErrorAction (action); }
  ~ErrorActionGuard () { setErrorAction (previous); }
  ErrorActionGuard (const ErrorActionGuard &)            = delete;
  ErrorActionGuard &operator= (const ErrorActionGuard &) = delete;
};

template <typename Callback>
void
expectRuntimeError (Callback callback)
{
  bool caught = false;
  try
    {
      callback ();
    }
  catch (const std::runtime_error &)
    {
      caught = true;
    }
  CHECK (caught) << "Expected std::runtime_error";
}

bool
isAligned (const void *ptr, std::size_t alignment)
{
  return reinterpret_cast<std::uintptr_t> (ptr) % alignment == 0;
}

// ---- device-independent tests ------------------------------------------

void
testZeroSizeAndNullCopy ()
{
  ErrorActionGuard ignore{ ErrorAction::Ignore };
  std::vector<char> host (64, 0x2a);

  // zero-byte copies succeed without touching the pointers
  CHECK (cudaCopyHtoD (nullptr, nullptr, 0));
  CHECK (cudaCopyDtoH (nullptr, nullptr, 0));
  CHECK (cudaCopyDtoD (nullptr, nullptr, 0));

  // null pointers with nonzero size are rejected
  CHECK (!cudaCopyHtoD (nullptr, host.data (), host.size ()));
  CHECK (!cudaCopyHtoD (host.data (), nullptr, host.size ()));
  CHECK (!cudaCopyDtoH (nullptr, host.data (), host.size ()));
}

void
testAlignmentValidation ()
{
  ErrorActionGuard ignore{ ErrorAction::Ignore };

  // alignment must be a power of two and at most 256
  CHECK (cudaAllocDev (1024, 3) == nullptr);
  CHECK (cudaAllocDev (1024, 0) == nullptr);
  CHECK (cudaAllocDev (1024, 512) == nullptr);
  CHECK (cudaAllocPin (1024, 1024) == nullptr);
}

void
testErrorActionThrow ()
{
  ErrorActionGuard throw_guard{ ErrorAction::Throw };
  expectRuntimeError ([] { (void)cudaAllocDev (1024, 3); });
}

// ---- device tests -------------------------------------------------------

void
testAllocationAlignment (int device)
{
  CHECK (vfem::cudaSetDevice (device));

  // zero bytes yields a null pointer, not an error
  CHECK (cudaAllocDev (0) == nullptr);
  CHECK (cudaAllocMan (0) == nullptr);
  CHECK (cudaAllocPin (0) == nullptr);

  void *dev = cudaAllocDev (1024);
  CHECK (dev != nullptr);
  CHECK (isAligned (dev, VFEM_CUDA_BLOCKS));
  cudaFreeDevice (dev);

  void *man = cudaAllocMan (1024);
  CHECK (man != nullptr);
  CHECK (isAligned (man, VFEM_CUDA_BLOCKS));
  cudaFreeDevice (man); // managed memory is released with cudaFree

  void *pin = cudaAllocPin (1024);
  CHECK (pin != nullptr);
  CHECK (isAligned (pin, 64));
  cudaFreePinned (pin);

  // double free of null is a no-op
  cudaFreeDevice (nullptr);
  cudaFreePinned (nullptr);
}

void
testHostDeviceRoundTrip ()
{
  constexpr std::size_t count = 256;
  std::vector<float> input (count);
  for (std::size_t i = 0; i < count; ++i)
    {
      input[i] = static_cast<float> (i) * 0.5f;
    }

  auto dev = makeCudaDeviceMemory (input.size () * sizeof (float));
  CHECK (dev.get () != nullptr);

  CHECK (cudaCopyHtoD (dev.get (), input.data (),
                       input.size () * sizeof (float)));

  std::vector<float> output (count, -1.0f);
  CHECK (cudaCopyDtoH (output.data (), dev.get (),
                       output.size () * sizeof (float)));

  CHECK (std::memcmp (input.data (), output.data (),
                      input.size () * sizeof (float))
         == 0);
}

void
testDeviceToDeviceCopy ()
{
  constexpr std::size_t bytes = 128;
  auto src                    = makeCudaDeviceMemory (bytes);
  auto dst                    = makeCudaDeviceMemory (bytes);
  CHECK (src.get () != nullptr);
  CHECK (dst.get () != nullptr);

  std::vector<char> host (bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    {
      host[i] = static_cast<char> (i);
    }
  CHECK (cudaCopyHtoD (src.get (), host.data (), bytes));
  CHECK (cudaCopyDtoD (dst.get (), src.get (), bytes));

  std::vector<char> output (bytes, 0);
  CHECK (cudaCopyDtoH (output.data (), dst.get (), bytes));
  CHECK (std::memcmp (host.data (), output.data (), bytes) == 0);

  // overlapping device ranges are rejected
  ErrorActionGuard ignore{ ErrorAction::Ignore };
  CHECK (
      !cudaCopyDtoD (static_cast<char *> (src.get ()) + 8, src.get (), bytes));
}

void
testManagedMemoryHostAccess ()
{
  auto man = makeCudaManagedMemory (256);
  CHECK (man.get () != nullptr);

  // managed memory is directly readable and writable from the host
  std::memset (man.get (), 0x5a, 256);
  const auto *bytes = static_cast<const unsigned char *> (man.get ());
  CHECK (bytes[0] == 0x5a);
  CHECK (bytes[255] == 0x5a);
}

void
testPinnedMemory ()
{
  auto pin = makeCudaPinnedMemory (256, 64);
  CHECK (pin.get () != nullptr);
  CHECK (isAligned (pin.get (), 64));
  std::memset (pin.get (), 0, 256);
}

void
testSynchronization ()
{
  CHECK (cudaStreamSync ());
  CHECK (cudaDeviceSync ());
}

} // namespace

int
main ()
{
  // device-independent tests always run
  testZeroSizeAndNullCopy ();
  testAlignmentValidation ();
  testErrorActionThrow ();

  int count = 0;
  {
    ErrorActionGuard ignore{ ErrorAction::Ignore };
    if (!cudaDeviceCount (count) || count == 0)
      {
        std::printf ("cuda_test: no CUDA device present, "
                     "device-dependent tests skipped\n");
        return 0;
      }
  }

  testAllocationAlignment (0);
  testHostDeviceRoundTrip ();
  testDeviceToDeviceCopy ();
  testManagedMemoryHostAccess ();
  testPinnedMemory ();
  testSynchronization ();

  std::printf ("cuda_test passed (%d device%s)\n", count,
               count == 1 ? "" : "s");
  return 0;
}
