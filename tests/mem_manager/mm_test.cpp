#include "base/mem_manager.hpp"
#include "utils/check.hpp"
#include <cstdint>

using namespace vfem;

static void
testDefaultConstruction ()
{
  Memory<int> m;
  CHECK (m.empty ());
  CHECK_EQ (m.capaCity (), 0);
}

static void
testSizedConstruction ()
{
  Memory<double> m (100);
  CHECK (!m.empty ());
  CHECK_EQ (m.capaCity (), 100);
}

static void
testTypedConstruction ()
{
  Memory<float> m (50, MemType::HOST);
  CHECK (!m.empty ());
  CHECK_EQ (m.capaCity (), 50);
  CHECK_EQ (m.getHostMemType (), MemType::HOST);
}

static void
testDualTypeConstruction ()
{
  Memory<int> m (32, MemType::HOST, MemType::DEVICE);
  CHECK (!m.empty ());
  CHECK_EQ (m.capaCity (), 32);
  CHECK_EQ (m.getHostMemType (), MemType::HOST);
  CHECK_EQ (m.getDeviceMemType (), MemType::DEVICE);
}

static void
testAllocate ()
{
  Memory<std::uint64_t> m;
  m.allocate (200);
  CHECK_EQ (m.capaCity (), 200);
  CHECK (!m.empty ());
}

static void
testAllocateWithType ()
{
  Memory<int> m;
  m.allocate (128, MemType::HOST);
  CHECK_EQ (m.capaCity (), 128);
  CHECK_EQ (m.getHostMemType (), MemType::HOST);
}

static void
testAllocateDualType ()
{
  Memory<double> m;
  m.allocate (64, MemType::HOST_PINNED, MemType::DEVICE);
  CHECK_EQ (m.capaCity (), 64);
  CHECK_EQ (m.getHostMemType (), MemType::HOST_PINNED);
  CHECK_EQ (m.getDeviceMemType (), MemType::DEVICE);
}

static void
testWrapNonOwning ()
{
  int buffer[100];
  Memory<int> m;
  m.wrap (buffer, 100, MemType::HOST, false);
  CHECK_EQ (m.capaCity (), 100);
  CHECK (!m.ownsHostPtr ());
}

static void
testWrapOwning ()
{
  int *buffer = new int[50];
  Memory<int> m;
  m.wrap (buffer, 50, MemType::HOST, true);
  CHECK_EQ (m.capaCity (), 50);
  CHECK (m.ownsHostPtr ());
}

static void
testWrapDualPointer ()
{
  int host_buffer[64];
  Memory<int> m;
  m.wrap (host_buffer, nullptr, 64, MemType::HOST, MemType::DEVICE, false,
          true, false);
  CHECK_EQ (m.capaCity (), 64);
  CHECK (m.hostIsValid ());
}

static void
testMakeAlias ()
{
  Memory<int> base (200);
  Memory<int> alias;
  alias.makeAlias (base, 50, 100);
  CHECK_EQ (alias.capaCity (), 100);
  CHECK (!alias.empty ());
}

static void
testMoveConstructor ()
{
  Memory<double> m1 (100);
  Memory<double> m2 (std::move (m1));
  CHECK (m1.empty ());
  CHECK_EQ (m2.capaCity (), 100);
}

static void
testMoveAssignment ()
{
  Memory<float> m1 (80);
  Memory<float> m2;
  m2 = std::move (m1);
  CHECK (m1.empty ());
  CHECK_EQ (m2.capaCity (), 80);
}

static void
testCopyConstructor ()
{
  Memory<int> m1 (150);
  const Memory<int> &m2 (m1);
  CHECK_EQ (m1.capaCity (), 150);
  CHECK_EQ (m2.capaCity (), 150);
}

static void
testCopyAssignment ()
{
  Memory<std::uint32_t> m1 (64);
  Memory<std::uint32_t> m2;
  m2 = m1;
  CHECK_EQ (m1.capaCity (), 64);
  CHECK_EQ (m2.capaCity (), 64);
}

static void
testSwap ()
{
  Memory<int> m1 (100);
  Memory<int> m2 (200);
  m1.swap (m2);
  CHECK_EQ (m1.capaCity (), 200);
  CHECK_EQ (m2.capaCity (), 100);
}

static void
testReset ()
{
  Memory<double> m (128);
  m.reset ();
  CHECK (m.empty ());
  CHECK_EQ (m.capaCity (), 0);
}

static void
testResetWithType ()
{
  Memory<int> m (50);
  m.reset (MemType::HOST_PINNED);
  CHECK (m.empty ());
  CHECK_EQ (m.getHostMemType (), MemType::HOST_PINNED);
}

static void
testZeroSizeAllocation ()
{
  Memory<float> m;
  m.allocate (0);
  CHECK (m.empty ());
  CHECK_EQ (m.capaCity (), 0);
}

static void
testZeroSizeWrap ()
{
  int *ptr = nullptr;
  Memory<int> m;
  m.wrap (ptr, 0, MemType::HOST, false);
  CHECK (m.empty ());
  CHECK_EQ (m.capaCity (), 0);
}

static void
testMemoryManagerSingleton ()
{
  MemoryManager &mm1 = MemoryManager::get ();
  MemoryManager &mm2 = MemoryManager::get ();
  CHECK_EQ (&mm1, &mm2);
}

static void
testUseDevice ()
{
  Memory<int> m (64, MemType::HOST, MemType::DEVICE);
  CHECK (!m.useDevice ());
  m.useDevice (true);
  CHECK (m.useDevice ());
  m.useDevice (false);
  CHECK (!m.useDevice ());
}

static void
testRelease ()
{
  Memory<double> m (256);
  m.release ();
  CHECK (m.empty ());
  CHECK_EQ (m.capaCity (), 0);
}

static void
testAliasFromOffset ()
{
  Memory<std::uint64_t> base (500);
  Memory<std::uint64_t> alias;
  alias.makeAlias (base, 100, 200);
  CHECK_EQ (alias.capaCity (), 200);
}

static void
testRepeatedAllocation ()
{
  Memory<int> m (50);
  CHECK_EQ (m.capaCity (), 50);
  m.allocate (100);
  CHECK_EQ (m.capaCity (), 100);
  m.allocate (25);
  CHECK_EQ (m.capaCity (), 25);
}

static void
testMemoryTypes ()
{
  Memory<int> m (32, MemType::HOST_32, MemType::DEVICE_DEBUG);
  CHECK_EQ (m.getHostMemType (), MemType::HOST_32);
  CHECK_EQ (m.getDeviceMemType (), MemType::DEVICE_DEBUG);
}

int
main ()
{
  testDefaultConstruction ();
  testSizedConstruction ();
  testTypedConstruction ();
  testDualTypeConstruction ();
  testAllocate ();
  testAllocateWithType ();
  testAllocateDualType ();
  // testWrapNonOwning ();
  // testWrapOwning ();
  // testWrapDualPointer ();
  testMakeAlias ();
  testMoveConstructor ();
  testMoveAssignment ();
  testCopyConstructor ();
  testCopyAssignment ();
  testSwap ();
  testReset ();
  testResetWithType ();
  testZeroSizeAllocation ();
  testZeroSizeWrap ();
  testMemoryManagerSingleton ();
  testUseDevice ();
  testRelease ();
  testAliasFromOffset ();
  testRepeatedAllocation ();
  testMemoryTypes ();
  return 0;
}
