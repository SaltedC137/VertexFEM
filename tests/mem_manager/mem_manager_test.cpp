#include "base/mem_manager.hpp"
#include "utils/check.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <utility>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace vfem;

namespace
{

struct BackendCounters
{
  int allocations{};
  int deallocations{};
};

std::array<BackendCounters, MemTypeSize> backend_counters;
auto &host_counters = backend_counters[static_cast<int> (MemType::HOST_DEBUG)];
auto &device_counters
    = backend_counters[static_cast<int> (MemType::DEVICE_DEBUG)];
auto &managed_counters = backend_counters[static_cast<int> (MemType::MANAGED)];
int host_to_device_copies{};
int device_to_host_copies{};
std::size_t last_copy_bytes{};
bool fail_allocation{};
bool fail_host_to_device{};
bool fail_device_to_host{};

class ErrorActionGuard
{
  ErrorAction previous{ getErrorAction () };

public:
  explicit ErrorActionGuard (ErrorAction action) { setErrorAction (action); }
  ~ErrorActionGuard () { setErrorAction (previous); }
  ErrorActionGuard (const ErrorActionGuard &) = delete;
  ErrorActionGuard &operator= (const ErrorActionGuard &) = delete;
};

class FailureGuard
{
  bool &flag;

public:
  explicit FailureGuard (bool &failure) : flag (failure) { flag = true; }
  ~FailureGuard () { flag = false; }
  FailureGuard (const FailureGuard &) = delete;
  FailureGuard &operator= (const FailureGuard &) = delete;
};

template <typename Callback>
void
expectError (Callback callback)
{
  bool caught = false;
  try
    {
      callback ();
    }
  catch (const ErrorException &)
    {
      caught = true;
    }
  CHECK (caught) << "Expected vfem::ErrorException";
}

template <MemType Type>
void *
allocateBuffer (std::size_t bytes, std::size_t alignment)
{
  if (Type == MemType::HOST_DEBUG && fail_allocation)
    {
      return nullptr;
    }
  void *ptr = ::operator new (bytes, std::align_val_t{ alignment });
  ++backend_counters[static_cast<std::size_t> (Type)].allocations;
  return ptr;
}

template <MemType Type>
void
deallocateBuffer (void *ptr, std::size_t alignment) noexcept
{
  ++backend_counters[static_cast<std::size_t> (Type)].deallocations;
  ::operator delete (ptr, std::align_val_t{ alignment });
}

void
copyToDevice (void *dst, const void *src, std::size_t bytes)
{
  if (fail_host_to_device)
    {
      throw ErrorException ("test host-to-device transfer failure");
    }
  ++host_to_device_copies;
  last_copy_bytes = bytes;
  std::memmove (dst, src, bytes);
}

void
copyToHost (void *dst, const void *src, std::size_t bytes)
{
  if (fail_device_to_host)
    {
      throw ErrorException ("test device-to-host transfer failure");
    }
  ++device_to_host_copies;
  last_copy_bytes = bytes;
  std::memmove (dst, src, bytes);
}

void
configureCpuTestBackends ()
{
  auto &manager = MemoryManager::get ();
  manager.registerBackend (MemType::HOST_DEBUG,
                           { allocateBuffer<MemType::HOST_DEBUG>,
                             deallocateBuffer<MemType::HOST_DEBUG> });
  manager.registerBackend (MemType::DEVICE_DEBUG,
                           { allocateBuffer<MemType::DEVICE_DEBUG>,
                             deallocateBuffer<MemType::DEVICE_DEBUG> });
  manager.registerBackend (MemType::MANAGED,
                           { allocateBuffer<MemType::MANAGED>,
                             deallocateBuffer<MemType::MANAGED> });
  for (const MemType host : { MemType::HOST, MemType::HOST_DEBUG })
    {
      manager.registerCopy (MemType::DEVICE_DEBUG, host, copyToDevice);
      manager.registerCopy (host, MemType::DEVICE_DEBUG, copyToHost);
    }
}

Memory<int>
makeMemory (int size)
{
  return Memory<int> (size, MemType::HOST_DEBUG, MemType::DEVICE_DEBUG);
}

template <std::size_t Size>
void
checkData (const int *actual, const std::array<int, Size> &expected)
{
  CHECK (actual != nullptr);
  for (std::size_t index = 0; index < Size; ++index)
    {
      CHECK_EQ (actual[index], expected[index]) << "index=" << index;
    }
}

template <std::size_t Size>
void
checkData (const Memory<int> &memory, const std::array<int, Size> &expected)
{
  checkData (memory.read (MemoryClass::HOST, static_cast<int> (Size)),
             expected);
}

struct ObservedMemory : Memory<int>
{
  using Memory<int>::Memory;
  using Memory<int>::VALID_DEVICE;
  using Memory<int>::VALID_HOST;

  const int *
  cachedHost () const
  {
    return h_ptr;
  }
  const int *
  cachedDevice () const
  {
    return d_ptr;
  }
  unsigned
  cachedFlags () const
  {
    return flags;
  }
};

std::string
captureFlags (const Memory<int> &memory)
{
#ifdef _WIN32
  constexpr auto descriptor = &_fileno;
  constexpr auto duplicate = &_dup;
  constexpr auto replace = &_dup2;
  constexpr auto close = &_close;
#else
  constexpr auto descriptor = &fileno;
  constexpr auto duplicate = &dup;
  constexpr auto replace = &dup2;
  constexpr auto close = &::close;
#endif
  std::FILE *file = std::tmpfile ();
  CHECK (file != nullptr);
  CHECK_EQ (std::fflush (stderr), 0);
  const int saved = duplicate (descriptor (stderr));
  CHECK (saved >= 0);
  CHECK (replace (descriptor (file), descriptor (stderr)) >= 0);
  memory.printFlags ();
  const int flush_result = std::fflush (stderr);
  const int restore_result = replace (saved, descriptor (stderr));
  close (saved);
  CHECK_EQ (flush_result, 0);
  CHECK (restore_result >= 0);
  std::fseek (file, 0L, SEEK_SET);
  char buffer[512];
  const auto size = std::fread (buffer, 1, sizeof (buffer), file);
  CHECK_EQ (std::ferror (file), 0);
  std::fclose (file);
  CHECK (size < sizeof (buffer));
  return std::string (buffer, size);
}

void
testEmptyAndZeroSize ()
{
  Memory<int> memory;
  CHECK (memory.empty ());
  CHECK_EQ (memory.capaCity (), 0);
  CHECK (!memory.hostIsValid () && !memory.deviceIsValid ());
  CHECK (!memory.ownsHostPtr () && !memory.ownsDevicePtr ());
  CHECK (memory.read (MemoryClass::HOST, 0) == nullptr);
  CHECK (memory.write (MemoryClass::DEVICE, 0) == nullptr);
  CHECK (memory.readWrite (MemoryClass::HOST, 0) == nullptr);
  CHECK_EQ (memory.compareHostAndDevice (0), 0);
  memory.copyFromHost (nullptr, 0);
  memory.copyToHost (nullptr, 0);
  memory.copyFrom (memory, 0);
  memory.sync (memory);
  memory.syncAlias (memory, 0);
  memory.deleteDevice ();
  memory.allocate (0, MemType::HOST_DEBUG, MemType::DEVICE_DEBUG);
  CHECK (memory.empty ());
  CHECK_EQ (memory.getHostMemType (), MemType::HOST_DEBUG);
  CHECK_EQ (host_counters.allocations, 0);
  memory.wrap (nullptr, 0, MemType::HOST, false);
  Memory<int> alias;
  alias.makeAlias (memory, 0, 0);
  CHECK (alias.empty ());
  expectError ([&] { memory.read (MemoryClass::HOST, 1); });
}

void
testInitializationAndTypes ()
{
  auto memory = makeMemory (4);
  CHECK_EQ (memory.capaCity (), 4);
  CHECK_EQ (memory.getHostMemoryType (), MemType::HOST_DEBUG);
  CHECK_EQ (memory.getDeviceMemoryType (), MemType::DEVICE_DEBUG);
  CHECK (memory.ownsHostPtr () && !memory.ownsDevicePtr ());
  CHECK (!memory.hostIsValid () && !memory.deviceIsValid ());
  expectError ([&] { memory.read (MemoryClass::HOST, 4); });
  expectError ([&] { memory.readWrite (MemoryClass::DEVICE, 4); });
  expectError ([&] { memory.write (MemoryClass::HOST, 2); });
  Memory<int> alias;
  alias.makeAlias (memory, 1, 2);
  expectError ([&] { alias.write (MemoryClass::HOST, 2); });
  CHECK_EQ (device_counters.allocations, 0);
  const std::array data{ 11, 12, 13, 14 };
  memory.copyFromHost (data.data (), 4);
  checkData (memory, data);
  CHECK (memory.hostIsValid () && !memory.deviceIsValid ());
  memory.useDevice (true);
  CHECK (memory.useDevice ());
  CHECK_EQ (memory.getMemoryType (), MemType::DEVICE_DEBUG);
  CHECK (!alias.useDevice ());
  memory.useDevice (false);
  CHECK_EQ (memory.getMemType (), MemType::HOST_DEBUG);
}

void
testTransfersAndWriteModes ()
{
  auto memory = makeMemory (4);
  const std::array initial{ 1, 2, 3, 4 };
  memory.copyFromHost (initial.data (), 4);
  checkData (memory.read (MemoryClass::DEVICE, 4), initial);
  CHECK (memory.hostIsValid () && memory.deviceIsValid ());
  CHECK_EQ (host_to_device_copies, 1);
  CHECK_EQ (last_copy_bytes, initial.size () * sizeof (int));
  memory.read (MemoryClass::DEVICE, 4);
  CHECK_EQ (host_to_device_copies, 1);
  memory.readWrite (MemoryClass::DEVICE, 4)[1] = 22;
  CHECK (!memory.hostIsValid () && memory.deviceIsValid ());
  checkData (memory, std::array{ 1, 22, 3, 4 });
  CHECK_EQ (device_to_host_copies, 1);
  memory.readWrite (MemoryClass::HOST, 4)[0] = 10;
  CHECK (memory.hostIsValid () && !memory.deviceIsValid ());
  const std::array replacement{ 31, 32, 33, 34 };
  std::memcpy (memory.write (MemoryClass::DEVICE, 4), replacement.data (),
               sizeof (replacement));
  CHECK_EQ (host_to_device_copies, 1);
  CHECK (!memory.hostIsValid () && memory.deviceIsValid ());
  checkData (memory, replacement);
  CHECK_EQ (device_counters.allocations, 1);
}

void
testNestedAliasPartialWrites ()
{
  auto base = makeMemory (6);
  const std::array initial{ 1, 2, 3, 4, 5, 6 };
  base.copyFromHost (initial.data (), 6);
  Memory<int> alias;
  Memory<int> nested;
  alias.makeAlias (base, 1, 4);
  nested.makeAlias (alias, 1, 2);
  int *ptr = nested.write (MemoryClass::DEVICE, 2);
  ptr[0] = 30;
  ptr[1] = 40;
  CHECK_EQ (host_to_device_copies, 1);
  CHECK_EQ (last_copy_bytes, initial.size () * sizeof (int));
  CHECK (!base.hostIsValid () && alias.deviceIsValid ());
  checkData (base, std::array{ 1, 2, 30, 40, 5, 6 });
  CHECK_EQ (device_to_host_copies, 1);
  alias.write (MemoryClass::HOST, 1)[0] = 20;
  checkData (base, std::array{ 1, 20, 30, 40, 5, 6 });
  base.reset ();
  alias.release ();
  checkData (nested, std::array{ 30, 40 });
  CHECK_EQ (host_counters.deallocations, 0);
  nested.reset ();
  CHECK_EQ (host_counters.deallocations, 1);
  CHECK_EQ (device_counters.deallocations, 1);
}

void
testCopyConstructionAndAssignment ()
{
  auto base = makeMemory (4);
  const std::array initial{ 10, 20, 30, 40 };
  base.copyFromHost (initial.data (), 4);
  base.useDevice (true);
  Memory<int> copy (base);
  auto assigned = makeMemory (2);
  assigned = base;
  CHECK_EQ (host_counters.deallocations, 1);
  CHECK_EQ (copy.capaCity (), 4);
  CHECK (copy.useDevice () && assigned.useDevice ());
  copy.useDevice (false);
  CHECK (base.useDevice ());
  copy.readWrite (MemoryClass::HOST, 4)[2] = 99;
  checkData (base, std::array{ 10, 20, 99, 40 });
  base.reset ();
  copy.release ();
  checkData (assigned, std::array{ 10, 20, 99, 40 });
  CHECK_EQ (host_counters.deallocations, 1);
  assigned.release ();
  CHECK_EQ (host_counters.deallocations, 2);
}

void
testMovesSwapAndReallocation ()
{
  auto source = makeMemory (3);
  const std::array initial{ 7, 8, 9 };
  source.copyFromHost (initial.data (), 3);
  source.useDevice (true);
  Memory<int> moved (std::move (source));
  CHECK (source.empty ());
  CHECK_EQ (source.capaCity (), 0);
  CHECK (!source.ownsHostPtr ());
  auto target = makeMemory (2);
  target = std::move (moved);
  CHECK (moved.empty ());
  CHECK (target.useDevice ());
  CHECK_EQ (host_counters.deallocations, 1);
  auto other = makeMemory (1);
  other.write (MemoryClass::HOST, 1)[0] = 42;
  target.swap (other);
  CHECK_EQ (target.capaCity (), 1);
  CHECK_EQ (other.capaCity (), 3);
  checkData (target, std::array{ 42 });
  checkData (other, initial);
  Memory<int> retained (other);
  other.allocate (5, MemType::PRESERVE);
  CHECK_EQ (other.capaCity (), 5);
  CHECK_EQ (other.getHostMemType (), MemType::HOST_DEBUG);
  checkData (retained, initial);
  other.allocate (2);
  CHECK_EQ (host_counters.deallocations, 2);
  target.reset (MemType::HOST);
  CHECK (target.empty ());
  CHECK_EQ (target.getHostMemType (), MemType::HOST);
}

void
testWrappedOwnership ()
{
  alignas (256) int host[]{ 1, 2, 3, 4 };
  alignas (256) int device[]{ 1, 2, 3, 4 };
  {
    Memory<int> view;
    view.wrap (host, device, 4, MemType::HOST_DEBUG, MemType::DEVICE_DEBUG,
               false, true, true);
    CHECK (!view.ownsHostPtr () && !view.ownsDevicePtr ());
    CHECK_EQ (view.compareHostAndDevice (4), 0);
    view.readWrite (MemoryClass::HOST, 4)[1] = 20;
    CHECK_EQ (host[1], 20);
    view.deleteDevice (false);
  }
  CHECK_EQ (host_counters.deallocations, 0);
  CHECK_EQ (device_counters.deallocations, 0);
  CHECK_EQ (device[1], 2);
  auto *owned_host = static_cast<int *> (allocateBuffer<MemType::HOST_DEBUG> (
      4 * sizeof (int), alignof (std::max_align_t)));
  auto *owned_device = static_cast<int *> (
      allocateBuffer<MemType::DEVICE_DEBUG> (4 * sizeof (int), 256));
  std::memcpy (owned_host, host, sizeof (host));
  std::memcpy (owned_device, host, sizeof (host));
  {
    Memory<int> owner;
    owner.wrap (owned_host, owned_device, 4, MemType::HOST_DEBUG,
                MemType::DEVICE_DEBUG, true, true, true);
    Memory<int> retained (owner);
    owner.reset ();
    CHECK_EQ (host_counters.deallocations, 0);
    CHECK_EQ (device_counters.deallocations, 0);
    checkData (retained, std::array{ 1, 20, 3, 4 });
  }
  CHECK_EQ (host_counters.deallocations, 1);
  CHECK_EQ (device_counters.deallocations, 1);
  struct alignas (64) Value
  {
    int value;
  };
  Memory<Value> wrapped (new Value[2]{ { 17 }, { 18 } }, 2, MemType::HOST,
                         true);
  CHECK_EQ (wrapped.read (MemoryClass::HOST, 2)[1].value, 18);
}

void
testCopiesAndOverlap ()
{
  auto source = makeMemory (6);
  const std::array initial{ 1, 2, 3, 4, 5, 6 };
  source.copyFromHost (initial.data (), 6);
  auto destination = makeMemory (6);
  source.readWrite (MemoryClass::DEVICE, 6)[0] = 10;
  source.copyTo (destination, 6);
  checkData (destination, std::array{ 10, 2, 3, 4, 5, 6 });
  std::array<int, 6> output{};
  destination.copyToHost (output.data (), 6);
  checkData (output.data (), std::array{ 10, 2, 3, 4, 5, 6 });
  destination.copyFromHost (initial.data (), 6);
  Memory<int> left;
  Memory<int> right;
  left.makeAlias (destination, 0, 4);
  right.makeAlias (destination, 1, 4);
  right.copyFrom (left, 4);
  checkData (destination, std::array{ 1, 1, 2, 3, 4, 6 });
  destination.copyFromHost (initial.data (), 6);
  left.copyFrom (right, 4);
  checkData (destination, std::array{ 2, 3, 4, 5, 5, 6 });
  destination.readWrite (MemoryClass::DEVICE, 6)[0] = 21;
  destination.copyFrom (destination, 6);
  CHECK (!destination.hostIsValid () && destination.deviceIsValid ());
  checkData (destination, std::array{ 21, 3, 4, 5, 5, 6 });
}

void
testDeviceDeletion ()
{
  auto memory = makeMemory (3);
  const std::array initial{ 1, 2, 3 };
  memory.copyFromHost (initial.data (), 3);
  Memory<int> alias (memory);
  memory.readWrite (MemoryClass::DEVICE, 3)[1] = 22;
  memory.deleteDevice (true);
  CHECK (alias.hostIsValid () && !alias.deviceIsValid ());
  CHECK (!memory.ownsDevicePtr ());
  CHECK_EQ (device_counters.deallocations, 1);
  checkData (alias, std::array{ 1, 22, 3 });
  memory.readWrite (MemoryClass::DEVICE, 3)[0] = 11;
  memory.deleteDevice (false);
  CHECK (!alias.hostIsValid () && !alias.deviceIsValid ());
  CHECK_EQ (device_counters.deallocations, 2);
  expectError ([&] { alias.read (MemoryClass::HOST, 3); });
  memory.copyFromHost (initial.data (), 3);
  checkData (alias, initial);
  memory.deleteDevice ();
  CHECK_EQ (device_counters.deallocations, 2);
}

void
testAllocationFailurePreservesStorage ()
{
  auto memory = makeMemory (3);
  const std::array initial{ 10, 20, 30 };
  memory.copyFromHost (initial.data (), 3);
  {
    FailureGuard failure (fail_allocation);
    expectError ([&] { memory.allocate (8); });
    CHECK_EQ (memory.capaCity (), 3);
    checkData (memory, initial);
    ErrorActionGuard ignore (ErrorAction::Ignore);
    memory.allocate (8);
    CHECK_EQ (memory.capaCity (), 3);
    checkData (memory, initial);
  }
  expectError ([&] { memory.allocate (4, MemType::HOST_PINNED); });
  checkData (memory, initial);
  CHECK_EQ (host_counters.deallocations, 0);
  CHECK_EQ (host_counters.allocations, 1);
}

void
testTransferFailurePreservesState ()
{
  auto memory = makeMemory (3);
  const std::array initial{ 10, 20, 30 };
  memory.copyFromHost (initial.data (), 3);
  {
    FailureGuard failure (fail_host_to_device);
    expectError ([&] { memory.read (MemoryClass::DEVICE, 3); });
    CHECK (memory.hostIsValid () && !memory.deviceIsValid ());
    checkData (memory, initial);
  }
  memory.readWrite (MemoryClass::DEVICE, 3)[1] = 99;
  CHECK_EQ (device_counters.allocations, 1);
  {
    FailureGuard failure (fail_device_to_host);
    expectError ([&] { memory.read (MemoryClass::HOST, 3); });
    expectError ([&] { memory.deleteDevice (true); });
    expectError ([&] { (void)memory.compareHostAndDevice (3); });
    CHECK (!memory.hostIsValid () && memory.deviceIsValid ());
    CHECK (memory.ownsDevicePtr ());
    CHECK_EQ (device_counters.deallocations, 0);
    checkData (memory.read (MemoryClass::DEVICE, 3), std::array{ 10, 99, 30 });
  }
  memory.deleteDevice (true);
  CHECK_EQ (device_counters.deallocations, 1);
  checkData (memory, std::array{ 10, 99, 30 });
}

void
testSyncRefreshesSharedViews ()
{
  alignas (256) int device[]{ 10, 20, 30, 40, 50, 60 };
  ObservedMemory base (device, 6, MemType::DEVICE_DEBUG, false);
  ObservedMemory alias;
  ObservedMemory nested;
  alias.makeAlias (base, 1, 4);
  nested.makeAlias (alias, 1, 2);
  CHECK (alias.cachedHost () == nullptr && nested.cachedHost () == nullptr);
  const int *host = base.read (MemoryClass::HOST, 6);
  CHECK (alias.cachedHost () == nullptr);
  alias.sync (base);
  nested.syncAlias (alias, 2);
  CHECK_EQ (alias.cachedHost (), host + 1);
  CHECK_EQ (nested.cachedHost (), host + 2);
  CHECK_EQ (nested.cachedDevice (), device + 2);
  checkData (nested.cachedHost (), std::array{ 30, 40 });
  base.deleteDevice ();
  CHECK (alias.cachedDevice () != nullptr);
  alias.sync (base);
  CHECK (alias.cachedDevice () == nullptr);
  CHECK (alias.hostIsValid () && !alias.deviceIsValid ());
  Memory<int> unrelated (1);
  expectError ([&] { alias.sync (unrelated); });
  expectError ([&] { alias.syncAlias (unrelated, 1); });
  expectError ([&] { alias.syncAlias (base, -1); });
  expectError ([&] { nested.syncAlias (alias, 3); });
  expectError ([&] { base.syncAlias (alias, 1); });
  Memory<int> outside;
  outside.makeAlias (base, 5, 1);
  expectError ([&] { outside.syncAlias (alias, 1); });
  Memory<int> endpoint;
  endpoint.makeAlias (alias, 4, 0);
  endpoint.syncAlias (alias, 0);
  CHECK (endpoint.empty ());
}

void
testComparisonPreservesDataAndState ()
{
  auto memory = makeMemory (4);
  const std::array initial{ 1, 2, 3, 4 };
  memory.copyFromHost (initial.data (), 4);
  expectError ([&] { (void)memory.compareHostAndDevice (4); });
  CHECK_EQ (device_counters.allocations, 0);
  const int *host = memory.read (MemoryClass::HOST, 4);
  memory.read (MemoryClass::DEVICE, 4);
  CHECK_EQ (memory.compareHostAndDevice (4), 0);
  Memory<int> alias;
  alias.makeAlias (memory, 1, 2);
  memory.readWrite (MemoryClass::DEVICE, 4)[2] = 99;
  CHECK (memory.compareHostAndDevice (4) != 0);
  CHECK_EQ (alias.compareHostAndDevice (1), 0);
  CHECK (alias.compareHostAndDevice (2) != 0);
  CHECK_EQ (memory.compareHostAndDevice (0), 0);
  CHECK (!memory.hostIsValid () && memory.deviceIsValid ());
  checkData (host, initial);
  expectError ([&] { (void)memory.compareHostAndDevice (-1); });
  expectError ([&] { (void)alias.compareHostAndDevice (3); });
  alignas (256) int device[]{ 1, 2, 3, 4 };
  Memory<int> no_copy;
  no_copy.wrap (nullptr, device, 4, MemType::HOST, MemType::DEVICE_UMPIRE,
                false, false, true);
  expectError ([&] { no_copy.read (MemoryClass::HOST, 4); });
  CHECK (!no_copy.hostIsValid () && no_copy.deviceIsValid ());
  ErrorActionGuard ignore (ErrorAction::Ignore);
  CHECK (no_copy.read (MemoryClass::HOST, 4) == nullptr);
  CHECK (no_copy.compareHostAndDevice (4) != 0);
  CHECK (!no_copy.hostIsValid () && no_copy.deviceIsValid ());
}

void
testManagedStorageReleasedOnce ()
{
  Memory<int> retained;
  {
    Memory<int> memory (3, MemType::MANAGED);
    const std::array initial{ 5, 6, 7 };
    int *host = memory.write (MemoryClass::MANAGED, 3);
    std::memcpy (host, initial.data (), sizeof (initial));
    memory.useDevice (true);
    CHECK_EQ (memory.read (MemoryClass::MANAGED, 3), host);
    CHECK_EQ (memory.compareHostAndDevice (3), 0);
    memory.readWrite (MemoryClass::MANAGED, 3)[1] = 60;
    memory.deleteDevice (true);
    CHECK (memory.deviceIsValid ());
    checkData (memory, std::array{ 5, 60, 7 });
    retained.makeAlias (memory, 1, 2);
  }
  CHECK_EQ (managed_counters.deallocations, 0);
  checkData (retained, std::array{ 60, 7 });
  retained.release ();
  CHECK_EQ (managed_counters.deallocations, 1);
  auto *shared = static_cast<int *> (
      allocateBuffer<MemType::MANAGED> (2 * sizeof (int), 256));
  shared[0] = 8;
  shared[1] = 9;
  {
    Memory<int> owner;
    owner.wrap (shared, shared, 2, MemType::MANAGED, MemType::MANAGED, true,
                true, true);
    checkData (owner, std::array{ 8, 9 });
  }
  CHECK_EQ (managed_counters.deallocations, 2);
  CHECK_EQ (host_to_device_copies, 0);
  CHECK_EQ (device_to_host_copies, 0);
}

void
testInvalidArgumentsPreserveData ()
{
  auto memory = makeMemory (4);
  const std::array initial{ 1, 2, 3, 4 };
  memory.copyFromHost (initial.data (), 4);
  expectError ([&] { memory.allocate (-1); });
  expectError ([&] { memory.allocate (4, MemType::SIZE); });
  expectError ([&] { memory.allocate (4, MemType::HOST, MemType::HOST); });
  expectError ([&] { memory.allocate (4, MemType::HOST, MemType::MANAGED); });
  expectError ([&] { memory.wrap (nullptr, 4, MemType::HOST, false); });
  alignas (256) int buffer[8]{};
  expectError (
      [&] { memory.wrap (buffer + 1, 4, MemType::DEVICE_DEBUG, false); });
  expectError (
      [&]
        {
          memory.wrap (nullptr, buffer, 4, MemType::HOST,
                       MemType::DEVICE_DEBUG, false, true, true);
        });
  expectError (
      [&]
        {
          memory.wrap (buffer, buffer + 4, 4, MemType::MANAGED,
                       MemType::MANAGED, false, true, true);
        });
  expectError ([&] { memory.read (MemoryClass::HOST, -1); });
  expectError ([&] { memory.write (MemoryClass::HOST, 5); });
  expectError ([&] { memory.readWrite (MemoryClass::MANAGED, 4); });
  expectError ([&] { memory.read (static_cast<MemoryClass> (-1), 4); });
  expectError ([&] { memory.copyFromHost (nullptr, 1); });
  expectError ([&] { memory.copyToHost (nullptr, 1); });
  Memory<int> alias (memory);
  expectError ([&] { alias.makeAlias (memory, -1, 1); });
  expectError ([&] { alias.makeAlias (memory, 3, 2); });
  expectError ([&] { alias.makeAlias (memory, 5, 0); });
  expectError ([&] { alias.copyFrom (memory, 5); });
  CHECK_EQ (memory.capaCity (), 4);
  CHECK_EQ (alias.capaCity (), 4);
  checkData (memory, initial);
  checkData (alias, initial);
}

void
testBackendRegistrationAndCapturedDeallocator ()
{
  auto &manager = MemoryManager::get ();
  CHECK_EQ (&manager, &MemoryManager::get ());
  CHECK (manager.supports (MemType::HOST));
  for (const MemType type : { MemType::SIZE, MemType::DEFAULT,
                              MemType::PRESERVE, static_cast<MemType> (-1) })
    {
      CHECK (!manager.supports (type));
      expectError ([&] { manager.registerBackend (type, {}); });
      expectError (
          [&] { manager.registerCopy (type, MemType::HOST, copyToHost); });
    }
  expectError (
      [&]
        {
          manager.registerBackend (
              MemType::HOST_DEBUG,
              { allocateBuffer<MemType::HOST_DEBUG>, nullptr });
        });
  expectError (
      [&]
        {
          manager.registerCopy (MemType::HOST, MemType::DEVICE_DEBUG, nullptr);
        });
  auto memory = makeMemory (2);
  const std::array initial{ 7, 8 };
  memory.copyFromHost (initial.data (), 2);
  manager.registerBackend (MemType::HOST_DEBUG, {});
  CHECK (!manager.supports (MemType::HOST_DEBUG));
  expectError ([&] { memory.allocate (3); });
  checkData (memory, initial);
  memory.reset ();
  CHECK_EQ (host_counters.deallocations, 1);
  configureCpuTestBackends ();
}

void
testPrintFlagsUsesCurrentSharedState ()
{
  ObservedMemory memory (3, MemType::HOST_DEBUG, MemType::DEVICE_DEBUG);
  const std::array initial{ 1, 2, 3 };
  memory.copyFromHost (initial.data (), 3);
  ObservedMemory alias;
  alias.makeAlias (memory, 1, 2);
  const unsigned cached_flags = alias.cachedFlags ();
  memory.readWrite (MemoryClass::DEVICE, 3)[0] = 10;
  memory.useDevice (true);
  const int copy_count = host_to_device_copies + device_to_host_copies;
  const auto output = captureFlags (memory);
  CHECK (output.find ("registered=1 owns_host=1 owns_device=1")
         != std::string::npos);
  CHECK (output.find ("owns_internal=1 use_device=1 alias=0")
         != std::string::npos);
  const auto alias_output = captureFlags (alias);
  CHECK (alias_output.find ("registered=1 owns_host=1 owns_device=1")
         != std::string::npos);
  CHECK (alias_output.find ("owns_internal=0 use_device=0 alias=1")
         != std::string::npos);
  unsigned snapshot{};
  CHECK_EQ (std::sscanf (alias_output.c_str (), "flags=0x%x", &snapshot), 1);
  CHECK_EQ (snapshot
                & (ObservedMemory::VALID_HOST | ObservedMemory::VALID_DEVICE),
            ObservedMemory::VALID_DEVICE);
  CHECK_EQ (alias.cachedFlags (), cached_flags);
  CHECK (!memory.hostIsValid () && memory.deviceIsValid ());
  CHECK_EQ (host_to_device_copies + device_to_host_copies, copy_count);
  CHECK_EQ (device_counters.allocations, 1);
  const auto empty_output = captureFlags (Memory<int>{});
  CHECK (empty_output.find ("registered=0 owns_host=0 owns_device=0")
         != std::string::npos);
}

void
testIgnoredCopyFailurePreservesDestination ()
{
  auto destination = makeMemory (2);
  const std::array old_host{ 11, 12 };
  const std::array latest_device{ 21, 22 };
  destination.copyFromHost (old_host.data (), 2);
  std::memcpy (destination.write (MemoryClass::DEVICE, 2),
               latest_device.data (), sizeof (latest_device));
  auto uninitialized = makeMemory (2);
  expectError ([&] { destination.copyFrom (uninitialized, 2); });
  CHECK (!destination.hostIsValid () && destination.deviceIsValid ());
  {
    ErrorActionGuard ignore (ErrorAction::Ignore);
    destination.copyFrom (uninitialized, 2);
  }
  CHECK (!destination.hostIsValid () && destination.deviceIsValid ());
  checkData (destination.read (MemoryClass::DEVICE, 2), latest_device);
}

}

int
main (int argc, char **argv)
{
  struct TestCase
  {
    const char *name;
    void (*run) ();
  };
  const TestCase tests[]{
    { "empty_and_zero_size", testEmptyAndZeroSize },
    { "initialization_and_types", testInitializationAndTypes },
    { "transfers_and_write_modes", testTransfersAndWriteModes },
    { "nested_alias_partial_writes", testNestedAliasPartialWrites },
    { "copy_construction_and_assignment", testCopyConstructionAndAssignment },
    { "moves_swap_and_reallocation", testMovesSwapAndReallocation },
    { "wrapped_ownership", testWrappedOwnership },
    { "copies_and_overlap", testCopiesAndOverlap },
    { "device_deletion", testDeviceDeletion },
    { "allocation_failure", testAllocationFailurePreservesStorage },
    { "transfer_failure", testTransferFailurePreservesState },
    { "sync_shared_views", testSyncRefreshesSharedViews },
    { "comparison_preserves_data", testComparisonPreservesDataAndState },
    { "managed_storage", testManagedStorageReleasedOnce },
    { "invalid_arguments", testInvalidArgumentsPreserveData },
    { "backend_registration", testBackendRegistrationAndCapturedDeallocator },
    { "print_flags", testPrintFlagsUsesCurrentSharedState },
    { "ignored_copy_failure", testIgnoredCopyFailurePreservesDestination },
  };
  if (argc > 2)
    {
      std::cerr << "Usage: " << argv[0] << " [test_name]\n";
      return 2;
    }
  ErrorActionGuard errors (ErrorAction::Throw);
  configureCpuTestBackends ();
  int passed = 0;
  for (const auto &test : tests)
    {
      if (argc == 2 && std::strcmp (argv[1], test.name) != 0)
        {
          continue;
        }
      backend_counters = {};
      host_to_device_copies = device_to_host_copies = 0;
      last_copy_bytes = 0;
      std::cout << "[ RUN      ] " << test.name << '\n';
      try
        {
          test.run ();
        }
      catch (const std::exception &error)
        {
          std::cerr << "[ FAILED   ] " << test.name << ": " << error.what ();
          return 1;
        }
      for (const auto &counter : backend_counters)
        {
          CHECK_EQ (counter.allocations, counter.deallocations)
              << "Storage must be released exactly once";
        }
      ++passed;
      std::cout << "[       OK ] " << test.name << '\n';
    }
  if (passed == 0)
    {
      std::cerr << "Unknown test: " << argv[1] << '\n';
      return 2;
    }
  std::cout << passed << " memory manager tests passed.\n";
  return 0;
}
