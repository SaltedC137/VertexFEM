#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

#include "config.hpp"
#include "error.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>

#ifdef VFEM_USE_MPI
#define HYPRE_TIMING
#include <HYPRE_utilities.h>
#if (21400 <= VFEM_HYPRE_VERSION) && (VFEM_HYPRE_VERSION < 21900)
#include <_hypre_utilities.h>
#endif
#endif

namespace vfem
{

enum class MemType
{
  HOST,         // 0 HOST memory (default)
  HOST_32,      // 1 32-bit addressable HOST memory
  HOST_64,      // 2 64-bit addressable HOST memory
  HOST_DEBUG,   // 3 HOST memory with debugging enabled
  HOST_UMPIRE,  // 4 HOST memory allocated via Umpire (if available)
  HOST_PINNED,  // 5 HOST pinned memory (page-locked, for faster GPU transfers)
  MANAGED,      // 6 MANAGED memory (accessible from both HOST and DEVICE)
  DEVICE,       // 7 DEVICE memory (GPU memory)
  DEVICE_DEBUG, // 8 DEVICE memory with debugging enabled
  DEVICE_UMPIRE,   // 9 DEVICE memory allocated via Umpire (if available)
  DEVICE_UMPIRE_2, // 10 DEVICE memory allocated via Umpire with different
                   // settings (e.g., different pool or arena)
  SIZE,            // 11 Number of memory types (must be last)
  PRESERVE, // 12 Preserve the existing memory type (used in certain operations
            // to indicate that the memory type should not be changed)
  DEFAULT   // 13 Default memory type (used in certain operations to indicate
            // that the default memory type should be used, which is typically
            // HOST or MANAGED depending on the context)
};

/// Static casts to 'int' and constexprs for MemType values
constexpr int MemTypeSize = static_cast<int> (MemType::SIZE);
constexpr int HostMemType = static_cast<int> (MemType::HOST);
constexpr int HostMemTypeSize = static_cast<int> (MemType::DEVICE);
constexpr int DeviceMemType = static_cast<int> (MemType::MANAGED);
constexpr int DeviceMemTypeSize = MemTypeSize - DeviceMemType;

extern VFEM_EXPORT const char *MemTypeName[MemTypeSize];

enum class MemoryClass
{
  HOST,
  HOST_32,
  HOST_64,
  DEVICE,
  MANAGED,
};

enum class MemorySide : std::uint8_t
{
  HOST,
  DEVICE
};

enum class AccessMode : std::uint8_t
{
  READ,
  WRITE,
  READ_WRITE
};

enum class MemoryState : std::uint8_t
{
  EMPTY,
  UNINITIALIZED,
  HOST_VALID,
  DEVICE_VALID,
  SYNCHRONIZED
};

enum class Transfer : std::uint8_t
{
  NONE,
  HOST_TO_DEVICE,
  DEVICE_TO_HOST
};

struct StateTransition
{
  bool allowed;
  Transfer transfer;
  MemoryState new_state;
};

// Get the exclusive memory state for a given memory side (host or device)
[[nodiscard]] constexpr MemoryState
exclusiveState (MemorySide side) noexcept
{
  return side == MemorySide::HOST ? MemoryState::HOST_VALID
                                  : MemoryState::DEVICE_VALID;
}

// check if the memory state is valid for the given side (host or device)
[[nodiscard]] constexpr bool
isValidOn (MemoryState state, MemorySide side) noexcept
{
  return state == MemoryState::SYNCHRONIZED || state == exclusiveState (side);
}

[[nodiscard]] constexpr StateTransition
nextState (MemoryState current, MemorySide target, AccessMode mode) noexcept
{
  const MemoryState target_state = exclusiveState (target);

  if (mode == AccessMode::WRITE)
    {
      return { true, Transfer::NONE, target_state };
    }

  if (current == MemoryState::EMPTY || current == MemoryState::UNINITIALIZED)
    {
      return { false, Transfer::NONE, current };
    }

  if (isValidOn (current, target))
    {
      return { true, Transfer::NONE,
               mode == AccessMode::READ ? current : target_state };
    }

  const Transfer transfer = target == MemorySide::HOST
                                ? Transfer::DEVICE_TO_HOST
                                : Transfer::HOST_TO_DEVICE;

  return { true, transfer,
           mode == AccessMode::READ ? MemoryState::SYNCHRONIZED
                                    : target_state };
}

[[nodiscard]] inline bool
isHostMemory (MemType type) noexcept
{
  switch (type)
    {
    case MemType::HOST:
    case MemType::HOST_32:
    case MemType::HOST_64:
    case MemType::HOST_DEBUG:
    case MemType::HOST_UMPIRE:
    case MemType::HOST_PINNED:
    case MemType::MANAGED:
      return true;
    default:
      return false;
    }
}

[[nodiscard]] inline bool
isDeviceMemory (MemType type) noexcept
{
  switch (type)
    {
    case MemType::MANAGED:
    case MemType::DEVICE:
    case MemType::DEVICE_DEBUG:
    case MemType::DEVICE_UMPIRE:
    case MemType::DEVICE_UMPIRE_2:
      return true;
    default:
      return false;
    }
}

VFEM_EXPORT MemType getMemType (MemoryClass mc, int index = 0);
VFEM_EXPORT bool memClassContainsType (MemoryClass mc, MemType type);
VFEM_EXPORT MemoryClass operator* (MemoryClass mc1, MemoryClass mc2);
VFEM_EXPORT void memoryPrintFlags (unsigned flags) noexcept;

/// Class used by VFEM to manage memory allocations and deallocations across
/// different memory types.

template <typename T>
concept DeviceCopyable
    = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

using DeallocateFunc = void (*) (void *, std::size_t alignment) noexcept;

struct MemoryRecord final
{
  void *h_ptr{};
  void *d_ptr{};
  std::size_t bytes{};
  std::size_t alignment{ alignof (std::max_align_t) };
  MemType h_mt{ MemType ::HOST };
  MemType d_mt{ MemType ::DEVICE };
  DeallocateFunc h_deallocate{};
  DeallocateFunc d_deallocate{};
  bool owns_h{};
  bool owns_d{};
  MemoryState state{ MemoryState::EMPTY };

  MemoryRecord () = default;
  MemoryRecord (const MemoryRecord &) = delete;
  MemoryRecord &operator= (const MemoryRecord &) = delete;
  ~MemoryRecord () noexcept;
};

class MemoryManager;

template <DeviceCopyable T> class Memory
{
protected:
  friend class MemoryManager;
  friend void memoryPrintFlags (unsigned flags) noexcept;

  enum FlagMask : unsigned
  {
#ifndef REGISTERED
    REGISTERED = 1 << 0,
#endif
    Registered = 1 << 0,
    OWNS_HOST = 1 << 1,
    OWNS_DEVICE = 1 << 2,
    OWNS_INTERNAL
    = 1 << 3, // Owns memory allocated internally by VFEM (not user-provided)
    VALID_HOST = 1 << 4,
    VALID_DEVICE = 1 << 5,
    USE_DEVICE = 1 << 6, // Indicates that the device pointer should be used
                         // for operations (e.g., when memory is mirrored)
    ALIAS = 1 << 7,      // Pointer is an alias
  };

  /// Pointer to the memory. Not owned.

  mutable T *h_ptr{};
  mutable T *d_ptr{};
  int capacity{};
  MemType h_mt{ MemType::HOST };
  MemType d_mt{ MemType::DEVICE };
  mutable unsigned flags{};

public:
  // Default constructor initializes to an empty, invalid state
  Memory () noexcept = default;

  Memory (Memory &&other) noexcept;
  Memory &operator= (Memory &&other) noexcept;

  Memory (const Memory &other) noexcept;
  Memory &operator= (const Memory &other) noexcept;

  // Create a new memory allocation of the given size. If the memory is already
  // allocated, it will be deallocated and reallocated. The memory type is set
  // to the default host memory type (HOST).

  explicit Memory (int size) { allocate (size); }

  // Create a new memory allocation of the given size and memory type. If the
  // memory is already allocated, it will be deallocated and reallocated.

  explicit Memory (MemType mt) { configureTypes (mt); }

  Memory (int size, MemType mt);
  Memory (int size, MemType h_mt, MemType d_mt);

  explicit Memory (T *ptr, int size, MemType mt, bool own);

  ~Memory () = default;

  // Note: The destructor is not marked noexcept because it may throw if the
  // destructor of T throws during deallocation.

  void swap (Memory &other) noexcept;

  // Reset memory to an empty state without releasing storage or changing
  // types.
  constexpr void reset () noexcept;

  // Reset memory to an empty state and configure its host/device memory types.
  void reset (MemType memory_type);

  bool empty () const noexcept;

  inline void allocate (int size);
  inline void allocate (int size, MemType mt);
  inline void allocate (int size, MemType host_mt, MemType device_mt);

  // Wrap the memory around an existing pointer. The memory will not be owned
  // by this object, and will not be deallocated when the object is destroyed.
  // The memory type is set to the default host memory type (HOST).

  inline void wrap (T *ptr, int size, MemType mt, bool own);
  inline void wrap (T *h_ptr, T *d_ptr, int size, MemType h_mt, MemType d_mt,
                    bool own, bool vaild_host = false,
                    bool valid_device = true);
  inline void makeAlias (const Memory &base, int offset, int size);

  inline T &operator[] (int index) noexcept;

  inline const T &operator[] (int index) const noexcept;

  operator T *() noexcept;
  operator const T *() const noexcept;
  template <typename U> inline explicit operator U *() noexcept;
  template <typename U> inline explicit operator const U *() const noexcept;

  void
  release () noexcept
  {
    reset ();
  };

  void deleteDevice (bool copy_to_host = true);

  int
  capaCity () const noexcept
  {
    return capacity;
  }

  // error checking for valid host/device pointers based on memory type and
  // ownership flags. This is used by MemoryManager and other internal
  // components to ensure that memory operations
  bool hostIsValid () const noexcept;
  bool deviceIsValid () const noexcept;
  bool ownsHostPtr () const noexcept;
  void setHostPtrOwner (bool own) noexcept;
  bool ownsDevicePtr () const noexcept;

  bool
  useDevice () const noexcept
  {
    return (flags & USE_DEVICE) != 0U;
  };

  void
  useDevice (bool use_dev) const noexcept
  {
    flags = use_dev ? (flags | USE_DEVICE) : (flags & ~USE_DEVICE);
  };

  inline void sync (const Memory &other) const;

  inline void syncAlias (const Memory &base, int alias_size) const;

  // Getters for memory type and ownership flags (used by MemoryManager and
  // other internal components)
  [[nodiscard]] MemType
  getHostMemType () const noexcept
  {
    return h_mt;
  }

  [[nodiscard]] MemType
  getDeviceMemType () const noexcept
  {
    return d_mt;
  }

  [[nodiscard]] MemType getMemType () const noexcept;

  // Memory access methods

  inline T *readWrite (MemoryClass mc, int size);
  inline const T *read (MemoryClass mc, int size) const;
  inline T *write (MemoryClass mc, int size);

  // Memory type query methods
  MemType
  getMemoryType () const noexcept
  {
    return getMemType ();
  };

  MemType
  getHostMemoryType () const noexcept
  {
    return h_mt;
  };

  MemType
  getDeviceMemoryType () const noexcept
  {
    return d_mt;
  };

  // Copy type methods
  inline void copyFrom (const Memory &other, int size);
  inline void copyTo (const Memory &other, int size) const;
  inline void copyFromHost (const T *host_ptr, int size);
  inline void copyToHost (T *host_ptr, int size) const;

  // Print the flags for debugging purposes

  inline void printFlags () const;

  [[nodiscard]] inline int compareHostAndDevice (int size) const;

private:
  std::shared_ptr<MemoryRecord> record_;
  std::size_t byte_offset_{};

  [[nodiscard]] static bool isPlainHostType (MemType type) noexcept;
  [[nodiscard]] static bool hasRequiredAlignment (const void *ptr,
                                                  MemType type) noexcept;
  static void deleteWrappedHost (void *ptr, std::size_t alignment) noexcept;

  [[nodiscard]] inline static std::size_t checkedBytes (int size);
  [[nodiscard]] inline bool validAccessSize (int size) const;
  [[nodiscard]] inline MemorySide sideFor (MemoryClass mc) const;
  [[nodiscard]] inline MemType typeFor (MemorySide side) const noexcept;
  [[nodiscard]] inline T *viewPointer (void *base) const noexcept;
  [[nodiscard]] inline const T *viewPointer (const void *base) const noexcept;

  void refreshView () const noexcept;
  void configureTypes (MemType memory_type);
  void moveFrom (Memory &&other) noexcept;

#if defined(VFEM_USE_CUDA) || defined(VFEM_USE_HIP)

  template <typename T, std::size_t new_align_bytes = alignas (T)>
  struct AllocDevice
  {
    static_assert (new_align_bytes <= 256, "cudaMalloc / hipMalloc ");
    static T *
    New (std::size_t size)
    {
      T *ptr = nullptr;

#ifdef VFEM_USE_HIP
      hipMalloc (reinterpret_cast<void **> (&ptr), size * sizeof (T));
#else
      cudaMalloc (reinterpret_cast<void **> (&ptr), size * sizeof (T));
#endif
      return ptr;
    }

    static void
    Delete (T *ptr)
    {
#ifdef VFEM_USE_HIP
      hipFree (ptr);
#else
      cudaFree (ptr);
#endif
    }
  };
#endif
};

// MemoryManager class manages memory allocations and deallocations across
// different memory types (HOST, DEVICE, MANAGED, etc.). It provides methods to
// register backends for memory allocation and deallocation, as well as methods
// to copy data between different memory types. The class also maintains a
// mapping of memory types to their corresponding backends and copy functions.

class VFEM_EXPORT MemoryManager
{
public:
  using AllocateFunc = void *(*)(std::size_t bytes, std::size_t alignment);
  using CopyFunc = void (*) (void *dst, const void *src, std::size_t bytes);
  using BackendDeallocateFunc = ::vfem::DeallocateFunc;

  struct Backend
  {
    AllocateFunc allocate{};
    BackendDeallocateFunc deallocate{};
  };

  /// Configure backends before any Memory object starts using the type.
  static MemoryManager &get ();

  void registerBackend (MemType type, Backend backend);
  void registerCopy (MemType dst, MemType src, CopyFunc copy);

  [[nodiscard]] bool supports (MemType type) const noexcept;

private:
  template <DeviceCopyable T> friend class Memory;

  MemoryManager ();

  [[nodiscard]] BackendDeallocateFunc
  getDeallocate (MemType type) const noexcept;

  void allocate (MemoryRecord &record, MemorySide side);
  void *access (MemoryRecord &record, MemorySide side, AccessMode mode);
  void deleteDevice (MemoryRecord &record, bool copy_to_host);
  void copy (void *dst, MemType dst_mt, const void *src, MemType src_mt,
             std::size_t bytes);

  [[nodiscard]] static bool isConcreteType (MemType type) noexcept;
  [[nodiscard]] static std::size_t typeIndex (MemType type);
  [[nodiscard]] static std::size_t
  requiredAlignment (MemType host_mt, MemType device_mt,
                     std::size_t type_alignment) noexcept;

  std::array<Backend, MemTypeSize> backends_{};
  std::array<CopyFunc, MemTypeSize * MemTypeSize> copies_{};
  mutable std::mutex backend_mutex_;
};

//============================================================================

template <DeviceCopyable T>
Memory<T>::Memory (const Memory &other) noexcept
    : h_ptr (other.h_ptr), d_ptr (other.d_ptr), capacity (other.capacity),
      h_mt (other.h_mt), d_mt (other.d_mt),
      flags ((other.flags & USE_DEVICE) | ALIAS), record_ (other.record_),
      byte_offset_ (other.byte_offset_)
{
  refreshView ();
}

template <DeviceCopyable T>
Memory<T> &
Memory<T>::operator= (const Memory &other) noexcept
{
  if (this == &other)
    {
      return *this;
    }
  record_ = other.record_;
  byte_offset_ = other.byte_offset_;
  capacity = other.capacity;
  h_mt = other.h_mt;
  d_mt = other.d_mt;
  flags = (other.flags & USE_DEVICE) | ALIAS;
  refreshView ();
  return *this;
}

template <DeviceCopyable T> Memory<T>::Memory (Memory &&other) noexcept
{
  moveFrom (std::move (other));
}

template <DeviceCopyable T>
Memory<T> &
Memory<T>::operator= (Memory &&other) noexcept
{
  if (this != &other)
    {
      reset ();
      moveFrom (std::move (other));
    }
  return *this;
}

template <DeviceCopyable T> Memory<T>::Memory (int size, MemType mt)
{
  allocate (size, mt);
}

template <DeviceCopyable T>
Memory<T>::Memory (int size, MemType h_mt, MemType d_mt)
{
  allocate (size, h_mt, d_mt);
}

template <DeviceCopyable T>
Memory<T>::Memory (T *ptr, int size, MemType mt, bool own)
{
  wrap (ptr, size, mt, own);
}

template <DeviceCopyable T>
void
Memory<T>::swap (Memory &other) noexcept
{
  using std::swap;
  swap (h_ptr, other.h_ptr);
  swap (d_ptr, other.d_ptr);
  swap (capacity, other.capacity);
  swap (h_mt, other.h_mt);
  swap (d_mt, other.d_mt);
  swap (flags, other.flags);
  swap (record_, other.record_);
  swap (byte_offset_, other.byte_offset_);
}

template <DeviceCopyable T>
constexpr void
Memory<T>::reset () noexcept
{
  record_.reset ();
  h_ptr = nullptr;
  d_ptr = nullptr;
  capacity = 0;
  byte_offset_ = 0;
  flags = 0;
}

template <DeviceCopyable T>
void
Memory<T>::reset (MemType memory_type)
{
  reset ();
  configureTypes (memory_type);
}

template <DeviceCopyable T>
void
Memory<T>::configureTypes (MemType memory_type)
{
  if (memory_type == MemType::PRESERVE)
    {
      return;
    }
  if (memory_type == MemType::DEFAULT)
    {
      memory_type = MemType::HOST;
    }
  if (!isHostMemory (memory_type) && !isDeviceMemory (memory_type))
    {
      vfemError ("memory type is not a concrete backend type");
      return;
    }

  if (memory_type == MemType::MANAGED)
    {
      h_mt = MemType::MANAGED;
      d_mt = MemType::MANAGED;
    }
  else if (isDeviceMemory (memory_type))
    {
      h_mt = MemType::HOST;
      d_mt = memory_type;
    }
  else
    {
      h_mt = memory_type;
      d_mt = MemType::DEVICE;
    }
}

template <DeviceCopyable T>
void
Memory<T>::deleteWrappedHost (void *ptr, std::size_t alignment) noexcept
{
  if (ptr != nullptr)
    {
      ::operator delete (ptr, std::align_val_t{ alignment });
    }
}

template <DeviceCopyable T>
std::size_t
Memory<T>::checkedBytes (int size)
{
  if (size < 0)
    {
      vfemError ("memory size must be non-negative");
      return 0;
    }

  const auto count = static_cast<std::size_t> (size);
  if (count > std::numeric_limits<std::size_t>::max () / sizeof (T))
    {
      vfemError ("memory allocation size overflow");
      return 0;
    }

  return count * sizeof (T);
}

template <DeviceCopyable T>
bool
Memory<T>::empty () const noexcept
{
  return record_ == nullptr || capacity == 0;
}

template <DeviceCopyable T>
void
Memory<T>::allocate (int size)
{
  allocate (size, h_mt, d_mt);
}

template <DeviceCopyable T>
void
Memory<T>::allocate (int size, MemType mt)
{
  configureTypes (mt);
  allocate (size, h_mt, d_mt);
}

template <DeviceCopyable T>
void
Memory<T>::allocate (int size, MemType host_mt, MemType device_mt)
{
  if (!isHostMemory (host_mt))
    {
      vfemError ("host memory type is not host-accessible");
      return;
    }
  if (!isDeviceMemory (device_mt))
    {
      vfemError ("device memory type is not device-accessible");
      return;
    }
  if (size == 0)
    {
      reset ();
      h_mt = host_mt;
      d_mt = device_mt;
      return;
    }
  const std::size_t bytes = checkedBytes (size);
  if (bytes == 0)
    {
      reset ();
      h_mt = host_mt;
      d_mt = device_mt;
      return;
    }
  constexpr std::size_t alignment = alignof (T) > alignof (std::max_align_t)
                                        ? alignof (T)
                                        : alignof (std::max_align_t);
  void *ptr = ::operator new (bytes, std::align_val_t{ alignment });
  auto record = std::make_shared<MemoryRecord> ();

  record->h_ptr = ptr;
  record->bytes = bytes;
  record->alignment = alignment;
  record->h_mt = host_mt;
  record->d_mt = device_mt;
  record->h_deallocate = &Memory<T>::deleteWrappedHost;
  record->owns_h = true;
  record->state = MemoryState::UNINITIALIZED;
  unsigned new_flags = Registered | OWNS_HOST | OWNS_INTERNAL;
  if (host_mt == MemType::MANAGED && device_mt == MemType::MANAGED)
    {
      record->d_ptr = ptr;
      record->d_deallocate = &Memory<T>::deleteWrappedHost;
      record->owns_d = true;
      new_flags |= OWNS_DEVICE;
    }
  reset ();
  record_ = std::move (record);
  byte_offset_ = 0;
  capacity = size;
  h_mt = host_mt;
  d_mt = device_mt;
  flags = new_flags;
  h_ptr = static_cast<T *> (record_->h_ptr);
  d_ptr = static_cast<T *> (record_->d_ptr);
}

template <DeviceCopyable T>
void
Memory<T>::wrap (T *host_ptr, T *device_ptr, int size, MemType host_mt,
                 MemType device_mt, bool own, bool valid_host,
                 bool valid_device)
{
  const std::size_t bytes = checkedBytes (size);
  if (size < 0)
    {
      return;
    }
  if (!isHostMemory (host_mt) || !isDeviceMemory (device_mt))
    {
      vfemError ("invalid host/device memory type pair");
      return;
    }
  if ((host_mt == MemType::MANAGED) != (device_mt == MemType::MANAGED))
    {
      vfemError ("MANAGED memory must be used on both sides");
      return;
    }
  if (host_mt == MemType::MANAGED && host_ptr != device_ptr)
    {
      vfemError ("managed host/device pointers must be identical");
      return;
    }
  if (size > 0 && host_ptr == nullptr && device_ptr == nullptr)
    {
      vfemError ("at least one wrapped pointer must be non-null");
      return;
    }
  if (valid_host && host_ptr == nullptr)
    {
      vfemError ("host pointer is marked valid but is null");
      return;
    }
  if (valid_device && device_ptr == nullptr)
    {
      vfemError ("device pointer is marked valid but is null");
      return;
    }
  if (!hasRequiredAlignment (host_ptr, host_mt)
      || !hasRequiredAlignment (device_ptr, device_mt))
    {
      vfemError ("wrapped pointer does not satisfy memory type alignment");
      return;
    }

  Memory replacement;
  replacement.h_mt = host_mt;
  replacement.d_mt = device_mt;
  if (size == 0)
    {
      swap (replacement);
      return;
    }

  auto record = std::make_shared<MemoryRecord> ();
  record->h_ptr = host_ptr;
  record->d_ptr = device_ptr;
  record->bytes = bytes;
  record->h_mt = host_mt;
  record->d_mt = device_mt;
  record->alignment
      = MemoryManager::requiredAlignment (host_mt, device_mt, alignof (T));

  record->h_deallocate = isPlainHostType (host_mt)
                             ? &Memory<T>::deleteWrappedHost
                             : MemoryManager::get ().getDeallocate (host_mt);
  record->d_deallocate = MemoryManager::get ().getDeallocate (device_mt);
  record->owns_h = own && host_ptr != nullptr;
  record->owns_d = own && device_ptr != nullptr && device_ptr != host_ptr;

  if (own
      && ((record->owns_h && record->h_deallocate == nullptr)
          || (record->owns_d && record->d_deallocate == nullptr)))
    {
      vfemError ("no deallocator is registered for owned wrapped memory");
      return;
    }

  if (valid_host && valid_device)
    {
      record->state = MemoryState::SYNCHRONIZED;
    }
  else if (valid_host)
    {
      record->state = MemoryState::HOST_VALID;
    }
  else if (valid_device)
    {
      record->state = MemoryState::DEVICE_VALID;
    }
  else
    {
      record->state = MemoryState::UNINITIALIZED;
    }

  replacement.record_ = std::move (record);
  replacement.capacity = size;
  replacement.refreshView ();
  swap (replacement);
}

} // namespace vfem

#endif
