#pragma once

#include <cstdint>
#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

#include "config.hpp"
#include "error.hpp"

#include <cstddef>
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

/// Class used by VFEM to manage memory allocations and deallocations across
/// different memory types.

template <typename T>
concept DeviceCopyable
    = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

template <DeviceCopyable T> class Memory
{
protected:
  friend class MemoryManager;
  friend void memoryPrintFlags (unsigned flags);

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

    ALIAS = 1 << 7, // Pointer is an alias
  };

  /// Pointer to the memory. Not owned.

  T *h_ptr;
  T *d_ptr;
  int capacity;
  MemType h_mt;
  MemType d_mt;
  mutable unsigned flags;

public:
  // Default constructor initializes to an empty, invalid state
  constexpr Memory () noexcept { reSet (); }

  [[deprecated ("Use MakeAlias or explicit Copy to avoid multiple ownership")]]

  Memory (const Memory &)
      = default;

  Memory (Memory &&other) noexcept
  {
    *this = other;
    other.reSet ();
  }

  [[deprecated ("Use MakeAlias or explicit Copy to avoid multiple ownership")]]

  Memory &operator= (const Memory &)
      = default;

  Memory &
  operator= (Memory &&other) noexcept
  {
    if (this == &other)
      {
        return *this;
      }
    *this = other;
    other.reSet ();
    return *this;
  }

  // Create a new memory allocation of the given size. If the memory is already
  // allocated, it will be deallocated and reallocated. The memory type is set
  // to the default host memory type (HOST).

  explicit Memory (int size) { alloCate (size); }

  // Create a new memory allocation of the given size and memory type. If the
  // memory is already allocated, it will be deallocated and reallocated.

  explicit Memory (MemType mt) { reSet (mt); }

  Memory (int size, MemType mt) { alloCate (size, mt); }

  Memory (int size, MemType h_mt, MemType d_mt)
  {
    alloCate (size, h_mt, d_mt);
  }

  explicit Memory (T *ptr, int size, MemType mt, bool own)
  {
    wrap (ptr, size, mt, own);
  }

  ~Memory () = default;

  // Note: The destructor is not marked noexcept because it may throw if the
  // destructor of T throws during deallocation.

  void
  sWap (Memory &other)
  {
    Memory temp (*this);
    *this = other;
    other = temp;
  }

  // Reset memory to an empty, invalid state. Does not deallocate any memory or
  // change the memory type.
  void reSet () noexcept;

  // Reset the host memory and update the memory type. Does not deallocate any
  // memory or change
  void reSet (MemType host_mt);

  bool
  emPty () const noexcept
  {
    return h_ptr == nullptr;
  }

  inline void alloCate (int size);

  inline void alloCate (int size, MemType mt);

  inline void alloCate (int size, MemType h_mt, MemType d_mt);

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

  inline operator T *() noexcept;

  inline operator const T *() const noexcept;

  template <typename U> inline explicit operator U *() noexcept;

  template <typename U> inline explicit operator const U *() const noexcept;

  void
  reLease () noexcept
  {
    if ((flags & OWNS_HOST) && h_ptr)
      {
        delete[] h_ptr;
      }
    reSet ();
  }

  void deleteDevice (bool copy_to_host = true);
  int
  caPacity () const noexcept
  {
    return capacity;
  }

  // error checking for valid host/device pointers based on memory type and
  // ownership flags. This is used by MemoryManager and other internal
  // components to ensure that memory operations
  bool hostIsValid () const noexcept;

  bool deviceIsValid () const noexcept;

  bool
  ownsHostPtr () const noexcept
  {
    return flags & OWNS_HOST;
  };

  void
  setHostPtrOwner (bool own) noexcept
  {
    flags = own ? (flags | OWNS_HOST) : (flags & ~OWNS_HOST);
  };

  bool
  ownsDevicePtr () const noexcept
  {
    return flags & OWNS_DEVICE;
  };

  bool
  useDevice () const noexcept
  {
    return flags & USE_DEVICE;
  };

  void
  useDevice (bool use_dev) const noexcept
  {
    flags = use_dev ? (flags | USE_DEVICE) : (flags & ~USE_DEVICE);
  };

  // Getters for memory type and ownership flags (used by MemoryManager and
  // other internal components)
  MemType getHostMemType () const noexcept;

  MemType getDeviceMemType () const noexcept;

  MemType getMemType () const noexcept;

  // Memory access methods

  inline T *readWrite (MemoryClass mc, int size);

  inline const T *read (MemoryClass mc, int size) const;

  inline T *write (MemoryClass mc, int size);

  inline void sync (const Memory &other) const;

  inline void syneAlias (const Memory &base, int alias_size) const;

  // Memory type query methods
  inline MemType getMemoryType () const;

  inline MemType getHostMemoryType () const;

  inline MemType getDeviceMemoryType () const;

  // Copy type methods
  inline void copyFrom (const Memory &other, int size);

  inline void copyTo (const Memory &other, int size) const;

  inline void copyFromHost (const T *host_ptr, int size);

  inline void copyToHost (T *host_ptr, int size) const;

  // Print the flags for debugging purposes

  inline void printFlags () const;

  inline int compareHostAndDevice (int size) const;

private:
  static constexpr std::size_t
  defAlignBytes ()
  {
    using namespace std;
    return alignof (max_align_t);
  }

  static constexpr std::size_t def_align_bytes = defAlignBytes ();

  static constexpr std::size_t new_align_bytes
      = alignof (T) > def_align_bytes ? alignof (T) : def_align_bytes;

  template <std::size_t align_bytes, bool dummy = true> struct Alloc
  {
    static T *
    New (std::size_t size)
    {
      return new T[size];
    }
    static void
    Delete (T *ptr)
    {
      delete[] ptr;
    }
  };

  // Specialization for aligned allocation using C++17's aligned new/delete

  static T *
  newHost (std::size_t size)
  {
    return Alloc<new_align_bytes>::New (size);
  }
  static void
  deleteHost (T *ptr)
  {
    Alloc<new_align_bytes>::Delete (ptr);
  }

  template <MemType mt, bool dummy = true> struct AllocMem
  {
    static T *
    New (std::size_t size)
    {
      return Alloc<new_align_bytes>::New (size);
    }
    static void
    Delete (T *ptr)
    {
      Alloc<new_align_bytes>::Delete (ptr);
    }
  };

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

class VFEM_EXPORT MemoryManager
{
private:
  typedef MemType MType;
  typedef Memory<int> MemInt;

  // MemoryManager is a singleton class that manages memory allocations and
  // deallocations across different memory types. It provides methods to
  // allocate, deallocate, and manage memory for different types of data.
  template <typename T> friend class Memory;

public:
  using allocateFunc = void *(*)(std::size_t size, MemType mt);
};



} // namespace vfem

#endif
