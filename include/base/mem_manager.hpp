#pragma once

#include <type_traits>
#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

#include "../config/config.hpp"

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

inline bool
IsHostMemory (MemType type)
{
  return type <= MemType::MANAGED;
}

inline bool
IsDeviceMemory (MemType type)
{
  return type >= MemType::MANAGED && type < MemType::SIZE;
}

MemType GetMemType (MemoryClass mc, int index);

bool MemClassContainsType (MemoryClass mc, MemType type);

MemoryClass operator* (MemoryClass mc1, MemoryClass mc2);

/// Class used by VFEM to manage memory allocations and deallocations across
/// different memory types.

template <typename T>
concept DeviceCopyable
    = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

template <DeviceCopyable T> class Memory
{
protected:
  friend class MemoryManager;
  friend void MemoryPrintFlags (unsigned flags);

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
  constexpr Memory () noexcept { Reset (); }

  [[deprecated ("Use MakeAlias or explicit Copy to avoid multiple ownership")]]

  Memory (const Memory &)
      = default;

  Memory (Memory &&other) noexcept
  {
    *this = other;
    other.Reset ();
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
    other.Reset ();
    return *this;
  }

  // Create a new memory allocation of the given size. If the memory is already
  // allocated, it will be deallocated and reallocated. The memory type is set
  // to the default host memory type (HOST).

  explicit Memory (int size) { New (size); }

  // Create a new memory allocation of the given size and memory type. If the
  // memory is already allocated, it will be deallocated and reallocated.

  explicit Memory (MemType mt) { Reset (mt); }

  Memory (int size, MemType mt) { New (size, mt); }

  Memory (int size, MemType h_mt, MemType d_mt) { New (size, h_mt, d_mt); }

  explicit Memory (T *ptr, int size, MemType mt, bool own)
  {
    Wrap (ptr, size, mt, own);
  }

  ~Memory () = default;

  // Note: The destructor is not marked noexcept because it may throw if the
  // destructor of T throws during deallocation.

  void
  Swap (Memory &other)
  {
    Memory temp (*this);
    *this = other;
    other = temp;
  }

  // Reset memory to an empty, invalid state. Does not deallocate any memory or
  // change the memory type.
  void Reset () noexcept;

  // Reset the host memory and update the memory type. Does not deallocate any
  // memory or change
  void Reset (MemType host_mt);

  bool
  Empty () const noexcept
  {
    return h_ptr == nullptr;
  }

  inline void New (int size);

  inline void New (int size, MemType mt);

  inline void New (int size, MemType h_mt, MemType d_mt);

  // Wrap the memory around an existing pointer. The memory will not be owned
  // by this object, and will not be deallocated when the object is destroyed.
  // The memory type is set to the default host memory type (HOST).

  inline void Wrap (T *ptr, int size);

  inline void Wrap (T *ptr, int size, bool own);

  inline void Wrap (T *ptr, int size, MemType mt, bool own);

  // void
  // Delete () noexcept
  // {
  //   if ((flags & OWNS_HOST) && h_ptr)
  //     {
  //       delete[] h_ptr;
  //     }
  //   Reset ();
  // }

  // void DeleteDevice (bool copy_to_host = true);

  int Capacity () const noexcept;

  // error checking for valid host/device pointers based on memory type and
  // ownership flags. This is used by MemoryManager and other internal
  // components to ensure that memory operations
  bool HostIsValid () const noexcept;

  bool DeviceIsValid () const noexcept;

  // OwnsHostPtr and OwnsDevicePtr check if the memory owns the host/device
  // pointer, which is important for determining whether the memory should be
  // deallocated when the Memory object is destroyed or reset. This is used by
  // MemoryManager and other internal components to manage memory lifetimes
  // correctly and avoid memory leaks
  
  bool OwnsHostPtr () const noexcept;
  bool OwnsDevicePtr () const noexcept;

  bool UseDevice () const noexcept;
  void UseDevice (bool use_dev) const noexcept;

  // Getters for memory type and ownership flags (used by MemoryManager and
  // other internal components)
  MemType GetHostMemType () const noexcept;

  MemType GetDeviceMemType () const noexcept;

  MemType GetMemType () const noexcept;

  // Memory access methods
  T *ReadWrite (MemoryClass mc, int size);
  const T *Read (MemoryClass mc, int size) const;
  T *Write (MemoryClass mc, int size);
};

} // namespace vfem

#endif
