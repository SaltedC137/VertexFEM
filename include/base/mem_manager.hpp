#pragma once

#include <type_traits>
#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

#include "../config/cconfig.hpp"
#include "../config/config.hpp"

#ifdef VFEM_USE_MPI
#define HYPRE_TIMING
#include <HYPRE_utilities.h>
#if (21400 <= VFEM_HYPRE_VERSION) && (VFEM_HYPRE_VERSION < 21900)
#include <_hypre_utilities.h>
#endif
#endif

namespace vfem {

enum class MemType {
  HOST,          // 0 HOST memory (default)
  HOST_32,       // 1 32-bit addressable HOST memory
  HOST_64,       // 2 64-bit addressable HOST memory
  HOST_DEBUG,    // 3 HOST memory with debugging enabled
  HOST_UMPIRE,   // 4 HOST memory allocated via Umpire (if available)
  HOST_PINNED,   // 5 HOST pinned memory (page-locked, for faster GPU transfers)
  MANAGED,       // 6 MANAGED memory (accessible from both HOST and DEVICE)
  DEVICE,        // 7 DEVICE memory (GPU memory)
  DEVICE_DEBUG,  // 8 DEVICE memory with debugging enabled
  DEVICE_UMPIRE, // 9 DEVICE memory allocated via Umpire (if available)
  DEVICE_UMPIRE_2, // 10 DEVICE memory allocated via Umpire with different
                   // settings (e.g., different pool or arena)
  SIZE,            // 11 Number of memory types (must be last)
  PRESERVE, // 12 Preserve the existing memory type (used in certain operations
            // to indicate that the memory type should not be changed)
  DEFAULT // 13 Default memory type (used in certain operations to indicate that
          // the default memory type should be used, which is typically HOST or
          // MANAGED depending on the context)
};

/// Static casts to 'int' and constexprs for MemType values
constexpr int MemTypeSize = static_cast<int>(MemType::SIZE);
constexpr int HostMemType = static_cast<int>(MemType::HOST);
constexpr int HostMemTypeSize = static_cast<int>(MemType::DEVICE);
constexpr int DeviceMemType = static_cast<int>(MemType::MANAGED);
constexpr int DeviceMemTypeSize = MemTypeSize - DeviceMemType;

extern VFEM_EXPORT const char *MemTypeName[MemTypeSize];

enum class MemoryClass {

  HOST,
  HOST_32,
  HOST_64,
  DEVICE,
  MANAGED,

};

inline bool IsHostMemory(MemType type) { return type <= MemType::MANAGED; }

inline bool IsDeviceMemory(MemType type) {
  return type >= MemType::MANAGED && type < MemType::SIZE;
}

MemType GetMemType(MemoryClass mc, int index);

bool MemClassContainsType(MemoryClass mc, MemType type);

MemoryClass operator*(MemoryClass mc1, MemoryClass mc2);

/// Class used by VFEM to manage memory allocations and deallocations across
/// different memory types.

template <typename T>
concept DeviceCopyable =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

template <DeviceCopyable T> class Memory {
protected:
  friend class MemoryManager;
  friend void MemoryPrintFlags(unsigned flags);

  enum FlagMask : unsigned {

#ifndef REGISTERED
    REGISTERED = 1 << 0,
#endif
    Registered = 1 << 0,

    OWNS_HOST = 1 << 1,
    OWNS_DEVICE = 1 << 2,

    OWNS_INTERNAL =
        1 << 3, // Owns memory allocated internally by VFEM (not user-provided)

    VALID_HOST = 1 << 4,
    VALID_DEVICE = 1 << 5,

    USE_DEVICE = 1 << 6, // Indicates that the device pointer should be used for
                         // operations (e.g., when memory is mirrored)

    ALIAS = 1 << 7, // Pointer is an alias
  };

  /// Pointer to the memory. Not owned.

  T *h_ptr;
  int capacity;
  MemType h_mt;
  mutable unsigned flags;

public:
  // Default constructor initializes to an empty, invalid state
  constexpr Memory() noexcept { Reset(); }

  [[deprecated("Use MakeAlias or explicit Copy to avoid multiple ownership")]]
  Memory(const Memory &) = default;

  Memory(Memory &&other) noexcept {
    *this = other;
    other.Reset();
  }

  [[deprecated("Use MakeAlias or explicit Copy to avoid multiple ownership")]]
  Memory &operator=(const Memory &) = default;

  Memory &operator=(Memory &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    *this = other;
    other.Reset();
    return *this;
  }

  
  void Reset() noexcept;
};

} // namespace vfem

#endif
