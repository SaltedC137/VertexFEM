#pragma once

#include <type_traits>
#ifndef VFEM_ARRAY_HPP
#define VFEM_ARRAY_HPP

#include "config.hpp"
#include "error.hpp"
#include "global.hpp"
#include "mem_manager.hpp"

#ifdef VFEM_USE_OPENMP
#include <omp.h>
#endif

namespace vfem {

template <typename T>
concept ArrayElement = DeviceCopyable<T>;

template <ArrayElement T> class Array;  // 1 demension
template <ArrayElement T> class Matrix; // 2 demensions
template <ArrayElement T> class Tensor; // 3 demensions

// AccessGuard is a helper class that provides safe access to the underlying
// data of an Array, Matrix, or Tensor object. It ensures that the data is
// properly synchronized between host and device memory, and that the
// appropriate access mode (read-only, write-only, or read-write) is
// enforced.

template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard;

template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard {
  static constexpr bool IsReadOnly = (M == AccessMode::READ_ONLY);
  using PointerType = std::conditional_t<IsReadOnly, const T *, T *>;

public:
  using value_type = T;
  using pointer    = PointerType;

  pointer data() const noexcept { return ptr_; }

private:
  friend class Array<T>;
  AccessGuard(pointer p, int n, Array<T> *owner) noexcept;
  pointer ptr_{nullptr};
  int size_{0};
  Array<T> *owner_{nullptr};
};

template <ArrayElement T> class Array {
public:
private:
};

template <ArrayElement T> class Matrix {
public:
private:
};

template <ArrayElement T> class Tensor {
public:
private:
};

} // namespace vfem
#endif // VFEM_ARRAY_HPP
