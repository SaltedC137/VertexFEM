#pragma once

#include <cstddef>
#include <type_traits>
#ifndef VFEM_ARRAY_HPP
#define VFEM_ARRAY_HPP

#include "config.hpp"
#include "error.hpp"
#include "global.hpp"
#include "mem_manager.hpp"

#include <cassert>
#include <cstring>
#include <span>

#ifdef VFEM_USE_OPENMP
#include <omp.h> // for parallelization
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

//
template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard {
  static constexpr bool IsReadOnly = (M == AccessMode::READ_ONLY);
  using PointerType = std::conditional_t<IsReadOnly, const T *, T *>;

public:
  //
  using value_type      = T;
  using pointer         = PointerType;
  using reference       = std::conditional_t<IsReadOnly, const T &, T &>;
  using iterator        = pointer;
  using difference_type = std::ptrdiff_t;

  // hot path for accessing the underlying data pointer
  pointer data() const noexcept { return ptr_; }
  int size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](int i) const noexcept {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "AccessGuard::operator[]: OOB");
    return ptr_[i];
  }

  iterator begin() const noexcept { return ptr_; }
  iterator end() const noexcept { return ptr_ + size_; }

  operator pointer() const noexcept { return ptr_; }

  std::span<std::conditional_t<IsReadOnly, const T, T>> span() const noexcept {
    return {ptr_, static_cast<std::size_t>(size_)};
  }

  // RAII for releasing the access guard
  AccessGuard(const AccessGuard &)             = delete;
  AccessGuard(&operator=(const AccessGuard &)) = delete;

  AccessGuard(AccessGuard &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_), owner_(other.owner_) {
    other.ptr_   = nullptr;
    other.size_  = 0;
    other.owner_ = nullptr;
  }
  ~AccessGuard() noexcept;

private:
  friend class Array<T>;
  AccessGuard(pointer p, int n, Array<T> *owner) noexcept;
  void release() noexcept;

  // pointer to control
  pointer ptr_{nullptr};
  int size_{0};
  Array<T> *owner_{nullptr};
};

// unfinish

template <ArrayElement T> class Array {

protected:
  Memory<T> memory_;

public:
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

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
