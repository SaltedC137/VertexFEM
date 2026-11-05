#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#ifndef VFEM_ARRAY_HPP
#define VFEM_ARRAY_HPP

#include "config.hpp" // IWYU pragma: keep
#include "error.hpp"
#include "global.hpp" // IWYU pragma: keep
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

// AccessGuard Lock
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

  reference operator[](int index) const noexcept {
    VFEM_HOT_ASSERT(index >= 0 && index < size_,
                    "AccessGuard::operator[]: OOB");
    return ptr_[index];
  }

  iterator begin() const noexcept { return ptr_; }
  iterator end() const noexcept { return ptr_ + size_; }

  operator pointer() const noexcept { return ptr_; }

  std::span<std::conditional_t<IsReadOnly, const T, T>> span() const noexcept {
    return {ptr_, static_cast<std::size_t>(size_)};
  }

  // RAII for releasing the access guard
  AccessGuard(const AccessGuard &)            = delete;
  AccessGuard &operator=(const AccessGuard &) = delete;

  AccessGuard(AccessGuard &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_), owner_(other.owner_) {
    other.ptr_   = nullptr;
    other.size_  = 0;
    other.owner_ = nullptr;
  }

  AccessGuard &operator=(AccessGuard &&other) noexcept {
    if (this != &other) {
      release();
      ptr_         = other.ptr_;
      size_        = other.size_;
      owner_       = other.owner_;
      other.ptr_   = nullptr;
      other.size_  = 0;
      other.owner_ = nullptr;
    }
    return *this;
  }

  ~AccessGuard() noexcept { release(); };

private:
  friend class Array<T>;
  AccessGuard(pointer p, int n, Array<T> *owner) noexcept
      : ptr_(p), size_(n), owner_(owner) {
    if (owner_) {
      owner_->active_guards_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void release() noexcept {
    if (owner_) {
      owner_->active_guards_.fetch_sub(1, std::memory_order_relaxed);
      owner_ = nullptr;
    }
    ptr_  = nullptr;
    size_ = 0;
  }

  // pointer to control
  pointer ptr_{nullptr};
  int size_{0};
  Array<T> *owner_{nullptr};
};

// RAII Access Guard Type Aliases
template <ArrayElement T>
using HostReadGuard = AccessGuard<T, MemorySide::HOST, AccessMode::READ_ONLY>;
template <ArrayElement T>
using HostWriteGuard = AccessGuard<T, MemorySide::HOST, AccessMode::WRITE_ONLY>;
template <ArrayElement T>
using HostRWGuard = AccessGuard<T, MemorySide::HOST, AccessMode::READ_WRITE>;
template <ArrayElement T>
using DeviceReadGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::READ_ONLY>;
template <ArrayElement T>
using DeviceWriteGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::WRITE_ONLY>;
template <ArrayElement T>
using DeviceRWGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::READ_WRITE>;

// unfinish
template <ArrayElement T> class Array {

protected:
  Memory<T> memory_;
  int size_{0};
  mutable std::atomic<int> active_guards_ = {0};

public:
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = T &;
  using const_reference = const T &;
  using pointer         = T *;
  using const_pointer   = const T *;
  // iterator on class array
  using iterator               = T *;
  using const_iterator         = const T *;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
  template <ArrayElement, MemorySide, AccessMode> friend class AccessGuard;

  // const T *acquireRead(MemorySide side, int count) const;
  // T *acquireWrite(MemorySide side, int count);
  // T *acquireRW(MemorySide side, int count);

  // void growTo(int min_capacity);
  // void shrinkTosize();

  void ensureNoActiveGuards(const char *op) const {
    if (active_guards_.load(std::memory_order_relaxed) > 0) [[unlikely]] {
      vfemError(std::string("Array::") + op +
                ": cannot modify structure while AccessGuard is alive");
    }
  }

  static constexpr MemoryClass toMc(MemorySide s) noexcept {
    return s == MemorySide::HOST ? MemoryClass::HOST : MemoryClass::DEVICE;
  }

  bool isUnintialized() const noexcept {
    return capacity() > 0 && !memory_.hostIsValid() && !memory_.deviceIsValid();
  }

  static Memory<T> allocateLike(const Memory<T> &proto, int cap) {
    if constexpr (requires {
                    proto.getHostMemType();
                    proto.getDeviceMemType();
                  }) {
      Memory<T> m(cap, proto.getHostMemType(), proto.getDeviceMemType());
      m.useDevice(proto.useDevice());
      return m;
    } else if constexpr (requires {
                           proto.hostMemType();
                           proto.deviceMemType();
                         }) {
      Memory<T> m(cap, proto.hostMemType(), proto.deviceMemType());
      m.useDevice(proto.useDevice());
      return m;
    } else {
      Memory<T> m(cap);
      m.useDevice(proto.useDevice());
      return m;
    }
  }

  const T *acquireRead(MemorySide side, int count) const {
    if (count < 0 || count > size_) [[unlikely]] {
      vfemError("Array::acquireRead: invalid count");
      return nullptr;
    }
    if (count == 0) {
      return nullptr;
    }
    return memory_.read(toMc(side), count);
  }

  T *acquireWrite(MemorySide side, int count) {
    if (count < 0 || count > size_) [[unlikely]] {
      vfemError("Array::acquireWrite: invalid count");
      return nullptr;
    }
    if (count == 0) {
      return nullptr;
    }
    if (isUnintialized()) {
      return memory_.write(toMc(side), capacity());
    }
    return memory_.write(toMc(side), count);
  }

  T *acquireRW(MemorySide side, int count) {
    if (count < 0 || count > size_) [[unlikely]] {
      vfemError("Array::acquireRW: invalid count");
      return nullptr;
    }
    if (count == 0) {
      return nullptr;
    }
    if (isUnintialized()) {
      return memory_.write(toMc(side), capacity());
    }
    return memory_.readWrite(toMc(side), count);
  }

  void growTo(int min_capacity) {
    const int cur_cap = capacity();
    const int new_cap = (cur_cap > std::numeric_limits<int>::max() / 2)
                            ? min_capacity
                            : std::max(min_capacity, 2 * cur_cap);

    Memory<T> new_mem = allocateLike(memory_, new_cap);
    if (size_ > 0) {
      if constexpr (requires { new_mem.copyFrom(memory_, size_); }) {
        new_mem.copyFrom(memory_, size_);
      } else {
        const T *old = acquireRead(MemorySide::HOST, size_);
        T *dst       = new_mem.write(MemoryClass::HOST, new_cap);
        std::memcpy(dst, old, byteSize());
      }
    }
    memory_ = std::move(new_mem);
  }

  void shrinkToSize() {
    if (size_ == capacity()) {
      return;
    }
    if (size_ == 0) {
      memory_.reset();
      return;
    }
    Memory<T> new_mem = allocateLike(memory_, size_);
    if constexpr (requires { new_mem.copyFrom(memory_, size_); }) {
      new_mem.copyFrom(memory_, size_);
    } else {
      const T *old = acquireRead(MemorySide::HOST, size_);
      T *dst       = new_mem.write(MemoryClass::HOST, size_);
      std::memcpy(dst, old, byteSize());
    }
    memory_ = std::move(new_mem);
  }

public:
  Array() noexcept = default;

  explicit Array(MemType mt) : memory_(mt) {}
  explicit Array(int n) : memory_(n), size_(n) {}
  Array(int n, MemType mt) : memory_(n, mt), size_(n) {}
  Array(int n, MemType h_mt, MemType d_mt) : memory_(n, h_mt, d_mt), size_(n) {}

  Array(T *p, int n, bool own) {
    memory_.wrap(p, n, MemType::HOST, own);
    size_ = n;
  }
  // Wraps an existing pointer with specified memory type and ownership
  Array(T *p, int n, MemType mt, bool own) {
    memory_.wrap(p, n, mt, own);
    size_ = n;
  }

  Array(std::initializer_list<T> init)
      : memory_(static_cast<int>(init.size())),
        size_(static_cast<int>(init.size())) {
    if (size_ > 0) {
      T *dst_ = acquireWrite(MemorySide::HOST, size_);
      std::ranges::copy(init, dst_);
    }
  }

  template <std::ranges::sized_range R>
    requires(!std::same_as<std::remove_cvref_t<R>, Array<T>> &&
             std::convertible_to<std::ranges::range_value_t<R>, T>)
  explicit Array(R &&range)
      : Array(static_cast<int>(std::ranges::size(range))) {
    if (size_ > 0) {
      T *dst = acquireWrite(MemorySide::HOST, size_);
      std::ranges::copy(range, dst);
    }
  }

  Array(const Array &other)
      : memory_(allocateLike(other.memory_, other.size_)), size_(other.size_) {
    if (size_ > 0) {
      if constexpr (requires { memory_.copyFrom(other.memory_, size_); }) {
        memory_.copyFrom(other.memory_, size_);
      } else {
        const T *src = other.acquireRead(MemorySide::HOST, size_);
        T *dst       = acquireWrite(MemorySide::HOST, size_);
        std::memcpy(dst, src, byteSize());
      }
    }
  }

  // Copy constructor for Array with different element type, provided that the
  // conversion is possible.
  template <ArrayElement CT>
    requires(!std::same_as<CT, T> && std::convertible_to<CT, T>)
  explicit Array(const Array<CT> &other)
      : memory_(other.size()), size_(other.size()) {
    memory_.useDevice(other.useDevice());
    if (size_ > 0) {
      auto src = other.hostRead();
      T *dst   = acquireWrite(MemorySide::HOST, size_);
      for (int i = 0; i < size_; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
  }

  Array(Array &&other) noexcept
      : memory_(std::move(other.memory_)), size_(other.size_) {
    VFEM_HOT_ASSERT(other.active_guards_ == 0,
                    "Array(Array&&): moving Array with active AccessGuard");
    other.size_ = 0;
  }

  ~Array() = default;

  //  operators

  Array &operator=(const Array &other) {
    if (this != &other) {
      ensureNoActiveGuards("operator=(const  Array&)");
      Array tmp(other);
      swap(tmp);
    }
    return *this;
  }

  Array &operator=(Array &&other) noexcept {
    if (this != &other) {
      ensureNoActiveGuards("operator=(const Array&&)");
      VFEM_HOT_ASSERT(
          other.active_guards_ == 0,
          "Array::operator=(Array&&): source has active AccessGuard");

      memory_     = std::move(other.memory_);
      size_       = other.size_;
      other.size_ = 0;
    }
    return *this;
  }

  template <ArrayElement CT>
    requires(!std::same_as<CT, T> && std::convertible_to<CT, T>)
  Array &operator=(const Array<CT> &other) {
    ensureNoActiveGuards("operator=(const Array<CT>&)");
    setSize(other.size());
    if (size_ > 0) {
      auto src = other.hostRead();
      T *dst   = acquireWrite(MemorySide::HOST, size_);
      for (int i = 0; i < size_; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
    return *this;
  }

  // factory reference rebinding methods

  [[nodiscard]] static Array<T> aliasOf(const Array &base) {
    Array out;
    out.memory_.makeAlias(base.memory_, 0, base.size_);
    out.size_ = base.size_;
    return out;
  }

  [[nodiscard]] static Array<T> aliasOf(const Array &base, int offset, int n) {
    if (offset < 0 || n < 0 || offset + n > base.size_) [[unlikely]] {
      vfemError("Array::alias_of: range out of bounds");
      return {};
    }
    Array out;
    out.memory_.makeAlias(base.memory_, offset, n);
    out.size_ = n;
    return out;
  }

  [[nodiscard]] static Array<T> wrap(T *p, int n, MemType mt, bool own) {
    Array out;
    out.memory_.wrap(p, n, mt, own);
    out.size_ = n;
    return out;
  }

  // HPC zero-copy

  void makeRef(const Array &base, int offset, int n) {
    ensureNoActiveGuards("makeRef");
    if (offset < 0 || n < 0 || offset + n > base.size_) [[unlikely]] {
      vfemError("Array::make_ref: range out of bounds");
      return;
    }
    memory_.makeAlias(base.memory_, offset, n);
    size_ = n;
  }

  void makeRef(const Array &base) { makeRef(base, 0, base.size_); }

  void makeRef(T *p, int n, MemType mt = MemType::HOST, bool own = false) {
    ensureNoActiveGuards("makeRef(ptr)");
    if (n < 0) [[unlikely]] {
      vfemError("Array::make_ref: negative size");
      return;
    }
    memory_.wrap(p, n, mt, own);
    size_ = n;
  }

  // size, capacity, empty, byteSize
  int size() const noexcept { return size_; }
  int capacity() const noexcept { return memory_.capaCity(); }
  bool empty() const noexcept { return size_ == 0; }

  // memory usage in bytes
  std::size_t byteSize() const noexcept {
    return static_cast<std::size_t>(size_) * sizeof(T);
  }
  std::size_t byteCapacity() const noexcept {
    return static_cast<std::size_t>(capacity()) * sizeof(T);
  }
  std::size_t memoryUsage() const noexcept { return byteCapacity(); }

  void reserve(int n) {
    ensureNoActiveGuards("reserve");
    if (n > capacity()) [[unlikely]] {
      growTo(n);
    }
  }

  void shrinkToFit() {
    ensureNoActiveGuards("shrinkToFit");
    shrinkToSize();
  }

  // resize function

  void resize(int n) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::resize: negative");
      return;
    }
    ensureNoActiveGuards("resize");
    if (n > capacity()) {
      growTo(n);
    }
    size_ = n;
  }

  void resize(int n, const T &v) {
    const int old = size_;
    resize(n);
    if (n > old) {
      T *p = (old == 0) ? acquireWrite(MemorySide::HOST, size_)
                        : acquireRW(MemorySide::HOST, size_);
      std::fill(p + old, p + n, v);
    }
  }

  void resize(int n, MemType mt) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::resize: negative");
      return;
    }
    ensureNoActiveGuards("resize(n, mt)");
    const bool use_dev = memory_.useDevice();
    Memory<T> new_mem(n, mt);
    new_mem.useDevice(use_dev);
    memory_ = std::move(new_mem);
    size_   = n;
  }

  void setSize(int n) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::set_size: negative");
      return;
    }
    ensureNoActiveGuards("setSize");
    if (n > capacity()) {
      Memory<T> new_mem = allocateLike(memory_, n);
      memory_           = std::move(new_mem);
    }
    size_ = n;
  }
  void setSize(int n, MemType mt) { resize(n, mt); }

  void clear() noexcept {
    VFEM_HOT_ASSERT(active_guards_ == 0, "Array::clear: active guard alive");
    size_ = 0;
  }

  // AccessGuard factory methods
  [[nodiscard]] HostReadGuard<T> hostRead() const {
    return {acquireRead(MemorySide::HOST, size_), size_,
            const_cast<Array *>(this)};
  }

  [[nodiscard]] HostWriteGuard<T> hostWrite() {
    return {acquireWrite(MemorySide::HOST, size_), size_, this};
  }

  [[nodiscard]] HostRWGuard<T> hostRW() {
    return {acquireRW(MemorySide::HOST, size_), size_, this};
  }

  [[nodiscard]] DeviceReadGuard<T> deviceRead() const {
    return {acquireRead(MemorySide::DEVICE, size_), size_,
            const_cast<Array *>(this)};
  }

  [[nodiscard]] DeviceWriteGuard<T> deviceWrite() {
    return {acquireWrite(MemorySide::DEVICE, size_), size_, this};
  }

  [[nodiscard]] DeviceRWGuard<T> deviceRW() {
    return {acquireRW(MemorySide::DEVICE, size_), size_, this};
  }

  [[nodiscard]] std::span<const T> view() const {
    if (size_ == 0) {
      return {};
    }
    return {acquireRead(MemorySide::HOST, size_),
            static_cast<std::size_t>(size_)};
  }

  [[nodiscard]] std::span<T> mutableView() {
    if (size_ == 0) {
      return {};
    }
    return {acquireRW(MemorySide::HOST, size_),
            static_cast<std::size_t>(size_)};
  }

  // HPC / OPENMP / HYPRE

  const T *readPtr(bool on_dev = false) const {
    return acquireRead(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }
  T *writePtr(bool on_dev = false) {
    return acquireWrite(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }
  T *rwPtr(bool on_dev = false) {
    return acquireRW(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }

  // convenience methods for host/device access
  const T *hostReadPtr() const { return acquireRead(MemorySide::HOST, size_); }
  T *hostWritePtr() { return acquireWrite(MemorySide::HOST, size_); }
  T *hostRWPtr() { return acquireRW(MemorySide::HOST, size_); }
  const T *deviceReadPtr() const {
    return acquireRead(MemorySide::DEVICE, size_);
  }
  T *deviceWritePtr() { return acquireWrite(MemorySide::DEVICE, size_); }
  T *deviceRWPtr() { return acquireRW(MemorySide::DEVICE, size_); }

  // element access and iterator

  T &operator[](int i) {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::operator[]: OOB");
    return acquireRW(MemorySide::HOST, size_)[i];
  }
  const T &operator[](int i) const {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::operator[]: OOB");
    return acquireRead(MemorySide::HOST, size_)[i];
  }
  T &at(int i) {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::at: OOB");
    return acquireRW(MemorySide::HOST, size_)[i];
  }
  const T &at(int i) const {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::at: OOB");
    return acquireRead(MemorySide::HOST, size_)[i];
  }
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
