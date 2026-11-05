#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

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

/**
 * @concept ArrayElement
 * @brief Constrains element types usable within Array, Matrix, and Tensor
 * containers.
 *
 * Requires `T` to satisfy `DeviceCopyable<T>`, mandating
 * `std::is_trivially_copyable_v<T>` and `std::is_standard_layout_v<T>`. This
 * ensures:
 * - Bitwise copyability (POD-like layout) required for Host <-> Device data
 * transfers (via cudaMemcpy, OpenMP target offload, or DMA) managed by
 * MemoryManager.
 * - Exclusion of types with virtual tables (vptr) or non-trivial destructors.
 * Pointer accessibility and host/device ABI compatibility remain caller
 * requirements.
 */
template <typename T>
concept ArrayElement = DeviceCopyable<T>;

template <ArrayElement T> class Array;
template <ArrayElement T> class Matrix;
template <ArrayElement T> class Tensor;

template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard;

/**
 * @class AccessGuard
 * @brief Scoped RAII access guard managing data synchronization, access
 * privilege, and structural locking.
 *
 * AccessGuard provides direct pointer access to the memory buffer of an Array
 * on a specified MemorySide with compile-time access mode enforcement.
 *
 * ### Dual-Memory Architecture & State Synchronization
 * When created via container accessors (e.g. hostRead(), deviceWrite()), the
 * underlying MemoryManager evaluates the container's MemoryState (EMPTY,
 * UNINITIALIZED, HOST_VALID, DEVICE_VALID, SYNCHRONIZED):
 * - @b READ_ONLY: If data on the requested side is stale, an automatic transfer
 * (D2H or H2D) is issued, transitioning the state to
 * `MemoryState::SYNCHRONIZED`.
 * - @b WRITE_ONLY: Bypasses data transfer, marks the target side as exclusively
 * valid
 *   (`HOST_VALID` or `DEVICE_VALID`), and invalidates the opposite side.
 * - @b READ_WRITE: Synchronizes stale data to the target side if necessary,
 * then marks the target side as exclusively valid, invalidating the opposite
 * side for subsequent writes.
 *
 * ### Access Privilege Enforcement
 * - If `M == AccessMode::READ_ONLY`, `pointer` resolves to `const T*` and
 * `reference` to `const T&`.
 * - If `M == AccessMode::WRITE_ONLY` or `AccessMode::READ_WRITE`, `pointer`
 * resolves to `T*` and `reference` to `T&`.
 *
 * ### Structural Modification Locking (Container Pinning)
 * Guards retain shared storage and a shared access counter: positive values
 * count readers, and -1 denotes an exclusive writer. While any guard is active,
 * structural mutations on
 * the Array (resize(), reserve(), setSize(), clear(), reallocation) are
 * prohibited and will raise an error via `ensureNoActiveGuards()`. Upon
 * destruction or release, the lease is released. Access acquisition and
 * structural operations require external host-thread serialization. Device
 * work must complete before its guard is released.
 *
 * AccessGuard is non-copyable (move-only) to guarantee unique RAII ownership of
 * each active lease.
 *
 * @tparam T Element type satisfying ArrayElement.
 * @tparam S Target memory side (MemorySide::HOST or MemorySide::DEVICE).
 * @tparam M Access mode (AccessMode::READ_ONLY, AccessMode::WRITE_ONLY, or
 * AccessMode::READ_WRITE).
 */
template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard {
  static constexpr bool IsReadOnly = (M == AccessMode::READ_ONLY);
  using PointerType = std::conditional_t<IsReadOnly, const T *, T *>;

public:
  using value_type      = T;
  using pointer         = PointerType;
  using reference       = std::conditional_t<IsReadOnly, const T &, T &>;
  using iterator        = pointer;
  using difference_type = std::ptrdiff_t;

  pointer data() const noexcept { return ptr_; }
  int size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](int index) const {
    VFEM_HOT_ASSERT(index >= 0 && index < size_,
                    "AccessGuard::operator[]: OOB");
    return ptr_[index];
  }

  iterator begin() const noexcept { return ptr_; }
  iterator end() const noexcept { return size_ > 0 ? ptr_ + size_ : ptr_; }

  operator pointer() const noexcept { return ptr_; }

  std::span<std::conditional_t<IsReadOnly, const T, T>> span() const noexcept {
    return {ptr_, static_cast<std::size_t>(size_)};
  }

  AccessGuard(const AccessGuard &)            = delete;
  AccessGuard &operator=(const AccessGuard &) = delete;

  AccessGuard(AccessGuard &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_), owner_(std::move(other.owner_)),
        memory_(std::move(other.memory_)) {
    other.ptr_   = nullptr;
    other.size_  = 0;
    other.owner_ = nullptr;
  }

  AccessGuard &operator=(AccessGuard &&other) noexcept {
    if (this != &other) {
      release();
      ptr_         = other.ptr_;
      size_        = other.size_;
      owner_       = std::move(other.owner_);
      memory_      = std::move(other.memory_);
      other.ptr_   = nullptr;
      other.size_  = 0;
      other.owner_ = nullptr;
    }
    return *this;
  }

  ~AccessGuard() noexcept { release(); };

private:
  friend class Array<T>;

  AccessGuard(int n, Array<T> *owner) : size_(n) {
    if (owner == nullptr) {
      size_ = 0;
      return;
    }
    if (!owner->active_guards_) {
      owner->active_guards_ = std::make_shared<std::atomic<int>>(0);
    }
    if constexpr (IsReadOnly) {
      ptr_ = owner->acquireRead(S, n);
    } else if constexpr (M == AccessMode::WRITE_ONLY) {
      ptr_ = owner->acquireWrite(S, n);
    } else {
      ptr_ = owner->acquireRW(S, n);
    }
    if (n > 0 && ptr_ == nullptr) {
      size_ = 0;
      return;
    }
    auto counter = owner->active_guards_;
    int expected = counter->load(std::memory_order_relaxed);
    if constexpr (IsReadOnly) {
      while (expected >= 0 && expected < std::numeric_limits<int>::max() &&
             !counter->compare_exchange_weak(expected, expected + 1,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
      }
      if (expected < 0 || expected == std::numeric_limits<int>::max()) {
        ptr_  = nullptr;
        size_ = 0;
        vfemError("AccessGuard: conflicting access or too many readers");
        return;
      }
    } else {
      expected = 0;
      if (!counter->compare_exchange_strong(expected, -1,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
        ptr_  = nullptr;
        size_ = 0;
        vfemError("AccessGuard: conflicting access");
        return;
      }
    }
    owner_  = std::move(counter);
    memory_ = owner->memory_;
  }

  void release() noexcept {
    if (owner_) {
      if constexpr (IsReadOnly) {
        owner_->fetch_sub(1, std::memory_order_release);
      } else {
        owner_->fetch_add(1, std::memory_order_release);
      }
      owner_.reset();
    }
    memory_.reset();
    ptr_  = nullptr;
    size_ = 0;
  }

  pointer ptr_{nullptr};
  int size_{0};
  std::shared_ptr<std::atomic<int>> owner_;
  Memory<T> memory_;
};

/**
 * @brief Host read-only access guard.
 *
 * Provides `const T*` pointer access to host memory. Triggers a Device-to-Host
 * (D2H) transfer if current state is `DEVICE_VALID`, updating state to
 * `MemoryState::SYNCHRONIZED`. Pins container structure against reallocation
 * while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using HostReadGuard = AccessGuard<T, MemorySide::HOST, AccessMode::READ_ONLY>;

/**
 * @brief Host write-only access guard.
 *
 * Provides mutable `T*` pointer access to host memory. Bypasses D2H transfer,
 * marks host data as exclusively valid (`MemoryState::HOST_VALID`), and
 * invalidates any device copy. Pins container structure against reallocation
 * while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using HostWriteGuard = AccessGuard<T, MemorySide::HOST, AccessMode::WRITE_ONLY>;

/**
 * @brief Host read-write access guard.
 *
 * Provides mutable `T*` pointer access to host memory. Triggers D2H transfer if
 * current state is `DEVICE_VALID`, then marks host data as exclusively valid
 * (`MemoryState::HOST_VALID`), invalidating the device copy. Pins container
 * structure against reallocation while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using HostRWGuard = AccessGuard<T, MemorySide::HOST, AccessMode::READ_WRITE>;

/**
 * @brief Device (GPU) read-only access guard.
 *
 * Provides `const T*` pointer access to device memory for GPU kernels or
 * accelerator routines. Triggers a Host-to-Device (H2D) transfer if current
 * state is `HOST_VALID`, updating state to `MemoryState::SYNCHRONIZED`. Pins
 * container structure against reallocation while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using DeviceReadGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::READ_ONLY>;

/**
 * @brief Device (GPU) write-only access guard.
 *
 * Provides mutable `T*` pointer access to device memory for GPU kernels.
 * Bypasses H2D transfer, marks device data as exclusively valid
 * (`MemoryState::DEVICE_VALID`), and invalidates the host copy. Pins container
 * structure against reallocation while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using DeviceWriteGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::WRITE_ONLY>;

/**
 * @brief Device (GPU) read-write access guard.
 *
 * Provides mutable `T*` pointer access to device memory. Triggers H2D transfer
 * if current state is `HOST_VALID`, then marks device data as exclusively valid
 * (`MemoryState::DEVICE_VALID`), invalidating the host copy. Pins container
 * structure against reallocation while active.
 *
 * @tparam T Element type satisfying ArrayElement.
 */
template <ArrayElement T>
using DeviceRWGuard =
    AccessGuard<T, MemorySide::DEVICE, AccessMode::READ_WRITE>;

/**
 * @class Array
 * @brief 1D dense array supporting heterogeneous memory, zero-copy wrapping,
 * lazy synchronization, and RAII guard tracking.
 *
 * @details Array manages a contiguous 1D buffer backed by @c Memory<T>,
 * abstracting dual-space memory management between Host (CPU) and Device
 * (GPU/accelerator). Key features include:
 * - Lazy data synchronization: Migrates data on demand between Host and Device
 * upon request, tracking state transitions (UNINITIALIZED, HOST_VALID,
 * DEVICE_VALID, SYNCHRONIZED).
 * - Zero-copy wrapping & aliasing: Wraps external pointers or creates sub-views
 * without copying.
 * - RAII access guard tracking: Tracks active @c AccessGuard references via
 * atomic @c active_guards_ to prevent structural reallocations while raw
 * pointers are in use.
 * - Standard container compatibility: Exposes standard STL type aliases and
 * contiguous iterators.
 *
 * @tparam T Element type, constrained by the @c ArrayElement concept (requires
 * @c DeviceCopyable<T>).
 */

template <typename T> class BlockArray;
template <ArrayElement T> void swap(Array<T> &a, Array<T> &b) noexcept;
template <ArrayElement T> void swap(Matrix<T> &a, Matrix<T> &b) noexcept;
template <ArrayElement T> void swap(Tensor<T> &a, Tensor<T> &b) noexcept;
template <typename T> void swap(BlockArray<T> &a, BlockArray<T> &b) noexcept;

template <ArrayElement T> class Array {

protected:
  Memory<T> memory_;
  int size_{0};
  mutable std::shared_ptr<std::atomic<int>> active_guards_;

public:
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = T &;
  using const_reference = const T &;
  using pointer         = T *;
  using const_pointer   = const T *;
  // STL-Compatible Iterators
  using iterator               = T *;
  using const_iterator         = const T *;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
  template <ArrayElement> friend class Array;
  template <ArrayElement, MemorySide, AccessMode> friend class AccessGuard;

  bool ensureNoActiveGuards(const char *op) const {
    if (active_guards_ && active_guards_->load(std::memory_order_acquire) != 0)
        [[unlikely]] {
      vfemError(std::string("Array::") + op +
                ": cannot modify structure while AccessGuard is alive");
      return false;
    }
    return true;
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
    if (active_guards_ && active_guards_->load(std::memory_order_acquire) < 0)
        [[unlikely]] {
      vfemError("Array::acquireRead: active writer");
      return nullptr;
    }
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
    if (!ensureNoActiveGuards("acquireWrite")) {
      return nullptr;
    }
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
    if (!ensureNoActiveGuards("acquireRW")) {
      return nullptr;
    }
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
    if (new_mem.capaCity() != new_cap) {
      return;
    }
    if (size_ > 0 && !isUnintialized()) {
      const T *old = acquireRead(MemorySide::HOST, size_);
      if (old == nullptr) {
        return;
      }
      T *dst = new_mem.write(MemoryClass::HOST, new_cap);
      if (dst == nullptr) {
        return;
      }
      std::memcpy(dst, old, byteSize());
    }
    memory_ = std::move(new_mem);
    active_guards_.reset();
  }

  void shrinkToSize() {
    if (size_ == 0) {
      memory_.reset();
      active_guards_.reset();
      return;
    }
    if (size_ == capacity()) {
      return;
    }
    Memory<T> new_mem = allocateLike(memory_, size_);
    if (new_mem.capaCity() != size_) {
      return;
    }
    if (!isUnintialized()) {
      const T *old = acquireRead(MemorySide::HOST, size_);
      if (old == nullptr) {
        return;
      }
      T *dst = new_mem.write(MemoryClass::HOST, size_);
      if (dst == nullptr) {
        return;
      }
      std::memcpy(dst, old, byteSize());
    }
    memory_ = std::move(new_mem);
    active_guards_.reset();
  }

public:
  Array() noexcept = default;

  explicit Array(MemType mt) : memory_(mt) {}
  explicit Array(int n) : memory_(n), size_(memory_.capaCity()) {}
  Array(int n, MemType mt) : memory_(n, mt), size_(memory_.capaCity()) {}
  Array(int n, MemType h_mt, MemType d_mt)
      : memory_(n, h_mt, d_mt), size_(memory_.capaCity()) {}

  Array(T *p, int n, bool own) : Array(p, n, MemType::HOST, own) {}

  Array(T *p, int n, MemType mt, bool own) {
    memory_.wrap(p, n == 0 && p != nullptr && own ? 1 : n, mt, own);
    if (n == 0) {
      memory_.reset();
    }
    size_ = memory_.capaCity();
  }

  Array(std::initializer_list<T> init) {
    if (init.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      vfemError("Array: initializer list exceeds maximum size");
      return;
    }
    memory_.allocate(static_cast<int>(init.size()));
    size_ = memory_.capaCity();
    if (size_ > 0) {
      T *dst_ = acquireWrite(MemorySide::HOST, size_);
      if (dst_ != nullptr) {
        std::ranges::copy(init, dst_);
      }
    }
  }

  template <std::ranges::sized_range R>
    requires(!std::same_as<std::remove_cvref_t<R>, Array<T>> &&
             std::convertible_to<std::ranges::range_value_t<R>, T>)
  explicit Array(R &&range) {
    const auto count = std::ranges::size(range);
    if (std::cmp_greater(count, std::numeric_limits<int>::max())) {
      vfemError("Array: range exceeds maximum size");
      return;
    }
    memory_.allocate(static_cast<int>(count));
    size_ = memory_.capaCity();
    if (size_ > 0) {
      T *dst = acquireWrite(MemorySide::HOST, size_);
      if (dst != nullptr) {
        std::ranges::copy(range, dst);
      }
    }
  }

  Array(const Array &other)
      : memory_(allocateLike(other.memory_, other.size_)),
        size_(memory_.capaCity()) {
    if (size_ > 0 && !other.isUnintialized()) {
      const T *src = other.acquireRead(MemorySide::HOST, size_);
      if (src == nullptr) {
        memory_.reset();
        size_ = 0;
        return;
      }
      T *dst = acquireWrite(MemorySide::HOST, size_);
      if (dst == nullptr) {
        memory_.reset();
        size_ = 0;
        return;
      }
      std::memcpy(dst, src, byteSize());
    }
  }

  template <ArrayElement CT>
    requires(!std::same_as<CT, T> && std::convertible_to<CT, T>)
  explicit Array(const Array<CT> &other)
      : memory_(other.size_, other.memory_.getHostMemType(),
                other.memory_.getDeviceMemType()),
        size_(memory_.capaCity()) {
    memory_.useDevice(other.memory_.useDevice());
    if (size_ > 0) {
      auto src = other.hostRead();
      if (src.size() != size_) {
        memory_.reset();
        size_ = 0;
        return;
      }
      T *dst = acquireWrite(MemorySide::HOST, size_);
      if (dst == nullptr) {
        memory_.reset();
        size_ = 0;
        return;
      }
      for (int i = 0; i < size_; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
  }

  Array(Array &&other) noexcept
      : memory_(std::move(other.memory_)), size_(other.size_),
        active_guards_(std::move(other.active_guards_)) {
    other.size_ = 0;
  }

  ~Array() = default;

  Array &operator=(const Array &other) {
    if (this != &other) {
      if (!ensureNoActiveGuards("operator=(const Array&)")) {
        return *this;
      }
      Array tmp(other);
      if (tmp.size_ != other.size_) {
        return *this;
      }
      memory_.swap(tmp.memory_);
      std::swap(size_, tmp.size_);
      active_guards_.swap(tmp.active_guards_);
    }
    return *this;
  }

  // NOLINTNEXTLINE(performance-noexcept-move-constructor,bugprone-exception-escape)
  Array &operator=(Array &&other) {
    if (this != &other) {
      if (!ensureNoActiveGuards("operator=(Array&&)")) {
        return *this;
      }

      memory_        = std::move(other.memory_);
      size_          = other.size_;
      active_guards_ = std::move(other.active_guards_);
      other.size_    = 0;
    }
    return *this;
  }

  template <ArrayElement CT>
    requires(!std::same_as<CT, T> && std::convertible_to<CT, T>)
  Array &operator=(const Array<CT> &other) {
    if (!ensureNoActiveGuards("operator=(const Array<CT>&)")) {
      return *this;
    }
    auto src = other.hostRead();
    if (src.size() != other.size()) {
      return *this;
    }
    setSize(other.size());
    if (size_ != other.size()) {
      return *this;
    }
    if (size_ > 0) {
      T *dst = acquireWrite(MemorySide::HOST, size_);
      if (dst == nullptr) {
        return *this;
      }
      for (int i = 0; i < size_; ++i) {
        dst[i] = static_cast<T>(src[i]);
      }
    }
    return *this;
  }

  [[nodiscard]] static Array<T> aliasOf(const Array &base) {
    return aliasOf(base, 0, base.size_);
  }

  [[nodiscard]] static Array<T> aliasOf(const Array &base, int offset, int n) {
    if (offset < 0 || n < 0 || offset > base.size_ || n > base.size_ - offset)
        [[unlikely]] {
      vfemError("Array::alias_of: range out of bounds");
      return {};
    }
    Array out;
    if (n == 0) {
      out.memory_ = allocateLike(base.memory_, 0);
      return out;
    }
    if (!base.active_guards_) {
      base.active_guards_ = std::make_shared<std::atomic<int>>(0);
    }
    out.memory_.makeAlias(base.memory_, offset, n);
    out.active_guards_ = base.active_guards_;
    out.size_          = n;
    return out;
  }

  [[nodiscard]] static Array<T> wrap(T *p, int n, MemType mt, bool own) {
    return Array(p, n, mt, own);
  }

  /**
   * @brief Rebinds this Array to reference a sub-range of another Array without
   * copying.
   *
   * Establishes a zero-copy alias of @p base starting at element index @p
   * offset with length @p n. Both arrays share the underlying MemoryRecord and
   * its host/device synchronization state.
   *
   * @param base Source Array whose memory is to be aliased.
   * @param offset Zero-based element offset into @p base.
   * @param n Number of elements in the sub-range view.
   *
   * @note Modifying elements through this Array will affect @p base and vice
   * versa.
   * @warning Active AccessGuards must not be alive on this instance when
   * rebinding.
   */
  void makeRef(const Array &base, int offset, int n) {
    if (!ensureNoActiveGuards("makeRef")) {
      return;
    }
    if (offset < 0 || n < 0 || offset > base.size_ || n > base.size_ - offset)
        [[unlikely]] {
      vfemError("Array::make_ref: range out of bounds");
      return;
    }
    Array replacement = aliasOf(base, offset, n);
    if (replacement.size_ != n) {
      return;
    }
    memory_        = std::move(replacement.memory_);
    active_guards_ = std::move(replacement.active_guards_);
    size_          = n;
  }

  /**
   * @brief Rebinds this Array to reference the entire storage of another Array.
   * Equivalent to `makeRef(base, 0, base.size())`.
   * @param base Source Array whose memory buffer is to be aliased.
   */
  void makeRef(const Array &base) { makeRef(base, 0, base.size_); }

  /**
   * @brief Rebinds this Array to wrap an externally allocated raw memory
   * buffer.
   *
   * Encapsulates external storage (e.g. MPI receive buffers, HYPRE vector
   * pointers, or stack memory) into VertexFEM's dual-memory management system.
   *
   * @param p Pointer to the external memory buffer.
   * @param n Number of elements in the buffer.
   * @param mt Memory space where @p p resides (HOST, DEVICE, MANAGED, etc.).
   * @param own If true, Array takes ownership and frees @p p on destruction;
   *            if false (default), the buffer is treated as a borrowed view.
   * @note Owned HOST pointers must come from new T[]; other types require a
   * matching registered deallocator. For pointers into this Array, use the
   * Array overload or a pointer within its current view on a valid memory side.
   */
  void makeRef(T *p, int n, MemType mt = MemType::HOST, bool own = false) {
    if (!ensureNoActiveGuards("makeRef(ptr)")) {
      return;
    }
    if (n < 0) [[unlikely]] {
      vfemError("Array::make_ref: negative size");
      return;
    }
    const T *current  = nullptr;
    bool owns_current = false;
    if (mt == memory_.getHostMemType() && memory_.hostIsValid()) {
      current      = memory_.read(MemoryClass::HOST, capacity());
      owns_current = memory_.ownsHostPtr();
    } else if (mt == memory_.getDeviceMemType() && memory_.deviceIsValid()) {
      current      = memory_.read(MemoryClass::DEVICE, capacity());
      owns_current = memory_.ownsDevicePtr();
    }
    if (p != nullptr && current != nullptr) {
      const auto address = reinterpret_cast<std::uintptr_t>(p);
      const auto start   = reinterpret_cast<std::uintptr_t>(current);
      if (address >= start && address - start <= byteCapacity()) {
        const auto offset_bytes = address - start;
        if (offset_bytes % sizeof(T) != 0 ||
            static_cast<std::size_t>(n) > static_cast<std::size_t>(capacity()) -
                                              offset_bytes / sizeof(T)) {
          vfemError("Array::make_ref: range exceeds current storage");
          return;
        }
        if (!own || owns_current) {
          if (n == 0) {
            memory_.reset();
            active_guards_.reset();
          } else {
            memory_.makeAlias(memory_,
                              static_cast<int>(offset_bytes / sizeof(T)), n);
          }
          size_ = n;
          return;
        }
        if (offset_bytes != 0) {
          vfemError(
              "Array::make_ref: cannot adopt an interior borrowed pointer");
          return;
        }
      }
    }
    Array replacement(p, n, mt, own);
    if (replacement.size_ != n) {
      return;
    }
    memory_ = std::move(replacement.memory_);
    active_guards_.reset();
    size_ = n;
  }

  int size() const noexcept { return size_; }

  int capacity() const noexcept { return memory_.capaCity(); }

  bool empty() const noexcept { return size_ == 0; }

  std::size_t byteSize() const noexcept {
    return static_cast<std::size_t>(size_) * sizeof(T);
  }

  std::size_t byteCapacity() const noexcept {
    return static_cast<std::size_t>(capacity()) * sizeof(T);
  }

  std::size_t memoryUsage() const noexcept { return byteCapacity(); }

  /**
   * @brief Pre-allocates storage capacity to accommodate at least @p n
   * elements.
   *
   * If @p n exceeds capacity(), triggers geometric growth and preserves
   * existing elements.
   * @param n Minimum desired capacity in elements.
   */
  void reserve(int n) {
    if (n < 0) {
      vfemError("Array::reserve: negative size");
      return;
    }
    if (!ensureNoActiveGuards("reserve")) {
      return;
    }
    if (n > capacity()) [[unlikely]] {
      growTo(n);
    }
  }

  /**
   * @brief Shrinks allocated buffer capacity to match the current size().
   * Releases excess memory back to the backend memory manager.
   */
  void shrinkToFit() {
    if (!ensureNoActiveGuards("shrinkToFit")) {
      return;
    }
    shrinkToSize();
  }

  /**
   * @brief Resizes the array to contain @p n elements.
   *
   * If @p n > capacity(), triggers capacity expansion and preserves existing
   * data. Extended elements are left uninitialized.
   * @param n Desired new size in elements.
   */
  void resize(int n) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::resize: negative");
      return;
    }
    if (!ensureNoActiveGuards("resize")) {
      return;
    }
    if (n > capacity()) {
      growTo(n);
      if (n > capacity()) {
        return;
      }
    }
    size_ = n;
  }

  /**
   * @brief Resizes the array to contain @p n elements, initializing new
   * elements with @p v.
   * @param n Desired new size in elements.
   * @param v Value to assign to all newly appended elements.
   */
  void resize(int n, const T &v) {
    const int old = size_;
    const T value = v;
    resize(n);
    if (size_ != n) {
      return;
    }
    if (n > old) {
      T *p = (old == 0) ? acquireWrite(MemorySide::HOST, size_)
                        : acquireRW(MemorySide::HOST, size_);
      if (p != nullptr) {
        std::fill(p + old, p + n, value);
      }
    }
  }

  /**
   * @brief Reallocates storage of exact size @p n with a specified memory type.
   * Discards previous contents and allocates a fresh buffer.
   * @param n Number of elements for the new allocation.
   * @param mt Backend memory type to allocate.
   */
  void resize(int n, MemType mt) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::resize: negative");
      return;
    }
    if (!ensureNoActiveGuards("resize(n, mt)")) {
      return;
    }
    const bool use_dev = memory_.useDevice();
    Memory<T> new_mem(n, mt);
    if (new_mem.capaCity() != n) {
      return;
    }
    new_mem.useDevice(use_dev);
    memory_ = std::move(new_mem);
    active_guards_.reset();
    size_ = n;
  }

  /**
   * @brief Adjusts array size, reallocating storage without preserving existing
   * data if capacity is exceeded.
   *
   * Fast size adjustment. If @p n > capacity(), setSize() allocates a fresh
   * buffer of exact size @p n without copying old elements, avoiding memory
   * copy overhead during assembly loops.
   * @param n Desired element count.
   */
  void setSize(int n) {
    if (n < 0) [[unlikely]] {
      vfemError("Array::set_size: negative");
      return;
    }
    if (!ensureNoActiveGuards("setSize")) {
      return;
    }
    if (n > capacity()) {
      Memory<T> new_mem = allocateLike(memory_, n);
      if (new_mem.capaCity() != n) {
        return;
      }
      memory_ = std::move(new_mem);
      active_guards_.reset();
    }
    size_ = n;
  }

  void setSize(int n, MemType mt) { resize(n, mt); }

  void clear() {
    if (!ensureNoActiveGuards("clear")) {
      return;
    }
    size_ = 0;
  }

  [[nodiscard]] HostReadGuard<T> hostRead() const {
    return {size_, const_cast<Array *>(this)};
  }

  [[nodiscard]] HostWriteGuard<T> hostWrite() { return {size_, this}; }

  [[nodiscard]] HostRWGuard<T> hostRW() { return {size_, this}; }

  [[nodiscard]] DeviceReadGuard<T> deviceRead() const {
    return {size_, const_cast<Array *>(this)};
  }

  [[nodiscard]] DeviceWriteGuard<T> deviceWrite() { return {size_, this}; }

  [[nodiscard]] DeviceRWGuard<T> deviceRW() { return {size_, this}; }

  [[nodiscard]] std::span<const T> view() const {
    if (size_ == 0) {
      return {};
    }
    const T *buffer = acquireRead(MemorySide::HOST, size_);
    return buffer != nullptr
               ? std::span<const T>{buffer, static_cast<std::size_t>(size_)}
               : std::span<const T>{};
  }

  [[nodiscard]] std::span<T> mutableView() {
    if (size_ == 0) {
      return {};
    }
    T *buffer = acquireRW(MemorySide::HOST, size_);
    return buffer != nullptr
               ? std::span<T>{buffer, static_cast<std::size_t>(size_)}
               : std::span<T>{};
  }

  // Architectural Rationale for Raw Pointers:
  // High-performance computing kernels and third-party scientific libraries
  // require raw pointers:
  //  1. MPI Communication: MPI_Send, MPI_Irecv require raw contiguous
  //  host/device buffers.
  //  2. OpenMP SIMD Loops: #pragma omp simd requires bare pointers for
  //  vectorization (AVX-512).
  //  3. HYPRE / PETSc Solvers: Direct integration with solver array interfaces.
  //  4. CUDA / HIP Launches: Direct kernel launches <<<grid, block>>> require
  //  raw device pointers.
  //
  // WARNING: Raw pointer accessors DO NOT register with active_guards_. The
  // caller must ensure the Array is not resized or destroyed while raw pointers
  // are held.

  const T *readPtr(bool on_dev = false) const {
    return acquireRead(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }

  T *writePtr(bool on_dev = false) {
    return acquireWrite(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }

  T *rwPtr(bool on_dev = false) {
    return acquireRW(on_dev ? MemorySide::DEVICE : MemorySide::HOST, size_);
  }

  const T *hostReadPtr() const { return acquireRead(MemorySide::HOST, size_); }
  T *hostWritePtr() { return acquireWrite(MemorySide::HOST, size_); }
  T *hostRWPtr() { return acquireRW(MemorySide::HOST, size_); }
  const T *deviceReadPtr() const {
    return acquireRead(MemorySide::DEVICE, size_);
  }
  T *deviceWritePtr() { return acquireWrite(MemorySide::DEVICE, size_); }
  T *deviceRWPtr() { return acquireRW(MemorySide::DEVICE, size_); }

  T &operator[](int i) {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::operator[]: OOB");
    T *buffer = acquireRW(MemorySide::HOST, size_);
    if (buffer == nullptr) [[unlikely]] {
      std::abort();
    }
    return buffer[i];
  }

  const T &operator[](int i) const {
    VFEM_HOT_ASSERT(i >= 0 && i < size_, "Array::operator[]: OOB");
    const T *buffer = acquireRead(MemorySide::HOST, size_);
    if (buffer == nullptr) [[unlikely]] {
      std::abort();
    }
    return buffer[i];
  }

  T &at(int i) {
    if (i < 0 || i >= size_) [[unlikely]] {
      vfemError("Array::at: OOB");
      std::abort();
    }
    T *buffer = acquireRW(MemorySide::HOST, size_);
    if (buffer == nullptr) [[unlikely]] {
      std::abort();
    }
    return buffer[i];
  }

  const T &at(int i) const {
    if (i < 0 || i >= size_) [[unlikely]] {
      vfemError("Array::at: OOB");
      std::abort();
    }
    const T *buffer = acquireRead(MemorySide::HOST, size_);
    if (buffer == nullptr) [[unlikely]] {
      std::abort();
    }
    return buffer[i];
  }

  T &front() { return (*this)[0]; }
  const T &front() const { return (*this)[0]; }
  T &back() { return (*this)[size_ - 1]; }
  const T &back() const { return (*this)[size_ - 1]; }
  T &last() { return back(); }
  const T &last() const { return back(); }

  iterator begin() {
    return size_ > 0 ? acquireRW(MemorySide::HOST, size_) : nullptr;
  }
  iterator end() {
    T *buffer = size_ > 0 ? acquireRW(MemorySide::HOST, size_) : nullptr;
    return buffer != nullptr ? buffer + size_ : nullptr;
  }
  const_iterator begin() const {
    return size_ > 0 ? acquireRead(MemorySide::HOST, size_) : nullptr;
  }
  const_iterator end() const {
    const T *buffer =
        size_ > 0 ? acquireRead(MemorySide::HOST, size_) : nullptr;
    return buffer != nullptr ? buffer + size_ : nullptr;
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  T *getData() { return begin(); }
  const T *getData() const { return begin(); }
};

template <ArrayElement T> class Matrix {
public:
private:
};

template <ArrayElement T> class Tensor {
public:
private:
};

template <typename T> class BlockArray {
public:
private:
};

} // namespace vfem
#endif // VFEM_ARRAY_HPP
