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
 * - Consistent byte layout, member offsets, and alignment across host and
 * device compilers.
 * - Exclusion of types with virtual tables (vptr), non-trivial destructors, or
 * dynamic heap pointers.
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
 * Upon acquisition, the guard atomically increments the parent Array's
 * `active_guards_` counter. While any guard is active, structural mutations on
 * the Array (resize(), reserve(), setSize(), clear(), reallocation) are
 * prohibited and will raise an error via `ensureNoActiveGuards()`. Upon
 * destruction or release, `active_guards_` is decremented.
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

  pointer ptr_{nullptr};
  int size_{0};
  Array<T> *owner_{nullptr};
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
template <ArrayElement T> class Array {

protected:
  Memory<T> memory_;
  int size_{0};
  mutable std::atomic<int> active_guards_ = {0};

public:
  using value_type             = T;
  using size_type              = std::size_t;
  using difference_type        = std::ptrdiff_t;
  using reference              = T &;
  using const_reference        = const T &;
  using pointer                = T *;
  using const_pointer          = const T *;
// STL-Compatible Iterators
  using iterator               = T *;
  using const_iterator         = const T *;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
  template <ArrayElement, MemorySide, AccessMode> friend class AccessGuard;

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
    ensureNoActiveGuards("makeRef");
    if (offset < 0 || n < 0 || offset + n > base.size_) [[unlikely]] {
      vfemError("Array::make_ref: range out of bounds");
      return;
    }
    memory_.makeAlias(base.memory_, offset, n);
    size_ = n;
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
   */
  void makeRef(T *p, int n, MemType mt = MemType::HOST, bool own = false) {
    ensureNoActiveGuards("makeRef(ptr)");
    if (n < 0) [[unlikely]] {
      vfemError("Array::make_ref: negative size");
      return;
    }
    memory_.wrap(p, n, mt, own);
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
    ensureNoActiveGuards("reserve");
    if (n > capacity()) [[unlikely]] {
      growTo(n);
    }
  }

  /**
   * @brief Shrinks allocated buffer capacity to match the current size().
   * Releases excess memory back to the backend memory manager.
   */
  void shrinkToFit() {
    ensureNoActiveGuards("shrinkToFit");
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
    ensureNoActiveGuards("resize");
    if (n > capacity()) {
      growTo(n);
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
    resize(n);
    if (n > old) {
      T *p = (old == 0) ? acquireWrite(MemorySide::HOST, size_)
                        : acquireRW(MemorySide::HOST, size_);
      std::fill(p + old, p + n, v);
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
    ensureNoActiveGuards("resize(n, mt)");
    const bool use_dev = memory_.useDevice();
    Memory<T> new_mem(n, mt);
    new_mem.useDevice(use_dev);
    memory_ = std::move(new_mem);
    size_   = n;
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
    return size_ > 0 ? acquireRW(MemorySide::HOST, size_) + size_ : nullptr;
  }
  const_iterator begin() const {
    return size_ > 0 ? acquireRead(MemorySide::HOST, size_) : nullptr;
  }
  const_iterator end() const {
    return size_ > 0 ? acquireRead(MemorySide::HOST, size_) + size_ : nullptr;
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

} // namespace vfem
#endif // VFEM_ARRAY_HPP
