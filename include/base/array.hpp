#pragma once

#ifndef VFEM_ARRAY_HPP
#define VFEM_ARRAY_HPP

#include "config.hpp"
#include "error.hpp"
#include "global.hpp"
#include "mem_manager.hpp"

namespace vfem
{

template <typename T>
concept ArrayElement = DeviceCopyable<T>;

template <ArrayElement T> class Array;  // 1 demension
template <ArrayElement T> class Matrix; // 2 demensions
template <ArrayElement T> class Tensor; // 3 demensions

// AccessGuard is a helper class that provides safe access to the underlying
// data of an Array, Matrix, or Tensor object. It ensures that the data is
// properly synchronized between host and device memory, and that the
// appropriate access mode (read-only, write-only, or read-write) is enforced.
template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard;

template <ArrayElement T, MemorySide S, AccessMode M> class AccessGuard
{
};

template <ArrayElement T> class Array
{
};

template <ArrayElement T> class Matrix
{
};

template <ArrayElement T> class Tensor
{
};

} // namespace vfem
#endif // VFEM_ARRAY_HPP
