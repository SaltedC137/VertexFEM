#pragma once

#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

namespace vfem {

enum class MemType {
  HOST, // new / delete

  HOST_32,

  HOST_64,

  HOST_DEBUG,

  HOST_UMPIRE,

  HOST_PINNED,

  MANAGED,

  DEVICE,

  DEVICE_DEBUG,

  DEVICE_UMPIRE,

  DEVICE_UMPIRE_2,

  SIZE,

  PRESERVE,

  DEFAULT
};

constexpr int MemTypeSize = static_cast<int>(MemType::SIZE);
constexpr int HostMemType = static_cast<int>(MemType::HOST);
constexpr int HostMemTypeSize = static_cast<int>(MemType::DEVICE);
constexpr int DeviceMemType = static_cast<int>(MemType::MANAGED);




} // namespace vfem

#endif
