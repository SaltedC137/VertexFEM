#include "base/mem_manager.hpp"

namespace vfem
{

const char *MemTypeName[MemTypeSize]{
  "HOST",         "HOST_32",       "HOST_64",         "HOST_DEBUG",
  "HOST_UMPIRE",  "HOST_PINNED",   "MANAGED",         "DEVICE",
  "DEVICE_DEBUG", "DEVICE_UMPIRE", "DEVICE_UMPIRE_2",
};

MemoryRecord::~MemoryRecord () noexcept
{
  if (h_ptr == d_ptr)
    {
      if (h_ptr != nullptr)
        {
          if (owns_h && h_deallocate != nullptr)
            {
              h_deallocate (h_ptr, h_alignment);
            }
          else if (owns_d && d_deallocate != nullptr)
            {
              d_deallocate (d_ptr, d_alignment);
            }
        }
      return;
    }

  if (owns_d && d_ptr != nullptr && d_deallocate != nullptr)
    {
      d_deallocate (d_ptr, d_alignment);
    }
  if (owns_h && h_ptr != nullptr && h_deallocate != nullptr)
    {
      h_deallocate (h_ptr, h_alignment);
    }
}

MemType
getMemType (MemoryClass mc, int index)
{
  if (index != 0)
    {
      vfemError ("memory class index is not supported");
      return MemType::HOST;
    }

  switch (mc)
    {
    case MemoryClass::HOST:
      return MemType::HOST;

    case MemoryClass::HOST_32:
      return MemType::HOST_32;

    case MemoryClass::HOST_64:
      return MemType::HOST_64;

    case MemoryClass::DEVICE:
      return MemType::DEVICE;

    case MemoryClass::MANAGED:
      return MemType::MANAGED;
    }

  vfemError ("invalid memory class");
  return MemType::HOST;
}

bool
memClassContainsType (MemoryClass mc, MemType type)
{
  switch (mc)
    {
    case MemoryClass::HOST:
      return isHostMemory (type) && type != MemType::MANAGED;
    case MemoryClass::HOST_32:
      return type == MemType::HOST_32;
    case MemoryClass::HOST_64:
      return type == MemType::HOST_64;
    case MemoryClass::DEVICE:
      return isDeviceMemory (type) && type != MemType::MANAGED;
    case MemoryClass::MANAGED:
      return type == MemType::MANAGED;
    }

  return false;
}
}
