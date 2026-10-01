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
  const bool shared_storage = h_ptr != nullptr && h_ptr == d_ptr;
  if (shared_storage)
    {
      // One physical allocation has one destruction, regardless of which side
      // originally acquired ownership. The callback must match that
      // allocation.
      if (owns_h && h_deallocate != nullptr)
        {
          h_deallocate (h_ptr, h_alignment);
        }
      else if (owns_d && d_deallocate != nullptr)
        {
          d_deallocate (d_ptr, d_alignment);
        }
    }
  else
    {
      if (owns_d && d_ptr != nullptr && d_deallocate != nullptr)
        {
          d_deallocate (d_ptr, d_alignment);
        }
      if (owns_h && h_ptr != nullptr && h_deallocate != nullptr)
        {
          h_deallocate (h_ptr, h_alignment);
        }
    }
  h_ptr = nullptr;
  d_ptr = nullptr;
  owns_d = false;
  owns_h = false;
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
      return isHostMemory (type);
    case MemoryClass::HOST_32:
      return type == MemType::HOST_32;
    case MemoryClass::HOST_64:
      return type == MemType::HOST_64;
    case MemoryClass::DEVICE:
      return isDeviceMemory (type);
    case MemoryClass::MANAGED:
      return type == MemType::MANAGED;
    }
  return false;
}

// MemoryClass operator* (MemoryClass mc1, MemoryClass mc2);

inline MemoryClass
operator* (MemoryClass lhs, MemoryClass rhs)
{
  return std::max (lhs, rhs);
}


void
memoryPrintFlags (unsigned flags) noexcept
{
  using Mem = Memory<int>;

  std::fprintf (
      stderr,
      "flags=0x%08x registered=%d owns_host=%d owns_device=%d "
      "owns_internal=%d use_device=%d alias=%d\n",
      flags, (flags & Mem::Registered) != 0U, (flags & Mem::OWNS_HOST) != 0U,
      (flags & Mem::OWNS_DEVICE) != 0U, (flags & Mem::OWNS_INTERNAL) != 0U,
      (flags & Mem::USE_DEVICE) != 0U, (flags & Mem::ALIAS) != 0U);
}

MemoryManager &
MemoryManager::get ()
{
  static MemoryManager instance;
  return instance;
}

MemoryManager::MemoryManager ()
{
  auto &backend = backends_[typeIndex (MemType::HOST)];
  backend.allocate = [] (std::size_t bytes, std::size_t alignment) -> void *
    { return ::operator new (bytes, std::align_val_t{ alignment }); };
  backend.deallocate = [] (void *ptr, std::size_t alignment) noexcept
    { ::operator delete (ptr, std::align_val_t{ alignment }); };
}

void
MemoryManager::registerBackend (MemType type, Backend backend)
{
  if (!isConcreteType (type))
    {
      vfemError ("memory type is not a concrete backend type");
      return;
    }
  if ((backend.allocate == nullptr) != (backend.deallocate == nullptr))
    {
      vfemError ("backend must provide both allocate and deallocate ");
      return;
    }
  std::lock_guard<std::mutex> lock (backend_mutex_);
  backends_[typeIndex (type)] = backend;
}

void
MemoryManager::registerCopy (MemType dst, MemType src, CopyFunc copy)
{
  if (!isConcreteType (dst) || !isConcreteType (src) || copy == nullptr)
    {
      vfemError ("invalid memory types or copy function");
      return;
    }
  std::lock_guard<std::mutex> lock (backend_mutex_);
  copies_[typeIndex (dst) * MemTypeSize + typeIndex (src)] = copy;
}

bool
MemoryManager::supports (MemType type) const noexcept
{
  std::lock_guard<std::mutex> lock (backend_mutex_);
  const std::size_t index = typeIndex (type);
  if (index >= backends_.size ())
    {
      return false;
    }
  return backends_[index].allocate != nullptr
         && backends_[index].deallocate != nullptr;
}

DeallocateFunc
MemoryManager::getDeallocate (MemType type) const noexcept
{
  std::lock_guard<std::mutex> lock (backend_mutex_);
  const std::size_t index = typeIndex (type);
  if (index >= backends_.size ())
    {
      return nullptr;
    }
  return backends_[index].deallocate;
}

std::size_t
MemoryManager::typeIndex (MemType type)
{
  const auto index = static_cast<std::size_t> (type);
  return index < static_cast<std::size_t> (MemTypeSize) ? index : MemTypeSize;
}

bool
MemoryManager::isConcreteType (MemType type) noexcept
{
  return isHostMemory (type) || isDeviceMemory (type);
}

std::size_t
MemoryManager::requiredAlignment (MemType host_mt, MemType device_mt,
                                  std::size_t type_alignment) noexcept
{
  std::size_t required = std::max (type_alignment, alignof (std::max_align_t));
  for (const MemType mt : { host_mt, device_mt })
    {
      switch (mt)
        {
        case MemType::MANAGED:
        case MemType::DEVICE:
        case MemType::DEVICE_DEBUG:
        case MemType::DEVICE_UMPIRE:
        case MemType::DEVICE_UMPIRE_2:
          required = std::max (required, std::size_t{ 256 });
          break;
        case MemType::HOST_PINNED:
          required = std::max (required, std::size_t{ 64 });
          break;
        default:
          break;
        }
    }
  return required;
}

void
MemoryManager::allocate (MemoryRecord &record, MemorySide side)
{
  const MemType mt = side == MemorySide::HOST ? record.h_mt : record.d_mt;
  const std::size_t index = typeIndex (mt);
  if (index >= backends_.size ())
    {
      vfemError ("invalid memory backend type");
      return;
    }
  void *&ptr = side == MemorySide::HOST ? record.h_ptr : record.d_ptr;
  if (ptr != nullptr)
    {
      return;
    }
  Backend backend;
  {
    std::lock_guard<std::mutex> lock (backend_mutex_);
    backend = backends_[index];
  }
  if (backend.allocate == nullptr || backend.deallocate == nullptr)
    {
      vfemError ("no complete backend registered for memory type");
      return;
    }
  const std::size_t alignment
      = side == MemorySide::HOST ? record.h_alignment : record.d_alignment;
  void *new_ptr = backend.allocate (record.bytes, alignment);
  if (new_ptr == nullptr)
    {
      vfemError ("memory backend allocation failed");
      return;
    }
  ptr = new_ptr;
  if (mt == MemType::MANAGED)
    {
      record.h_ptr = new_ptr;
      record.d_ptr = new_ptr;
    }
  if (side == MemorySide::HOST)
    {
      record.h_deallocate = backend.deallocate;
      record.owns_h = true;
    }
  else
    {
      record.d_deallocate = backend.deallocate;
      record.owns_d = true;
    }
}

void *
MemoryManager::access (MemoryRecord &record, MemorySide side, AccessMode mode)
{
  // step the state of the memory record based on the requested access
  const StateTransition step = nextState (record.state, side, mode);
  if (!step.allowed)
    {
      vfemError ("invalid memory access: state transition not allowed");
      return nullptr;
    }
  void *&target_ptr = side == MemorySide::HOST ? record.h_ptr : record.d_ptr;
  if (target_ptr == nullptr)
    {
      allocate (record, side);
      if (target_ptr == nullptr)
        {
          return nullptr;
        }
    }
  if (step.transfer == Transfer::HOST_TO_DEVICE)
    {
      if (record.h_ptr == nullptr
          || !copy (record.d_ptr, record.d_mt, record.h_ptr, record.h_mt,
                    record.bytes))
        {
          return nullptr;
        }
    }
  else if (step.transfer == Transfer::DEVICE_TO_HOST)
    {
      if (record.d_ptr == nullptr
          || !copy (record.h_ptr, record.h_mt, record.d_ptr, record.d_mt,
                    record.bytes))
        {
          return nullptr;
        }
    }
  record.state = step.new_state;
  return target_ptr;
}

bool
MemoryManager::copy (void *dst, MemType dst_mt, const void *src,
                     MemType src_mt, std::size_t bytes)
{
  if (bytes == 0 || dst == src)
    {
      return true;
    }
  if (dst == nullptr || src == nullptr)
    {
      vfemError ("null pointer passed to memory copy");
      return false;
    }
  const std::size_t dst_index = typeIndex (dst_mt);
  const std::size_t src_index = typeIndex (src_mt);
  if (dst_index >= MemTypeSize || src_index >= MemTypeSize)
    {
      vfemError ("invalid memory type passed to memory copy");
      return false;
    }
  CopyFunc fn = nullptr;
  {
    std::lock_guard<std::mutex> lock (backend_mutex_);
    fn = copies_[dst_index * MemTypeSize + src_index];
  }
  if (fn != nullptr)
    {
      fn (dst, src, bytes);
      return true;
    }
  if (isHostMemory (dst_mt) && isHostMemory (src_mt))
    {
      std::memmove (dst, src, bytes);
      return true;
    }
  vfemError ("no copy function registered for memory types");
  return false;
}

void
MemoryManager::deleteDevice (MemoryRecord &record, bool copy_to_host)
{
  if (record.d_ptr == nullptr || record.d_ptr == record.h_ptr)
    {
      return;
    }
  if (copy_to_host && record.state == MemoryState::DEVICE_VALID)
    {
      if (record.h_ptr == nullptr)
        {
          allocate (record, MemorySide::HOST);
        }
      if (record.h_ptr == nullptr
          || !copy (record.h_ptr, record.h_mt, record.d_ptr, record.d_mt,
                    record.bytes))
        {
          return;
        }
    }
  if (record.owns_d && record.d_deallocate != nullptr)
    {
      record.d_deallocate (record.d_ptr, record.d_alignment);
    }
  record.d_ptr = nullptr;
  record.owns_d = false;
  record.d_deallocate = nullptr;
  if (record.state == MemoryState::DEVICE_VALID)
    {
      record.state = copy_to_host ? MemoryState::HOST_VALID
                                  : MemoryState::UNINITIALIZED;
    }
  else if (record.state == MemoryState::SYNCHRONIZED)
    {
      record.state = MemoryState::HOST_VALID;
    }
}

} // namespace vfem
