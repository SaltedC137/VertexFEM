#pragma once

#ifndef VFEM_GLOBAL_HPP
#define VFEM_GLOBAL_HPP

#include "config.hpp" // IWYU pragma: keep
#include <iostream>
#include <streambuf>
#include <string>

#ifdef VFEM_USE_MPI
#include <mpi.h>
#endif

namespace vfem
{

class OutStream : public std::ostream
{
protected:
  std::streambuf *m_rdbuf{nullptr};
  std::ostream *m_tie{nullptr};

  void init ();

public:
  explicit OutStream (std::ostream &stream);
  ~OutStream () override = default;

  // reload the operator= and copy constructor to delete them, so that the
  // stream cannot be copied or moved
  OutStream (const OutStream &) = delete;
  OutStream &operator= (const OutStream &) = delete;
  OutStream (OutStream &&) = delete;
  OutStream &operator= (OutStream &&) = delete;

  void setStream (std::ostream &os);
  void enable ();
  void disable ();

  [[nodiscard]] bool isEnabled () const noexcept;
};

// for the library to use standard output and error streams
extern VFEM_EXPORT OutStream out;
extern VFEM_EXPORT OutStream err;

[[nodiscard]] std::string makeParFilename (const std::string &prefix, int id_,
                                           const std::string &suffix = {},
                                           int width = 6);

// MPI communicator for parallel execution
#ifdef VFEM_USE_MPI

MPI_Comm getGlobalMPIComm ();
void setGlobalMPIComm (MPI_Comm communicator);

#endif

[[nodiscard]] const char *getEnv (const char *name) noexcept;

VFEM_HOST_DEVICE constexpr int
flipIndexSign (int index) noexcept
{
  if (index >= 0)
    {
      return -1 - index;
    }
  return -(index + 1);
}

VFEM_HOST_DEVICE constexpr int
unsignIndex (int index) noexcept
{
  return index < 0 ? -(index + 1) : index;
}

} // namespace vfem

#endif // VFEM_GLOBAL_HPP