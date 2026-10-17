
#ifdef VFEM_USE_MPI
#include <mutex>  
#endif

#include <global.hpp>
#include <ostream>

void
vfem::OutStream::setStream (std::ostream &os)
{
  rdbuf (m_rdbuf = os.rdbuf ());
  tie (m_tie = os.tie ());
  Init ();
}

bool
vfem::OutStream::isEnabled () const noexcept
{
  return (rdbuf () == m_rdbuf);
}

void
vfem::OutStream::enable ()
{
  if (!vfem::OutStream::isEnabled ())
    {
    }
}