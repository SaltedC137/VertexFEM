
#include <cstdlib>
#include <sstream>
#ifdef VFEM_USE_MPI
#include <mutex>
#endif

#include "global.hpp"
#include <iomanip>
#include <ostream>

namespace vfem {

namespace {
#ifdef VFEM_USE_MPI
// default MPI communicator for parallel execution
std::mutex global_mpi_mutex;
MPI_Comm global_mpi_communicator = MPI_COMM_WORLD;
#endif
} // namespace

OutStream::OutStream(std::ostream &stream) : std::ostream(stream.rdbuf()) {
  setStream(stream);
}

OutStream out(std::cout);
OutStream err(std::cerr);

void OutStream::setStream(std::ostream &os) {
  rdbuf(m_rdbuf = os.rdbuf());
  tie(m_tie = os.tie());
  Init();
}

bool OutStream::isEnabled() const noexcept { return (rdbuf() == m_rdbuf); }

void OutStream::enable() {
  if (isEnabled() || m_rdbuf == nullptr) {
    return;
  }
  rdbuf(m_rdbuf);
  tie(m_tie);
  init();
}

void OutStream::disable() {
  if (!isEnabled()) {
    return;
  }
  m_rdbuf = rdbuf(nullptr);
  m_tie   = tie(nullptr);
}

void OutStream::init() { clear(); }

std::string makeParFilename(const std::string &prefix, int id_,
                            const std::string &suffix, int width) {
  const int normalized_width = width > 0 ? width : 6;
  std::ostringstream filename;
  filename << prefix << std::setfill('0') << std::setw(normalized_width) << id_
           << suffix;
  return filename.str();
}

#ifdef VFEM_USE_MPI // when MPI is enabled, we need to provide a way to set and
                    // get the global MPI communicator
MPI_Comm getGlobalMPIComm() {
  std::lock_guard<std::mutex> lock(global_mpi_mutex);
  return global_mpi_communicator;
}
void setGlobalMPIComm(MPI_Comm communicator) {
  std::lock_guard<std::mutex> lock(global_mpi_mutex);
  global_mpi_communicator = communicator;
}
#endif

const char *getEnv(const char *name) noexcept {
  if (name == nullptr) {
    return nullptr;
  }
  return std::getenv(name);
}

} // namespace vfem