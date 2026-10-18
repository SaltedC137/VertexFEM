if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

option(VFEM_USE_DOUBLE "Use double precision" ON)
option(VFEM_SAFE_THREADS "Enable safe threads" OFF)

# VertexFEM options
option(VFEM_USE_OPENMP "Enable OPENMP" OFF)
option(VFEM_USE_CUDA "Enable CUDA" OFF)
option(VFEM_USE_HIP "Enable HIP" OFF)
option(VFEM_USE_MPI "Enable MPI" OFF)
option(VFEM_USE_HYPRE "Enable HYPRE" OFF)
option(MFEM_USE_SIMD "Enable use of SIMD intrinsics" OFF)

# Debugging options
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(BUILD_TESTING "Build the testing tree" ON)
