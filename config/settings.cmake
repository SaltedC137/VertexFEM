# VertexFEM Build Configuration Options
#
# This file defines build-time options and feature flags for VertexFEM, including:
#   - Numerical precision and global index width
#   - Parallel backends and accelerators (OpenMP, CUDA, HIP, MPI)
#   - Vectorization intrinsics (SIMD) and third-party solver libraries (HYPRE)
#   - Thread safety diagnostics, sanitizers, and unit testing toggles

option(VFEM_USE_DOUBLE       "Use double precision (FP64) instead of single (FP32)" ON)
option(VFEM_USE_64BIT_INDEX  "Enable 64-bit global index integers for exascale grids" OFF)


option(VFEM_USE_OPENMP       "Enable OpenMP multi-threading and SIMD" ON)
option(VFEM_USE_CUDA         "Enable NVIDIA CUDA accelerator backend" ON)
option(VFEM_USE_HIP          "Enable AMD ROCm/HIP accelerator backend" OFF)
option(VFEM_USE_MPI          "Enable MPI distributed computing backend" ON)
option(VFEM_USE_HYPRE        "Enable HYPRE linear solver integration" ON)
option(VFEM_USE_SIMD         "Enable AVX2/AVX-512 SIMD vectorization intrinsics" ON)


option(VFEM_SAFE_THREADS     "Enable thread-safety verification checks" OFF)
option(ENABLE_ASAN           "Enable AddressSanitizer and LeakSanitizer" OFF)
option(BUILD_TESTING         "Build VertexFEM unit tests and benchmark suite" ON)
