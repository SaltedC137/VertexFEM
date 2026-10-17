if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

option(VFEM_USE_DOUBLE "Use double precision" ON)

option(VFEM_USE_CUDA "Enable CUDA backend" OFF)
option(VFEM_USE_HIP "Enable HIP backend" OFF)
option(VFEM_USE_MPI "Enable MPI backend" OFF)

option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(BUILD_TESTING "Build the testing tree" OFF)
