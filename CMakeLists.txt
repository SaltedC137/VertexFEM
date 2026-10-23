cmake_minimum_required(VERSION 3.15...4.0.0)

project(VertexFEM
  VERSION 0.1.0
  DESCRIPTION "Vertex-centered Finite Element Method library"
  LANGUAGES C CXX) #C/C++

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "FORCE C++ standard requirement")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  
# Include settings.cmake for build options 
include("${CMAKE_CURRENT_SOURCE_DIR}/config/settings.cmake")
  
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math)
  elseif(MSVC)
    add_compile_options(/O2 /arch:AVX2)
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Config : Debug build")
    add_compile_options(-g -O0)
  elseif(MSVC)
    add_compile_options(/Zi /Od)
  endif()
endif()

file(GLOB VERTEXFEM_SOURCES CONFIGURE_DEPENDS
  src/asm/*.cpp
  src/base/*.cpp
  src/fem/*.cpp
  src/io/*.cpp
  src/linalg/*.cpp
  src/mesh/*.cpp
  src/solver/*.cpp
  src/utils/*.cpp
)

add_library(vertexfem ${VERTEXFEM_SOURCES})

include(GNUInstallDirs)
target_include_directories(vertexfem
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

file(GLOB_RECURSE VFEM_INCLUDE_ENTRIES
  LIST_DIRECTORIES true
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*"
)

foreach(VFEM_INCLUDE_ENTRY IN LISTS VFEM_INCLUDE_ENTRIES)
  if(IS_DIRECTORY "${VFEM_INCLUDE_ENTRY}")
    target_include_directories(vertexfem PRIVATE
      "${VFEM_INCLUDE_ENTRY}"
    )
  endif()
endforeach()

target_compile_features(vertexfem PUBLIC cxx_std_20)

# CUDA support
if(VFEM_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_CUDA)
  target_include_directories(vertexfem PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(vertexfem PUBLIC CUDA::cudart)
  message(STATUS "Config : Enable CUDA backend")
endif()

if(VFEM_USE_HIP)
  find_package(HIP REQUIRED)
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_HIP)
  target_include_directories(vertexfem PUBLIC ${HIP_INCLUDE_DIRS})
  target_link_libraries(vertexfem PUBLIC HIP::device)
  message(STATUS "Config : Enable HIP backend")
endif()

# Detect MPI and set up the target accordingly
if(VFEM_USE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(NOT MPI_FOUND)
    message(STATUS "Info : try to load Environment Modules")
    find_package(EnvModules)
    if(EnvModules_FOUND)
      env_module(load mpi/openmpi-x86_64)
      find_package(MPI COMPONENTS CXX)
    endif()
    if(NOT MPI_FOUND)
      message(FATAL_ERROR "Error : Failed to find MPI")
    endif()
  endif()
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_MPI)
  target_link_libraries(vertexfem PUBLIC MPI::MPI_CXX)
  message(STATUS "Config : Enable MPI backend")
endif()

if(VFEM_USE_OPENMP)
  find_package(OpenMP COMPONENTS C CXX)
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_OPENMP)
  target_link_libraries(vertexfem PUBLIC OpenMP::OpenMP_CXX)
endif()

if(VFEM_USE_HYPRE)
  # HYPRE was built with OpenMP; its exported target references
  # OpenMP::OpenMP_C, which must exist before HYPRETargets.cmake loads.
  # Load hypre-Openmpi-devel.x86_64 
  find_package(HYPRE CONFIG REQUIRED)
  if(NOT TARGET HYPRE::HYPRE)
    message(FATAL_ERROR
      "The selected HYPRE package does not export HYPRE::HYPRE")
  endif()
  set(VFEM_HYPRE_VERSION "${HYPRE_VERSION}")
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_HYPRE)
  target_link_libraries(vertexfem PUBLIC HYPRE::HYPRE)
  message(STATUS "Config: Enable HYPRE ${VFEM_HYPRE_VERSION}")
endif()

if(VFEM_USE_DOUBLE)
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_DOUBLE)
else()
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_SINGLE)
endif()

if(ENABLE_ASAN)
    if(MSVC)
    # CHECK MSVC
        add_compile_options(/fsanitize=address /Zi /RTC-)
        add_link_options(/fsanitize=address)
    else()
    # CHECK GCC/CLANG
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

add_subdirectory(config)
target_sources(vertexfem PRIVATE ${VFEM_CONFIG_HEADERS})

# Installation
install(TARGETS vertexfem
  EXPORT vertexfem-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY config/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/config)
install(EXPORT vertexfem-targets
  NAMESPACE vertexfem::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vertexfem
)

# CTest configuration
# Standard CMake switch (also probed by super-projects/IDEs via BUILD_TESTING).
set(ASAN_ENV "ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:symbolize=1:protect_shadow_gap=0")

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

