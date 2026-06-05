cmake_minimum_required(VERSION 3.14)

project(VertexFEM
  VERSION 0.1.0
  DESCRIPTION "Vertex-centered Finite Element Method library"
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

option(VERTEXFEM_USE_DOUBLE "Use double precision" ON)
option(VERTEXFEM_BUILD_TESTS "Build test programs" ON)

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

file(GLOB VERTEXFEM_SOURCES
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

target_include_directories(vertexfem
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(vertexfem PUBLIC cxx_std_20)

if(VERTEXFEM_USE_DOUBLE)
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_DOUBLE)
else()
  target_compile_definitions(vertexfem PUBLIC VFEM_USE_SINGLE)
endif()

add_subdirectory(config)
target_sources(vertexfem PUBLIC ${VFEM_CONFIG_HEADERS})

if(VERTEXFEM_BUILD_TESTS)
  add_subdirectory(test)
endif()


# Installation
include(GNUInstallDirs)
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
