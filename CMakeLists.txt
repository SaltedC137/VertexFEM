cmake_minimum_required(VERSION 3.14)


project(VertexFEM
  VERSION 0.1.0
  DESCRIPTION "Vertex-centered Finite Element Method library"
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/out)

option(VERTEXFEM_BUILD_TESTS "Build test programs" ON)

file(GLOB VERTEXFEM_SOURCES
  src/asm/*.cpp
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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(vertexfem PUBLIC cxx_std_20)


if(VERTEXFEM_BUILD_TESTS)
  add_subdirectory(test)
endif()


include(GNUInstallDirs)
install(TARGETS vertexfem
  EXPORT vertexfem-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vertexfem-targets
  NAMESPACE vertexfem::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vertexfem
)
