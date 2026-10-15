set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/tests)

add_subdirectory(benchmarks)
add_subdirectory(mem_manager)

if(VFEM_USE_CUDA)
  add_subdirectory(cuda)
endif()
