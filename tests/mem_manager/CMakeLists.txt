add_executable(mem_manager_test mem_manager_test.cpp)
target_link_libraries(mem_manager_test vertexfem)
add_test(NAME mem_manager_test COMMAND $<TARGET_FILE:mem_manager_test>)
set_tests_properties(mem_manager_test PROPERTIES ENVIRONMENT "${ASAN_ENV}")

