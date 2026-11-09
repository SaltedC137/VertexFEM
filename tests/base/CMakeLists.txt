add_executable(array_test array_test.cpp)
target_link_libraries(array_test vertexfem)
add_test(NAME array_test COMMAND $<TARGET_FILE:array_test>)
set_tests_properties(array_test PROPERTIES ENVIRONMENT "${ASAN_ENV}")
