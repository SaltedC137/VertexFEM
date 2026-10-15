add_executable(cuda_test cuda_test.cpp)
target_link_libraries(cuda_test vertexfem)
add_test(NAME cuda_test COMMAND $<TARGET_FILE:cuda_test>)
set_tests_properties(cuda_test PROPERTIES ENVIRONMENT "${ASAN_ENV}")