add_executable(bench_test bench_test.cpp)
target_link_libraries(bench_test vertexfem)
add_test(NAME bench_test COMMAND $<TARGET_FILE:bench_test>)
set_tests_properties(bench_test PROPERTIES ENVIRONMENT "${ASAN_ENV}")