# Tests

From the repository root, configure, build, and run the CTest suite:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DENABLE_ASAN=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The suite registers `mem_manager_test` and `bench_test`. To run only the memory manager test, use `ctest --test-dir build -R mem_manager_test --output-on-failure`.

Set `-DENABLE_ASAN=ON` when configuring to enable AddressSanitizer.
