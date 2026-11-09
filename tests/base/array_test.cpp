#include "base/array.hpp"
#include "base/mem_manager.hpp"
#include "utils/check.hpp"

#include <vector>

using namespace vfem;

namespace {

class ErrorActionGuard {
  ErrorAction previous{getErrorAction()};

public:
  explicit ErrorActionGuard(ErrorAction action) { setErrorAction(action); }
  ~ErrorActionGuard() { setErrorAction(previous); }
  ErrorActionGuard(const ErrorActionGuard &)            = delete;
  ErrorActionGuard &operator=(const ErrorActionGuard &) = delete;
};

void testConstructorsAndSize() {
  Array<double> empty;
  CHECK(empty.empty());
  CHECK_EQ(empty.size(), 0);
  CHECK_EQ(empty.capacity(), 0);
  CHECK_EQ(empty.byteSize(), 0);

  Array<double> sized(10);
  CHECK(!sized.empty());
  CHECK_EQ(sized.size(), 10);
  CHECK_EQ(sized.capacity(), 10);
  CHECK_EQ(sized.byteSize(), 10 * sizeof(double));
  CHECK_EQ(sized.memoryUsage(), 10 * sizeof(double));

  Array<int> init{1, 2, 3, 4, 5};
  CHECK_EQ(init.size(), 5);
  auto r_init = init.hostRead();
  CHECK_EQ(r_init[0], 1);
  CHECK_EQ(r_init[4], 5);

  std::vector<int> std_vec{10, 20, 30};
  Array<int> range_arr(std_vec);
  CHECK_EQ(range_arr.size(), 3);
  auto r_range = range_arr.hostRead();
  CHECK_EQ(r_range[0], 10);
  CHECK_EQ(r_range[1], 20);
  CHECK_EQ(r_range[2], 30);
}

void testHostWriteReadAndGuard() {
  Array<double> a(5);
  {
    auto w = a.hostWrite();
    CHECK_EQ(w.size(), 5);
    for (int i = 0; i < 5; ++i) {
      w[i] = (i + 1) * 1.5;
    }
  }

  {
    auto r = a.hostRead();
    CHECK_EQ(r.size(), 5);
    CHECK_EQ(r[0], 1.5);
    CHECK_EQ(r[4], 7.5);
  }

  {
    auto r = a.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    a.resize(20);
    CHECK_EQ(a.size(), 5);
  }

  a.resize(20);
  CHECK_EQ(a.size(), 20);
}

void testResizeAndGrow() {
  Array<double> a(3);
  {
    auto w = a.hostWrite();
    w[0] = 10.0;
    w[1] = 20.0;
    w[2] = 30.0;
  }

  a.resize(7);
  CHECK_EQ(a.size(), 7);
  CHECK(a.capacity() >= 7);

  {
    auto r = a.hostRead();
    CHECK_EQ(r[0], 10.0);
    CHECK_EQ(r[1], 20.0);
    CHECK_EQ(r[2], 30.0);
  }

  a.resize(9, 99.0);
  CHECK_EQ(a.size(), 9);
  {
    auto r2 = a.hostRead();
    CHECK_EQ(r2[7], 99.0);
    CHECK_EQ(r2[8], 99.0);
  }
}

void testAliasingAndSharedLocks() {
  Array<double> base(10);
  {
    auto w = base.hostWrite();
    for (int i = 0; i < 10; ++i) {
      w[i] = i * 1.0;
    }
  }

  Array<double> slice = Array<double>::aliasOf(base, 2, 4);
  CHECK_EQ(slice.size(), 4);
  {
    auto r = slice.hostRead();
    CHECK_EQ(r[0], 2.0);
    CHECK_EQ(r[3], 5.0);
  }

  {
    auto w = slice.hostRW();
    w[0] = 888.0;
  }
  {
    auto r = base.hostRead();
    CHECK_EQ(r[2], 888.0);
  }
}

void testPointerWrap() {
  double *raw = new double[5]{1.0, 2.0, 3.0, 4.0, 5.0};
  {
    Array<double> wrapped(raw, 5, MemType::HOST, true);
    CHECK_EQ(wrapped.size(), 5);
    auto r = wrapped.hostRead();
    CHECK_EQ(r[0], 1.0);
    CHECK_EQ(r[4], 5.0);
  }
}

void testIteratorsAndStlCompatibility() {
  Array<int> a(4);
  a.fill(42);
  CHECK_EQ(a.front(), 42);
  CHECK_EQ(a.back(), 42);
  CHECK_EQ(a.last(), 42);

  int sum = 0;
  for (int x : a) {
    sum += x;
  }
  CHECK_EQ(sum, 42 * 4);
}

} // namespace

int main() {
  testConstructorsAndSize();
  testHostWriteReadAndGuard();
  testResizeAndGrow();
  testAliasingAndSharedLocks();
  testPointerWrap();
  testIteratorsAndStlCompatibility();
  return 0;
}
