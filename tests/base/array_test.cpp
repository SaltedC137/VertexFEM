#include "base/array.hpp"
#include "base/mem_manager.hpp"
#include "utils/check.hpp"

#include <deque>
#include <functional>
#include <limits>
#include <thread>
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
    w[0]   = 10.0;
    w[1]   = 20.0;
    w[2]   = 30.0;
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
    w[0]   = 888.0;
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

void testAppendRange() {
  Array<int> arr;
  std::vector<int> empty_vec;
  CHECK_EQ(arr.appendrange(empty_vec), 0);
  CHECK(arr.empty());

  std::vector<int> contiguous_data{1, 2, 3};
  CHECK_EQ(arr.appendrange(contiguous_data), 3);
  CHECK_EQ(arr.size(), 3);
  {
    auto r = arr.hostRead();
    CHECK_EQ(r[0], 1);
    CHECK_EQ(r[1], 2);
    CHECK_EQ(r[2], 3);
  }

  std::deque<int> deque_data{4, 5};
  CHECK_EQ(arr.appendrange(deque_data), 5);
  CHECK_EQ(arr.size(), 5);
  {
    auto r = arr.hostRead();
    CHECK_EQ(r[3], 4);
    CHECK_EQ(r[4], 5);
  }

  Array<double> d_arr{10.5};
  std::vector<int> int_data{20, 30};
  d_arr.appendrange(int_data);
  CHECK_EQ(d_arr.size(), 3);
  {
    auto r = d_arr.hostRead();
    CHECK_EQ(r[0], 10.5);
    CHECK_EQ(r[1], 20.0);
    CHECK_EQ(r[2], 30.0);
  }

  {
    auto r = arr.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    CHECK_EQ(arr.appendrange(contiguous_data), 5);
  }

  // popBack testing
  Array<int> pop_arr{1, 2, 3};
  pop_arr.popBack();
  CHECK_EQ(pop_arr.size(), 2);
  CHECK_EQ(pop_arr.back(), 2);
  pop_arr.popBack();
  pop_arr.popBack();
  CHECK(pop_arr.empty());
  {
    ErrorActionGuard guard(ErrorAction::Ignore);
    pop_arr.popBack();
    CHECK(pop_arr.empty());
  }
}

void testFindAndInsertUnique() {
  Array<int> empty;
  CHECK_EQ(empty.find(42), -1);
  CHECK_EQ(empty.findSorted(42), -1);

  Array<int> arr{10, 20, 30, 20, 50};
  CHECK_EQ(arr.find(10), 0);
  CHECK_EQ(arr.find(20), 1);
  CHECK_EQ(arr.find(50), 4);
  CHECK_EQ(arr.find(999), -1);

  Array<int> sorted{5, 15, 25, 35, 45};
  CHECK_EQ(sorted.findSorted(5), 0);
  CHECK_EQ(sorted.findSorted(25), 2);
  CHECK_EQ(sorted.findSorted(45), 4);
  CHECK_EQ(sorted.findSorted(1), -1);
  CHECK_EQ(sorted.findSorted(20), -1);
  CHECK_EQ(sorted.findSorted(100), -1);

  Array<int> u;
  CHECK_EQ(u.insertUnique(100), 0);
  CHECK_EQ(u.size(), 1);
  CHECK_EQ(u.insertUnique(200), 1);
  CHECK_EQ(u.size(), 2);
  CHECK_EQ(u.insertUnique(100), 0); // duplicate
  CHECK_EQ(u.size(), 2);
  CHECK_EQ(u.insertUnique(300), 2);
  CHECK_EQ(u.size(), 3);
  CHECK_EQ(u.insertUnique(200), 1); // duplicate
  CHECK_EQ(u.size(), 3);

  // Active guard prevents insertUnique from inserting new element
  {
    auto r = u.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    CHECK_EQ(u.insertUnique(999), -1);
    CHECK_EQ(u.size(), 3);
  }
}

void testCopy() {
  Array<int> a(5);
  for (int i = 0; i < 5; ++i) {
    a[i] = (i + 1) * 10;
  }

  // Copy to uninitialized / empty array
  Array<int> b;
  a.copy(b);
  CHECK_EQ(b.size(), 5);
  for (int i = 0; i < 5; ++i) {
    CHECK_EQ(b[i], (i + 1) * 10);
  }

  // Self copy
  b.copy(b);
  CHECK_EQ(b.size(), 5);
  for (int i = 0; i < 5; ++i) {
    CHECK_EQ(b[i], (i + 1) * 10);
  }

  // Empty copy
  Array<int> empty;
  Array<int> c;
  empty.copy(c);
  CHECK_EQ(c.size(), 0);

  // copyFrom / copyTo
  std::vector<int> src = {1, 2, 3};
  Array<int> d(3);
  d.copyFrom(src.data());
  CHECK_EQ(d[0], 1);
  CHECK_EQ(d[1], 2);
  CHECK_EQ(d[2], 3);

  std::vector<int> dst(3, 0);
  d.copyTo(dst.data());
  CHECK_EQ(dst[0], 1);
  CHECK_EQ(dst[1], 2);
  CHECK_EQ(dst[2], 3);

  // getSubArray
  Array<int> sub;
  a.getSubArray(1, 3, sub);
  CHECK_EQ(sub.size(), 3);
  CHECK_EQ(sub[0], 20);
  CHECK_EQ(sub[1], 30);
  CHECK_EQ(sub[2], 40);
}

void testPredicatesAndStats() {
  Array<int> empty;
  CHECK(empty.isSorted());
  CHECK(empty.isConstant());
  CHECK_EQ(empty.sum(), 0);
  {
    ErrorActionGuard guard(ErrorAction::Ignore);
    CHECK_EQ(empty.max(), 0);
    CHECK_EQ(empty.min(), 0);
  }

  Array<int> single{42};
  CHECK(single.isSorted());
  CHECK(single.isConstant());
  CHECK_EQ(single.max(), 42);
  CHECK_EQ(single.min(), 42);
  CHECK_EQ(single.sum(), 42);

  Array<int> sorted{1, 2, 2, 3, 5};
  CHECK(sorted.isSorted());
  CHECK(!sorted.isConstant());
  CHECK_EQ(sorted.max(), 5);
  CHECK_EQ(sorted.min(), 1);
  CHECK_EQ(sorted.sum(), 13);

  Array<int> unsorted{5, 1, 4, 2};
  CHECK(!unsorted.isSorted());
  CHECK(!unsorted.isConstant());
  CHECK_EQ(unsorted.max(), 5);
  CHECK_EQ(unsorted.min(), 1);
  CHECK_EQ(unsorted.sum(), 12);

  Array<int> constant{7, 7, 7, 7};
  CHECK(constant.isSorted());
  CHECK(constant.isConstant());
  CHECK_EQ(constant.max(), 7);
  CHECK_EQ(constant.min(), 7);
  CHECK_EQ(constant.sum(), 28);

  Array<double> d_arr{1.5, 2.5, 6.0};
  CHECK(d_arr.isSorted());
  CHECK(!d_arr.isConstant());
  CHECK_EQ(d_arr.max(), 6.0);
  CHECK_EQ(d_arr.min(), 1.5);
  CHECK_EQ(d_arr.sum(), 10.0);

  Array<int> neg{-8, -3, -5};
  CHECK(!neg.isSorted());
  CHECK(!neg.isConstant());
  CHECK_EQ(neg.max(), -3);
  CHECK_EQ(neg.min(), -8);
  CHECK_EQ(neg.sum(), -16);

  Array<int> neg_sorted{-10, -5, -1};
  CHECK(neg_sorted.isSorted());
  CHECK(!neg_sorted.isConstant());
  CHECK_EQ(neg_sorted.max(), -1);
  CHECK_EQ(neg_sorted.min(), -10);
  CHECK_EQ(neg_sorted.sum(), -16);
}

void testEraseMethods() {
  Array<int> arr{10, 20, 30, 20, 40};
  arr.eraseFirst(999);
  CHECK_EQ(arr.size(), 5);

  arr.eraseFirst(20);
  CHECK_EQ(arr.size(), 4);
  CHECK_EQ(arr[0], 10);
  CHECK_EQ(arr[1], 30);
  CHECK_EQ(arr[2], 20);
  CHECK_EQ(arr[3], 40);

  {
    auto r = arr.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    arr.eraseFirst(10);
    CHECK_EQ(arr.size(), 4);
  }

  Array<int> arr2{0, 1, 2, 3, 4, 5, 6};
  arr2.eraseAt(std::vector<int>{});
  CHECK_EQ(arr2.size(), 7);

  arr2.eraseAt(std::vector<int>{4, 1, 1, 5});
  CHECK_EQ(arr2.size(), 4);
  CHECK_EQ(arr2[0], 0);
  CHECK_EQ(arr2[1], 2);
  CHECK_EQ(arr2[2], 3);
  CHECK_EQ(arr2[3], 6);

  Array<int> indices{2, 0};
  arr2.eraseAt(indices);
  CHECK_EQ(arr2.size(), 2);
  CHECK_EQ(arr2[0], 2);
  CHECK_EQ(arr2[1], 6);

  {
    auto r = arr2.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    arr2.eraseAt(std::vector<int>{0});
    CHECK_EQ(arr2.size(), 2);
    arr2.eraseAll();
    CHECK_EQ(arr2.size(), 2);
  }

  arr2.eraseAll();
  CHECK_EQ(arr2.size(), 0);
  CHECK(arr2.empty());

  Array<int> u;
  CHECK_EQ(u.unionWith(10), 0);
  CHECK_EQ(u.unionWith(20), 1);
  CHECK_EQ(u.unionWith(10), 0);
  CHECK_EQ(u.size(), 2);

  Array<int> empty_erase;
  empty_erase.eraseFirst(10);
  CHECK_EQ(empty_erase.size(), 0);
  empty_erase.eraseAt(std::vector<int>{0});
  CHECK_EQ(empty_erase.size(), 0);

  Array<int> oob_arr{1, 2, 3};
  {
    ErrorActionGuard guard(ErrorAction::Ignore);
    oob_arr.eraseAt(std::vector<int>{-1});
    CHECK_EQ(oob_arr.size(), 3);
    oob_arr.eraseAt(std::vector<int>{5});
    CHECK_EQ(oob_arr.size(), 3);
    oob_arr.eraseAt(std::vector<int>{-5, 10});
    CHECK_EQ(oob_arr.size(), 3);
  }

  Array<int> del_all{10, 20, 30};
  del_all.eraseAt(std::vector<int>{0, 1, 2});
  CHECK_EQ(del_all.size(), 0);
  CHECK(del_all.empty());
}

void testSortUniqueSwapAndConcurrency() {
  Array<int> s{4, 1, 3, 1, 2, 3, 4};
  s.sort();
  CHECK(s.isSorted());
  CHECK_EQ(s[0], 1);
  CHECK_EQ(s[6], 4);

  {
    auto r = s.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    s.unique();
    CHECK_EQ(s.size(), 7);
    s.sort(std::greater<int>{});
    CHECK_EQ(r[0], 1);
  }

  s.unique();
  CHECK_EQ(s.size(), 4);
  CHECK_EQ(s[0], 1);
  CHECK_EQ(s[1], 2);
  CHECK_EQ(s[2], 3);
  CHECK_EQ(s[3], 4);

  s.sort(std::greater<int>{});
  CHECK_EQ(s[0], 4);
  CHECK_EQ(s[3], 1);

  {
    Array<int> uninit(5);
    ErrorActionGuard guard(ErrorAction::Ignore);
    uninit.sort();
    uninit.unique();
    CHECK(!uninit.isConstant());
    CHECK(!uninit.isSorted());
    CHECK_EQ(uninit.max(), 0);
    CHECK_EQ(uninit.min(), 0);
    CHECK_EQ(uninit.sum(), 0);
  }

  Array<int> lhs{10, 20, 30};
  {
    auto r = lhs.hostRead();
    CHECK_EQ(r[0], 10);
  }
  Array<int> rhs;
  swap(lhs, rhs);
  CHECK(lhs.empty());
  CHECK_EQ(rhs.size(), 3);
  {
    auto r1 = lhs.hostRead();
    auto r2 = rhs.hostRead();
    CHECK(r1.empty());
    CHECK_EQ(r2[2], 30);
  }

  rhs.reserve(16);
  rhs.shrinkToFit();
  CHECK_EQ(rhs.capacity(), 3);
  CHECK_EQ(rhs.sum(), 60);

  Array<int> moved(std::move(rhs));
  CHECK(rhs.empty());
  CHECK_EQ(moved.size(), 3);
  {
    auto r = moved.hostRead();
    ErrorActionGuard guard(ErrorAction::Ignore);
    Array<int> target{9};
    target = std::move(moved);
    CHECK_EQ(target.size(), 1);
    CHECK_EQ(moved.size(), 3);
  }
  lhs = moved;
  CHECK_EQ(lhs.sum(), 60);
  rhs = std::move(moved);
  CHECK(moved.empty());
  CHECK_EQ(rhs.sum(), 60);

  Array<int> self_sub{5, 10, 15, 20, 25};
  {
    auto r = self_sub.hostRead();
    Array<int> sub_out;
    self_sub.getSubArray(1, 3, sub_out);
    CHECK_EQ(sub_out.size(), 3);
    CHECK_EQ(sub_out[0], 10);
    CHECK_EQ(sub_out[2], 20);
  }
  self_sub.getSubArray(1, 3, self_sub);
  CHECK_EQ(self_sub.size(), 3);
  CHECK_EQ(self_sub[0], 10);
  CHECK_EQ(self_sub[2], 20);
  {
    ErrorActionGuard guard(ErrorAction::Ignore);
    Array<int> out;
    self_sub.getSubArray(std::numeric_limits<int>::max(), 2, out);
    CHECK(out.empty());
  }

  const Array<int> shared_const{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<std::thread> workers;
  workers.reserve(8);
  for (int t = 0; t < 8; ++t) {
    workers.emplace_back([&shared_const]() {
      for (int iter = 0; iter < 64; ++iter) {
        auto g = shared_const.hostRead();
        CHECK_EQ(g.size(), 8);
        CHECK_EQ(g[0] + g[7], 9);
      }
    });
  }
  for (auto &th : workers) {
    th.join();
  }
  CHECK_EQ(shared_const.sum(), 36);
}

} // namespace

int main() {
  testConstructorsAndSize();
  testHostWriteReadAndGuard();
  testResizeAndGrow();
  testAliasingAndSharedLocks();
  testPointerWrap();
  testIteratorsAndStlCompatibility();
  testAppendRange();
  testFindAndInsertUnique();
  testCopy();
  testPredicatesAndStats();
  testEraseMethods();
  testSortUniqueSwapAndConcurrency();
  return 0;
}
