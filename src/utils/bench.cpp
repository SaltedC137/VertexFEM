#include <algorithm>
#include <cstdio>
#include <vector>

#include <utils/bench.hpp>

static std::vector<BenchEntry> &
registry ()
{
  static std::vector<BenchEntry> v;
  return v;
}

void
register_bench (const char *name, void (*fn) ())
{
  registry ().push_back ({ name, fn, 0.0, 0.0, 0 });
}

void
run_all_benchmarks (int warmup, int iters)
{
  for (auto &b : registry ())
    {
      // warmup
      for (int i = 0; i < warmup; ++i)
        {
          b.fn ();
          BENCH_DO_NOT_OPTIMIZE (0);
        }

      // measure
      double sum = 0.0;
      double min = 1e18;
      for (int i = 0; i < iters; ++i)
        {
          uint64_t t0 = ticks ();
          b.fn ();
          BENCH_DO_NOT_OPTIMIZE (0);
          uint64_t t1 = ticks ();
          double ns = ticks_to_ns (t1 - t0);
          sum += ns;
          min = std::min (ns, min);
        }
      b.avg_ns = sum / (double)iters;
      b.min_ns = min;
      b.iters = iters;
    }

  // print results
  std::printf ("%-40s %12s %12s %8s\n", "name", "avg(ns)", "min(ns)", "iters");
  std::printf ("%.*s\n", 76,
               "-------------------------------------------"
               "-------------------------------------------");
  for (const auto &b : registry ())
    {
      std::printf ("%-40s %12.1f %12.1f %8d\n", b.name, b.avg_ns, b.min_ns,
                   b.iters);
    }
}
