#include <cmath>
#include <cstdlib>
#include "utils/bench.hpp"








int
main()
{
  std::printf("TSC frequency: %.1f MHz\n\n", tsc_freq_mhz());
  run_all_benchmarks(/*warmup=*/5, /*iters=*/100000);
}
