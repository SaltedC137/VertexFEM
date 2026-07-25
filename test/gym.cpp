#include "utils/bench.hpp"


int
main ()
{
  std::printf ("TSC frequency: %.1f MHz\n\n", tscFreqMhz ());
  runAllBenchmarks (/*warmup=*/5, /*iters=*/100000);
}

