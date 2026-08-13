#include "utils/bench.hpp"

static std::uint64_t bench_seed = 1;
static void
benchIntegerMix ()
{
  std::uint64_t value = bench_seed;
  for (int i = 0; i < 64; ++i)
    {
      value = value * 1664525u + 1013904223u;
      value ^= value >> 17;
    }
  bench_seed = value;
  BENCH_KEEP_VAR (bench_seed);
}
const int bench_integer_mix_registered
    = (registerBench ("benchIntegerMix", benchIntegerMix), 0);

int
main ()
{
  std::printf ("TSC frequency: %.1f MHz\n\n", tscFreqMhz ());
  runAllBenchmarks (/*warmup=*/5, /*iters=*/100000);
}
