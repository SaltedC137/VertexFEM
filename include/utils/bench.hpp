#pragma once

#ifndef BENCH_HPP
#define BENCH_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>

// Detect TSC frequency in MHz, cached after first call.
// Priority: Intel CPUID → AMD CPUID → Linux sysfs → runtime calibration
inline double tsc_freq_mhz() {
  static const double mhz = []() {
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;

    // Intel: CPUID.15H.ECX = nominal TSC crystal frequency (Hz)
    if (__get_cpuid(0x15, &eax, &ebx, &ecx, &edx) && ecx != 0) {
      return (double)ecx * 1e-6;
    }

    // AMD Family 17h+: CPUID.80000007H.EBX[30:0] = TscFreq (KHz)
    if (__get_cpuid(0x80000007, &eax, &ebx, &ecx, &edx)) {
      unsigned int khz = ebx & 0x7FFFFFFFu;
      if (khz > 100000 && khz < 10000000) { // 100 MHz – 10 GHz
        return (double)khz * 1e-3;
      }
    }
#endif

#if defined(__linux__)
    // sysfs cpuinfo_max_freq typically matches TSC rate (KHz)
    if (FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/"
                        "cpuinfo_max_freq",
                        "r")) {
      unsigned int khz = 0;
      if (fscanf(f, "%u", &khz) == 1 && khz > 0) {
        fclose(f);
        return (double)khz * 1e-3;
      }
      fclose(f);
    }
#endif

    // Last resort: calibrate against high_resolution_clock (5 × 20 ms)
    using namespace std::chrono;
    constexpr int cal_ms = 20;
    constexpr int N = 5;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      _mm_lfence();
      uint64_t tsc0 = __rdtsc();
      auto t0 = high_resolution_clock::now();
      std::this_thread::sleep_for(milliseconds(cal_ms));
      auto t1 = high_resolution_clock::now();
      _mm_lfence();
      uint64_t tsc1 = __rdtsc();

      double ns = (double)duration_cast<nanoseconds>(t1 - t0).count();
      sum += (double)(tsc1 - tsc0) / ns;
    }
    return (sum / (double)N) * 1000.0;
  }();
  return mhz;
}

inline uint64_t ticks() { return __rdtsc(); }

inline double ticks_to_ns(uint64_t ticks) {
  return (double)ticks * 1000.0 / tsc_freq_mhz();
}

#else

inline uint64_t now_ticks() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}
inline double ticks_to_ns(uint64_t ns) { return static_cast<double>(ns); }
#endif

#if defined(_MSC_VER)
#define BENCH_DO_NOT_OPTIMIZE(x) _ReadWriteBarrier()
#define BENCH_KEEP_VAR(x)                                                      \
  _ReadWriteBarrier();                                                         \
  (void)x
#else
#define BENCH_DO_NOT_OPTIMIZE(x) __asm__ __volatile__("" ::"r"(x) : "memory")
#define BENCH_KEEP_VAR(x) __asm__ __volatile__("" ::"r"(x) : "memory")
#endif

struct BenchEntry {
  const char *name;
  void (*fn)();
  double avg_ns;
  double min_ns;
  int iters;
};

void register_bench(const char *name, void (*fn)());
void run_all_benchmarks(int warmup = 5, int iters = 1000);

#define BENCHMARK(func)                                                        \
  void func();                                                                 \
  static int _reg_##func = (register_bench(#func, func), 0);                   \
  void func()

#endif // BENCH_HPP
