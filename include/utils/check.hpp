/**
 * @file check.hpp
 * @author Aska Lyn
 * @brief Assertion utilities for DL4Cpp library
 * @details Provides CHECK macros for runtime validation of conditions.
 *          Built on top of log.hpp. When a CHECK condition fails,
 *          the program logs a FATAL error with detailed information
 *          and aborts execution. These macros are intended for catching
 *          programming errors and invariant violations during development.
 * @date 2026-03-11 21:28:21
 */
#pragma once

#ifndef CHECK_HPP_
#define CHECK_HPP_

#include "utils/log.hpp" // IWYU pragma: export

/**
 * @brief Compiler-specific branch prediction hints
 * @details LIKELY hints that a condition is expected to be true,
 *          allowing the compiler to optimize the common case.
 *          UNLIKELY hints that a condition is expected to be false.
 *          On GCC/Clang, these use __builtin_expect for optimization.
 *          On other compilers, they evaluate to the condition unchanged.
 */
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/**
 * @brief Check a general condition
 * @param conditions The boolean expression to validate
 * @return void (or no-op if condition is true)
 * @details If the condition is false, logs a FATAL message with
 *          the failed expression text and aborts the program.
 *
 * Usage example:
 * @code
 * CHECK(!tensor.empty()) << "Tensor must not be empty";
 * CHECK(ptr != nullptr) << "Null pointer detected";
 * @endcode
 */
#define CHECK(conditions)                                                      \
  LIKELY(conditions)                                                           \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #conditions " "

/**
 * @brief Check equality of two values
 * @param val1 The first value to compare
 * @param val2 The second value to compare
 * @return void (or no-op if values are equal)
 * @details If val1 != val2, logs a FATAL message showing both values
 *          and aborts the program. Useful for validating expected results.
 *
 * Usage example:
 * @code
 * CHECK_EQ(tensor.size(), expected_size);
 * CHECK_EQ(status, 0);
 * @endcode
 */
#define CHECK_EQ(val1, val2)                                                   \
  LIKELY((val1) == (val2))                                                     \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #val1 " == " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "

/**
 * @brief Check that val1 is less than val2
 * @param val1 The first value to compare
 * @param val2 The second value to compare
 * @return void (or no-op if val1 < val2)
 * @details If val1 >= val2, logs a FATAL message showing both values
 *          and aborts the program. Useful for bounds checking.
 *
 * Usage example:
 * @code
 * CHECK_LT(index, array_size);
 * CHECK_LT(batch_size, max_batch_size);
 * @endcode
 */
#define CHECK_LT(val1, val2)                                                   \
  LIKELY((val1) < (val2))                                                      \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #val1 " < " #val2 " (" << (val1) << " vs "   \
              << (val2) << ") "

/**
 * @brief Check that val1 is less than or equal to val2
 * @param val1 The first value to compare
 * @param val2 The second value to compare
 * @return void (or no-op if val1 <= val2)
 * @details If val1 > val2, logs a FATAL message showing both values
 *          and aborts the program.
 *
 * Usage example:
 * @code
 * CHECK_LE(current_size, max_size);
 * CHECK_LE(score, max_score);
 * @endcode
 */
#define CHECK_LE(val1, val2)                                                   \
  LIKELY((val1) <= (val2))                                                     \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #val1 " <= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "

/**
 * @brief Check that val1 is greater than or equal to val2
 * @param val1 The first value to compare
 * @param val2 The second value to compare
 * @return void (or no-op if val1 >= val2)
 * @details If val1 < val2, logs a FATAL message showing both values
 *          and aborts the program.
 *
 * Usage example:
 * @code
 * CHECK_GE(version, min_version);
 * CHECK_GE(data_size, header_size);
 * @endcode
 */
#define CHECK_GE(val1, val2)                                                   \
  LIKELY((val1) >= (val2))                                                     \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #val1 " >= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "



#define CHECK_GT(val1, val2)                                                   \
  LIKELY((val1) > (val2))                                                      \
  ? (void)0                                                                    \
  : ctl::FMessageVoidify() &                                                   \
          ctl::LogMessage(ctl::LogLevel::FATAL, __FILE__, __LINE__).stream()   \
              << "Check failed: " #val1 " > " #val2 " (" << (val1) << " vs "   \
              << (val2) << ") "

#endif // CHECK_HPP_