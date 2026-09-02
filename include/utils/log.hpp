/**
 * @file log.hpp
 * @author Aska Lyn
 * @brief Logging utilities for DL4Cpp library
 * @details Provides a flexible logging system with multiple severity levels
 *          (DEBUG, INFO, WARNING, ERROR, FATAL). Log messages include
 * timestamps, source file, and line number information for easy debugging.
 * @date 2026-03-11 21:28:25
 */
#pragma once

#ifndef LOG_HPP_
#define LOG_HPP_

#include <iostream>
#include <sstream>

namespace ctl
{

/**
 * @brief Enumeration of logging severity levels
 * @details Levels are ordered from least to most severe:
 *          DEBUG < INFO < WARNING < ERROR < FATAL
 *          Messages below the current SystemLogLevel threshold are suppressed.
 */
enum class LogLevel
{
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  FATAL
};

/**
 * @brief Global system log level threshold
 * @details Messages with severity below this level will not be logged.
 *          Can be set at program startup to control verbosity.
 *          Default: LogLevel::INFO
 */
extern LogLevel SystemLogLevel;

/**
 * @brief Log message builder class
 * @details Constructs and outputs log messages with automatic formatting.
 *          Each message includes the log level, timestamp, source file, and
 * line number. For FATAL level messages, the program will abort after logging.
 */
class LogMessage
{
public:
  /**
   * @brief Construct a log message with metadata
   * @param level The severity level of this log message
   * @param file The source file name (__FILE__)
   * @param line The source line number (__LINE__)
   * @details Initializes the message with timestamp and location info.
   *          The actual message content is written via the stream() method.
   */
  LogMessage (LogLevel level, const char *file, int line);

  /**
   * @brief Destructor - outputs the log message
   * @details Automatically flushes and outputs the accumulated message
   *          when the LogMessage object goes out of scope.
   *          ERROR and FATAL messages go to stderr; others to stdout.
   *          FATAL messages trigger program abortion after logging.
   */
  ~LogMessage ();

  /**
   * @brief Get the output stream for writing message content
   * @return std::ostream& Reference to the internal string stream
   * @details Use this stream to append custom message content.
   *          Example: LOG(INFO) << "Value = " << x;
   */
  std::ostream &stream ();

private:
  LogLevel level_;            ///< The severity level of this message
  std::ostringstream stream_; ///< Accumulates the message content
};

/**
 * @brief Helper class to convert log expression to void
 * @details Used internally by the LOG macro to create valid expressions
 *          that can be used in ternary operators. The operator& overload
 *          accepts the stream reference and discards it safely.
 */
class FMessageVoidify
{
public:
  FMessageVoidify () = default;

  /**
   * @brief Consume the stream reference and return void
   * @param stream The ostream reference to discard
   * @details This operator allows the LOG macro to work correctly
   *          in conditional expressions by converting the result to void.
   */
  void
  operator& (std::ostream &stream)
  {
  }
};

/**
 * @brief Logging macro with severity-based filtering
 * @param level The log level (DEBUG, INFO, WARNING, ERROR, FATAL)
 * @return A stream reference for writing the log message, or void if filtered
 * @details If the specified level is below the current SystemLogLevel,
 *          the macro evaluates to (void)0 and no logging occurs.
 *          Otherwise, it creates a LogMessage object and returns its stream.
 *
 * Usage examples:
 * @code
 * LOG(DEBUG) << "Debug value: " << x;
 * LOG(INFO) << "Process started";
 * LOG(ERROR) << "Error code: " << err;
 * LOG(FATAL) << "Critical failure, aborting";
 * @endcode
 */
#define LOG(level)                                                            \
  ((ctl::LogLevel::level < ctl::SystemLogLevel)                               \
       ? (void)0                                                              \
       : ctl::FMessageVoidify ()                                              \
             & ctl::LogMessage (ctl::LogLevel::level, __FILE__, __LINE__)     \
                   .stream ())

/**
 * @brief Conditional logging macro with severity-based filtering
 * @param level The log level (DEBUG, INFO, WARNING, ERROR, FATAL)
 * @param condition The boolean condition to check
 * @return A stream reference for writing the log message, or void if filtered
 * @details If the specified level is below the current SystemLogLevel,
 *          the macro evaluates to (void)0 and no logging occurs.
 *          Otherwise, it creates a LogMessage object and returns its stream.
 *
 * Usage example:
 * @code
 * LOG_IF(INFO, x > 0) << "Value is positive";
 * @endcode
 */

#define LOG_IF(level, condition)                                              \
  ((!(condition)) ? (void)0                                                   \
   : (ctl::LogLevel::level < ctl::SystemLogLevel)                             \
       ? (void)0                                                              \
       : ctl::FMessageVoidify ()                                              \
             & ctl::LogMessage (ctl::LogLevel::level, __FILE__, __LINE__)     \
                   .stream ())
} // namespace ctl

#endif // LOG_HPP_