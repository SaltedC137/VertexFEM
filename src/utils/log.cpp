/**
 * @file log.cpp
 * @author Aska Lyn
 * @date 2026-03-11 19:35:36
 */

#include "utils/log.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <mutex>

namespace ctl
{

#define COLOR_RESET "\033[0m"    //  reset
#define COLOR_DEBUG "\033[36m"   //  (Cyan)
#define COLOR_INFO "\033[32m"    //  (Green)
#define COLOR_WARNING "\033[33m" //  (Yellow)
#define COLOR_ERROR "\033[31m"   //  (Red)
#define COLOR_FATAL "\033[1;31m" //  (Bold Red)

// Default log level set to INFO
LogLevel SystemLogLevel = LogLevel::INFO;

// Initialize log message with timestamp, level, and source location
LogMessage::LogMessage (LogLevel level, const char *file, int line)
    : level_ (level)
{
  auto now = std::chrono::system_clock::now ();
  auto time_t_now = std::chrono::system_clock::to_time_t (now);
  std::tm *tm_now = std::localtime (&time_t_now);

  const char *level_str = "UNKNOWN";
  const char *color_str = COLOR_RESET;

  switch (level)
    {
    case LogLevel::DEBUG:
      level_str = "DEBUG";
      color_str = COLOR_DEBUG;
      break;
    case LogLevel::INFO:
      level_str = "INFO";
      color_str = COLOR_INFO;
      break;
    case LogLevel::WARNING:
      level_str = "WARN";
      color_str = COLOR_WARNING;
      break;
    case LogLevel::ERROR:
      level_str = "ERROR";
      color_str = COLOR_ERROR;
      break;
    case LogLevel::FATAL:
      level_str = "FATAL";
      color_str = COLOR_FATAL;
      break;
    }

  stream_ << color_str << "[" << level_str << "] "
          << std::put_time (tm_now, "%Y-%m-%d %H:%M:%S") << " [" << file << ":"
          << line << "] ";
}

// Output log message on destruction, abort if FATAL level
LogMessage::~LogMessage ()
{
  static std::mutex log_mutex;
  {
    std::lock_guard<std::mutex> lock (log_mutex);
    if (level_ == LogLevel::FATAL || level_ == LogLevel::ERROR)
      {
        std::cerr << stream_.str () << COLOR_RESET << '\n';
      }
    else
      {
        std::cout << stream_.str () << COLOR_RESET << '\n';
      }
  }

  if (level_ == LogLevel::FATAL)
    {
      std::abort ();
    }
}

// Return the stream for chaining log output
std::ostream &
LogMessage::stream ()
{
  return stream_;
}

} // namespace ctl