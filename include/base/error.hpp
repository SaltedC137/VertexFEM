#pragma once

#ifndef VFEM_ERROR_HPP
#define VFEM_ERROR_HPP

#include <atomic>
#include <cstdlib>

#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vfem
{

enum class ErrorAction
{
  Abort = 0,
  Throw,
  Ignore,
};

inline constexpr ErrorAction VFEM_ERROR_ABORT = ErrorAction::Abort;
inline constexpr ErrorAction VFEM_ERROR_THROW = ErrorAction::Throw;
inline constexpr ErrorAction VFEM_ERROR_THROW_EXCEPTION = ErrorAction::Throw;
inline constexpr ErrorAction VFEM_ERROR_ACTION_IGNORE = ErrorAction::Ignore;

class ErrorException : public std::runtime_error
{
public:
  explicit ErrorException (const std::string &message)
      : std::runtime_error (message)
  {
  }
};

namespace detail
{

inline std::atomic<ErrorAction> current_error_action{
#ifdef VFEM_USE_EXCEPTIONS
  ErrorAction::Throw
#else
  ErrorAction::Abort
#endif
};

inline std::string
withLocation (std::string_view message, const std::source_location &location)
{
  std::ostringstream stream;
  stream << message << "\n ... in function: " << location.function_name ()
         << "\n ... in file: " << location.file_name () << ':'
         << location.line () << '\n';
  return stream.str ();
}

} // namespace detail

inline void
setErrorAction (ErrorAction action) noexcept
{
  detail::current_error_action.store (action, std::memory_order_relaxed);
}

inline ErrorAction
getErrorAction () noexcept
{
  return detail::current_error_action.load (std::memory_order_relaxed);
}

inline void
vfemError (std::string_view message = {}, const std::source_location &location
                                           = std::source_location::current ())
{
  if (getErrorAction () == ErrorAction::Ignore)
    {
      return;
    }

  const std::string full_message = detail::withLocation (message, location);

  if (getErrorAction () == ErrorAction::Throw)
    {
#if defined(__cpp_exceptions)
      throw ErrorException (full_message);
#else
      std::fputs (full_message.c_str (), stderr);
      std::abort ();
#endif
    }

  std::fputs (full_message.c_str (), stderr);
  std::abort ();
}

inline void
vfemWarning (std::string_view message = {},
              const std::source_location &location
              = std::source_location::current ())
{
  const std::string full_message = detail::withLocation (message, location);
  std::fputs (full_message.c_str (), stderr);
}

} // namespace vfem

#define VFEM_DETAIL_MESSAGE(prefix, msg, fn)                                  \
  do                                                                          \
    {                                                                         \
      std::ostringstream vfemMsgStream;                                       \
      vfemMsgStream << std::setprecision (16) << std::scientific;             \
      vfemMsgStream << (prefix) << (msg);                                     \
      fn (vfemMsgStream.str (), std::source_location::current ());            \
    }                                                                         \
  while (false)

#define VFEM_ABORT(msg)                                                       \
  VFEM_DETAIL_MESSAGE ("VFEM abort: ", msg, ::vfem::vfem_error)

#define VFEM_VERIFY(expr, msg)                                                \
  do                                                                          \
    {                                                                         \
      if (!(expr))                                                            \
        {                                                                     \
          VFEM_DETAIL_MESSAGE ("Verification failed: (" #expr                 \
                               ") is false:\n --> ",                          \
                               msg, ::vfem::vfem_error);                      \
        }                                                                     \
    }                                                                         \
  while (false)

#define VFEM_CONTRACT_VAR(x) (void)(x)

#if defined(VFEM_DEBUG) || !defined(NDEBUG)
#define VFEM_ASSERT(expr, msg)                                                \
  do                                                                          \
    {                                                                         \
      if (!(expr))                                                            \
        {                                                                     \
          VFEM_DETAIL_MESSAGE ("Assertion failed: (" #expr                    \
                               ") is false:\n --> ",                          \
                               msg, ::vfem::vfem_error);                      \
        }                                                                     \
    }                                                                         \
  while (false)
#define VFEM_DEBUG_DO(expr) expr
#else
#define VFEM_ASSERT(expr, msg)                                                \
  do                                                                          \
    {                                                                         \
    }                                                                         \
  while (false)
#define VFEM_DEBUG_DO(expr)                                                   \
  do                                                                          \
    {                                                                         \
    }                                                                         \
  while (false)
#endif

#define VFEM_WARNING(msg)                                                     \
  VFEM_DETAIL_MESSAGE ("VFEM warning: ", msg, ::vfem::vfem_warning)

#define VFEM_ASSERT_INDEX_IN_RANGE(index, begin, end)                         \
  VFEM_ASSERT ((begin) <= (index) && (index) < (end),                         \
               "invalid index " #index " = "                                  \
                   << (index) << ", valid range is [" << (begin) << ','       \
                   << (end) << ')')

#endif // VFEM_ERROR_HPP
