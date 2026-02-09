#ifndef KAMAYAN_CALLBACK_REGISTRATION_HPP_
#define KAMAYAN_CALLBACK_REGISTRATION_HPP_

#include <string>
#include <utility>
#include <vector>

namespace kamayan {

/// Wrapper for callbacks with dependency information.
///
/// This template wraps a callback function along with metadata about its
/// dependencies on other units' callbacks. It supports expressing both
/// "run after" and "run before" relationships.
///
/// @tparam Func The function type (typically std::function<...>)
template <typename Func>
struct CallbackRegistration {
  /// The actual callback function
  Func callback = nullptr;

  /// Names of units whose callbacks must execute before this one ("run after")
  std::vector<std::string> depends_on;

  /// Names of units whose callbacks must execute after this one ("run before")
  std::vector<std::string> required_by;

  /// Assignment operator to allow direct function assignment.
  ///
  /// This enables syntax like: callback_registration = some_function;
  /// The function is registered without any dependencies.
  ///
  /// @param fn The callback function to assign
  /// @return Reference to this registration
  CallbackRegistration &operator=(Func fn) {
    callback = fn;
    depends_on.clear();
    required_by.clear();
    return *this;
  }

  /// Register a callback with optional dependencies.
  ///
  /// @param fn The callback function to register
  /// @param after Units that must execute before this callback
  /// @param before Units that must execute after this callback
  /// @return Reference to this registration for potential chaining
  CallbackRegistration &Register(Func fn, const std::vector<std::string> &after = {},
                                 const std::vector<std::string> &before = {}) {
    callback = fn;
    depends_on = after;
    required_by = before;
    return *this;
  }

  /// Check if a callback has been registered.
  /// @return true if callback is not nullptr
  bool IsRegistered() const { return callback != nullptr; }

  /// Call the underlying callback function.
  ///
  /// Allows the registration to be used like a function object.
  /// @param args Arguments to forward to the callback
  /// @return Whatever the callback returns
  template <typename... Args>
  auto operator()(Args &&...args) const {
    return callback(std::forward<Args>(args)...);
  }

  /// Implicit conversion to bool for checking registration status.
  /// @return true if callback is registered
  explicit operator bool() const { return IsRegistered(); }
};

}  // namespace kamayan

#endif  // KAMAYAN_CALLBACK_REGISTRATION_HPP_
