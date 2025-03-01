#ifndef UTILS_STRINGS_HPP_
#define UTILS_STRINGS_HPP_
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace kamayan::strings {

template <std::size_t N>
constexpr bool strInList(std::string_view s, std::array<std::string_view, N> sArr) {
  for (const auto &tst : sArr) {
    if (s == tst) return true;
  }
  return false;
}

std::string ltrim(const std::string &s);
std::string rtrim(const std::string &s);
std::string trim(const std::string &s);

std::string lower(const std::string &s);

std::vector<std::string> split(const std::string &s, char delimiter);

inline constexpr std::size_t getLen(const char s[]) {
  int n = 0;
  char c = 0;
  constexpr std::size_t max_str_len = 999;
  for (int i = 0; i < max_str_len; i++) {
    if (s[i] == ',') n++;
    if (s[i] == '\0') break;
  }
  return n + 1;
}

// get compile time array of string_views of the comma separated substrings
template <std::size_t N>
inline constexpr std::array<std::string_view, N> splitStrView(std::string_view s) {
  std::array<std::string_view, N> out;
  int pos1 = 0, pos2 = 0;
  for (int i = 0; i < N - 1; i++) {
    pos2 = s.find(',', pos1);
    int strt = s.find_first_not_of(' ', pos1);
    int end = s.find_last_not_of(' ', pos2);
    out[i] = s.substr(strt, end - strt);
    pos1 = pos2 + 1;
  }
  int strt = s.find_first_not_of(' ', pos1);
  int end = s.find_last_not_of(' ');
  out[N - 1] = s.substr(strt, s.size() - strt);
  return out;
}
} // namespace kamayan::strings

#endif // UTILS_STRINGS_HPP_
