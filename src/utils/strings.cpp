
#include <sstream>
#include <string>
#include <vector>

#include "strings.hpp"

namespace kamayan::strings {
const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) { return rtrim(ltrim(s)); }

std::string lower(const std::string &s) {
  std::string outstr = s;
  std::transform(outstr.begin(), outstr.end(), outstr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return outstr;
}

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(trim(token));
  }
  return tokens;
}

} // namespace kamayan::strings
