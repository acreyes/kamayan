#include <array>
#include <string>

#include <gtest/gtest.h>

#include "strings.hpp"

namespace kamayan::strings {
TEST(strings, trim) {
  std::string test_str = "  foo bar  ";

  EXPECT_EQ(ltrim(test_str), std::string("foo bar  "));
  EXPECT_EQ(rtrim(test_str), std::string("  foo bar"));
  EXPECT_EQ(trim(test_str), std::string("foo bar"));
}

TEST(strings, case) {
  std::string test_str = "CamelCase";
  EXPECT_EQ(lower(test_str), "camelcase");
}

TEST(strings, list_of_strings) {
  constexpr std::string_view test_str = "one,two,three";
  EXPECT_EQ(getLen(std::string(test_str).c_str()), 3);
  constexpr std::array<std::string_view, 3> test_str_arr{"one", "two", "three"};
  EXPECT_EQ(splitStrView<3>(test_str), test_str_arr);
}
} // namespace kamayan::strings
