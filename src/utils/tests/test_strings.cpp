#include <gtest/gtest.h>

#include <array>
#include <string>

#include "utils/strings.hpp"

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
  std::string test_str = "one,two,three";
  auto split_str = split(test_str, ',');
  EXPECT_EQ(split_str[0], "one");
  EXPECT_EQ(split_str[1], "two");
  EXPECT_EQ(split_str[2], "three");

  // compile time determination of length of list of comma separated strings
  static_assert(getLen("one,two,three") == 3);

  // compile time split our comma separated list into an array of string_views
  constexpr auto test_str_arr = splitStrView<3>("one,two,three");
  static_assert(test_str_arr[0] == "one");
  static_assert(test_str_arr[1] == "two");
  static_assert(test_str_arr[2] == "three");

  // check that strings are in the list
  static_assert(strInList("one", test_str_arr));
  static_assert(strInList("two", test_str_arr));
  static_assert(strInList("three", test_str_arr));
  static_assert(!strInList("four", test_str_arr));
}
}  // namespace kamayan::strings
