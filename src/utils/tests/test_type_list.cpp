#include <gtest/gtest.h>
#include <type_traits>

#include "utils/type_list.hpp"

namespace kamayan {
struct foo {};
struct bar {};
struct fizz {};
struct buzz {};

TEST(type_lists, TypeList) {
  using test_list = TypeList<foo, bar, fizz, buzz>;
  static_assert(SizeOfList(test_list()) == 4);

  // get index of a given type in a TL
  static_assert(test_list::Idx(fizz()) == 2);
  static_assert(test_list::template Idx<fizz>() == 2);

  // get an array of indexes to selected types
  constexpr auto idx_arr = test_list::GetIdxArr(TypeList<bar, buzz, foo, fizz>());
  static_assert(idx_arr == decltype(idx_arr){1, 3, 0, 2});

  using foo_bar = TypeList<foo, bar>;
  using fizz_buzz = TypeList<fizz, buzz>;
  static_assert(std::is_same_v<ConcatTypeLists_t<foo_bar, fizz_buzz>, test_list>);
}
} // namespace kamayan
