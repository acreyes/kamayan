#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "grid/grid_types.hpp"
#include "grid/scratch_variables.hpp"

namespace kamayan {
using TT = TopologicalType;
using scratch_cell_1 = ScratchVariable<"one", TT::Cell, 3>;
using scratch_cell_2 = ScratchVariable<"two", TT::Cell, 2, 4>;
using scratch_list = ScratchVariableList<TT::Cell, scratch_cell_1, scratch_cell_2>;
using sc1 = scratch_list::type<scratch_cell_1>;
using sc2 = scratch_list::type<scratch_cell_2>;

TEST(grid, scratchvariablelist) {
  // they're added into the underlying TypeList going from right to left
  // but from a user perspective this is just an implementation detail
  // never going to need to know this
  static_assert(std::is_same_v<scratch_list::type<scratch_cell_2>,
                               ScratchVariable_impl<scratch_cell_2, 0>>);
  static_assert(std::is_same_v<scratch_list::type<scratch_cell_1>,
                               ScratchVariable_impl<scratch_cell_1, 8>>);

  // check that the regex generated for the list will correctly match
  std::string base_str = "scratch_cell_";
  auto name1 = sc1::name();
  auto name2 = sc2::name();
  int nmatch1 = 0;
  int nmatch2 = 0;
  for (int i = 0; i < 100; i++) {
    std::string test_str = base_str + std::to_string(i);
    if (std::regex_match(test_str, std::regex(name1))) nmatch1 += 1;
    if (std::regex_match(test_str, std::regex(name2))) nmatch2 += 1;
  }
  EXPECT_EQ(nmatch1, scratch_cell_1::size);
  EXPECT_EQ(nmatch2, scratch_cell_2::size);
}
}  // namespace kamayan
