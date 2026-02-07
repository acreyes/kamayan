#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "grid/grid_types.hpp"
#include "grid/scratch_variables.hpp"

namespace kamayan {
using TT = TopologicalType;
using scratch_cell_1 = ScratchVariable<"one", TT::Cell, 3>;
using scratch_cell_2 = ScratchVariable<"two", TT::Cell, 2, 4>;
using scratch_list = ScratchVariableList<scratch_cell_1, scratch_cell_2>;
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

TEST(grid, scratchvariable_indexing) {
  // Test that scratch variable constructors correctly apply the lower bound offset
  // scratch_cell_2 has lb=0, ub=7 (size=8)
  // scratch_cell_1 has lb=8, ub=10 (size=3)

  // Test sc2: lb=0, so sc2(i) should have idx=i
  for (int i = 0; i < scratch_cell_2::size; i++) {
    sc2 var(i);
    EXPECT_EQ(var.idx, i) << "sc2(" << i << ") should have idx=" << i;
  }

  // Test sc1: lb=8, so sc1(i) should have idx=8+i
  for (int i = 0; i < scratch_cell_1::size; i++) {
    sc1 var(i);
    EXPECT_EQ(var.idx, sc1::lb + i)
        << "sc1(" << i << ") should have idx=" << (sc1::lb + i);
  }

  // Test that default constructor works
  sc1 var_default;
  EXPECT_EQ(var_default.idx, sc1::lb) << "Default constructor should set idx=lb";
}
}  // namespace kamayan
