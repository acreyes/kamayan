#include <gtest/gtest.h>

#include "dispatcher/options.hpp"

namespace kamayan {
POLYMORPHIC_PARM(Full, a, b, c, d);
#define OPT_Part b, d
POLYMORPHIC_PARM(Part, a, b, c, d);
#undef OPT_Part

TEST(options, comptime_optlist) {
  // when OPT_Full is not defined then we should
  // get the entire parm list as requested
  static_assert(!OptInfo<Full>::isdef);
  static_assert(OptInfo<Full>::nopts == 4);
  constexpr auto full_list = OptInfo<Full>::ParmList();
  static_assert(full_list == std::array<Full, 4>{Full::a, Full::b, Full::c, Full::d});

  // we've defined OPT_Part so should get a truncated parm list
  static_assert(OptInfo<Part>::isdef);
  static_assert(OptInfo<Part>::nopts == 2);
  constexpr auto part_list = OptInfo<Part>::ParmList();
  static_assert(part_list == std::array<Part, 2>{Part::b, Part::d});
}
} // namespace kamayan
