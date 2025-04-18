#include <gtest/gtest.h>

#include "kamayan/fields.hpp"
#include "utils/type_list_array.hpp"

namespace kamayan {
using Fields = TypeList<DENS, MOMENTUM, ENER>;

// this will wrap a 5 element Kokkos::Array indexed by
// DENS, MOMENTUM_0, MOMENTUM_1, MOMENTUM_2, ENER
//    0,          1,          2,          3,    4
Kokkos::Array<Real, 5> data{0., 1., 2., 3., 4.};
TypeListArray<Fields> tl_arr(data);

TEST(type_list_array, type_list_array) {
  EXPECT_EQ(tl_arr(DENS()), data[0]);
  EXPECT_EQ(tl_arr(MOMENTUM(0)), data[1]);
  EXPECT_EQ(tl_arr(MOMENTUM(1)), data[2]);
  EXPECT_EQ(tl_arr(MOMENTUM(2)), data[3]);
  EXPECT_EQ(tl_arr(ENER()), data[4]);
}

TEST(type_list_array, type_for) {
  // --8<-- [start:type_for]
  // we can also use a type_for to loop over
  // the types in our TypeList
  int idx = 0;
  type_for(Fields(), [&]<typename V>(V v) {
    // each of our fields can in principle be multi-component tensors
    // and so we can loop over the components as well through the static
    // V::n_comps member
    // As an example MOMENTUM is declared with shape {3}
    // VARIABLE(MOMENTUM, 3)
    // and each field is by default shape {1}
    for (int comp = 0; comp < V::n_comps; comp++) {
      EXPECT_EQ(tl_arr(V(comp)), data[idx]);
      idx++;
    }
  });
  // --8<-- [end:type_for]
  EXPECT_EQ(idx, 5);
}
}  // namespace kamayan
