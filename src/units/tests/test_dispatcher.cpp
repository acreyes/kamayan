#include <string>

#include <gtest/gtest.h>

#include "units/dispatcher.hpp"
#include "utils/type_list.hpp"

namespace kamayan {

POLYMORPHIC_PARM(Foo, a, b);
POLYMORPHIC_PARM(Bar, d, e);
POLYMORPHIC_PARM(Baz, f, g);

template <Foo opt>
int foo_func() {
  return 0;
}

template <Foo opt>
  requires is_opt<opt, Foo::a>::value
int foo_func() {
  return 1;
}

template <Bar opt>
int bar_func() {
  return 0;
}

template <Bar opt>
  requires is_opt<opt, Bar::e>::value
int bar_func() {
  return 1;
}

template <Baz opt>
int baz_func() {
  return 0;
}

template <Baz opt>
  requires is_opt<opt, Baz::f>::value
int baz_func() {
  return 1;
}

struct MyFunctor {
  using options =
      TypeList<OPT_LIST<opt_Foo, Foo::a, Foo::b>, OPT_LIST<opt_Bar, Bar::d, Bar::e>,
               OPT_LIST<opt_Baz, Baz::f, Baz::g>>;
  using value = void;

  template <typename FOO_T, typename BAR_T, typename BAZ_T>
  value dispatch(int foo, int bar, int baz) const {
    EXPECT_EQ(foo_func<FOO_T::value>(), foo);
    EXPECT_EQ(bar_func<BAR_T::value>(), bar);
    EXPECT_EQ(baz_func<BAZ_T::value>(), baz);
  }
};

void test_dispatch(Foo foo, Bar bar, Baz baz) {
  int foo_v = foo == Foo::a ? 1 : 0;
  int bar_v = bar == Bar::e ? 1 : 0;
  int baz_v = baz == Baz::f ? 1 : 0;
  Dispatcher<MyFunctor>(foo, bar, baz).execute(foo_v, bar_v, baz_v);
}

TEST(dispatcher, dispatch) {
  test_dispatch(Foo::a, Bar::e, Baz::f);
  test_dispatch(Foo::b, Bar::e, Baz::f);
  test_dispatch(Foo::a, Bar::d, Baz::f);
  test_dispatch(Foo::b, Bar::d, Baz::f);
  test_dispatch(Foo::a, Bar::e, Baz::g);
  test_dispatch(Foo::b, Bar::e, Baz::g);
  test_dispatch(Foo::a, Bar::d, Baz::g);
  test_dispatch(Foo::b, Bar::d, Baz::g);
}

} // namespace kamayan
