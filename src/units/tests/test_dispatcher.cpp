#include <string>

#include <gtest/gtest.h>
#include <parthenon/parthenon.hpp>

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
  using options = TypeList<OptList<Foo, Foo::a, Foo::b>, OptList<Bar, Bar::d, Bar::e>,
                           OptList<Baz, Baz::f, Baz::g>>;
  using value = void;

  template <Foo FOO, Bar BAR, Baz BAZ>
  value dispatch(int foo, int bar, int baz) const {
    EXPECT_EQ(foo_func<FOO>(), foo);
    EXPECT_EQ(bar_func<BAR>(), bar);
    EXPECT_EQ(baz_func<BAZ>(), baz);
  }
};

TEST(dispatcher, manual_dispatch) {
  MyFunctor().dispatch<Foo::a, Bar::e, Baz::f>(1, 1, 1);
}
#if 0

void test_dispatch(Foo foo, Bar bar, Baz baz) {
  int foo_v = foo == Foo::a ? 1 : 0;
  int bar_v = bar == Bar::e ? 1 : 0;
  int baz_v = baz == Baz::f ? 1 : 0;
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, foo, bar, baz).execute(foo_v, bar_v, baz_v);
}

TEST(dispatcher, dispatch_aef) { test_dispatch(Foo::a, Bar::e, Baz::f); }
TEST(dispatcher, dispatch_bef) { test_dispatch(Foo::b, Bar::e, Baz::f); }
TEST(dispatcher, dispatch_adf) { test_dispatch(Foo::a, Bar::d, Baz::f); }
TEST(dispatcher, dispatch_bdf) { test_dispatch(Foo::b, Bar::d, Baz::f); }
TEST(dispatcher, dispatch_aeg) { test_dispatch(Foo::a, Bar::e, Baz::g); }
TEST(dispatcher, dispatch_beg) { test_dispatch(Foo::b, Bar::e, Baz::g); }
TEST(dispatcher, dispatch_adg) { test_dispatch(Foo::a, Bar::d, Baz::g); }
TEST(dispatcher, dispatch_bdg) { test_dispatch(Foo::b, Bar::d, Baz::g); }
#endif

} // namespace kamayan
