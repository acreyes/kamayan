#include <gtest/gtest.h>

#include <memory>

#include <parthenon/parthenon.hpp>

#include "dispatcher/dispatcher.hpp"
#include "dispatcher/options.hpp"
#include "kamayan/config.hpp"

namespace kamayan {

// define our enums, the macro will take care of
// specializing OptInfo for each type that can give
// debug output when dispatcher fails
POLYMORPHIC_PARM(Foo, a, b);
POLYMORPHIC_PARM(Bar, d, e);
POLYMORPHIC_PARM(Baz, f, g);

// we can also template our dispatches on types that are templated on
// multiple enum options
template <Foo f, Bar b>
struct CompositeOption {
  static constexpr auto foo = f;
  static constexpr auto bar = b;
};

// In order to have composite type options we need to define
// a factory that can tell dispatch how to build our type
struct CompositeFactory : OptionFactory {
  using options = OptTypeList<OptList<Foo, Foo::a, Foo::b>, OptList<Bar, Bar::d, Bar::e>>;
  template <Foo f, Bar b>
  using composite = CompositeOption<f, b>;
  using type = CompositeFactory;
};

template <Foo opt>
int foo_func() {
  return 0;
}

template <Foo opt>
requires(opt == Foo::a)
int foo_func() {
  return 1;
}

template <Bar opt>
int bar_func() {
  return 0;
}

template <Bar opt>
requires(opt == Bar::e)
int bar_func() {
  return 1;
}

template <Baz opt>
int baz_func() {
  return 0;
}

template <Baz opt>
requires(opt == Baz::f)
int baz_func() {
  return 1;
}

struct MyFunctor {
  using options = OptTypeList<OptList<Foo, Foo::a, Foo::b>, OptList<Bar, Bar::d, Bar::e>,
                              OptList<Baz, Baz::f, Baz::g>>;
  using value = void;

  template <Foo FOO, Bar BAR, Baz BAZ>
  value dispatch(int foo, int bar, int baz) const {
    EXPECT_EQ(foo_func<FOO>(), foo);
    EXPECT_EQ(bar_func<BAR>(), bar);
    EXPECT_EQ(baz_func<BAZ>(), baz);
  }
};

struct MyCompositeFunctor {
  // composite types need to provide the factory rather than an OptList. Composite types
  // always come first
  using options = OptTypeList<CompositeFactory, OptList<Baz, Baz::f, Baz::g>>;
  using value = void;

  template <typename Composite, Baz BAZ>
  value dispatch(int foo, int bar, int baz) const {
    constexpr auto FOO = Composite::foo;
    constexpr auto BAR = Composite::bar;
    EXPECT_EQ(foo_func<FOO>(), foo);
    EXPECT_EQ(bar_func<BAR>(), bar);
    EXPECT_EQ(baz_func<BAZ>(), baz);
  }
};

struct MyCompositeFunctor_R {
  using options = OptTypeList<CompositeFactory, OptList<Baz, Baz::f, Baz::g>>;
  using value = int;

  template <typename Composite, Baz BAZ>
  value dispatch(int foo, int bar, int baz) const {
    constexpr auto FOO = Composite::foo;
    constexpr auto BAR = Composite::bar;
    return foo_func<FOO>() + bar_func<BAR>() + baz_func<BAZ>();
  }
};
struct MyFunctor_R {
  using options = OptTypeList<OptList<Foo, Foo::a, Foo::b>, OptList<Bar, Bar::d, Bar::e>,
                              OptList<Baz, Baz::f, Baz::g>>;
  using value = int;

  template <Foo FOO, Bar BAR, Baz BAZ>
  value dispatch(int foo, int bar, int baz) const {
    return foo_func<FOO>() + bar_func<BAR>() + baz_func<BAZ>();
  }
};

TEST(dispatcher, manual_dispatch) {
  MyFunctor().dispatch<Foo::a, Bar::e, Baz::f>(1, 1, 1);
}

void test_dispatch(Foo foo, Bar bar, Baz baz) {
  int foo_v = foo == Foo::a ? 1 : 0;
  int bar_v = bar == Bar::e ? 1 : 0;
  int baz_v = baz == Baz::f ? 1 : 0;
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, foo, bar, baz).execute(foo_v, bar_v, baz_v);
}

void test_dispatchR(Foo foo, Bar bar, Baz baz) {
  int foo_v = foo == Foo::a ? 1 : 0;
  int bar_v = bar == Bar::e ? 1 : 0;
  int baz_v = baz == Baz::f ? 1 : 0;
  int val = Dispatcher<MyFunctor_R>(PARTHENON_AUTO_LABEL, foo, bar, baz)
                .execute(foo_v, bar_v, baz_v);
  EXPECT_EQ(val, foo_v + bar_v + baz_v);
}

void test_dispatchCompositeR(Foo foo, Bar bar, Baz baz) {
  int foo_v = foo == Foo::a ? 1 : 0;
  int bar_v = bar == Bar::e ? 1 : 0;
  int baz_v = baz == Baz::f ? 1 : 0;
  int val = Dispatcher<MyCompositeFunctor_R>(PARTHENON_AUTO_LABEL, foo, bar, baz)
                .execute(foo_v, bar_v, baz_v);
  EXPECT_EQ(val, foo_v + bar_v + baz_v);
}

TEST(dispatcher, dispatch_aef) { test_dispatch(Foo::a, Bar::e, Baz::f); }
TEST(dispatcher, dispatch_bef) { test_dispatch(Foo::b, Bar::e, Baz::f); }
TEST(dispatcher, dispatch_adf) { test_dispatch(Foo::a, Bar::d, Baz::f); }
TEST(dispatcher, dispatch_bdf) { test_dispatch(Foo::b, Bar::d, Baz::f); }
TEST(dispatcher, dispatch_aeg) { test_dispatch(Foo::a, Bar::e, Baz::g); }
TEST(dispatcher, dispatch_beg) { test_dispatch(Foo::b, Bar::e, Baz::g); }
TEST(dispatcher, dispatch_adg) { test_dispatch(Foo::a, Bar::d, Baz::g); }
TEST(dispatcher, dispatch_bdg) { test_dispatch(Foo::b, Bar::d, Baz::g); }

TEST(dispatcher, dispatchR_aef) { test_dispatch(Foo::a, Bar::e, Baz::f); }
TEST(dispatcher, dispatchR_bef) { test_dispatch(Foo::b, Bar::e, Baz::f); }
TEST(dispatcher, dispatchR_adf) { test_dispatch(Foo::a, Bar::d, Baz::f); }
TEST(dispatcher, dispatchR_bdf) { test_dispatch(Foo::b, Bar::d, Baz::f); }
TEST(dispatcher, dispatchR_aeg) { test_dispatch(Foo::a, Bar::e, Baz::g); }
TEST(dispatcher, dispatchR_beg) { test_dispatch(Foo::b, Bar::e, Baz::g); }
TEST(dispatcher, dispatchR_adg) { test_dispatch(Foo::a, Bar::d, Baz::g); }
TEST(dispatcher, dispatchR_bdg) { test_dispatch(Foo::b, Bar::d, Baz::g); }

TEST(dispatcher, dispatch_config) {
  auto config = std::make_shared<Config>();
  config->Add(Foo::a);
  config->Add(Bar::d);
  config->Add(Baz::f);
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, config).execute(1, 0, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Foo::b);
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 0, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Bar::e);
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 1, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Baz::g);
  Dispatcher<MyFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 1, 0);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
}

TEST(dispatcher, dispatch_composite) {
  auto config = std::make_shared<Config>();
  config->Add(Foo::a);
  config->Add(Bar::d);
  config->Add(Baz::f);
  Dispatcher<MyCompositeFunctor>(PARTHENON_AUTO_LABEL, config).execute(1, 0, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Foo::b);
  Dispatcher<MyCompositeFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 0, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Bar::e);
  Dispatcher<MyCompositeFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 1, 1);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
  config->Update(Baz::g);
  Dispatcher<MyCompositeFunctor>(PARTHENON_AUTO_LABEL, config).execute(0, 1, 0);
  test_dispatchCompositeR(config->Get<Foo>(), config->Get<Bar>(), config->Get<Baz>());
}

}  // namespace kamayan
