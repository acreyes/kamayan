#include <gtest/gtest.h>
#include <parthenon/parthenon.hpp>

#include "dispatcher/options.hpp"
#include "kamayan/config.hpp"

namespace kamayan {
POLYMORPHIC_PARM(Foo, a, b);
POLYMORPHIC_PARM(Bar, d, e);
POLYMORPHIC_PARM(Baz, f, g);

TEST(Config, config_params) {
  Config config;
  config.Add(Foo::a);
  config.Add(Bar::d);
  config.Add(Baz::f);

  EXPECT_EQ(config.Get<Foo>(), Foo::a);
  EXPECT_EQ(config.Get<Bar>(), Bar::d);
  EXPECT_EQ(config.Get<Baz>(), Baz::f);

  config.Update(Foo::b);
  config.Update(Bar::e);
  config.Update(Baz::g);
  EXPECT_EQ(config.Get<Foo>(), Foo::b);
  EXPECT_EQ(config.Get<Bar>(), Bar::e);
  EXPECT_EQ(config.Get<Baz>(), Baz::g);
}
}  // namespace kamayan
