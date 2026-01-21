#include <gtest/gtest.h>

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit_data.hpp"

namespace kamayan {

POLYMORPHIC_PARM(Foo, bar, baz);

class UnitDataTest : public testing::Test {
  using RuntimeParameters = runtime_parameters::RuntimeParameters;

 protected:
  UnitDataTest() : unit_data("block1") {
    in = std::make_unique<parthenon::ParameterInput>();
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var0 = hello" << std::endl
       << "var1 = 8" << std::endl
       << "var2 = true" << std::endl
       << "var3 = -4.6" << std::endl
       << "Foo = baz" << std::endl;

    std::istringstream s(ss.str());
    in->LoadFromStream(s);
    runtime_parameters = std::make_shared<RuntimeParameters>(in.get());
    config = std::make_shared<Config>();
    pkg = std::make_shared<StateDescriptor>("test");

    unit_data.AddParm<std::string>("var0", "world", "This is block1/var1 std::string", {},
                                   UnitData::Mutability::Mutable);
    unit_data.AddParm<int>("var1", 0, "This is block1/var1 int", {},
                           UnitData::Mutability::Mutable);
    unit_data.AddParm<bool>("var2", false, "This is block1/var2 bool", {},
                            UnitData::Mutability::Mutable);
    unit_data.AddParm<Real>("var3", 131.68, "This is block1/var3 Real", {{-5., 200.}},
                            UnitData::Mutability::Mutable);
    unit_data.AddParm<Real>("var4", 138.68, "This is block1/var3 Real");
    unit_data.AddParm<Foo>("Foo", "bar", "some config var",
                           {{"bar", Foo::bar}, {"baz", Foo::baz}});

    unit_data.Setup(runtime_parameters, config);
    unit_data.Initialize(pkg);
    unit_data.SetupComplete();
  }

  std::shared_ptr<RuntimeParameters> runtime_parameters;
  std::unique_ptr<parthenon::ParameterInput> in;
  std::shared_ptr<Config> config;
  std::shared_ptr<StateDescriptor> pkg;
  UnitData unit_data;
};

TEST_F(UnitDataTest, Value) {
  EXPECT_EQ(unit_data.Get<std::string>("var0"), "hello");
  EXPECT_EQ(unit_data.Get<int>("var1"), 8);
  EXPECT_EQ(unit_data.Get<bool>("var2"), true);
  EXPECT_EQ(unit_data.Get<Real>("var3"), -4.6);
  EXPECT_EQ(unit_data.Get<std::string>("Foo"), "baz");
}

TEST_F(UnitDataTest, Parm) {
  EXPECT_EQ(runtime_parameters->Get<std::string>("block1", "var0"), "hello");
  EXPECT_EQ(runtime_parameters->Get<int>("block1", "var1"), 8);
  EXPECT_EQ(runtime_parameters->Get<bool>("block1", "var2"), true);
  EXPECT_EQ(runtime_parameters->Get<Real>("block1", "var3"), -4.6);
}

TEST_F(UnitDataTest, Params) {
  EXPECT_EQ(pkg->Param<std::string>("block1/var0"), "hello");
  EXPECT_EQ(pkg->Param<int>("block1/var1"), 8);
  EXPECT_EQ(pkg->Param<bool>("block1/var2"), true);
  EXPECT_EQ(pkg->Param<Real>("block1/var3"), -4.6);
}

TEST_F(UnitDataTest, Config) { EXPECT_EQ(config->Get<Foo>(), Foo::baz); }

TEST_F(UnitDataTest, Update) {
  unit_data.UpdateParm("var0", "world");
  unit_data.UpdateParm("var1", 0);
  unit_data.UpdateParm("var2", false);
  unit_data.UpdateParm("var3", 130.0);
  unit_data.UpdateParm("Foo", "bar");

  EXPECT_EQ(pkg->Param<std::string>("block1/var0"), "world");
  EXPECT_EQ(pkg->Param<int>("block1/var1"), 0);
  EXPECT_EQ(pkg->Param<bool>("block1/var2"), false);
  EXPECT_EQ(pkg->Param<Real>("block1/var3"), 130.0);
  EXPECT_EQ(config->Get<Foo>(), Foo::bar);

  // throw if we violate the rules
  EXPECT_THROW({ unit_data.UpdateParm("var3", 201.0); }, std::runtime_error);
  // expect to throw when updating an immutable param
  EXPECT_THROW({ unit_data.UpdateParm("var4", 130.0); }, std::runtime_error);
}

}  // namespace kamayan
