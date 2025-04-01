#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <parthenon/parthenon.hpp>

#include "kamayan/runtime_parameters.hpp"
#include "parameter_input.hpp"

namespace kamayan::runtime_parameters {

class RuntimeParametersTest : public testing::Test {
 protected:
  RuntimeParametersTest() {
    in = std::make_unique<parthenon::ParameterInput>();
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var0 = hello" << std::endl
       << "var1 = 8" << std::endl
       << "var2 = true" << std::endl
       << "var3 = -4.6" << std::endl
       << "<block2>" << std::endl
       << "var4 = 8" << std::endl
       << "var5 = 8" << std::endl
       << "var6 = strVar" << std::endl
       << "var7 = -4.6" << std::endl
       << "var8 = -4.6" << std::endl;

    std::istringstream s(ss.str());
    in->LoadFromStream(s);
    runtime_parameters = RuntimeParameters(in.get());

    runtime_parameters.Add<std::string>("block0", "def0", "testStr",
                                        "This is block0/def1 int");
    runtime_parameters.Add<int>("block0", "def1", 0, "This is block0/def1 int");
    runtime_parameters.Add<bool>("block0", "def2", false, "This is block0/def2 bool");
    runtime_parameters.Add<Real>("block0", "def3", 131.68, "This is block0/def2 Real");

    runtime_parameters.Add<std::string>("block1", "var0", "world",
                                        "This is block1/var1 std::string");
    runtime_parameters.Add<int>("block1", "var1", 0, "This is block1/var1 int");
    runtime_parameters.Add<bool>("block1", "var2", false, "This is block1/var2 bool");
    runtime_parameters.Add<Real>("block1", "var3", 131.68, "This is block1/var3 Real");
  }

  RuntimeParameters runtime_parameters;
  std::unique_ptr<parthenon::ParameterInput> in;
};

TEST_F(RuntimeParametersTest, GetDefaults) {
  EXPECT_EQ(runtime_parameters.GetOrAdd<std::string>("block0", "def0", "testStr",
                                                     "This is block0/def1 int"),
            "teststr");
  EXPECT_EQ(
      runtime_parameters.GetOrAdd<int>("block0", "def1", 0, "This is block0/def1 int"),
      0);
  EXPECT_EQ(runtime_parameters.GetOrAdd<bool>("block0", "def2", false,
                                              "This is block0/def2 bool"),
            false);
  EXPECT_EQ(runtime_parameters.GetOrAdd<Real>("block0", "def3", 131.68,
                                              "This is block0/def2 Real"),
            131.68);
}

TEST_F(RuntimeParametersTest, GetSet) {
  EXPECT_EQ(runtime_parameters.Get<std::string>("block1", "var0"), "hello");
  EXPECT_EQ(runtime_parameters.Get<int>("block1", "var1"), 8);
  EXPECT_EQ(runtime_parameters.Get<bool>("block1", "var2"), true);
  EXPECT_EQ(runtime_parameters.Get<Real>("block1", "var3"), -4.6);
}

TEST_F(RuntimeParametersTest, Rules) {
  EXPECT_THROW(
      {
        runtime_parameters.Add<int>("block2", "var4", 0, "This is block2/var4 int",
                                    {0, {3, 6}});
      },
      std::runtime_error);
  EXPECT_NO_THROW({
    runtime_parameters.Add<int>("block2", "var4", 0, "This is block2/var4 int",
                                {8, {3, 6}});
  });
  EXPECT_NO_THROW({
    runtime_parameters.Add<int>("block2", "var5", 0, "This is block2/var4 int",
                                {0, {3, 9}});
  });

  EXPECT_THROW(
      {
        runtime_parameters.Add<std::string>(
            "block2", "var6", "hello", "This is block2/var5 string", {"hello", "world"});
      },
      std::runtime_error);
  EXPECT_NO_THROW({
    // we will make everything lower case, so even though the case doesn't match between
    // the set value & the rule value, everything will be ok
    runtime_parameters.Add<std::string>("block2", "var6", "hello",
                                        "This is block2/var5 string",
                                        {"hello", "world", "STRVAR"});
  });

  EXPECT_THROW(
      {
        runtime_parameters.Add<Real>("block2", "var7", 0.0, "This is block2/var6 Real",
                                     {0.0, {3.8, 615.9}});
      },
      std::runtime_error);
  EXPECT_NO_THROW({
    runtime_parameters.Add<Real>("block2", "var7", 0.0, "This is block2/var6 Real",
                                 {-4.6, {3.8, 615.9}});
  });
  EXPECT_NO_THROW({
    runtime_parameters.Add<Real>("block2", "var8", 0.0, "This is block2/var6 Real",
                                 {0.0, {-38.8, 615.9}});
  });
}

TEST_F(RuntimeParametersTest, AddN) {
  runtime_parameters.Add<int>("block3", "var_", 5, 0, "add_n vars", {0, {5, 8}});
  EXPECT_NO_THROW({ auto var = runtime_parameters.Get<int>("block3", "var_4"); });
  EXPECT_NO_THROW({
    for (int i = 0; i < 5; i++) {
      auto var = runtime_parameters.Get<int>("block3", "var_" + std::to_string(i));
    }
  });
}
}  // namespace kamayan::runtime_parameters
