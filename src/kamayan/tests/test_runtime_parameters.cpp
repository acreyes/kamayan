#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <parthenon/parthenon.hpp>

#include "kamayan/runtime_parameters.hpp"

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
    // --8<-- [start:int]
    // var4 == 8 or 3 <= var4 <= 6
    runtime_parameters.Add<int>("block2", "var4", 0, "This is block2/var4 int",
                                {8, {3, 6}});
    // --8<-- [end:int]
  });
  EXPECT_NO_THROW({
    runtime_parameters.Add<int>("block2", "var5", 0, "This is block2/var4 int",
                                {0, {3, 9}});
  });

  EXPECT_THROW(
      {
        // --8<-- [start:string]
        runtime_parameters.Add<std::string>(
            "block2", "var6", "hello", "This is block2/var5 string", {"hello", "world"});
        // --8<-- [end:string]
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

TEST_F(RuntimeParametersTest, DirectParameterAssignment) {
  // Test int parameter assignment
  auto int_param = Parameter<int>("test", "key", "doc", 5, {0, {3, 8}}, 5);
  EXPECT_NO_THROW(int_param = 4);
  EXPECT_EQ(int_param.value, 4);

  // Test invalid int assignment throws
  EXPECT_THROW(int_param = 10, std::runtime_error);

  // Test Real parameter assignment
  auto real_param = Parameter<Real>("test", "key", "doc", 4.0, {0.0, {2.0, 6.0}}, 4.0);
  EXPECT_NO_THROW(real_param = 3.0);
  EXPECT_EQ(real_param.value, 3.0);

  // Test invalid Real assignment throws
  EXPECT_THROW(real_param = 7.0, std::runtime_error);

  // Test string parameter assignment
  auto str_param =
      Parameter<std::string>("test", "key", "doc", "hello", {"hello", "world"}, "hello");
  EXPECT_NO_THROW(str_param = "world");
  EXPECT_EQ(str_param.value, "world");

  // Test invalid string assignment throws
  EXPECT_THROW(str_param = "invalid", std::runtime_error);

  // Test bool parameter assignment (no rules, should always work)
  auto bool_param =
      Parameter<bool>("test", "key", "doc", false, std::vector<Rule<bool>>{}, false);
  EXPECT_NO_THROW(bool_param = true);
  EXPECT_EQ(bool_param.value, true);
  EXPECT_NO_THROW(bool_param = false);
  EXPECT_EQ(bool_param.value, false);
}

TEST_F(RuntimeParametersTest, SetValidation) {
  // Add parameters with rules
  runtime_parameters.Add<int>("block4", "int_param", 5, "int with rules", {0, {3, 8}});
  runtime_parameters.Add<Real>("block4", "real_param", 4.0, "real with rules",
                               {0.0, {2.0, 6.0}});
  runtime_parameters.Add<std::string>("block4", "str_param", "hello", "string with rules",
                                      {"hello", "world"});
  runtime_parameters.template Add<bool>("block4", "bool_param", false,
                                        "bool without rules", std::vector<Rule<bool>>{});

  // Test valid assignments
  EXPECT_NO_THROW(runtime_parameters.Set<int>("block4", "int_param", 4));
  EXPECT_EQ(runtime_parameters.Get<int>("block4", "int_param"), 4);

  EXPECT_NO_THROW(runtime_parameters.Set<Real>("block4", "real_param", 3.0));
  EXPECT_EQ(runtime_parameters.Get<Real>("block4", "real_param"), 3.0);

  EXPECT_NO_THROW(runtime_parameters.Set<std::string>("block4", "str_param", "world"));
  EXPECT_EQ(runtime_parameters.Get<std::string>("block4", "str_param"), "world");

  EXPECT_NO_THROW(runtime_parameters.Set<bool>("block4", "bool_param", true));
  EXPECT_EQ(runtime_parameters.Get<bool>("block4", "bool_param"), true);

  // Test invalid assignments throw
  EXPECT_THROW(runtime_parameters.Set<int>("block4", "int_param", 10),
               std::runtime_error);
  EXPECT_THROW(runtime_parameters.Set<Real>("block4", "real_param", 7.0),
               std::runtime_error);
  EXPECT_THROW(runtime_parameters.Set<std::string>("block4", "str_param", "invalid"),
               std::runtime_error);

  // Verify values remain unchanged after failed assignments
  EXPECT_EQ(runtime_parameters.Get<int>("block4", "int_param"), 4);
  EXPECT_EQ(runtime_parameters.Get<Real>("block4", "real_param"), 3.0);
  EXPECT_EQ(runtime_parameters.Get<std::string>("block4", "str_param"), "world");
  EXPECT_EQ(runtime_parameters.Get<bool>("block4", "bool_param"), true);
}

TEST_F(RuntimeParametersTest, AssignmentChaining) {
  auto param = Parameter<int>("test", "key", "doc", 5, {0, {3, 8}}, 5);

  // Test chaining returns reference
  (param = 4, param = 3);
  EXPECT_EQ(param.value, 3);

  // Test validation still works in chaining
  EXPECT_THROW((param = 10, param = 4), std::runtime_error);

  // Test with Real parameter
  auto real_param = Parameter<Real>("test", "key", "doc", 4.0, {0.0, {2.0, 6.0}}, 4.0);
  (real_param = 3.0, real_param = 5.0);
  EXPECT_EQ(real_param.value, 5.0);
  EXPECT_THROW((real_param = 7.0, real_param = 3.0), std::runtime_error);
}

TEST_F(RuntimeParametersTest, SetValuePersistence) {
  runtime_parameters.Add<int>("block5", "persistent", 5, "persistent param", {0, {3, 8}});
  runtime_parameters.Add<Real>("block5", "real_persistent", 4.0, "persistent real param",
                               {0.0, {2.0, 6.0}});
  runtime_parameters.Add<std::string>("block5", "str_persistent", "hello",
                                      "persistent string param", {"hello", "world"});
  runtime_parameters.template Add<bool>("block5", "bool_persistent", false,
                                        "persistent bool param",
                                        std::vector<Rule<bool>>{});

  // Set values and verify they persist
  runtime_parameters.Set<int>("block5", "persistent", 6);
  EXPECT_EQ(runtime_parameters.Get<int>("block5", "persistent"), 6);

  runtime_parameters.Set<Real>("block5", "real_persistent", 3.0);
  EXPECT_EQ(runtime_parameters.Get<Real>("block5", "real_persistent"), 3.0);

  runtime_parameters.Set<std::string>("block5", "str_persistent", "world");
  EXPECT_EQ(runtime_parameters.Get<std::string>("block5", "str_persistent"), "world");

  runtime_parameters.Set<bool>("block5", "bool_persistent", true);
  EXPECT_EQ(runtime_parameters.Get<bool>("block5", "bool_persistent"), true);

  // Multiple sets should work
  runtime_parameters.Set<int>("block5", "persistent", 7);
  EXPECT_EQ(runtime_parameters.Get<int>("block5", "persistent"), 7);

  runtime_parameters.Set<Real>("block5", "real_persistent", 5.5);
  EXPECT_EQ(runtime_parameters.Get<Real>("block5", "real_persistent"), 5.5);

  runtime_parameters.Set<std::string>("block5", "str_persistent", "hello");
  EXPECT_EQ(runtime_parameters.Get<std::string>("block5", "str_persistent"), "hello");

  runtime_parameters.Set<bool>("block5", "bool_persistent", false);
  EXPECT_EQ(runtime_parameters.Get<bool>("block5", "bool_persistent"), false);
}
}  // namespace kamayan::runtime_parameters
