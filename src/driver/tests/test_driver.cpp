#include <gtest/gtest.h>

#include <list>
#include <memory>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "types.hpp"

namespace kamayan {

namespace RP = runtime_parameters;
POLYMORPHIC_PARM(option1, a, b);
POLYMORPHIC_PARM(option2, c, d);

KamayanDriver get_test_driver() {
  auto in = std::make_shared<ParameterInput>();
  std::stringstream ss;
  ss << "<block1>" << std::endl
     << "option1 = 0" << std::endl
     << "option2 = 1" << std::endl;

  std::istringstream s(ss.str());
  in->LoadFromStream(s);

  auto app_in = std::make_unique<ApplicationInput>();
  std::unique_ptr<Mesh> pm;

  return KamayanDriver(in, app_in.get(), pm.get());
}

class DriverTest : public testing::Test {
 protected:
  DriverTest() : driver(get_test_driver()) {}

  KamayanDriver driver;
};

void SetupTest(Config *cfg, RP::RuntimeParameters *rp) {
  auto opt1 = rp->GetOrAdd<int>("block1", "option1", 0, "docstring1");
  EXPECT_EQ(opt1, 0);
  auto opt2 = rp->GetOrAdd<int>("block1", "option2", 0, "docstring2");
  EXPECT_EQ(opt2, 1);
  cfg->Add(opt1 == 0 ? option1::a : option1::b);
  cfg->Add(opt2 == 0 ? option2::c : option2::d);
}

std::shared_ptr<StateDescriptor> InitializeTest(const RP::RuntimeParameters *rp) {
  auto package = std::make_shared<StateDescriptor>("test_unit");
  package->AddParam("data", 111);

  return package;
}

std::shared_ptr<KamayanUnit> TestUnit() {
  auto test_unit = std::make_shared<KamayanUnit>();
  test_unit->Setup = SetupTest;
  test_unit->Initialize = InitializeTest;
  return test_unit;
}

std::list<std::shared_ptr<KamayanUnit>> ProcessUnits() {
  std::list<std::shared_ptr<KamayanUnit>> unit_list;
  unit_list.push_back(TestUnit());
  return unit_list;
}

TEST_F(DriverTest, register_units) {
  EXPECT_NO_THROW({
    driver.ProcessUnits = ProcessUnits;
    driver.Setup();
  });

  auto cfg = driver.GetConfig();
  EXPECT_EQ(cfg->Get<option1>(), option1::a);
  EXPECT_EQ(cfg->Get<option2>(), option2::d);
}

}  // namespace kamayan
