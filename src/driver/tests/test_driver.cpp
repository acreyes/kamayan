#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <list>
#include <memory>
#include <string>

#include "dispatcher/options.hpp"
#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "types.hpp"

namespace kamayan {
namespace RP = runtime_parameters;

class UnitMock {
 public:
  explicit UnitMock() {}

  MOCK_METHOD(void, Setup, (Config *, RP::RuntimeParameters *));
  MOCK_METHOD(std::shared_ptr<StateDescriptor>, Initialize,
              (const RP::RuntimeParameters *));
};

KamayanDriver get_test_driver() {
  auto in = std::make_shared<ParameterInput>();

  auto app_in = std::make_unique<ApplicationInput>();
  std::unique_ptr<Mesh> pm;

  return KamayanDriver(in, app_in.get(), pm.get());
}

class DriverTest : public testing::Test {
 protected:
  DriverTest() : driver(get_test_driver()) {}

  KamayanDriver driver;
  UnitMock mock;
};

std::shared_ptr<KamayanUnit> MockUnit(UnitMock *mock) {
  auto mock_unit = std::make_shared<KamayanUnit>();
  mock_unit->Setup = [=](Config *cfg, RP::RuntimeParameters *rp) {
    mock->Setup(cfg, rp);
  };
  mock_unit->Initialize = [=](const RP::RuntimeParameters *rp) {
    return mock->Initialize(rp);
  };

  return mock_unit;
}

using ::testing::_;
using ::testing::Exactly;

TEST_F(DriverTest, RegisterUnits) {
  driver.ProcessUnits = [&]() {
    std::list<std::shared_ptr<KamayanUnit>> unit_list;
    unit_list.push_back(MockUnit(&mock));
    unit_list.push_back(MockUnit(&mock));
    unit_list.push_back(MockUnit(&mock));
    return unit_list;
  };

  EXPECT_CALL(mock, Setup(_, _)).Times(Exactly(3));
  driver.Setup();
}

}  // namespace kamayan
