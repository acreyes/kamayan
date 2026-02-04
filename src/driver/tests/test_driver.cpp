#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <list>
#include <memory>

#include "driver/kamayan_driver.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit.hpp"
#include "kamayan/unit_data.hpp"

namespace kamayan {

class UnitMock {
 public:
  explicit UnitMock() {}

  MOCK_METHOD(void, SetupParams, (KamayanUnit * unit));
  MOCK_METHOD(TaskID, AddTasksOneStep, (TaskID, TaskList &, MeshData *, MeshData *));
  MOCK_METHOD(TaskID, AddTaskSplit, (TaskID, TaskList &, MeshData *, const Real &));
};

std::shared_ptr<KamayanUnit> MockUnit(UnitMock *mock,
                                      std::shared_ptr<UnitCollection> units = nullptr) {
  auto mock_unit = std::make_shared<KamayanUnit>("mock");
  if (units != nullptr) {
    mock_unit->SetUnits(units);
  }
  mock_unit->SetupParams = [=](KamayanUnit *unit) { mock->SetupParams(unit); };

  mock_unit->AddTasksOneStep = [=](TaskID prev, TaskList &tl, MeshData *md,
                                   MeshData *dudt) {
    return mock->AddTasksOneStep(prev, tl, md, dudt);
  };

  mock_unit->AddTasksSplit = [=](TaskID prev, TaskList &tl, MeshData *md,
                                 const Real &dt) {
    return mock->AddTaskSplit(prev, tl, md, dt);
  };

  return mock_unit;
}

KamayanDriver get_test_driver(UnitMock &mock) {
  auto in = std::make_unique<ParameterInput>();

  auto app_in = std::make_unique<ApplicationInput>();
  std::unique_ptr<Mesh> pm;

  auto unit_list = std::make_shared<UnitCollection>();
  (*unit_list)["mock1"] = MockUnit(&mock, unit_list);
  (*unit_list)["mock2"] = MockUnit(&mock, unit_list);
  (*unit_list)["mock3"] = MockUnit(&mock, unit_list);
  unit_list->rk_stage = {"mock1", "mock2", "mock3"};
  unit_list->operator_split = {"mock1", "mock2", "mock3"};
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>(in.get());

  return KamayanDriver(unit_list, rps, app_in.get(), pm.get());
}

class DriverTest : public testing::Test {
 protected:
  DriverTest() : driver(get_test_driver(mock)) {}

  KamayanDriver driver;
  UnitMock mock;
};

using ::testing::_;
using ::testing::Exactly;

void test_build_task_list(const KamayanDriver &driver, const Real &dt, const Real &beta,
                          std::shared_ptr<MeshData> mbase, std::shared_ptr<MeshData> md0,
                          std::shared_ptr<MeshData> md1,
                          std::shared_ptr<MeshData> mdudt) {
  const int nstages = 3;
  TaskRegion task_region(1);
  auto &tl = task_region[0];
  for (int stage = 0; stage < nstages; stage++) {
    driver.BuildTaskList(tl, dt, beta, stage, mbase, md0, md1, mdudt);
  }
}

TEST_F(DriverTest, RegisterUnits) {
  EXPECT_CALL(mock, SetupParams(_)).Times(Exactly(3));
  driver.Setup();

  // 3 stages * 3 units
  EXPECT_CALL(mock, AddTasksOneStep(_, _, _, _)).Times(Exactly(9));
  EXPECT_CALL(mock, AddTaskSplit(_, _, _, _)).Times(Exactly(3));
  // MeshData md;
  auto md = std::make_shared<MeshData>();
  test_build_task_list(driver, 0., 0., md, md, md, md);
}

}  // namespace kamayan
