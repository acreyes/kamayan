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

using ::testing::_;
using ::testing::Exactly;

TEST(DriverTest, RegisterUnits) {
  UnitMock mock;

  auto in = std::make_unique<ParameterInput>();
  in->SetReal("parthenon/mesh", "x1min", 0.0);
  in->SetReal("parthenon/mesh", "x1max", 1.0);
  in->SetReal("parthenon/mesh", "x2min", 0.0);
  in->SetReal("parthenon/mesh", "x2max", 1.0);
  in->SetReal("parthenon/mesh", "x3min", 0.0);
  in->SetReal("parthenon/mesh", "x3max", 1.0);
  in->SetInteger("parthenon/mesh", "nx1", 8);
  in->SetInteger("parthenon/mesh", "nx2", 8);
  in->SetInteger("parthenon/mesh", "nx3", 1);
  in->SetString("parthenon/mesh", "refinement", "none");
  auto app_in = std::make_unique<ApplicationInput>();
  auto unit_list = std::make_shared<UnitCollection>();
  (*unit_list)["mock1"] = MockUnit(&mock, unit_list);
  (*unit_list)["mock2"] = MockUnit(&mock, unit_list);
  (*unit_list)["mock3"] = MockUnit(&mock, unit_list);
  auto rps = std::make_shared<runtime_parameters::RuntimeParameters>(in.get());

  parthenon::Packages_t packages;
  auto outputs_pkg = std::make_shared<parthenon::StateDescriptor>("Outputs");
  packages.Add(outputs_pkg);
  auto fake_mesh = std::make_unique<parthenon::Mesh>(in.get(), app_in.get(), packages, 1);

  auto driver = KamayanDriver(unit_list, rps, app_in.get(), fake_mesh.get());

  EXPECT_CALL(mock, SetupParams(_)).Times(Exactly(3));
  driver.Setup();

  EXPECT_CALL(mock, AddTasksOneStep(_, _, _, _)).Times(Exactly(9));
  EXPECT_CALL(mock, AddTaskSplit(_, _, _, _)).Times(Exactly(3));
  {
    auto md = std::make_shared<MeshData>();
    const int nstages = 3;
    TaskRegion task_region(1);
    auto &tl = task_region[0];
    for (int stage = 0; stage < nstages; stage++) {
      driver.BuildTaskList(tl, 0., 0., stage, md, md, md, md);
    }
  }
}

}  // namespace kamayan
