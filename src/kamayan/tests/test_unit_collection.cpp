#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/unit.hpp"
#include "tasks/tasks.hpp"

namespace kamayan::mock {
class UnitMock {
 public:
  explicit UnitMock() {}

  MOCK_METHOD(TaskID, AddTasksOneStep, (TaskID, TaskList &, MeshData *, MeshData *));
  MOCK_METHOD(TaskID, AddTaskSplit, (TaskID, TaskList &, MeshData *, const Real &));
};

std::shared_ptr<KamayanUnit> MockUnit(UnitMock *mock) {
  auto mock_unit = std::make_shared<KamayanUnit>();

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
using ::testing::InSequence;

TEST(KamayanUnit, UnitCollection) {
  UnitMock mock1;
  UnitMock mock2;
  UnitMock mock3;

  auto unit1 = MockUnit(&mock1);
  auto unit2 = MockUnit(&mock2);
  auto unit3 = MockUnit(&mock3);

  // build a collection of our mock units
  UnitCollection unit_collection;
  unit_collection["one"] = unit1;
  unit_collection["two"] = unit2;
  unit_collection["three"] = unit3;

  // set the order we want these to be called in
  unit_collection.rk_stage = {"three", "one", "two"};
  unit_collection.operator_split = {"two", "one", "three"};

  {
    InSequence seq;
    EXPECT_CALL(mock3, AddTasksOneStep(_, _, _, _));
    EXPECT_CALL(mock1, AddTasksOneStep(_, _, _, _));
    EXPECT_CALL(mock2, AddTasksOneStep(_, _, _, _));
  }
  TaskID none(0);
  TaskList tl;
  MeshData md;
  for (auto &unit : unit_collection.rk_stage) {
    if (unit->AddTasksOneStep != nullptr) {
      auto tid = unit->AddTasksOneStep(none, tl, &md, &md);
    }
  }

  {
    InSequence seq;
    EXPECT_CALL(mock2, AddTaskSplit(_, _, _, _));
    EXPECT_CALL(mock1, AddTaskSplit(_, _, _, _));
    EXPECT_CALL(mock3, AddTaskSplit(_, _, _, _));
  }

  for (auto &unit : unit_collection.operator_split) {
    if (unit->AddTasksSplit != nullptr) {
      auto tid = unit->AddTasksSplit(none, tl, &md, 0.);
    }
  }
}
}  // namespace kamayan::mock
