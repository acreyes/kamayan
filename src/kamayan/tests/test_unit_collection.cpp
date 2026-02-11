#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "driver/kamayan_driver_types.hpp"
#include "kamayan/unit.hpp"

namespace kamayan::mock {
class UnitMock {
 public:
  explicit UnitMock() {}

  MOCK_METHOD(TaskID, AddTasksOneStep, (TaskID, TaskList &, MeshData *, MeshData *));
  MOCK_METHOD(TaskID, AddTaskSplit, (TaskID, TaskList &, MeshData *, const Real &));
};

std::shared_ptr<KamayanUnit> MockUnit(UnitMock *mock,
                                      const std::vector<std::string> &before_one,
                                      const std::vector<std::string> &before_split) {
  auto mock_unit = std::make_shared<KamayanUnit>("mock");

  mock_unit->AddTasksOneStep.Register(
      [=](TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt) {
        return mock->AddTasksOneStep(prev, tl, md, dudt);
      },
      {}, before_one);

  mock_unit->AddTasksSplit.Register(
      [=](TaskID prev, TaskList &tl, MeshData *md, const Real &dt) {
        return mock->AddTaskSplit(prev, tl, md, dt);
      },
      {}, before_split);

  return mock_unit;
}

using ::testing::_;
using ::testing::InSequence;

TEST(KamayanUnit, UnitCollection) {
  UnitMock mock1;
  UnitMock mock2;
  UnitMock mock3;

  // specificy units to run after
  auto unit1 = MockUnit(&mock1, {"two"}, {"three"});
  auto unit2 = MockUnit(&mock2, {}, {"one"});
  auto unit3 = MockUnit(&mock3, {"one"}, {});

  // build a collection of our mock units
  UnitCollection unit_collection;
  unit_collection["one"] = unit1;
  unit_collection["two"] = unit2;
  unit_collection["three"] = unit3;

  {
    InSequence seq;
    EXPECT_CALL(mock3, AddTasksOneStep(_, _, _, _));
    EXPECT_CALL(mock1, AddTasksOneStep(_, _, _, _));
    EXPECT_CALL(mock2, AddTasksOneStep(_, _, _, _));
  }
  TaskID none(0);
  TaskList tl;
  MeshData md;
  unit_collection.AddTasksDAG(
      [](KamayanUnit *u) -> auto & { return u->AddTasksOneStep; },
      [&](KamayanUnit *u) { auto tid = u->AddTasksOneStep(none, tl, &md, &md); },
      "OneStep");

  {
    InSequence seq;
    EXPECT_CALL(mock2, AddTaskSplit(_, _, _, _));
    EXPECT_CALL(mock1, AddTaskSplit(_, _, _, _));
    EXPECT_CALL(mock3, AddTaskSplit(_, _, _, _));
  }

  unit_collection.AddTasksDAG(
      [](KamayanUnit *u) -> auto & { return u->AddTasksSplit; },
      [&](KamayanUnit *u) { auto tid = u->AddTasksSplit(none, tl, &md, 0.0); },
      "OneStep");
}
}  // namespace kamayan::mock
