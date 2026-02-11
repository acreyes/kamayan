#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "kamayan/callback_dag.hpp"

namespace kamayan {

TEST(CallbackDAG, EmptyGraph) {
  CallbackDAG dag;
  auto order = dag.TopologicalSort();
  EXPECT_EQ(order.size(), 0);
}

TEST(CallbackDAG, SingleNode) {
  CallbackDAG dag;
  dag.AddNode("a");
  auto order = dag.TopologicalSort();
  ASSERT_EQ(order.size(), 1);
  EXPECT_EQ(order[0], "a");
}

TEST(CallbackDAG, LinearOrder) {
  // a -> b -> c
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("b", "c");

  auto order = dag.TopologicalSort();
  ASSERT_EQ(order.size(), 3);
  EXPECT_EQ(order[0], "a");
  EXPECT_EQ(order[1], "b");
  EXPECT_EQ(order[2], "c");
}

TEST(CallbackDAG, DiamondDependency) {
  // Diamond:
  //     a
  //    / \
  //   b   c
  //    \ /
  //     d
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("a", "c");
  dag.AddEdge("b", "d");
  dag.AddEdge("c", "d");

  auto order = dag.TopologicalSort();
  ASSERT_EQ(order.size(), 4);

  // 'a' must come first
  EXPECT_EQ(order[0], "a");

  // 'd' must come last
  EXPECT_EQ(order[3], "d");

  // 'b' and 'c' can be in either order, but both must come after 'a' and before 'd'
  EXPECT_TRUE((order[1] == "b" && order[2] == "c") ||
              (order[1] == "c" && order[2] == "b"));
}

TEST(CallbackDAG, DisconnectedComponents) {
  // Two separate chains: a -> b and c -> d
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("c", "d");

  auto order = dag.TopologicalSort();
  ASSERT_EQ(order.size(), 4);

  // Check that dependencies within each chain are preserved
  auto pos_a = std::find(order.begin(), order.end(), "a") - order.begin();
  auto pos_b = std::find(order.begin(), order.end(), "b") - order.begin();
  auto pos_c = std::find(order.begin(), order.end(), "c") - order.begin();
  auto pos_d = std::find(order.begin(), order.end(), "d") - order.begin();

  EXPECT_LT(pos_a, pos_b);  // a before b
  EXPECT_LT(pos_c, pos_d);  // c before d
}

TEST(CallbackDAG, SimpleCycle) {
  // a -> b -> a (cycle)
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("b", "a");

  EXPECT_THROW({ auto order = dag.TopologicalSort(); }, std::runtime_error);
}

TEST(CallbackDAG, ThreeNodeCycle) {
  // a -> b -> c -> a (cycle)
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("b", "c");
  dag.AddEdge("c", "a");

  EXPECT_THROW({ auto order = dag.TopologicalSort(); }, std::runtime_error);
}

TEST(CallbackDAG, CycleWithBranch) {
  // More complex case with cycle:
  //   a -> b -> c
  //   ^         |
  //   |_________|
  //        d (no cycle, separate)
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("b", "c");
  dag.AddEdge("c", "a");  // Cycle
  dag.AddNode("d");       // Separate node

  EXPECT_THROW({ auto order = dag.TopologicalSort(); }, std::runtime_error);
}

TEST(CallbackDAG, CycleErrorMessage) {
  // Verify that cycle error messages contain useful information
  CallbackDAG dag;
  dag.AddEdge("hydro", "eos");
  dag.AddEdge("eos", "multispecies");
  dag.AddEdge("multispecies", "hydro");  // Creates cycle

  try {
    auto order = dag.TopologicalSort();
    FAIL() << "Expected std::runtime_error to be thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_THAT(msg, ::testing::HasSubstr("Cyclic dependency"));
    EXPECT_THAT(msg, ::testing::HasSubstr("hydro"));
    EXPECT_THAT(msg, ::testing::HasSubstr("eos"));
    EXPECT_THAT(msg, ::testing::HasSubstr("multispecies"));
  }
}

TEST(CallbackDAG, ComplexDAG) {
  // More realistic scenario:
  //   grid -> hydro -> eos
  //              |      |
  //              v      v
  //         multispecies
  //              |
  //              v
  //           driver
  CallbackDAG dag;
  dag.AddEdge("grid", "hydro");
  dag.AddEdge("hydro", "eos");
  dag.AddEdge("hydro", "multispecies");
  dag.AddEdge("eos", "multispecies");
  dag.AddEdge("multispecies", "driver");

  auto order = dag.TopologicalSort();
  ASSERT_EQ(order.size(), 5);

  auto pos_grid = std::find(order.begin(), order.end(), "grid") - order.begin();
  auto pos_hydro = std::find(order.begin(), order.end(), "hydro") - order.begin();
  auto pos_eos = std::find(order.begin(), order.end(), "eos") - order.begin();
  auto pos_multi = std::find(order.begin(), order.end(), "multispecies") - order.begin();
  auto pos_driver = std::find(order.begin(), order.end(), "driver") - order.begin();

  EXPECT_LT(pos_grid, pos_hydro);
  EXPECT_LT(pos_hydro, pos_eos);
  EXPECT_LT(pos_hydro, pos_multi);
  EXPECT_LT(pos_eos, pos_multi);
  EXPECT_LT(pos_multi, pos_driver);
}

TEST(CallbackDAG, GraphVizOutput) {
  CallbackDAG dag;
  dag.AddEdge("a", "b");
  dag.AddEdge("b", "c");

  std::ostringstream oss;
  dag.WriteGraphViz(oss);

  std::string output = oss.str();
  EXPECT_THAT(output, ::testing::HasSubstr("digraph"));
  EXPECT_THAT(output, ::testing::HasSubstr("\"a\""));
  EXPECT_THAT(output, ::testing::HasSubstr("\"b\""));
  EXPECT_THAT(output, ::testing::HasSubstr("\"c\""));
  EXPECT_THAT(output, ::testing::HasSubstr("\"a\" -> \"b\""));
  EXPECT_THAT(output, ::testing::HasSubstr("\"b\" -> \"c\""));
}

TEST(CallbackDAG, StreamOperator) {
  CallbackDAG dag;
  dag.AddEdge("x", "y");

  std::ostringstream oss;
  oss << dag;

  std::string output = oss.str();
  EXPECT_THAT(output, ::testing::HasSubstr("digraph"));
  EXPECT_THAT(output, ::testing::HasSubstr("\"x\" -> \"y\""));
}

}  // namespace kamayan
