#ifndef KAMAYAN_UNIT_HPP_
#define KAMAYAN_UNIT_HPP_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/callback_dag.hpp"
#include "kamayan/callback_registration.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "kamayan/unit_data.hpp"

namespace kamayan {

struct UnitCollection;
struct KamayanUnit;

}  // namespace kamayan

namespace kamayan {

struct KamayanUnit : public StateDescriptor,
                     public std::enable_shared_from_this<KamayanUnit> {
  explicit KamayanUnit(std::string name) : StateDescriptor(name), name_(name) {}

  ~KamayanUnit() = default;

  // Setup is called to add options into the kamayan configuration and to register
  // runtime parameters owned by the unit
  CallbackRegistration<std::function<void(KamayanUnit *unit)>> SetupParams;

  // Initialize is responsible for setting up the parthenon StateDescriptor, registering
  // params , adding fields owned by the unit & registering any callbacks known to
  // parthenon
  CallbackRegistration<std::function<void(KamayanUnit *unit)>> InitializeData;

  // Used as a callback during problem generation on the mesh
  CallbackRegistration<std::function<void(MeshBlock *)>> ProblemGeneratorMeshBlock;

  // makes sure the conserved variables are ready before applying dudt
  CallbackRegistration<std::function<TaskStatus(MeshData *md)>> PrepareConserved;

  // make sure primitive variables are ready after updating conserved
  CallbackRegistration<std::function<TaskStatus(MeshData *md)>> PreparePrimitive;

  // Accumulates the fluxes in md, and the driver will handle the flux
  // correction and dudt
  CallbackRegistration<std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md)>>
      AddFluxTasks;

  // These tasks get added to the tasklist that accumulate dudt for this unit based
  // on the current state in md, returning the TaskID of the final task for a single
  // stage in the multi-stage driver
  CallbackRegistration<
      std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt)>>
      AddTasksOneStep;

  // These tasks are used to advance md by dt as one of the operators in the
  // operator splitting
  CallbackRegistration<
      std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, const Real &dt)>>
      AddTasksSplit;

  const std::string Name() const { return name_; }

  // get a reference to the UnitData configured for a particular block
  const UnitData &Data(const std::string &key) const;
  UnitData &AddData(const std::string &block);
  bool HasData(const std::string &block) const;
  auto &AllData() { return unit_data_; }

  std::shared_ptr<Config> Configuration() { return config_; }
  std::shared_ptr<runtime_parameters::RuntimeParameters> RuntimeParameters() {
    return runtime_parameters_;
  }

  void InitResources(std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
                     std::shared_ptr<Config> cfg);
  void InitializePackage(std::shared_ptr<StateDescriptor> pkg);

  void SetUnits(std::shared_ptr<const UnitCollection> units);
  const KamayanUnit &GetUnit(const std::string &name) const;
  std::shared_ptr<const KamayanUnit> GetUnitPtr(const std::string &name) const;

  static std::shared_ptr<KamayanUnit> GetFromMesh(MeshData *md, const std::string &name);

 private:
  std::string name_;
  std::map<std::string, UnitData> unit_data_;
  std::shared_ptr<Config> config_;
  std::shared_ptr<runtime_parameters::RuntimeParameters> runtime_parameters_;
  std::weak_ptr<const UnitCollection> units_;
};
// --8<-- [end:unit]

struct UnitCollection {
  std::list<std::string> rk_fluxes, rk_stage, prepare_prim, operator_split;

  UnitCollection(const UnitCollection &uc)
      : units(uc.units), rk_fluxes(uc.rk_fluxes), rk_stage(uc.rk_stage),
        prepare_prim(uc.prepare_prim), operator_split(uc.operator_split) {}
  UnitCollection() {}

  std::shared_ptr<KamayanUnit> Get(const std::string &key) const { return units.at(key); }
  std::shared_ptr<KamayanUnit> &operator[](const std::string &key) { return units[key]; }

  auto GetMap() const { return &units; }

  // iterator goes over all registered units
  auto begin() const { return units.begin(); }
  auto end() const { return units.end(); }

  void AddTasks(std::list<std::string> unit_list,
                std::function<void(KamayanUnit *)> function) const;

  /// Execute callbacks in DAG-determined order based on dependencies.
  ///
  /// This method builds a dependency graph from callback registrations,
  /// computes a topological ordering, and executes the callbacks in that order.
  ///
  /// @tparam CallbackGetter Function that extracts the callback registration from a unit
  /// @param getter Lambda that returns reference to CallbackRegistration from KamayanUnit
  /// @param executor Function to invoke for each unit with registered callback
  /// @param callback_name Name of callback type (for error messages and diagnostics)
  template <typename CallbackGetter>
  void AddTasksDAG(CallbackGetter getter, std::function<void(KamayanUnit *)> executor,
                   const std::string &callback_name) const;

  /// Build execution order for a callback type based on DAG dependencies.
  ///
  /// @tparam CallbackGetter Function that extracts the callback registration from a unit
  /// @param getter Lambda that returns reference to CallbackRegistration from KamayanUnit
  /// @param callback_name Name of callback type (for error messages)
  /// @return Vector of unit names in execution order
  template <typename CallbackGetter>
  std::vector<std::string> BuildExecutionOrder(CallbackGetter getter,
                                               const std::string &callback_name) const;

  /// Write callback dependency graph in GraphViz DOT format.
  ///
  /// @tparam CallbackGetter Function that extracts the callback registration from a unit
  /// @param stream Output stream to write to
  /// @param getter Lambda that returns reference to CallbackRegistration from KamayanUnit
  /// @param callback_name Name of callback type (for annotation)
  template <typename CallbackGetter>
  void WriteCallbackGraph(std::ostream &stream, CallbackGetter getter,
                          const std::string &callback_name) const;

  void Add(std::shared_ptr<KamayanUnit> kamayan_unit);

 private:
  std::map<std::string, std::shared_ptr<KamayanUnit>> units;
};

// Template method implementations (must be in header for template instantiation)

template <typename CallbackGetter>
std::vector<std::string>
UnitCollection::BuildExecutionOrder(CallbackGetter getter,
                                    const std::string &callback_name) const {
  CallbackDAG dag;

  // Add all units with this callback registered as nodes
  for (const auto &[name, unit] : units) {
    auto &registration = getter(unit.get());
    if (registration.IsRegistered()) {
      dag.AddNode(name);
    }
  }

  // Build edges from dependency specifications
  for (const auto &[name, unit] : units) {
    auto &registration = getter(unit.get());
    if (!registration.IsRegistered()) continue;

    // "depends_on" means this unit runs AFTER those units
    for (const auto &dependency : registration.depends_on) {
      // Only add edge if the dependency unit also has this callback registered
      if (units.count(dependency) > 0) {
        auto dep_unit = units.at(dependency);
        auto &dep_registration = getter(dep_unit.get());
        if (dep_registration.IsRegistered()) {
          // Edge: dependency -> name (dependency executes first)
          dag.AddEdge(dependency, name);
        }
      }
    }

    // "required_by" means this unit runs BEFORE those units
    for (const auto &dependent : registration.required_by) {
      // Only add edge if the dependent unit also has this callback registered
      if (units.count(dependent) > 0) {
        auto dep_unit = units.at(dependent);
        auto &dep_registration = getter(dep_unit.get());
        if (dep_registration.IsRegistered()) {
          // Edge: name -> dependent (this executes first)
          dag.AddEdge(name, dependent);
        }
      }
    }
  }

  // Compute topological order (may throw on cycle)
  try {
    return dag.TopologicalSort();
  } catch (const std::exception &e) {
    PARTHENON_THROW("Error building execution order for " + callback_name +
                    " callbacks: " + e.what());
  }
}

template <typename CallbackGetter>
void UnitCollection::AddTasksDAG(CallbackGetter getter,
                                 std::function<void(KamayanUnit *)> executor,
                                 const std::string &callback_name) const {
  auto order = BuildExecutionOrder(getter, callback_name);

  for (const auto &unit_name : order) {
    auto unit = Get(unit_name).get();
    auto &registration = getter(unit);
    if (registration.IsRegistered()) {
      executor(unit);
    }
  }
}

template <typename CallbackGetter>
void UnitCollection::WriteCallbackGraph(std::ostream &stream, CallbackGetter getter,
                                        const std::string &callback_name) const {
  CallbackDAG dag;

  // Build DAG same way as BuildExecutionOrder
  for (const auto &[name, unit] : units) {
    auto &registration = getter(unit.get());
    if (registration.IsRegistered()) {
      dag.AddNode(name);
    }
  }

  for (const auto &[name, unit] : units) {
    auto &registration = getter(unit.get());
    if (!registration.IsRegistered()) continue;

    for (const auto &dependency : registration.depends_on) {
      dag.AddEdge(dependency, name);
    }

    for (const auto &dependent : registration.required_by) {
      dag.AddEdge(name, dependent);
    }
  }

  // Output in GraphViz format
  stream << "// Callback execution order for: " << callback_name << "\n";
  stream << dag;
}

// gather up all the units in kamayan
UnitCollection ProcessUnits();

// write out all the doc strings for runtime parameters
// registered by a given unit
std::stringstream RuntimeParameterDocs(KamayanUnit *unit, ParameterInput *pin);

}  // namespace kamayan

#endif  // KAMAYAN_UNIT_HPP_
