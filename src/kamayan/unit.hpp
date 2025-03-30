#ifndef KAMAYAN_UNIT_HPP_
#define KAMAYAN_UNIT_HPP_
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "driver/kamayan_driver_types.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/runtime_parameters.hpp"
#include "utils/error_checking.hpp"

namespace kamayan {
struct KamayanUnit {
  // Setup is called to add options into the kamayan configuration and to register
  // runtime parameters owned by the unit
  std::function<void(Config *, runtime_parameters::RuntimeParameters *)> Setup = nullptr;

  // Initialize is responsible for setting up the parthenon StateDescriptor, registering
  // params , adding fields owned by the unit & registering any callbacks known to
  // parthenon
  std::function<std::shared_ptr<StateDescriptor>(
      const Config *, const runtime_parameters::RuntimeParameters *)>
      Initialize = nullptr;

  // These tasks get added to the tasklist that accumulate dudt for this unit based
  // on the current state in md, returning the TaskID of the final task for a single
  // stage in the multi-stage driver
  std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, MeshData *dudt)>
      AddTasksOneStep = nullptr;

  // These tasks are used to advance md by dt as one of the operators in the
  // operator splitting
  std::function<TaskID(TaskID prev, TaskList &tl, MeshData *md, const Real &dt)>
      AddTasksSplit = nullptr;
};

// container that is used to iterate through an ordered list of keys
// used in a map
template <typename T>
struct MapList {
  explicit MapList(std::map<std::string, T> &units) : map(units) {}
  explicit MapList(std::list<std::string> unit_list, std::map<std::string, T> &units)
      : keys(unit_list), map(units) {}

  void push_back(std::string k) {
    PARTHENON_REQUIRE_THROWS(map.contains(k), "Trying to add a key not contained in map")
    keys.push_back(k);
  }

  MapList &operator=(std::list<std::string> new_keys) {
    for (auto &k : new_keys) {
      PARTHENON_REQUIRE_THROWS(map.contains(k),
                               "Trying to add a key not contained in map")
    }
    keys = new_keys;
    return *this;
  }

  struct IterItem {
    std::list<std::string>::iterator it;
    std::map<std::string, T> &map;

    friend bool operator==(const IterItem &a, const IterItem &b) { return a.it == b.it; }
    friend bool operator!=(const IterItem &a, const IterItem &b) { return a.it != b.it; }

    IterItem(std::list<std::string>::iterator i, std::map<std::string, T> &m)
        : it(i), map(m) {}
  };

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = IterItem;
    using reference = T &;

    explicit Iterator(IterItem ptr) : m_ptr(ptr) {}

    reference operator*() const { return m_ptr.map[*(m_ptr.it)]; }
    pointer operator->() { return m_ptr; }
    Iterator &operator++() {
      m_ptr.it++;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.m_ptr == b.m_ptr;
    }
    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.m_ptr != b.m_ptr;
    }

   private:
    IterItem m_ptr;
  };

  Iterator begin() { return Iterator(IterItem(keys.begin(), map)); }
  Iterator end() { return Iterator(IterItem(keys.end(), map)); }

 private:
  std::list<std::string> keys;
  std::map<std::string, T> &map;
};

struct UnitCollection {
  using MapList_t = MapList<std::shared_ptr<KamayanUnit>>;
  MapList_t rk_stage, operator_split;

  UnitCollection() : rk_stage(units), operator_split(units) {}

  std::shared_ptr<KamayanUnit> &operator[](const std::string &key) { return units[key]; }

  // iterator goes over all registered units
  auto begin() { return units.begin(); }
  auto end() { return units.end(); }

  // private:
  std::map<std::string, std::shared_ptr<KamayanUnit>> units;
};

// gather up all the units in kamayan
UnitCollection ProcessUnits();

}  // namespace kamayan

#endif  // KAMAYAN_UNIT_HPP_
