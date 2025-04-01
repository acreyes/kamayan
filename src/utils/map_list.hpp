#ifndef UTILS_MAP_LIST_HPP_
#define UTILS_MAP_LIST_HPP_

#include <list>
#include <map>

#include <parthenon/package.hpp>

// A wrapper around a std::map that lets a subset of the
// map get accessed with an iterator in a requested order
// by using a std::list of keys
template <typename U, typename T>
struct MapList {
  explicit MapList(std::map<U, T> &units) : map(units) {}
  explicit MapList(std::list<U> unit_list, std::map<U, T> &units)
      : keys(unit_list), map(units) {}

  void push_back(U k) {
    PARTHENON_REQUIRE_THROWS(map.contains(k), "Trying to add a key not contained in map")
    keys.push_back(k);
  }

  std::list<U> &Keys() { return keys; }

  MapList &operator=(MapList new_map_list) { return new_map_list; }

  MapList &operator=(std::list<U> new_keys) {
    for (auto &k : new_keys) {
      PARTHENON_REQUIRE_THROWS(map.contains(k),
                               "Trying to add a key not contained in map")
    }
    keys = new_keys;
    return *this;
  }

  struct IterItem {
    std::list<U>::const_iterator it;
    const std::map<U, T> &map;

    friend bool operator==(const IterItem &a, const IterItem &b) { return a.it == b.it; }
    friend bool operator!=(const IterItem &a, const IterItem &b) { return a.it != b.it; }

    IterItem(std::list<U>::const_iterator i, const std::map<U, T> &m) : it(i), map(m) {}

    const T &operator[](const U &key) const { return map.at(key); }
  };

  struct Iterator {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = IterItem;
    using reference = const T &;

    explicit Iterator(IterItem ptr) : m_ptr(ptr) {}

    reference operator*() const { return m_ptr[*(m_ptr.it)]; }
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

  Iterator begin() const { return Iterator(IterItem(keys.cbegin(), map)); }
  Iterator end() const { return Iterator(IterItem(keys.cend(), map)); }

 private:
  std::list<U> keys;
  const std::map<U, T> &map;
};

#endif  // UTILS_MAP_LIST_HPP_
