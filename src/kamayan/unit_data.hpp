#ifndef KAMAYAN_UNIT_DATA_HPP_
#define KAMAYAN_UNIT_DATA_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <utils/error_checking.hpp>

#include "dispatcher/option_types.hpp"
#include "kamayan/runtime_parameters.hpp"
namespace kamayan {
struct UnitData {
  using RPs = runtime_parameters::RuntimeParameters;
  using DataType = std::variant<Real, int, bool, std::string>;
  const std::string Block() const { return block; }

  struct UnitParm {
    void AddRP(RPs *rps) const {
      if (add_rp != nullptr) add_rp(rps);
    }
    void AddParam() const {
      if (add_param != nullptr) add_param();
    }
    void Update(const DataType &value) const {
      if (update_param != nullptr) update_param(value);
    }

   private:
    auto Params() { return parent->params.lock(); }
    auto Config() { return parent->config.lock(); }

   public:
    template <typename T>
    requires(runtime_parameters::Rparm<T>)
    UnitParm(UnitData *parent_ptr, const std::string &key, const T &value,
             const std::string &docstring,
             std::initializer_list<runtime_parameters::Rule<T>> rules = {})
        : parent(parent_ptr) {
      PARTHENON_REQUIRE_THROWS(std::holds_alternative<T>(value),
                               "Parameter types must match.")
      add_rp = [=, this](RPs *rps) {
        value_ = rps->GetOrAdd(parent->block, key, value, docstring, rules);
      };

      add_param = [=, this]() {
        auto params = Params();
        params->AddParam(key, value_);
      };

      update_param = [=, this](const DataType &new_value) {
        auto params = Params();
        params->UpdateParam(key, new_value);
      };
    }

    template <typename T>
    requires(PolyOpt<T>)
    UnitParm(UnitData *parent_ptr, const std::string &key, const T &value,
             const std::string &docstring, std::map<std::string, T> mapping) {
      PARTHENON_REQUIRE_THROWS(std::holds_alternative<std::string>(value),
                               "Config must be set with a string.")
      std::vector<std::string> rules;
      for (const auto &pair : mapping) {
        rules.push_back(pair.first);
      }

      add_rp = [=, this](RPs *rps) {
        auto val = rps->GetOrAdd(parent->block, key, value, docstring, rules);
        auto cfg = Config();
        cfg->Add(val);
      };

      update_param = [=, this](const DataType &new_value) {
        PARTHENON_REQUIRE_THROWS(std::holds_alternative<std::string>(new_value),
                                 "Config must be set with a string.")
        auto cfg = Config();
        cfg->Update(mapping[new_value]);
      };
    }

   private:
    UnitData *parent;
    DataType value_;
    std::function<void(RPs *rps)> add_rp = nullptr;
    std::function<void()> add_param = nullptr;
    std::function<void(const DataType)> update_param = nullptr;
  };

  explicit(false) UnitData(const std::string &name) : block(name) {}

  void Setup(std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
             std::shared_ptr<Config> cfg);
  void Initialize(std::shared_ptr<StateDescriptor> pkg);

  template <typename... Args>
  void AddParm(Args &&...args) {
    parameters.push_back(UnitParm(this, std::forward<Args>(args)...));
  }

 private:
  std::string block;
  std::weak_ptr<Config> config;
  std::weak_ptr<StateDescriptor> params;
  std::vector<UnitParm> parameters;
};
}  // namespace kamayan
#endif  // KAMAYAN_UNIT_DATA_HPP_
