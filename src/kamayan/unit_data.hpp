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
    const auto Get() const { return value_; }
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
    template <typename T>
    void Validator(const std::string &key,
                   std::vector<runtime_parameters::Rule<T>> rules) {
      std::stringstream err_msg;
      err_msg << "[UnitParm] Invalid runtime value for parameter ";
      err_msg << "<" + parent->block + ">/" << key << "\n";
      err_msg << "Valid values are: "
              << runtime_parameters::impl::ToDocString("\n", rules);
      auto err_msg_str = err_msg.str();
      validate = [=](const DataType &value) {
        bool valid_value = (rules.size() == 0);
        for (const auto &rule : rules) {
          valid_value = valid_value || rule.validate(std::get<T>(value));
        }
        PARTHENON_REQUIRE_THROWS(valid_value, err_msg_str.c_str())
      };
    }

   public:
    template <typename T>
    requires(runtime_parameters::Rparm<T>)
    void Init(const std::string &key, const std::string &docstring,
              std::vector<runtime_parameters::Rule<T>> rules = {}) {
      Validator(key, rules);
      add_rp = [=, this](RPs *rps) {
        value_ = rps->GetOrAdd(parent->block, key, std::get<T>(value_), docstring, rules);
      };

      add_param = [=, this]() {
        auto params = Params();
        params->AddParam(key, std::get<T>(value_),
                         parthenon::Params::Mutability::Mutable);
      };

      update_param = [=, this](const DataType &new_value) {
        validate(new_value);
        auto params = Params();
        params->UpdateParam(key, std::get<T>(new_value));
      };
    }

    template <typename T>
    requires(PolyOpt<T>)
    void Init(const std::string &key, const std::string &docstring,
              std::map<std::string, T> mapping) {
      using Rule = runtime_parameters::Rule<std::string>;

      std::vector<Rule> rules;
      for (const auto &pair : mapping) {
        rules.push_back(pair.first);
      }

      Validator(key, rules);
      add_rp = [=, this](RPs *rps) {
        value_ = rps->GetOrAdd<std::string>(
            parent->block, key, std::get<std::string>(value_), docstring, rules);
        auto cfg = Config();
        cfg->Add(mapping.at(std::get<std::string>(value_)));
      };

      update_param = [=, this](const DataType &new_value) {
        PARTHENON_REQUIRE_THROWS(std::holds_alternative<std::string>(new_value),
                                 "Config must be set with a string.")
        validate(new_value);
        auto cfg = Config();
        cfg->Update(mapping.at(std::get<std::string>(new_value)));
      };
    }

    template <typename T>
    requires(runtime_parameters::Rparm<T>)
    UnitParm(UnitData *parent_ptr, const std::string &key, const T &value)
        : value_(value), parent(parent_ptr) {}

    template <typename T>
    requires(PolyOpt<T>)
    UnitParm(UnitData *parent_ptr, const std::string &key, const std::string &value)
        : parent(parent_ptr), value_(value) {}

   private:
    UnitData *parent;
    DataType value_;
    std::function<void(RPs *rps)> add_rp = nullptr;
    std::function<void()> add_param = nullptr;
    std::function<void(const DataType)> update_param = nullptr, validate = nullptr;
  };

  explicit(false) UnitData(const std::string &name) : block(name) {}

  void Setup(std::shared_ptr<runtime_parameters::RuntimeParameters> rps,
             std::shared_ptr<Config> cfg);
  void Initialize(std::shared_ptr<StateDescriptor> pkg);

  template <typename T>
  requires(runtime_parameters::Rparm<T>)
  void AddParm(const std::string &key, const T &value, const std::string &docstring,
               std::vector<runtime_parameters::Rule<T>> rules = {}) {
    parameters.emplace(key, UnitParm(this, key, value));
    parameters.at(key).Init<T>(key, docstring, rules);
  }

  template <typename T>
  requires(PolyOpt<T>)
  void AddParm(const std::string &key, const std::string &value,
               const std::string &docstring, std::map<std::string, T> mapping) {
    parameters.emplace(key, UnitParm(this, key, value));
    parameters.at(key).Init<T>(key, docstring, mapping);
  }

  void UpdateParm(const std::string &key, const DataType &value) {
    parameters.at(key).Update(value);
  }

  template <typename T>
  const auto Get(const std::string &key) const {
    return std::get<T>(parameters.at(key).Get());
  }

 private:
  std::string block;
  std::weak_ptr<Config> config;
  std::weak_ptr<StateDescriptor> params;
  std::map<std::string, UnitParm> parameters;
};
}  // namespace kamayan
#endif  // KAMAYAN_UNIT_DATA_HPP_
