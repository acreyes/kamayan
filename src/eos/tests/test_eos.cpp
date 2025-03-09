#include <gtest/gtest.h>

#include <array>

#include "eos/eos_singularity.hpp"
#include "eos/eos_types.hpp"
#include "kamayan/fields.hpp"

namespace kamayan::eos {

template <typename>
struct EosTestData {};

template <typename... Ts>
struct EosTestData<TypeList<Ts...>> {
 public:
  using types = TypeList<Ts...>;
  static constexpr std::size_t size = types::n_types;

  explicit EosTestData(std::vector<Real> data_) : data(data_) {}

  Real &operator[](int idx) { return data[idx]; }

  template <typename T>
  requires(types::template Contains<T>())
  Real &operator()(T) {
    return data[types::Idx(T())];
  }

 private:
  std::vector<Real> data;
};

template <eosType eos_type, typename... Args>
requires(sizeof...(Args) == EosVars<eos_type>::types::n_types)
auto EosData(Args &&...args) {
  return EosTestData<typename EosVars<eos_type>::types>(std::vector<Real>{args...});
}

class EosTest : public testing::Test {
 protected:
  EosTest() {}

  singularity::EOS eos;
};

TEST(Eos, IdealGas) {
  //
  singularity::EOS eos = singularity::IdealGas(0.4, 1.0);
  auto eos_data = EosData<eosType::oneT>(1., 0., 0., 1., 0., 0.);
  // eint = P / dens / (gamma - 1)
  EosSingle<eosMode::pres>(eos_data, eos);
  EXPECT_EQ(eos_data(EINT()), 1. / 0.4);
  EosSingle<eosMode::ener>(eos_data, eos);
  EXPECT_EQ(eos_data(PRES()), 1.);
}

}  // namespace kamayan::eos
