#include <gtest/gtest.h>

#include <vector>

#include "kamayan/fields.hpp"
#include "physics/eos/eos_singularity.hpp"
#include "physics/eos/eos_types.hpp"

namespace kamayan::eos {

template <typename>
struct EosTestData {};

template <typename... Ts>
struct EosTestData<TypeList<Ts...>> {
 public:
  using types = TypeList<Ts...>;
  static constexpr std::size_t size = types::n_types;
  using Arr_t = std::array<Real, size>;

  explicit EosTestData(Arr_t &data_) : data(data_) {}

  Real &operator[](int idx) { return data[idx]; }

  template <typename T>
  requires(types::template Contains<T>())
  Real &operator()(T) {
    return data[types::Idx(T())];
  }

 private:
  Arr_t &data;
};

template <eosType eos_type>
auto EosData(std::array<Real, 6> &data) {
  return EosTestData<typename EosVars<eos_type>::types>(data);
}

class EosTest : public testing::Test {
 protected:
  EosTest() {}

  singularity::EOS eos;
};

TEST(Eos, IdealGas) {
  //
  singularity::EOS eos = singularity::IdealGas(0.4, 1.0);
  auto eos_arr = std::array<Real, 6>{1., 0., 0., 1., 0., 0.};
  auto eos_data = EosData<eosType::oneT>(eos_arr);
  std::vector<Real> lambda(eos.nlambda());

  // eint = P / dens / (gamma - 1)
  EosSingle<eosMode::pres>(eos_data, eos, lambda);
  EXPECT_EQ(eos_data(EINT()), 1. / 0.4);
  EosSingle<eosMode::ener>(eos_data, eos, lambda);
  EXPECT_EQ(eos_data(PRES()), 1.);
}

}  // namespace kamayan::eos
