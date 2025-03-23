#include <gtest/gtest.h>

#include <vector>

#include "kamayan/fields.hpp"
#include "physics/eos/eos_types.hpp"
#include "physics/eos/equation_of_state.hpp"
#include "physics/physics_types.hpp"

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

template <EosComponent component>
auto EosData(std::array<Real, 6> &data) {
  return EosTestData<typename EosVars<component>::types>(data);
}

class EosTest : public testing::Test {
 protected:
  EosTest() {}

  singularity::EOS eos;
};

TEST(Eos, IdealGas) {
  auto eos = EquationOfState<EosModel::gamma>(1.4, 1.0);
  auto eos_arr = std::array<Real, 6>{1., 0., 0., 1., 0., 0.};
  auto eos_data = EosData<EosComponent::oneT>(eos_arr);
  std::vector<Real> lambda(eos.nlambda());

  // eint = P / dens / (gamma - 1)
  EosCall<Fluid::oneT, EosMode::pres>(eos, eos_data, lambda);
  EXPECT_NEAR(eos_data(EINT()), 1. / 0.4, 1.e-14);
  EosCall<Fluid::oneT, EosMode::ener>(eos, eos_data, lambda);
  EXPECT_EQ(eos_data(PRES()), 1.);
}

TEST(Eos, EOS_t) {
  EOS_t eos(EquationOfState<EosModel::gamma>(1.4, 1.0));
  auto eos_arr = std::array<Real, 6>{1., 0., 0., 1., 0., 0.};
  auto eos_data = EosData<EosComponent::oneT>(eos_arr);
  std::vector<Real> lambda(eos.nlambda());

  using fill_mode_pres = SingularityEosFill<EosMode::pres>;
  using fill_mode_ener = SingularityEosFill<EosMode::ener>;
  using eos_vars = EosVars<EosComponent::oneT>;

  // eint = P / dens / (gamma - 1)
  eos.Call(eos_vars(), fill_mode_pres(), eos_data, lambda);
  EXPECT_NEAR(eos_data(EINT()), 1. / 0.4, 1.e-14);
  eos.Call(eos_vars(), fill_mode_ener(), eos_data, lambda);
  EXPECT_EQ(eos_data(PRES()), 1.);
}

}  // namespace kamayan::eos
