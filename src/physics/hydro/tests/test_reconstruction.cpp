#include <cstdlib>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/reconstruction.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

const Kokkos::Array<Real, 5> coeffs{10., 3., 1., 0.5};
const Real xc = 20.;

Real PolynomialAvg(const std::size_t &order, const int &idx) {
  // average a polynomial of order order over the stencil point
  // centered at idx
  Real value = coeffs[0];  // constant value
  // f(x) = A x^n
  // -> integral(f(x), idx-1/2, idx+1/2) = A / (n+1) * x^{n+1} | _idx-1/2 ^ idx+1/2
  for (int i = 1; i < order + 1; i++) {
    value += coeffs[i] / static_cast<Real>(i + 1) *
             (Kokkos::pow(static_cast<Real>(idx) + 0.5 - xc, i + 1) -
              Kokkos::pow(static_cast<Real>(idx) - 0.5 - xc, i + 1));
  }

  return value;
}

template <std::size_t size>
auto GetData(std::size_t order) {
  std::vector<Real> data;
  data.resize(2 * size + 1);
  for (int i = 0; i < 2 * size + 1; i++) {
    data[i] = PolynomialAvg(order, i - size);
  }

  return data;
}

template <std::size_t size>
struct DataOneD {
  explicit DataOneD(std::size_t order_) : order(order_) { data = GetData<size>(order); }
  // I have no idea why I have to do this... copy constructor
  // for vector<Real> doesn't seem to be working on this machine
  DataOneD(const DataOneD &in) {
    data = GetData<size>(in.order);
    order = in.order;
  }

  Real operator()(const int &idx) { return data[size + idx]; }

  std::vector<Real> data;
  std::size_t order;
};

Real Polynomial(std::size_t order, const Real &x) {
  Real value = coeffs[0];
  for (int i = 1; i < order + 1; i++) {
    value += coeffs[i] * Kokkos::pow(x - xc, i);
  }

  return value;
}

template <typename T>
requires(NonTypeTemplateSpecialization<T, ReconstructTraits>)
class ReconstructionTest : public testing::Test {};

class ReconstructionTestNamer {
 public:
  template <typename T>
  requires(NonTypeTemplateSpecialization<T, ReconstructTraits>)
  static std::string GetName(int) {
    return PolyOpt_t<Reconstruction>::opt<T::reconstruction>::Label() + "-" +
           PolyOpt_t<SlopeLimiter>::opt<T::slope_limiter>::Label();
  }
};

constexpr std::size_t GetSize(Reconstruction recon) {
  if (recon == Reconstruction::plm) return 1;
  if (recon == Reconstruction::ppm || recon == Reconstruction::wenoz) return 2;
  return 0;
}

constexpr std::size_t GetOrder(Reconstruction recon) {
  if (recon == Reconstruction::plm) return 1;
  if (recon == Reconstruction::ppm) return 2;
  if (recon == Reconstruction::wenoz) return 3;
  return 0;
}

using minmod = ReconstructTraits<Reconstruction::plm, SlopeLimiter::minmod>;
using mc = ReconstructTraits<Reconstruction::plm, SlopeLimiter::mc>;
using van_leer = ReconstructTraits<Reconstruction::plm, SlopeLimiter::van_leer>;
using minmod_ppm = ReconstructTraits<Reconstruction::ppm, SlopeLimiter::minmod>;
using mc_ppm = ReconstructTraits<Reconstruction::ppm, SlopeLimiter::mc>;
using van_leer_ppm = ReconstructTraits<Reconstruction::ppm, SlopeLimiter::van_leer>;
using wenoz = ReconstructTraits<Reconstruction::wenoz, SlopeLimiter::van_leer>;
using ReconTypes =
    ::testing::Types<minmod, mc, van_leer, minmod_ppm, mc_ppm, van_leer_ppm, wenoz>;
TYPED_TEST_SUITE(ReconstructionTest, ReconTypes);
// why does using the namer cause ctest not to catch failures???
// TYPED_TEST_SUITE(ReconstructionTest, ReconTypes, ReconstructionTestNamer);

TYPED_TEST(ReconstructionTest, PlmSlopeLimiters) {
  auto order = GetOrder(TypeParam::reconstruction);
  DataOneD<GetSize(TypeParam::reconstruction)> data(order);
  Real vP, vM;
  Reconstruct<TypeParam>(data, vM, vP);
  constexpr Real eps = 5.e-6;
  EXPECT_LT(std::abs((vM - Polynomial(order, -0.5)) / Polynomial(order, -0.5)), eps);
  EXPECT_LT(std::abs((vP - Polynomial(order, 0.5)) / Polynomial(order, 0.5)), eps);
}

}  // namespace kamayan::hydro
