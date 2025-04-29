#include <gtest/gtest.h>

#include <vector>

#include <Kokkos_Macros.hpp>

#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/reconstruction.hpp"

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
auto GetData(const std::size_t &order) {
  std::vector<Real> data;
  data.resize(size);
  for (int i = 0; i < 2 * size + 1; i++) {
    data[i] = PolynomialAvg(order, i - size);
  }

  return data;
}

template <std::size_t size>
struct DataOneD {
  explicit DataOneD(const std::size_t &order_) : order(order_) {
    data = GetData<size>(order);
  }
  // I have no idea why I have to do this... copy constructor
  // for vector<Real> doesn't seem to be working on this machine
  DataOneD(const DataOneD &in) { data = GetData<size>(in.order); }

  Real operator()(const int &idx) { return data[size + idx]; }

  std::vector<Real> data;
  std::size_t order;
};

Real Polynomial(const std::size_t &order, const Real &x) {
  Real value = coeffs[0];
  for (int i = 1; i < order + 1; i++) {
    value += coeffs[i] * Kokkos::pow(x - xc, i);
  }

  return value;
}

TEST(Reconstruction, PLMSlopeLimiters) {
  using minmod = ReconstructTraits<Reconstruction::plm, SlopeLimiter::minmod>;
  using mc = ReconstructTraits<Reconstruction::plm, SlopeLimiter::mc>;
  using van_leer = ReconstructTraits<Reconstruction::plm, SlopeLimiter::van_leer>;
  DataOneD<1> data(1);
  Real vP, vM;
  Reconstruct<minmod>(data, vM, vP);
  EXPECT_EQ(vM, Polynomial(1, -0.5));
  EXPECT_EQ(vP, Polynomial(1, 0.5));

  Reconstruct<mc>(data, vM, vP);
  EXPECT_EQ(vM, Polynomial(1, -0.5));
  EXPECT_EQ(vP, Polynomial(1, 0.5));

  Reconstruct<van_leer>(data, vM, vP);
  EXPECT_EQ(vM, Polynomial(1, -0.5));
  EXPECT_EQ(vP, Polynomial(1, 0.5));
}
}  // namespace kamayan::hydro
