#include "gll.h"
#include <algorithm>
#include <cmath>

/**
 * @breif Computes Bonnet recursion formula to approximate Legendre poly at
 * order P
 * @param P Order of the LP
 * @param xi double valiu [-1,1]
 * @return double
 */
inline double Bonnet(int P, double xi) {
  if (P == 0)
    return 1.;
  if (P == 1)
    return xi;
  double L_nm2 = 1.;
  double L_nm1 = xi;
  double L_n = 0.;
  for (int n = 2; n < P + 1; n++) {
    L_n = ((2. * n - 1.) * xi * L_nm1 - (n - 1.) * L_nm2) / n;
    L_nm2 = L_nm1;
    L_nm1 = L_n;
  }
  return L_n;
}

/**
 * @brief Computes the derivative of the LP of order P at point xo
 * @param P Order of the LP
 * @param xi Value in [-1,1]
 * @return double
 */
inline double Lpp(int P, double xi) {
  return (P / (1 - xi * xi)) * (Bonnet(P - 1, xi) - xi * Bonnet(P, xi));
}

/**
 * @brief Computes the double derivative of the LP of order P at point xo
 * @param P Order of the LP
 * @param xi Value in [-1,1]
 * @return double
 */
inline double Lppp(int P, double xi) {
  return -(P * (P + 1) / (1 - xi * xi)) * Bonnet(P, xi);
}

/**
 * @brief Computes the Derivative matrix D
 * @param D Matrix [P+1, P+1] storing the derivative coeffs
 * @param quads Positions of the quadrature points
 * @param p Order of the LP
 * @param q Number of quadrature points
 * @return void
 */
void computeDerivative(double *D, double *quads, const int p, const int q) {

  int N = p;

  for (int i = 0; i < q; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < q; ++j) {
      if (i != j) {
        // Off-diagonal formula using Legendre values at the nodes
        double Li = Bonnet(N, quads[i]);
        double Lj = Bonnet(N, quads[j]);

        D[i * q + j] = (Li / Lj) * (1.0 / (quads[i] - quads[j]));
        rowSum += D[i * q + j];
      }
    }
    D[i * q + i] = -rowSum;
  }
}

/**
 * @brief Computes the quadrature points of the LP of order P on [-1,1]
 * @param quads Quadrature points array
 * @param P Order of the LP
 * @return void
 */
void setQuads(double *quads, const int Q) {

  quads[0] = -1.0;
  quads[Q] = 1.0;

  for (int k = 1; k <= Q / 2; ++k) {
    double xi = -cos(M_PI * k / Q);
    double eps;
    double temp;
    for (int iter = 0; iter < 50; ++iter) {
      temp = xi - (Lpp(Q, xi) / Lppp(Q, xi));
      eps = fabs(temp - xi);
      xi = temp;
      if (eps < 1e-15)
        break;
    }
    quads[k] = xi;
    quads[Q - k] = -xi;
  }
}

/**
 * @brief Computes the weights of the quadratures of the LP of order P on [-1,1]
 * @param weights Weights of the quadratures
 * @param quad Quadratures of the LP
 * @param P Order of the LP
 * @return void
 */
void setWeights(double *weights, const double *quads, const int Q) {
  for (int i = 0; i <= Q; ++i) {
    double Lpi = Bonnet(Q, quads[i]);
    weights[i] = 2 / (Q * (Q + 1) * (Lpi * Lpi));
  }
}

namespace gll {
Basis::Basis(const int P, const int Q) : p(P), q(Q) {
  quads = new double[q];
  weights = new double[q];
  D = new double[(q) * (q)];
  setQuads(quads, q - 1);
  setWeights(weights, quads, q - 1);
  computeDerivative(D, quads, p, q);
}

Basis::~Basis() {
  delete[] quads;
  delete[] weights;
}

std::ostream &operator<<(std::ostream &os, const Basis &b) {
  os << "----- BASIS -----" << std::endl
     << "ORDER  : " << b.p << std::endl
     << "NQUADS : " << b.q << std::endl;
  return os;
}
} // namespace gll
