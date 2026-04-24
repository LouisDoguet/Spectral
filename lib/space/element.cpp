#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "../math/math.h"
#include "../phy/physics.h"
#include "element.h"

/**
 * @brief BLAS of the derivative (dgemv)
 * @param dFdx DivFlux to overwrite
 * @param D Derivative matrix
 * @param F Flux
 * @param invJ 1/J, where J jacobian for the base change
 * @param n Size of the vector
 */
void divF(double *dFdx, const double *D, const double *F, const double invJ,
          const int n) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ, D, n, F, 1, 0., dFdx, 1);
}

namespace elem {

/**
 * @brief Sets the Jacobian from the element position
 */
void Element::setJ(double xL, double xR) {
  this->xL = xL;
  this->xR = xR;
  double dx = xR - xL;
  this->J = dx / 2.;
  this->invJ = 1 / J;
}

/// Core constructor: all public constructors delegate here
Element::Element(int id, gll::Basis *basis, double xL, double xR,
                 double *rho, double *rhou, double *e, bool ownsMemory)
    : id(id), basis(basis), rho(rho), rhou(rhou), e(e), ownsMemory(ownsMemory) {
  setJ(xL, xR);
  int n = basis->getOrder() + 1;
  F1 = new double[n]; F2 = new double[n]; F3 = new double[n];
  divF1 = new double[n]; divF2 = new double[n]; divF3 = new double[n];
  legendreCoefficients = new double[n];
}

/// No-values: allocates zeroed U arrays, delegates to core
Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR)
    : Element(id, sharedBasis, xL, xR,
              new double[sharedBasis->getOrder() + 1](),
              new double[sharedBasis->getOrder() + 1](),
              new double[sharedBasis->getOrder() + 1](),
              true) {}

/// Scalar init: delegates to no-values, then fills U arrays
Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
                 double rho_init, double rhou_init, double e_init)
    : Element(id, sharedBasis, xL, xR) {
  int n = sharedBasis->getOrder() + 1;
  std::fill(rho,  rho  + n, rho_init);
  std::fill(rhou, rhou + n, rhou_init);
  std::fill(e,    e    + n, e_init);
}

/// External buffers: delegates to core with ownsMemory=false
Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
                 double *external_rho, double *external_rhou, double *external_e)
    : Element(id, sharedBasis, xL, xR, external_rho, external_rhou, external_e, false) {}

/**
 * @brief Sets the flux from the Euler system solved
 */
void Element::setFlux() {
  int n = basis->getOrder() + 1;
  double *p = new double[n];
  phy::getP(p, rho, rhou, e, n);
  phy::computeFlux(F1, F2, F3, rho, rhou, e, p, n);
  delete[] p;
}

void Element::computeLegendreCoefficients() {
  mat::computeLegendreCoeffs(legendreCoefficients, rho,
                             basis->getQuads(), basis->getWeights(),
                             basis->getOrder());
}

void Element::computeDivFlux() {
  divF(divF1, basis->getD(), F1, invJ, basis->getOrder() + 1);
  divF(divF2, basis->getD(), F2, invJ, basis->getOrder() + 1);
  divF(divF3, basis->getD(), F3, invJ, basis->getOrder() + 1);
}

Element::~Element() {
  if (ownsMemory) {
    delete[] rho;
    delete[] rhou;
    delete[] e;
  }
  delete[] F1;    delete[] F2;    delete[] F3;
  delete[] divF1; delete[] divF2; delete[] divF3;
  delete[] legendreCoefficients;
}

std::ostream &operator<<(std::ostream &os, const Element &e) {
  os << "----- ELEM -----" << std::endl
     << "ID  : " << e.id << std::endl
     << "xL  : " << e.xL << std::endl
     << "xR  : " << e.xR << std::endl;
  return os;
}
} // namespace elem
