#include <cblas.h>
#include <iostream>

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
          const int q) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, q, q, invJ, D, q, F, 1, 0., dFdx, 1);
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

Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR)
    : id(id), basis(sharedBasis) {
  this->setJ(xL, xR);
  this->ownsMemory = true;
  const int q = basis->getQ();
  rho = new double[q];
  rhou = new double[q];
  e = new double[q];
  F1 = new double[q];
  F2 = new double[q];
  F3 = new double[q];
  divF1 = new double[q];
  divF2 = new double[q];
  divF3 = new double[q];
}

Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
                 double rho_init, double rhou_init, double e_init)
    : id(id), basis(sharedBasis) {

  this->ownsMemory = true;
  const int q = basis->getQ();
  rho = new double[q];
  rhou = new double[q];
  e = new double[q];

  for (int i = 0; i < q; i++) {
    rho[i] = rho_init;
    rhou[i] = rhou_init;
    e[i] = e_init;
  }

  this->setJ(xL, xR);
  F1 = new double[q];
  F2 = new double[q];
  F3 = new double[q];
  divF1 = new double[q];
  divF2 = new double[q];
  divF3 = new double[q];
}

Element::Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
                 double *external_rho, double *external_rhou,
                 double *external_e)
    : id(id), basis(sharedBasis), rho(external_rho), rhou(external_rhou),
      e(external_e) {

  this->setJ(xL, xR);
  this->ownsMemory = false;
  const int q = basis->getQ();
  F1 = new double[q];
  F2 = new double[q];
  F3 = new double[q];
  divF1 = new double[q];
  divF2 = new double[q];
  divF3 = new double[q];
}

/**
 * @brief Sets the flux from the Euler system solved
 */
void Element::setFlux() {
  const int q = basis->getQ();
  double *p = new double[q];
  phy::getP(p, rho, rhou, e, q);
  phy::computeFlux(F1, F2, F3, rho, rhou, e, p, q);
  delete[] p;
}

void Element::computeDivFlux() {
  const int q = basis->getQ();
  divF(divF1, basis->getD(), F1, invJ, q);
  divF(divF2, basis->getD(), F2, invJ, q);
  divF(divF3, basis->getD(), F3, invJ, q);
}

Element::~Element() {
  if (ownsMemory) {
    delete[] rho;
    delete[] rhou;
    delete[] e;
  }
  delete[] F1;
  delete[] F2;
  delete[] F3;
  delete[] divF1;
  delete[] divF2;
  delete[] divF3;
}

std::ostream &operator<<(std::ostream &os, const Element &e) {
  os << "----- ELEM -----" << std::endl
     << "ID  : " << e.id << std::endl
     << "xL  : " << e.xL << std::endl
     << "xR  : " << e.xR << std::endl;
  return os;
}
} // namespace elem
